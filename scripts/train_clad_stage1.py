#!/usr/bin/env python3
"""Train Stage 1 Cross-Modal Latent Dynamics from cached VLM features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from clad.data import (
    CachedLiberoWindowDataset,
    DecisionNCEFeatureCache,
    LiberoDatasetConfig,
    LiberoWindowDataset,
)
from clad.models import CLaDStage1Config, CLaDStage1Model
from clad.training import Stage1Trainer, Stage1TrainerConfig, build_stage1_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLaD Stage 1 using a precomputed DecisionNCE feature cache."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/libero_long.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/clad_stage1.yaml"),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train/stage1.yaml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/decisionnce/libero_long"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clad_stage1"))
    parser.add_argument("--file-pattern")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--amp-init-scale", type=float)
    parser.add_argument("--max-consecutive-optimizer-skips", type=int)
    parser.add_argument("--attention-layers", type=int)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise TypeError(f"Configuration must contain a mapping: {resolved}")
    return dict(values)


def _trainer_config(args: argparse.Namespace) -> Stage1TrainerConfig:
    values = _load_yaml(args.train_config)
    overrides = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_workers": args.num_workers,
        "log_interval": args.log_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "amp_init_scale": args.amp_init_scale,
        "max_consecutive_optimizer_skips": args.max_consecutive_optimizer_skips,
        "amp_enabled": args.amp,
        "save_final_checkpoint": args.save_final_checkpoint,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})

    # A short smoke run should not fail only because the full-run warmup is
    # longer. An explicit --warmup-steps remains authoritative.
    if args.max_steps is not None and args.warmup_steps is None:
        values["warmup_steps"] = min(int(values["warmup_steps"]), args.max_steps - 1)
    return Stage1TrainerConfig.from_mapping(values)


def _dataset_config(args: argparse.Namespace) -> LiberoDatasetConfig:
    values = _load_yaml(args.data_config)
    if args.dataset_dir is not None:
        values["dataset_dir"] = args.dataset_dir
    if args.file_pattern is not None:
        values["file_pattern"] = args.file_pattern
    values["include_images"] = False
    known = set(LiberoDatasetConfig.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown LIBERO dataset settings: {unknown}")
    return LiberoDatasetConfig(**values)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parameter_counts(model: CLaDStage1Model) -> dict[str, int]:
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "parameters_ema_target": sum(
            parameter.numel() for parameter in model.target_encoders.parameters()
        ),
    }


def main() -> None:
    args = parse_args()
    trainer_config = _trainer_config(args)
    dataset_config = _dataset_config(args)
    model_config = CLaDStage1Config.from_mapping(_load_yaml(args.model_config))
    if args.attention_layers is not None:
        model_config = replace(
            model_config,
            attention=replace(model_config.attention, num_layers=args.attention_layers),
        )
    if dataset_config.horizon != model_config.inputs.horizon:
        raise ValueError(
            "Dataset and model horizons must match: "
            f"{dataset_config.horizon} != {model_config.inputs.horizon}"
        )

    device = _resolve_device(args.device)
    generator = Stage1Trainer.seed_everything(trainer_config.seed)
    base_dataset = LiberoWindowDataset(dataset_config)
    dataset = CachedLiberoWindowDataset(
        base_dataset=base_dataset,
        feature_cache=DecisionNCEFeatureCache(args.cache_dir),
    )
    try:
        dataloader = build_stage1_dataloader(
            dataset,
            trainer_config,
            generator=generator,
        )
        model = CLaDStage1Model(model_config)
        trainer = Stage1Trainer(
            model=model,
            dataloader=dataloader,
            config=trainer_config,
            device=device,
            output_dir=args.output_dir,
        )
        if args.resume is not None:
            resumed_step = trainer.load_checkpoint(args.resume)
            print(f"resumed checkpoint at step {resumed_step}", flush=True)

        run_summary = {
            "dataset_windows": len(dataset),
            "device": str(device),
            "model_config": asdict(model_config),
            "trainer_config": asdict(trainer_config),
            **_parameter_counts(model),
        }
        print(json.dumps(run_summary, sort_keys=True), flush=True)
        result = trainer.train()
        print(
            json.dumps(
                {
                    "completed_step": result.global_step,
                    "attempt_step": result.attempt_step,
                    "skipped_optimizer_steps": result.skipped_optimizer_steps,
                    "checkpoint": (
                        str(result.checkpoint_path) if result.checkpoint_path is not None else None
                    ),
                    "latest_metrics": result.latest_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
