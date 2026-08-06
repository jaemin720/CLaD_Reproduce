#!/usr/bin/env python3
"""Train the Stage 2 foresight-conditioned diffusion policy."""

from __future__ import annotations

import argparse
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
    compute_libero_action_bounds,
)
from clad.models import (
    CLaDDiffusionPolicy,
    CLaDForesightBackbone,
    CLaDStage2Conditioner,
    DiffusionPolicyConfig,
    PolicyOnlyConditioner,
    PolicyOnlyConditionerConfig,
    Stage2ConditionerConfig,
)
from clad.proprioception import proprioception_spec
from clad.training import (
    ForesightCheckpointIdentity,
    Stage2MetricLogger,
    Stage2Trainer,
    Stage2TrainerConfig,
    build_stage2_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/libero_long.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/clad_stage2.yaml"),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train/stage2.yaml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/decisionnce/libero_long"),
    )
    parser.add_argument(
        "--foresight-checkpoint",
        type=Path,
        default=Path("outputs/clad_stage1_official/stage1_foresight.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clad_stage2_official"),
    )
    parser.add_argument("--file-pattern")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--policy-variant",
        choices=("clad", "policy_only"),
        default="clad",
        help="Use frozen CLaD foresight or the current-observation-only ablation.",
    )
    parser.add_argument(
        "--frozen-backbone-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
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
    parser.add_argument(
        "--down-dims",
        type=int,
        nargs="+",
        help="Debug override for U-Net widths; changes the reproduction model.",
    )
    parser.add_argument(
        "--diffusion-timesteps",
        type=int,
        help="Debug override for DDPM steps; changes the reproduction model.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
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


def _trainer_config(args: argparse.Namespace) -> Stage2TrainerConfig:
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
        "ema_enabled": args.ema,
        "save_final_checkpoint": args.save_final_checkpoint,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    if args.max_steps is not None and args.warmup_steps is None:
        values["warmup_steps"] = min(int(values["warmup_steps"]), args.max_steps - 1)
    return Stage2TrainerConfig.from_mapping(values)


def _dataset_config(args: argparse.Namespace) -> LiberoDatasetConfig:
    values = _load_yaml(args.data_config)
    if args.dataset_dir is not None:
        values["dataset_dir"] = args.dataset_dir
    if args.file_pattern is not None:
        values["file_pattern"] = args.file_pattern
    values["include_images"] = False
    unknown = sorted(set(values) - set(LiberoDatasetConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown LIBERO dataset settings: {unknown}")
    return LiberoDatasetConfig(**values)


def _model_configs(
    args: argparse.Namespace,
) -> tuple[Stage2ConditionerConfig | PolicyOnlyConditionerConfig, DiffusionPolicyConfig]:
    values = _load_yaml(args.model_config)
    expected = {"conditioning", "diffusion"}
    if set(values) != expected:
        raise ValueError(
            f"Stage 2 model config must contain exactly {sorted(expected)}, got {sorted(values)}"
        )
    conditioning_values = values["conditioning"]
    diffusion_values = values["diffusion"]
    if not isinstance(conditioning_values, Mapping) or not isinstance(diffusion_values, Mapping):
        raise TypeError("conditioning and diffusion configs must be mappings")
    if args.policy_variant == "clad":
        conditioner_config = Stage2ConditionerConfig.from_mapping(conditioning_values)
    else:
        conditioner_config = PolicyOnlyConditionerConfig.from_mapping(conditioning_values)
    diffusion_config = DiffusionPolicyConfig.from_mapping(diffusion_values)
    if args.down_dims is not None:
        diffusion_config = replace(diffusion_config, down_dims=tuple(args.down_dims))
    if args.diffusion_timesteps is not None:
        diffusion_config = replace(
            diffusion_config,
            num_train_timesteps=args.diffusion_timesteps,
        )
    return conditioner_config, diffusion_config


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _backbone_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def _parameter_counts(model: CLaDDiffusionPolicy) -> dict[str, int]:
    frozen_clad = 0
    backbone = getattr(model.conditioner, "backbone", None)
    if backbone is not None:
        frozen_clad = sum(parameter.numel() for parameter in backbone.parameters())
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "parameters_frozen_clad": frozen_clad,
        "parameters_denoiser": sum(parameter.numel() for parameter in model.denoiser.parameters()),
        "parameters_conditioner_trainable": sum(
            parameter.numel()
            for parameter in model.conditioner.parameters()
            if parameter.requires_grad
        ),
        # Retain the original run-config field for downstream log readers.
        "parameters_conditioning_film": sum(
            parameter.numel()
            for name, parameter in model.conditioner.named_parameters()
            if parameter.requires_grad and "film." in name
        ),
    }


def main() -> None:
    args = parse_args()
    trainer_config = _trainer_config(args)
    dataset_config = _dataset_config(args)
    conditioner_config, diffusion_config = _model_configs(args)
    if dataset_config.horizon != diffusion_config.horizon:
        raise ValueError(
            "Dataset and policy horizons must match: "
            f"{dataset_config.horizon} != {diffusion_config.horizon}"
        )

    device = _resolve_device(args.device)
    generator = Stage2Trainer.seed_everything(trainer_config.seed)
    base_dataset = LiberoWindowDataset(dataset_config)
    action_bounds = compute_libero_action_bounds(
        base_dataset,
        expected_action_dim=diffusion_config.action_dim,
    )
    dataset = CachedLiberoWindowDataset(
        base_dataset=base_dataset,
        feature_cache=DecisionNCEFeatureCache(args.cache_dir),
        include_future_features=False,
    )
    if args.policy_variant == "policy_only":
        assert isinstance(conditioner_config, PolicyOnlyConditionerConfig)
        if dataset.view_names != conditioner_config.camera_views:
            raise ValueError(
                "Policy-only model and dataset camera views must match exactly: "
                f"model={conditioner_config.camera_views}, dataset={dataset.view_names}"
            )
        if dataset_config.proprioception != conditioner_config.proprioception:
            raise ValueError(
                "Policy-only model and dataset proprioception contracts must match: "
                f"model={conditioner_config.proprioception!r}, "
                f"dataset={dataset_config.proprioception!r}"
            )
        proprio_dim = proprioception_spec(dataset_config.proprioception).dimension
        if proprio_dim != conditioner_config.proprio_dim:
            raise ValueError(
                "Policy-only proprioception dimension does not match the model: "
                f"contract={proprio_dim}, model={conditioner_config.proprio_dim}"
            )
    metric_logger: Stage2MetricLogger | None = None
    try:
        dataloader = build_stage2_dataloader(
            dataset,
            trainer_config,
            generator=generator,
        )
        foresight_identity: ForesightCheckpointIdentity | None = None
        backbone_dtype: str | None = None
        if args.policy_variant == "clad":
            assert isinstance(conditioner_config, Stage2ConditionerConfig)
            foresight_identity = ForesightCheckpointIdentity.from_path(
                args.foresight_checkpoint
            )
            backbone = CLaDForesightBackbone.from_checkpoint(
                args.foresight_checkpoint,
                dtype=_backbone_dtype(args.frozen_backbone_dtype, device),
            )
            backbone_dtype = str(next(backbone.parameters()).dtype)
            if dataset_config.proprioception != backbone.config.inputs.proprioception:
                raise ValueError(
                    "CLaD Stage 1 and Stage 2 dataset proprioception contracts must match: "
                    f"stage1={backbone.config.inputs.proprioception!r}, "
                    f"dataset={dataset_config.proprioception!r}"
                )
            conditioner = CLaDStage2Conditioner(
                backbone=backbone,
                config=conditioner_config,
            )
        else:
            assert isinstance(conditioner_config, PolicyOnlyConditionerConfig)
            conditioner = PolicyOnlyConditioner(conditioner_config)
        model = CLaDDiffusionPolicy(
            conditioner=conditioner,
            config=diffusion_config,
        )
        model.action_normalizer.fit_from_bounds(
            action_bounds.minimum,
            action_bounds.maximum,
        )
        parameter_counts = _parameter_counts(model)
        metric_logger = Stage2MetricLogger(
            output_dir=args.output_dir,
            max_steps=trainer_config.max_steps,
        )
        trainer = Stage2Trainer(
            model=model,
            dataloader=dataloader,
            config=trainer_config,
            device=device,
            output_dir=args.output_dir,
            foresight_checkpoint=foresight_identity,
            metric_callback=metric_logger,
        )
        if args.resume is not None:
            resumed_step = trainer.load_checkpoint(args.resume)
            print(f"Resumed checkpoint at optimizer step {resumed_step}.", flush=True)
        metric_logger.start(trainer.global_step)

        run_summary = {
            "starting_global_step": trainer.global_step,
            "starting_attempt_step": trainer.attempt_step,
            "dataset_windows": len(dataset),
            "source_action_count": action_bounds.count,
            "action_minimum": action_bounds.minimum.tolist(),
            "action_maximum": action_bounds.maximum.tolist(),
            "device": str(device),
            "policy_variant": args.policy_variant,
            "frozen_backbone_dtype": backbone_dtype,
            "dataset_config": asdict(dataset_config),
            "conditioner_config": asdict(conditioner_config),
            "diffusion_config": asdict(diffusion_config),
            "trainer_config": asdict(trainer_config),
            "foresight_checkpoint": (
                asdict(foresight_identity) if foresight_identity is not None else None
            ),
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            "resume": str(args.resume.expanduser().resolve()) if args.resume else None,
            **parameter_counts,
        }
        config_path = metric_logger.write_run_config(run_summary)
        effective_batch_size = (
            trainer_config.batch_size * trainer_config.gradient_accumulation_steps
        )
        print(f"{args.policy_variant} diffusion policy training", flush=True)
        print(
            f"  device={device} | windows={len(dataset):,} | "
            f"trainable_params={parameter_counts['parameters_trainable']:,}",
            flush=True,
        )
        print(
            f"  steps={trainer_config.max_steps:,} | "
            f"micro_batch={trainer_config.batch_size} | "
            f"accumulation={trainer_config.gradient_accumulation_steps} | "
            f"effective_batch={effective_batch_size}",
            flush=True,
        )
        upstream = (
            f"foresight_sha256={foresight_identity.sha256}"
            if foresight_identity is not None
            else "foresight=disabled"
        )
        print(
            f"  action_samples={action_bounds.count:,} | {upstream}",
            flush=True,
        )
        print(f"  metrics={metric_logger.metrics_path}", flush=True)
        print(f"  config={config_path}", flush=True)
        result = trainer.train()
        print(
            "Stage 2 finished | "
            f"step={result.global_step} | attempts={result.attempt_step} | "
            f"skips={result.skipped_optimizer_steps} | "
            f"checkpoint={result.checkpoint_path}",
            flush=True,
        )
    finally:
        if metric_logger is not None:
            metric_logger.close()
        dataset.close()


if __name__ == "__main__":
    main()
