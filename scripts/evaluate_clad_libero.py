#!/usr/bin/env python3
"""Evaluate a Stage 2 CLaD EMA policy on fixed LIBERO initial states."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from clad.data.feature_cache import MANIFEST_NAME, sha256_file
from clad.evaluation.checkpoint import load_stage2_policy
from clad.evaluation.libero_rollout import (
    EvaluationRecorder,
    LiberoRolloutConfig,
    evaluate_libero,
    require_libero_runtime,
)
from clad.evaluation.libero_setup import activate_libero_config
from clad.evaluation.online_policy import CLaDOnlinePolicy, OnlineDecisionNCEEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("configs/eval/libero_long.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/clad_stage2_official/stage2_latest.pt"),
    )
    parser.add_argument(
        "--foresight-checkpoint",
        type=Path,
        default=Path("outputs/clad_stage1_official/stage1_foresight.pt"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/decisionnce/libero_long"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clad_evaluation_official"),
    )
    parser.add_argument(
        "--libero-config-dir",
        type=Path,
        default=Path(".cache/libero"),
        help=(
            "Directory containing LIBERO config.yaml. An explicit "
            "LIBERO_CONFIG_PATH environment variable takes precedence."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--require-step",
        type=int,
        help="Refuse evaluation unless checkpoint global_step equals this value.",
    )
    parser.add_argument(
        "--frozen-backbone-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--rollouts-per-task", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--warmup-gripper-action", type=float)
    parser.add_argument("--execution-steps", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--render-gpu-device-id", type=int)
    parser.add_argument(
        "--camera",
        action="append",
        metavar="VIEW=OBS_KEY",
        help=(
            "Map a cached camera view to a live LIBERO observation key. "
            "May be repeated; defaults cover agentview_rgb and eye_in_hand_rgb."
        ),
    )
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Load and validate the policy checkpoint, then exit without LIBERO or DecisionNCE.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise TypeError(f"Configuration must contain a mapping: {resolved}")
    return dict(values)


def _rollout_config(args: argparse.Namespace) -> LiberoRolloutConfig:
    values = _load_yaml(args.eval_config)
    overrides = {
        "task_ids": args.task_ids,
        "rollouts_per_task": args.rollouts_per_task,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "warmup_gripper_action": args.warmup_gripper_action,
        "execution_steps": args.execution_steps,
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "seed": args.seed,
        "environment_seed": args.environment_seed,
        "render_gpu_device_id": args.render_gpu_device_id,
        "save_videos": args.save_videos,
        "resume": args.resume,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    return LiberoRolloutConfig.from_mapping(values)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _camera_mapping(values: list[str] | None) -> dict[str, str] | None:
    if values is None:
        return None
    mapping: dict[str, str] = {}
    for value in values:
        view, separator, observation_key = value.partition("=")
        if not separator or not view or not observation_key:
            raise ValueError(f"--camera must use VIEW=OBS_KEY syntax, got {value!r}")
        if view in mapping:
            raise ValueError(f"Duplicate --camera view: {view!r}")
        mapping[view] = observation_key
    return mapping


def _checkpoint_summary(loaded: Any) -> dict[str, Any]:
    info = loaded.info
    summary = {
        "checkpoint": str(info.path),
        "checkpoint_sha256": info.sha256,
        "global_step": info.global_step,
        "attempt_step": info.attempt_step,
        "weights": info.weights,
        "ema_optimization_step": info.ema_optimization_step,
        "policy_variant": info.policy_variant,
        "policy_config": asdict(loaded.model.config),
        "conditioner_config": asdict(loaded.model.conditioner.config),
        "parameters_total": sum(parameter.numel() for parameter in loaded.model.parameters()),
    }
    if info.foresight_checkpoint is not None:
        summary.update(
            {
                "foresight_checkpoint": str(info.foresight_checkpoint.path),
                "foresight_sha256": info.foresight_checkpoint.sha256,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    config = _rollout_config(args)
    if not args.checkpoint_only and config.num_envs > 1:
        start_method = multiprocessing.get_start_method(allow_none=True)
        if start_method is None:
            multiprocessing.set_start_method("spawn")
        elif start_method != "spawn":
            raise RuntimeError(
                "Parallel LIBERO evaluation requires multiprocessing start method "
                f"'spawn', but {start_method!r} is already active"
            )
    device = _resolve_device(args.device)
    if not args.checkpoint_only:
        # Fail quickly before constructing a 0.66B-parameter model.
        activate_libero_config(args.libero_config_dir)
        require_libero_runtime()
    loaded = load_stage2_policy(
        args.checkpoint,
        foresight_checkpoint=args.foresight_checkpoint,
        device=device,
        weights=args.weights,
        backbone_dtype=args.frozen_backbone_dtype,
    )
    if args.require_step is not None and loaded.info.global_step != args.require_step:
        raise ValueError(
            "Checkpoint global step does not match --require-step: "
            f"{loaded.info.global_step} != {args.require_step}"
        )
    checkpoint_summary = _checkpoint_summary(loaded)
    print("Diffusion policy checkpoint loaded", flush=True)
    print(
        f"  variant={loaded.info.policy_variant} | step={loaded.info.global_step:,} | "
        f"weights={loaded.info.weights} | "
        f"device={device} | parameters={checkpoint_summary['parameters_total']:,}",
        flush=True,
    )
    print(f"  checkpoint_sha256={loaded.info.sha256}", flush=True)
    if loaded.info.foresight_checkpoint is not None:
        print(
            f"  foresight_sha256={loaded.info.foresight_checkpoint.sha256}",
            flush=True,
        )
    if args.checkpoint_only:
        print(json.dumps(checkpoint_summary, indent=2, sort_keys=True), flush=True)
        return

    if config.execution_steps > loaded.model.config.horizon:
        raise ValueError(
            f"execution_steps={config.execution_steps} exceeds policy horizon "
            f"{loaded.model.config.horizon}"
        )
    camera_mapping = _camera_mapping(args.camera)
    encoder = OnlineDecisionNCEEncoder.from_feature_cache(
        args.cache_dir,
        device=str(device),
        camera_observation_keys=camera_mapping,
        proprioception=loaded.model.conditioner.input_config.proprioception,
    )
    try:
        evaluation_image_size = (config.camera_height, config.camera_width)
        if (
            encoder.source_image_size is not None
            and evaluation_image_size != encoder.source_image_size
        ):
            raise ValueError(
                "Evaluation render size must match the regenerated training data: "
                f"evaluation={evaluation_image_size}, "
                f"training={encoder.source_image_size}. Pass --camera-height and "
                "--camera-width explicitly."
            )
        policy = CLaDOnlinePolicy(
            model=loaded.model,
            encoder=encoder,
            execution_steps=config.execution_steps,
            amp_enabled=args.amp,
        )
        manifest_path = args.cache_dir.expanduser().resolve() / MANIFEST_NAME
        run_identity = {
            **checkpoint_summary,
            "feature_cache_manifest": str(manifest_path),
            "feature_cache_manifest_sha256": sha256_file(manifest_path),
            "rollout_protocol": {
                name: value
                for name, value in asdict(config).items()
                if name not in {"rollouts_per_task", "task_ids", "save_videos", "resume"}
            },
            "camera_observation_keys": (
                camera_mapping if camera_mapping is not None else encoder.camera_observation_keys
            ),
            "proprioception": encoder.proprioception,
            "image_transform": encoder.image_transform,
            "source_image_size": encoder.source_image_size,
        }
        recorder = EvaluationRecorder(
            args.output_dir,
            run_identity=run_identity,
            resume=config.resume,
        )
        print(
            f"{loaded.info.policy_variant} LIBERO rollout evaluation",
            flush=True,
        )
        print(
            f"  suite={config.suite_name} | tasks="
            f"{list(config.task_ids) if config.task_ids else 'all'} | "
            f"rollouts_per_task={config.rollouts_per_task} | num_envs={config.num_envs}",
            flush=True,
        )
        print(
            f"  max_steps={config.max_steps} | warmup={config.warmup_steps} | "
            f"warmup_gripper={config.warmup_gripper_action:+.1f} | "
            f"execute={config.execution_steps}/{loaded.model.config.horizon}",
            flush=True,
        )
        print(
            f"  environment_seed={config.environment_seed} | "
            f"policy_seed_base={config.seed}",
            flush=True,
        )
        print(
            f"  render={config.camera_height}x{config.camera_width} | "
            f"image_transform={encoder.image_transform}",
            flush=True,
        )
        print(f"  results={recorder.results_path}", flush=True)
        summary = evaluate_libero(policy=policy, config=config, recorder=recorder)
        macro_rate = summary["macro_task_success_rate"]
        print(
            "Evaluation finished | "
            f"episodes={summary['completed_rollouts']} | "
            f"macro_success_rate={100.0 * macro_rate:.1f}% | "
            f"summary={recorder.summary_path}",
            flush=True,
        )
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
