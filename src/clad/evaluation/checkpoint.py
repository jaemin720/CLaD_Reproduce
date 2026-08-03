"""Inference-only restoration of Stage 2 raw or EMA policy weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from clad.data.feature_cache import sha256_file
from clad.models import (
    CLaDDiffusionPolicy,
    CLaDForesightBackbone,
    CLaDStage2Conditioner,
    DiffusionPolicyConfig,
    Stage2ConditionerConfig,
)
from clad.training.stage2_trainer import (
    STAGE2_CHECKPOINT_SCHEMA_VERSION,
    ForesightCheckpointIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage2PolicyCheckpointInfo:
    """Identity and training position of one restored policy artifact."""

    path: Path
    sha256: str
    global_step: int
    attempt_step: int
    weights: str
    ema_optimization_step: int | None
    foresight_checkpoint: ForesightCheckpointIdentity


@dataclass(frozen=True, slots=True)
class LoadedStage2Policy:
    """Frozen inference policy and the metadata used to construct it."""

    model: CLaDDiffusionPolicy
    info: Stage2PolicyCheckpointInfo


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Stage 2 checkpoint {path} must contain a mapping")
    return payload


def _mapping_field(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage 2 checkpoint must contain a {name!r} mapping")
    return value


@torch.no_grad()
def _load_trainable_parameters(
    model: CLaDDiffusionPolicy,
    state: Mapping[str, Any],
) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if state.keys() != parameters.keys():
        missing = sorted(parameters.keys() - state.keys())
        unexpected = sorted(state.keys() - parameters.keys())
        raise ValueError(
            "Stage 2 inference parameter names do not match: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f"Invalid Stage 2 inference tensor for {name!r}")
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _resolve_backbone_dtype(
    value: str | torch.dtype,
    device: torch.device,
) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[value]
    except KeyError as error:
        raise ValueError(
            "backbone_dtype must be auto, float16, bfloat16, or float32"
        ) from error


def load_stage2_policy(
    checkpoint: str | Path,
    *,
    foresight_checkpoint: str | Path,
    device: torch.device | str = "cpu",
    weights: str = "ema",
    backbone_dtype: str | torch.dtype = "auto",
    verify_foresight: bool = True,
) -> LoadedStage2Policy:
    """Reconstruct a policy without loading optimizer state onto the device.

    The trainer checkpoint stores only Stage 2 trainable parameters and refers
    to the frozen Stage 1 artifact by hash. The full checkpoint is memory-mapped
    when supported, so the multi-gigabyte optimizer state is not materialized
    for inference.
    """

    if weights not in {"ema", "raw"}:
        raise ValueError(f"weights must be 'ema' or 'raw', got {weights!r}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage 2 checkpoint does not exist: {checkpoint_path}")
    foresight_path = Path(foresight_checkpoint).expanduser().resolve()
    if not foresight_path.is_file():
        raise FileNotFoundError(f"Frozen foresight checkpoint does not exist: {foresight_path}")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path)
    if int(payload.get("schema_version", -1)) != STAGE2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Stage 2 checkpoint schema: {payload.get('schema_version')!r}"
        )

    stored_foresight = ForesightCheckpointIdentity.from_mapping(
        _mapping_field(payload, "foresight_checkpoint")
    )
    actual_foresight = ForesightCheckpointIdentity.from_path(foresight_path)
    if verify_foresight and not actual_foresight.matches(stored_foresight):
        raise ValueError(
            "Frozen foresight checkpoint does not match the Stage 2 checkpoint: "
            f"expected sha256={stored_foresight.sha256}, "
            f"actual sha256={actual_foresight.sha256}"
        )

    policy_config = DiffusionPolicyConfig.from_mapping(
        _mapping_field(payload, "policy_config")
    )
    conditioner_config = Stage2ConditionerConfig.from_mapping(
        _mapping_field(payload, "conditioner_config")
    )
    backbone = CLaDForesightBackbone.from_checkpoint(
        foresight_path,
        dtype=_resolve_backbone_dtype(backbone_dtype, resolved_device),
    )
    model = CLaDDiffusionPolicy(
        conditioner=CLaDStage2Conditioner(
            backbone=backbone,
            config=conditioner_config,
        ),
        config=policy_config,
    )

    ema_optimization_step: int | None = None
    if weights == "ema":
        ema = _mapping_field(payload, "ema")
        state = _mapping_field(ema, "shadow")
        ema_optimization_step = int(ema.get("optimization_step", -1))
        if ema_optimization_step != int(payload["global_step"]):
            raise ValueError(
                "EMA optimization step does not match checkpoint global step: "
                f"{ema_optimization_step} != {payload['global_step']}"
            )
    else:
        state = _mapping_field(payload, "model_trainable")

    _load_trainable_parameters(model, state)
    model.action_normalizer.load_state_dict(_mapping_field(payload, "action_normalizer"))
    if not bool(model.action_normalizer.fitted.item()):
        raise ValueError("Stage 2 checkpoint contains an unfitted action normalizer")

    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)
    info = Stage2PolicyCheckpointInfo(
        path=checkpoint_path,
        sha256=checkpoint_sha256,
        global_step=int(payload["global_step"]),
        attempt_step=int(payload["attempt_step"]),
        weights=weights,
        ema_optimization_step=ema_optimization_step,
        foresight_checkpoint=actual_foresight,
    )
    return LoadedStage2Policy(model=model, info=info)
