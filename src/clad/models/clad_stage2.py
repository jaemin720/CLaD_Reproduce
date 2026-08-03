"""Frozen CLaD inference and Stage 2 foresight conditioning.

This module implements the bridge between Stage 1 and the diffusion policy:
the history-only CLaD path predicts modality-specific future latents, and
equations (20)--(21) modulate those latents with the current observations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from clad.models.clad_dynamics import (
    CrossModalDynamicsEncoder,
    CrossModalDynamicsOutput,
)
from clad.models.clad_foresight import (
    GroundedForesightPredictor,
    LatentForesight,
)
from clad.models.clad_inputs import (
    ActionTokenOutput,
    CLaDInputEncoders,
    VisionFeatures,
)
from clad.models.clad_stage1 import CLaDStage1Config
from clad.models.clad_transition import (
    CLaDTransitionEncoders,
    CLaDTransitionOutput,
)

FROZEN_FORESIGHT_ARTIFACT_TYPE = "clad_frozen_foresight"
FROZEN_FORESIGHT_SCHEMA_VERSION = 1
_FORESIGHT_STATE_PREFIXES = (
    "inputs.",
    "transitions.",
    "dynamics.",
    "foresight_predictor.",
)


@dataclass(frozen=True, slots=True)
class CLaDHistoryBatch:
    """CLaD inputs available online, without future supervision fields."""

    vision_prev: VisionFeatures
    vision_now: VisionFeatures
    text_features: torch.Tensor
    proprio_prev: torch.Tensor
    proprio_now: torch.Tensor
    past_actions: torch.Tensor

    @staticmethod
    def _vision_at(
        vision_features: Mapping[str, Any],
        timestep: str,
    ) -> dict[str, torch.Tensor]:
        views: dict[str, torch.Tensor] = {}
        for view_name, timeline in vision_features.items():
            if not isinstance(timeline, Mapping) or timestep not in timeline:
                raise ValueError(f"vision_features[{view_name!r}] must contain {timestep!r}")
            feature = timeline[timestep]
            if not isinstance(feature, torch.Tensor):
                raise TypeError(f"vision_features[{view_name!r}][{timestep!r}] must be a Tensor")
            views[view_name] = feature
        if not views:
            raise ValueError("vision_features cannot be empty")
        return views

    @classmethod
    def from_mapping(cls, batch: Mapping[str, Any]) -> CLaDHistoryBatch:
        """Extract only the history fields from a collated dataset batch."""

        vision_features = batch.get("vision_features")
        if not isinstance(vision_features, Mapping):
            raise ValueError("batch must contain a vision_features mapping")
        tensor_fields = {
            "text_features": batch.get("text_feature"),
            "proprio_prev": batch.get("proprio_prev"),
            "proprio_now": batch.get("proprio_now"),
            "past_actions": batch.get("past_actions"),
        }
        for name, value in tensor_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"batch field {name!r} must be a Tensor")
        return cls(
            vision_prev=cls._vision_at(vision_features, "prev"),
            vision_now=cls._vision_at(vision_features, "now"),
            text_features=tensor_fields["text_features"],
            proprio_prev=tensor_fields["proprio_prev"],
            proprio_now=tensor_fields["proprio_now"],
            past_actions=tensor_fields["past_actions"],
        )

    @staticmethod
    def _move_vision(
        features: VisionFeatures,
        device: torch.device | str,
        *,
        non_blocking: bool,
    ) -> VisionFeatures:
        if isinstance(features, Mapping):
            return {
                name: tensor.to(device=device, non_blocking=non_blocking)
                for name, tensor in features.items()
            }
        return features.to(device=device, non_blocking=non_blocking)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CLaDHistoryBatch:
        """Move all online-history tensors to one device."""

        return CLaDHistoryBatch(
            vision_prev=self._move_vision(
                self.vision_prev,
                device,
                non_blocking=non_blocking,
            ),
            vision_now=self._move_vision(
                self.vision_now,
                device,
                non_blocking=non_blocking,
            ),
            text_features=self.text_features.to(
                device=device,
                non_blocking=non_blocking,
            ),
            proprio_prev=self.proprio_prev.to(
                device=device,
                non_blocking=non_blocking,
            ),
            proprio_now=self.proprio_now.to(
                device=device,
                non_blocking=non_blocking,
            ),
            past_actions=self.past_actions.to(
                device=device,
                non_blocking=non_blocking,
            ),
        )


@dataclass(frozen=True, slots=True)
class CLaDForesightBackboneOutput:
    """Frozen Stage 1 products required by policy conditioning."""

    foresight: LatentForesight
    proprio_now: torch.Tensor
    semantic_now: torch.Tensor
    transitions: CLaDTransitionOutput
    dynamics: CrossModalDynamicsOutput
    actions: ActionTokenOutput


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        # PyTorch 2.2 accepts mmap only when the filename is a plain string.
        # mmap avoids eagerly materializing the multi-gigabyte optimizer state
        # while exporting the already-completed Stage 1 checkpoint.
        payload = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {path} must contain a mapping")
    return payload


def _checkpoint_config(payload: Mapping[str, Any]) -> CLaDStage1Config:
    values = payload.get("model_config")
    if not isinstance(values, Mapping):
        raise ValueError("Checkpoint must contain a model_config mapping")
    return CLaDStage1Config.from_checkpoint_mapping(values)


def _foresight_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint must contain a model state mapping")
    selected: dict[str, torch.Tensor] = {}
    for name, value in model_state.items():
        if not isinstance(name, str):
            raise TypeError("Model state keys must be strings")
        if name.startswith(_FORESIGHT_STATE_PREFIXES):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Model state value {name!r} must be a Tensor")
            selected[name] = value
    missing_groups = [
        prefix
        for prefix in _FORESIGHT_STATE_PREFIXES
        if not any(name.startswith(prefix) for name in selected)
    ]
    if missing_groups:
        raise ValueError(
            f"Checkpoint is missing frozen foresight parameter groups: {missing_groups}"
        )
    return selected


class CLaDForesightBackbone(nn.Module):
    """Inference-only subset of Stage 1 CLaD.

    Future target encoders, reconstruction heads, and losses are deliberately
    excluded. Action masking is always disabled because Stage 2 consumes the
    observed action history at both training and execution time.
    """

    def __init__(self, config: CLaDStage1Config) -> None:
        super().__init__()
        self.config = config
        self.inputs = CLaDInputEncoders(config.inputs)
        self.transitions = CLaDTransitionEncoders(config.attention)
        self.dynamics = CrossModalDynamicsEncoder(config.attention)
        self.foresight_predictor = GroundedForesightPredictor(config.foresight)
        self._freeze()

    def _freeze(self) -> None:
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> CLaDForesightBackbone:
        """Keep the Stage 1 backbone frozen and deterministic."""

        del mode
        self._freeze()
        return self

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CLaDForesightBackbone:
        """Load either a full Stage 1 or compact foresight checkpoint."""

        checkpoint_path = Path(path).expanduser().resolve()
        payload = _load_checkpoint(checkpoint_path)
        artifact_type = payload.get("artifact_type")
        if artifact_type is not None:
            if artifact_type != FROZEN_FORESIGHT_ARTIFACT_TYPE:
                raise ValueError(f"Unsupported checkpoint artifact type: {artifact_type!r}")
            if int(payload.get("schema_version", -1)) != FROZEN_FORESIGHT_SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported frozen foresight checkpoint schema: "
                    f"{payload.get('schema_version')!r}"
                )
        elif int(payload.get("schema_version", -1)) not in {1, 2}:
            raise ValueError(
                f"Unsupported Stage 1 checkpoint schema: {payload.get('schema_version')!r}"
            )

        backbone = cls(_checkpoint_config(payload))
        backbone.load_state_dict(_foresight_state_dict(payload), strict=True)
        if device is not None or dtype is not None:
            backbone.to(device=device, dtype=dtype)
        backbone._freeze()
        return backbone

    @torch.no_grad()
    def forward(
        self,
        batch: CLaDHistoryBatch | Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> CLaDForesightBackboneOutput:
        if isinstance(batch, Mapping):
            batch = CLaDHistoryBatch.from_mapping(batch)
        if not isinstance(batch, CLaDHistoryBatch):
            raise TypeError("batch must be CLaDHistoryBatch or a collated mapping")

        semantic_prev = self.inputs.semantic(
            batch.vision_prev,
            batch.text_features,
        )
        semantic_now = self.inputs.semantic(
            batch.vision_now,
            batch.text_features,
        )
        proprio_prev = self.inputs.proprio(batch.proprio_prev)
        proprio_now = self.inputs.proprio(batch.proprio_now)
        actions = self.inputs.action(batch.past_actions, mask_actions=False)
        transitions = self.transitions(
            proprio_now=proprio_now,
            proprio_past=proprio_prev,
            semantic_now=semantic_now,
            semantic_past=semantic_prev,
            action_tokens=actions.tokens,
            return_attention=return_attention,
        )
        dynamics = self.dynamics(
            transitions.proprio.tokens,
            transitions.semantic.tokens,
            return_attention=return_attention,
        )
        return CLaDForesightBackboneOutput(
            foresight=self.foresight_predictor(dynamics.z_dyn),
            proprio_now=proprio_now,
            semantic_now=semantic_now,
            transitions=transitions,
            dynamics=dynamics,
            actions=actions,
        )


@dataclass(frozen=True, slots=True)
class Stage2ConditionerConfig:
    """Configuration for observation-modulated foresight."""

    observation_pooling: str = "mean"
    film_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.observation_pooling != "mean":
            raise ValueError(
                "Only observation_pooling='mean' is currently implemented, "
                f"got {self.observation_pooling!r}"
            )
        if not 0.0 <= self.film_dropout < 1.0:
            raise ValueError(f"film_dropout must be in [0, 1), got {self.film_dropout}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Stage2ConditionerConfig:
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown Stage 2 conditioner settings: {unknown}")
        return cls(**dict(values))


class LatentFiLM(nn.Module):
    """Affine-modulate one latent using its current observation embedding."""

    def __init__(
        self,
        *,
        feature_dim: int,
        condition_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or condition_dim <= 0:
            raise ValueError("feature_dim and condition_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.condition_dropout = nn.Dropout(dropout)
        self.affine = nn.Linear(condition_dim, 2 * feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Start equation (21) as an identity map. Stage 2 can then learn how
        # much present context should adjust each already-trained foresight.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [B, {self.feature_dim}], got {tuple(features.shape)}"
            )
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if features.shape[0] != condition.shape[0]:
            raise ValueError("Feature and condition batch sizes must match")
        parameter = self.affine.weight
        features = features.to(dtype=parameter.dtype)
        condition = condition.to(dtype=parameter.dtype)
        delta_scale, shift = self.affine(self.condition_dropout(condition)).chunk(2, dim=-1)
        return features * (1.0 + delta_scale) + shift


@dataclass(frozen=True, slots=True)
class Stage2ConditioningOutput:
    """Equation (21) outputs supplied to the diffusion policy."""

    proprio: torch.Tensor
    semantic: torch.Tensor
    proprio_observation: torch.Tensor
    semantic_observation: torch.Tensor
    foresight: LatentForesight

    @property
    def combined(self) -> torch.Tensor:
        return torch.cat((self.proprio, self.semantic), dim=-1)


class CLaDStage2Conditioner(nn.Module):
    """Produce ``g_p`` and ``g_s`` from a frozen Stage 1 CLaD model."""

    def __init__(
        self,
        *,
        backbone: CLaDForesightBackbone,
        config: Stage2ConditionerConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or Stage2ConditionerConfig()
        hidden_dim = backbone.config.inputs.hidden_dim
        self.proprio_film = LatentFiLM(
            feature_dim=hidden_dim,
            condition_dim=hidden_dim,
            dropout=self.config.film_dropout,
        )
        self.semantic_film = LatentFiLM(
            feature_dim=hidden_dim,
            condition_dim=hidden_dim,
            dropout=self.config.film_dropout,
        )

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(
        self,
        batch: CLaDHistoryBatch | Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> Stage2ConditioningOutput:
        frozen = self.backbone(batch, return_attention=return_attention)
        proprio_observation = self._pool(frozen.proprio_now)
        semantic_observation = self._pool(frozen.semantic_now)
        return Stage2ConditioningOutput(
            proprio=self.proprio_film(
                frozen.foresight.proprio,
                proprio_observation,
            ),
            semantic=self.semantic_film(
                frozen.foresight.semantic,
                semantic_observation,
            ),
            proprio_observation=proprio_observation,
            semantic_observation=semantic_observation,
            foresight=frozen.foresight,
        )


@dataclass(frozen=True, slots=True)
class ForesightCheckpointInfo:
    """Summary returned after exporting a compact inference artifact."""

    path: Path
    source_path: Path
    global_step: int
    tensor_count: int


def export_frozen_foresight_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> ForesightCheckpointInfo:
    """Strip optimizer and Stage 1-only modules from a training checkpoint."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("Source and destination checkpoints must be different")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination_path}; pass overwrite=True to replace it"
        )

    payload = _load_checkpoint(source_path)
    if payload.get("artifact_type") is not None:
        raise ValueError("Source must be a full Stage 1 training checkpoint")
    if int(payload.get("schema_version", -1)) not in {1, 2}:
        raise ValueError(
            f"Unsupported Stage 1 checkpoint schema: {payload.get('schema_version')!r}"
        )
    config = _checkpoint_config(payload)
    model_state = _foresight_state_dict(payload)
    global_step = int(payload.get("global_step", 0))

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    compact_payload = {
        "artifact_type": FROZEN_FORESIGHT_ARTIFACT_TYPE,
        "schema_version": FROZEN_FORESIGHT_SCHEMA_VERSION,
        "source_checkpoint": str(source_path),
        "source_schema_version": int(payload["schema_version"]),
        "global_step": global_step,
        "model_config": asdict(config),
        "model": model_state,
    }
    try:
        torch.save(compact_payload, temporary)
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return ForesightCheckpointInfo(
        path=destination_path,
        source_path=source_path,
        global_step=global_step,
        tensor_count=len(model_state),
    )
