"""Composed Stage 1 CLaD model and its training-batch contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from clad.models.clad_dynamics import (
    CrossModalDynamicsEncoder,
    CrossModalDynamicsOutput,
)
from clad.models.clad_foresight import (
    CLaDLossOutput,
    CLaDObjective,
    EMAStateEncoders,
    ForesightConfig,
    ForesightReconstructionHeads,
    ForesightReconstructions,
    ForesightTargets,
    GroundedForesightPredictor,
    LatentForesight,
)
from clad.models.clad_inputs import (
    ActionTokenOutput,
    CLaDInputEncoderConfig,
    CLaDInputEncoders,
    VisionFeatures,
)
from clad.models.clad_transition import (
    CLaDTransitionEncoders,
    CLaDTransitionOutput,
    CrossAttentionConfig,
)


@dataclass(frozen=True, slots=True)
class CLaDStage1Config:
    """Consistent nested configuration for every Stage 1 component."""

    inputs: CLaDInputEncoderConfig
    attention: CrossAttentionConfig
    foresight: ForesightConfig

    def __post_init__(self) -> None:
        hidden_dimensions = {
            self.inputs.hidden_dim,
            self.attention.hidden_dim,
            self.foresight.hidden_dim,
        }
        if len(hidden_dimensions) != 1:
            raise ValueError(
                "Input, attention, and foresight hidden dimensions must match, "
                f"got {sorted(hidden_dimensions)}"
            )
        if self.inputs.proprio_dim != self.foresight.proprio_dim:
            raise ValueError(
                "Input and foresight proprio dimensions must match: "
                f"{self.inputs.proprio_dim} != {self.foresight.proprio_dim}"
            )
        if self.inputs.vision_feature_dim != self.foresight.semantic_visual_dim:
            raise ValueError(
                "Vision feature and semantic reconstruction dimensions must match: "
                f"{self.inputs.vision_feature_dim} != "
                f"{self.foresight.semantic_visual_dim}"
            )


@dataclass(frozen=True, slots=True)
class CLaDStage1Batch:
    """Tensor-only view of a collated ``CachedLiberoWindowDataset`` batch."""

    vision_prev: VisionFeatures
    vision_now: VisionFeatures
    vision_future: VisionFeatures
    text_features: torch.Tensor
    proprio_prev: torch.Tensor
    proprio_now: torch.Tensor
    proprio_future: torch.Tensor
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
    def from_mapping(cls, batch: Mapping[str, Any]) -> CLaDStage1Batch:
        """Extract model inputs from PyTorch's collated dataset mapping."""

        vision_features = batch.get("vision_features")
        if not isinstance(vision_features, Mapping):
            raise ValueError("batch must contain a vision_features mapping")
        tensor_fields = {
            "text_features": batch.get("text_feature"),
            "proprio_prev": batch.get("proprio_prev"),
            "proprio_now": batch.get("proprio_now"),
            "proprio_future": batch.get("proprio_future"),
            "past_actions": batch.get("past_actions"),
        }
        for name, value in tensor_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"batch field {name!r} must be a Tensor")

        return cls(
            vision_prev=cls._vision_at(vision_features, "prev"),
            vision_now=cls._vision_at(vision_features, "now"),
            vision_future=cls._vision_at(vision_features, "future"),
            text_features=tensor_fields["text_features"],
            proprio_prev=tensor_fields["proprio_prev"],
            proprio_now=tensor_fields["proprio_now"],
            proprio_future=tensor_fields["proprio_future"],
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
    ) -> CLaDStage1Batch:
        """Move every batch tensor while preserving cached feature dtypes."""

        return CLaDStage1Batch(
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
            vision_future=self._move_vision(
                self.vision_future,
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
            proprio_future=self.proprio_future.to(
                device=device,
                non_blocking=non_blocking,
            ),
            past_actions=self.past_actions.to(
                device=device,
                non_blocking=non_blocking,
            ),
        )


@dataclass(frozen=True, slots=True)
class CLaDStage1Output:
    """All Stage 1 products needed for optimization and diagnostics."""

    losses: CLaDLossOutput
    transitions: CLaDTransitionOutput
    dynamics: CrossModalDynamicsOutput
    foresight: LatentForesight
    targets: ForesightTargets
    reconstructions: ForesightReconstructions
    actions: ActionTokenOutput


class CLaDStage1Model(nn.Module):
    """Compose equations (5)--(19) into one trainable forward pass."""

    def __init__(self, config: CLaDStage1Config) -> None:
        super().__init__()
        self.config = config
        self.inputs = CLaDInputEncoders(config.inputs)
        self.transitions = CLaDTransitionEncoders(config.attention)
        self.dynamics = CrossModalDynamicsEncoder(config.attention)
        self.foresight_predictor = GroundedForesightPredictor(config.foresight)
        self.reconstruction_heads = ForesightReconstructionHeads(config.foresight)
        self.target_encoders = EMAStateEncoders(
            semantic_encoder=self.inputs.semantic,
            proprio_encoder=self.inputs.proprio,
            momentum=config.foresight.ema_momentum,
            target_pooling=config.foresight.target_pooling,
        )
        self.objective = CLaDObjective(config.foresight)

    def forward(
        self,
        batch: CLaDStage1Batch | Mapping[str, Any],
        *,
        action_mask: torch.Tensor | None = None,
        mask_actions: bool | None = None,
        generator: torch.Generator | None = None,
        return_attention: bool = False,
    ) -> CLaDStage1Output:
        if isinstance(batch, Mapping):
            batch = CLaDStage1Batch.from_mapping(batch)
        if not isinstance(batch, CLaDStage1Batch):
            raise TypeError("batch must be CLaDStage1Batch or a collated mapping")

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
        actions = self.inputs.action(
            batch.past_actions,
            action_mask=action_mask,
            mask_actions=mask_actions,
            generator=generator,
        )
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
        foresight = self.foresight_predictor(dynamics.z_dyn)
        targets = self.target_encoders(
            vision_future=batch.vision_future,
            text_features=batch.text_features,
            proprio_future=batch.proprio_future,
        )
        reconstructions = self.reconstruction_heads(foresight)
        semantic_visual_future = self.inputs.semantic.film.fuse_views(batch.vision_future)
        losses = self.objective(
            foresight=foresight,
            targets=targets,
            reconstructions=reconstructions,
            proprio_future=batch.proprio_future,
            semantic_visual_future=semantic_visual_future,
        )
        return CLaDStage1Output(
            losses=losses,
            transitions=transitions,
            dynamics=dynamics,
            foresight=foresight,
            targets=targets,
            reconstructions=reconstructions,
            actions=actions,
        )

    @torch.no_grad()
    def update_ema(self, *, momentum: float | None = None) -> None:
        """Update target state encoders after the online optimizer step."""

        self.target_encoders.update(
            semantic_encoder=self.inputs.semantic,
            proprio_encoder=self.inputs.proprio,
            momentum=momentum,
        )
