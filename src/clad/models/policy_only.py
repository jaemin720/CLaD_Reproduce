"""Observation-conditioned diffusion baseline without CLaD foresight."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from clad.models.clad_inputs import FeatureFiLM
from clad.models.clad_stage2 import CLaDHistoryBatch, Stage2ConditioningOutput
from clad.proprioception import (
    LEGACY_ROBOT_STATE,
    LIBERO_JOINT_GRIPPER,
    proprioception_spec,
)


@dataclass(frozen=True, slots=True)
class PolicyOnlyConditionerConfig:
    """Trainable current-observation encoder for the policy-only ablation."""

    vision_feature_dim: int = 1024
    text_feature_dim: int = 1024
    proprio_dim: int = 9
    action_dim: int = 7
    hidden_dim: int = 1024
    horizon: int = 6
    mlp_hidden_dim: int = 1024
    dropout: float = 0.0
    view_fusion: str = "mean"
    camera_views: tuple[str, ...] = ("agentview_rgb",)
    proprioception: str = LIBERO_JOINT_GRIPPER

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_views", tuple(self.camera_views))
        for name, value in {
            "vision_feature_dim": self.vision_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "horizon": self.horizon,
            "mlp_hidden_dim": self.mlp_hidden_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.view_fusion != "mean":
            raise ValueError(
                f"Only view_fusion='mean' is currently implemented, got {self.view_fusion!r}"
            )
        if not self.camera_views:
            raise ValueError("camera_views cannot be empty")
        if any(not isinstance(view, str) or not view for view in self.camera_views):
            raise ValueError("camera_views must contain non-empty strings")
        if len(set(self.camera_views)) != len(self.camera_views):
            raise ValueError(f"camera_views contains duplicates: {self.camera_views}")
        proprioception_spec(self.proprioception)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PolicyOnlyConditionerConfig:
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown policy-only conditioner settings: {unknown}")
        return cls(**dict(values))

    @classmethod
    def from_checkpoint_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PolicyOnlyConditionerConfig:
        """Read a checkpoint config, preserving the historical state layout."""

        migrated = dict(values)
        # Checkpoints created before this field existed always consumed the
        # 9D robot_states vector, irrespective of the new-training default.
        migrated.setdefault("proprioception", LEGACY_ROBOT_STATE)
        return cls.from_mapping(migrated)


class ObservationMLP(nn.Module):
    """Encode one current observation vector into one policy condition."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"observation must have shape [B, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        return self.network(inputs.to(dtype=self.network[0].weight.dtype))


class PolicyOnlyConditioner(nn.Module):
    """Condition action diffusion directly on the current observation.

    Frozen DecisionNCE still supplies visual and text features. Unlike CLaD
    Stage 2, this module has no Stage 1 checkpoint, action history, latent
    dynamics, or future foresight. Its encoders are trained with the denoiser.
    """

    def __init__(self, config: PolicyOnlyConditionerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyOnlyConditionerConfig()
        self.semantic_film = FeatureFiLM(
            feature_dim=self.config.vision_feature_dim,
            condition_dim=self.config.text_feature_dim,
            view_fusion=self.config.view_fusion,
        )
        self.proprio_encoder = ObservationMLP(
            input_dim=self.config.proprio_dim,
            hidden_dim=self.config.hidden_dim,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            dropout=self.config.dropout,
        )
        self.semantic_encoder = ObservationMLP(
            input_dim=self.config.vision_feature_dim,
            hidden_dim=self.config.hidden_dim,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            dropout=self.config.dropout,
        )

    @property
    def policy_variant(self) -> str:
        return "policy_only"

    @property
    def input_config(self) -> PolicyOnlyConditionerConfig:
        return self.config

    def forward(
        self,
        batch: CLaDHistoryBatch | Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> Stage2ConditioningOutput:
        if return_attention:
            raise ValueError("Policy-only conditioning has no attention maps")
        if isinstance(batch, Mapping):
            batch = CLaDHistoryBatch.from_mapping(batch)
        if not isinstance(batch, CLaDHistoryBatch):
            raise TypeError("batch must be CLaDHistoryBatch or a collated mapping")
        actual_views = tuple(batch.vision_now)
        if actual_views != self.config.camera_views:
            raise ValueError(
                "Policy-only current camera views do not match the training contract: "
                f"expected={self.config.camera_views}, actual={actual_views}"
            )

        semantic_state = self.semantic_film(
            batch.vision_now,
            batch.text_features,
        )
        proprio = self.proprio_encoder(batch.proprio_now)
        semantic = self.semantic_encoder(semantic_state)
        return Stage2ConditioningOutput(
            proprio=proprio,
            semantic=semantic,
            proprio_observation=proprio,
            semantic_observation=semantic,
            foresight=None,
        )
