"""Input encoders for Stage 1 Cross-Modal Latent Dynamics.

This module implements the paper's equations (5) and (6): language-conditioned
semantic states and MLP-tokenized semantic/proprioceptive states. It also
encodes the action history used by equations (7) and (8), including stochastic
replacement of action tokens during training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

VisionFeatures = torch.Tensor | Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class CLaDInputEncoderConfig:
    """Dimensions and token counts for the Stage 1 input encoders."""

    vision_feature_dim: int = 1024
    text_feature_dim: int = 1024
    proprio_dim: int = 9
    action_dim: int = 7
    hidden_dim: int = 1024
    tokenizer_mlp_hidden_dim: int = 1024
    num_proprio_tokens: int = 4
    num_semantic_tokens: int = 4
    horizon: int = 6
    action_mask_ratio: float = 0.3
    tokenizer_dropout: float = 0.0
    view_fusion: str = "mean"

    def __post_init__(self) -> None:
        positive_dimensions = {
            "vision_feature_dim": self.vision_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "tokenizer_mlp_hidden_dim": self.tokenizer_mlp_hidden_dim,
            "num_proprio_tokens": self.num_proprio_tokens,
            "num_semantic_tokens": self.num_semantic_tokens,
            "horizon": self.horizon,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.action_mask_ratio <= 1.0:
            raise ValueError(f"action_mask_ratio must be in [0, 1], got {self.action_mask_ratio}")
        if not 0.0 <= self.tokenizer_dropout < 1.0:
            raise ValueError(f"tokenizer_dropout must be in [0, 1), got {self.tokenizer_dropout}")
        if self.view_fusion != "mean":
            raise ValueError(
                f"Only view_fusion='mean' is currently implemented, got {self.view_fusion!r}"
            )


@dataclass(frozen=True, slots=True)
class ActionTokenOutput:
    """Encoded actions and the token positions replaced by the mask token."""

    tokens: torch.Tensor
    mask: torch.Tensor


class FeatureFiLM(nn.Module):
    """Fuse frozen image and language features with affine modulation.

    The paper uses one image view. For forward compatibility, image features
    may also be supplied as ``[B, V, D]`` or a mapping of view name to
    ``[B, D]``. Multiple views are mean-fused before FiLM; the one-view result
    is unchanged.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        condition_dim: int,
        view_fusion: str = "mean",
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or condition_dim <= 0:
            raise ValueError("feature_dim and condition_dim must be positive")
        if view_fusion != "mean":
            raise ValueError(f"Unsupported view fusion: {view_fusion!r}")

        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.view_fusion = view_fusion
        self.affine = nn.Linear(condition_dim, 2 * feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Residual FiLM starts as the identity while still allowing gradients
        # to immediately train language-dependent scale and shift parameters.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def _validate_view(self, view: torch.Tensor, *, name: str) -> None:
        if not isinstance(view, torch.Tensor):
            raise TypeError(f"Vision view {name!r} must be a Tensor")
        if view.ndim != 2 or view.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Vision view {name!r} must have shape [B, {self.feature_dim}], "
                f"got {tuple(view.shape)}"
            )

    def _fuse_views(self, features: VisionFeatures) -> torch.Tensor:
        if isinstance(features, Mapping):
            if not features:
                raise ValueError("vision feature mapping cannot be empty")
            views = list(features.items())
            for name, view in views:
                self._validate_view(view, name=name)
            reference_shape = views[0][1].shape
            if any(view.shape != reference_shape for _, view in views[1:]):
                raise ValueError("All vision views must have the same [B, D] shape")
            if len(views) == 1:
                return views[0][1]
            return torch.stack([view for _, view in views], dim=1).mean(dim=1)

        if not isinstance(features, torch.Tensor):
            raise TypeError("vision features must be a Tensor or mapping of view tensors")
        if features.ndim == 2:
            self._validate_view(features, name="input")
            return features
        if features.ndim == 3 and features.shape[1] > 0 and features.shape[-1] == self.feature_dim:
            return features.mean(dim=1)
        raise ValueError(
            "vision features must have shape "
            f"[B, {self.feature_dim}] or [B, V, {self.feature_dim}], "
            f"got {tuple(features.shape)}"
        )

    def forward(
        self,
        vision_features: VisionFeatures,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        vision = self._fuse_views(vision_features)
        if text_features.ndim != 2 or text_features.shape[-1] != self.condition_dim:
            raise ValueError(
                f"text_features must have shape [B, {self.condition_dim}], "
                f"got {tuple(text_features.shape)}"
            )
        if vision.shape[0] != text_features.shape[0]:
            raise ValueError(
                "Vision and text batch sizes must match: "
                f"{vision.shape[0]} != {text_features.shape[0]}"
            )

        parameter = self.affine.weight
        vision = vision.to(dtype=parameter.dtype)
        text_features = text_features.to(dtype=parameter.dtype)
        delta_scale, shift = self.affine(text_features).chunk(2, dim=-1)
        return vision * (1.0 + delta_scale) + shift


class MLPTokenizer(nn.Module):
    """Map one vector per sample into a fixed sequence of latent tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_tokens: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, num_tokens, hidden_dim, mlp_hidden_dim) <= 0:
            raise ValueError("Tokenizer dimensions and num_tokens must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_tokens * hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Tokenizer inputs must have shape [B, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        inputs = inputs.to(dtype=self.network[0].weight.dtype)
        outputs = self.network(inputs)
        return outputs.reshape(inputs.shape[0], self.num_tokens, self.hidden_dim)


class SemanticStateEncoder(nn.Module):
    """Compute ``s_t = f_s(FiLM(v_t, l))`` from cached VLM features."""

    def __init__(self, config: CLaDInputEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.film = FeatureFiLM(
            feature_dim=config.vision_feature_dim,
            condition_dim=config.text_feature_dim,
            view_fusion=config.view_fusion,
        )
        self.tokenizer = MLPTokenizer(
            input_dim=config.vision_feature_dim,
            num_tokens=config.num_semantic_tokens,
            hidden_dim=config.hidden_dim,
            mlp_hidden_dim=config.tokenizer_mlp_hidden_dim,
            dropout=config.tokenizer_dropout,
        )

    def semantic_state(
        self,
        vision_features: VisionFeatures,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the language-conditioned vector before tokenization."""

        return self.film(vision_features, text_features)

    def forward(
        self,
        vision_features: VisionFeatures,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.tokenizer(self.semantic_state(vision_features, text_features))


class ProprioStateEncoder(nn.Module):
    """Compute the tokenized proprioceptive state ``p_t = f_p(p_t)``."""

    def __init__(self, config: CLaDInputEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = MLPTokenizer(
            input_dim=config.proprio_dim,
            num_tokens=config.num_proprio_tokens,
            hidden_dim=config.hidden_dim,
            mlp_hidden_dim=config.tokenizer_mlp_hidden_dim,
            dropout=config.tokenizer_dropout,
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.tokenizer(proprio)


class ActionSequenceEncoder(nn.Module):
    """Encode ``a_(t-tau):t`` and stochastically mask action tokens."""

    def __init__(self, config: CLaDInputEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = MLPTokenizer(
            input_dim=config.action_dim,
            num_tokens=1,
            hidden_dim=config.hidden_dim,
            mlp_hidden_dim=config.tokenizer_mlp_hidden_dim,
            dropout=config.tokenizer_dropout,
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, config.horizon, config.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _sample_mask(
        self,
        *,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.config.action_mask_ratio == 0.0:
            return torch.zeros(
                batch_size,
                self.config.horizon,
                dtype=torch.bool,
                device=device,
            )
        return (
            torch.rand(
                batch_size,
                self.config.horizon,
                device=device,
                generator=generator,
            )
            < self.config.action_mask_ratio
        )

    def forward(
        self,
        actions: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        mask_actions: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> ActionTokenOutput:
        expected_shape = (self.config.horizon, self.config.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected_shape:
            raise ValueError(
                "actions must have shape "
                f"[B, {self.config.horizon}, {self.config.action_dim}], "
                f"got {tuple(actions.shape)}"
            )

        batch_size = actions.shape[0]
        content_tokens = self.tokenizer(
            actions.reshape(batch_size * self.config.horizon, self.config.action_dim)
        ).reshape(batch_size, self.config.horizon, self.config.hidden_dim)

        if action_mask is not None:
            if action_mask.shape != actions.shape[:2]:
                raise ValueError(
                    f"action_mask must have shape {tuple(actions.shape[:2])}, "
                    f"got {tuple(action_mask.shape)}"
                )
            mask = action_mask.to(device=content_tokens.device, dtype=torch.bool)
        else:
            should_mask = self.training if mask_actions is None else mask_actions
            if should_mask:
                mask = self._sample_mask(
                    batch_size=batch_size,
                    device=content_tokens.device,
                    generator=generator,
                )
            else:
                mask = torch.zeros(
                    batch_size,
                    self.config.horizon,
                    dtype=torch.bool,
                    device=content_tokens.device,
                )

        mask_tokens = self.mask_token.expand(batch_size, self.config.horizon, -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_tokens, content_tokens)
        tokens = tokens + self.position_embedding
        return ActionTokenOutput(tokens=tokens, mask=mask)


class CLaDInputEncoders(nn.Module):
    """Container for the three online encoders used by Stage 1 CLaD."""

    def __init__(self, config: CLaDInputEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.semantic = SemanticStateEncoder(config)
        self.proprio = ProprioStateEncoder(config)
        self.action = ActionSequenceEncoder(config)
