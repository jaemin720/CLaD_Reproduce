"""Grounded latent foresight, EMA targets, and Stage 1 objectives.

This module implements equations (11)--(19): modality-specific future latent
prediction from ``z_dyn``, EMA target state encoders, observable-state
reconstruction, and the combined CLaD pre-training loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from clad.models.clad_inputs import (
    ProprioStateEncoder,
    SemanticStateEncoder,
    VisionFeatures,
)


@dataclass(frozen=True, slots=True)
class ForesightConfig:
    """Architecture and objective settings for grounded latent foresight."""

    hidden_dim: int = 1024
    predictor_hidden_dim: int = 1024
    decoder_hidden_dim: int = 1024
    proprio_dim: int = 9
    semantic_visual_dim: int = 1024
    dropout: float = 0.0
    ema_momentum: float = 0.995
    reconstruction_weight: float = 0.1
    normalization_eps: float = 1e-6
    target_pooling: str = "mean"

    def __post_init__(self) -> None:
        for name, value in {
            "hidden_dim": self.hidden_dim,
            "predictor_hidden_dim": self.predictor_hidden_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "proprio_dim": self.proprio_dim,
            "semantic_visual_dim": self.semantic_visual_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1], got {self.ema_momentum}")
        if self.reconstruction_weight < 0.0:
            raise ValueError(
                f"reconstruction_weight must be non-negative, got {self.reconstruction_weight}"
            )
        if self.normalization_eps <= 0.0:
            raise ValueError(f"normalization_eps must be positive, got {self.normalization_eps}")
        if self.target_pooling != "mean":
            raise ValueError(
                f"Only target_pooling='mean' is currently implemented, got {self.target_pooling!r}"
            )


@dataclass(frozen=True, slots=True)
class LatentForesight:
    """Predicted proprioceptive and semantic future latent vectors."""

    proprio: torch.Tensor
    semantic: torch.Tensor

    @property
    def combined(self) -> torch.Tensor:
        """Return equation (13), ``[z_hat_p; z_hat_s]``."""

        return torch.cat((self.proprio, self.semantic), dim=-1)


@dataclass(frozen=True, slots=True)
class ForesightTargets:
    """Stop-gradient future latent vectors produced by EMA encoders."""

    proprio: torch.Tensor
    semantic: torch.Tensor


@dataclass(frozen=True, slots=True)
class ForesightReconstructions:
    """Future observable quantities decoded from predicted foresight."""

    proprio: torch.Tensor
    semantic_visual: torch.Tensor


@dataclass(frozen=True, slots=True)
class CLaDLossOutput:
    """Combined Stage 1 loss and its modality-specific components."""

    total: torch.Tensor
    latent: torch.Tensor
    reconstruction: torch.Tensor
    latent_proprio: torch.Tensor
    latent_semantic: torch.Tensor
    reconstruction_proprio: torch.Tensor
    reconstruction_semantic: torch.Tensor


class _TwoLayerMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"MLP inputs must have shape [B, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        return self.network(inputs.to(dtype=self.network[0].weight.dtype))


class GroundedForesightPredictor(nn.Module):
    """Predict modality-specific future latent vectors from shared dynamics."""

    def __init__(self, config: ForesightConfig) -> None:
        super().__init__()
        self.config = config
        self.proprio_predictor = _TwoLayerMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.predictor_hidden_dim,
            output_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.semantic_predictor = _TwoLayerMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.predictor_hidden_dim,
            output_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, z_dyn: torch.Tensor) -> LatentForesight:
        return LatentForesight(
            proprio=self.proprio_predictor(z_dyn),
            semantic=self.semantic_predictor(z_dyn),
        )


class ForesightReconstructionHeads(nn.Module):
    """Decode predicted latent foresights to future observable states."""

    def __init__(self, config: ForesightConfig) -> None:
        super().__init__()
        self.config = config
        self.proprio_decoder = _TwoLayerMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.decoder_hidden_dim,
            output_dim=config.proprio_dim,
            dropout=config.dropout,
        )
        self.semantic_decoder = _TwoLayerMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.decoder_hidden_dim,
            output_dim=config.semantic_visual_dim,
            dropout=config.dropout,
        )

    def forward(self, foresight: LatentForesight) -> ForesightReconstructions:
        return ForesightReconstructions(
            proprio=self.proprio_decoder(foresight.proprio),
            semantic_visual=self.semantic_decoder(foresight.semantic),
        )


class EMAStateEncoders(nn.Module):
    """Frozen moving-average copies of the online state encoders."""

    def __init__(
        self,
        *,
        semantic_encoder: SemanticStateEncoder,
        proprio_encoder: ProprioStateEncoder,
        momentum: float = 0.995,
        target_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        if target_pooling != "mean":
            raise ValueError(
                f"Only target_pooling='mean' is currently implemented, got {target_pooling!r}"
            )

        self.momentum = momentum
        self.target_pooling = target_pooling
        self.semantic = copy.deepcopy(semantic_encoder)
        self.proprio = copy.deepcopy(proprio_encoder)
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> EMAStateEncoders:
        """Keep target encoders deterministic when a parent module trains."""

        del mode
        self._freeze_targets()
        return self

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    @torch.no_grad()
    def forward(
        self,
        *,
        vision_future: VisionFeatures,
        text_features: torch.Tensor,
        proprio_future: torch.Tensor,
    ) -> ForesightTargets:
        semantic_tokens = self.semantic(vision_future, text_features)
        proprio_tokens = self.proprio(proprio_future)
        return ForesightTargets(
            proprio=self._pool_tokens(proprio_tokens),
            semantic=self._pool_tokens(semantic_tokens),
        )

    @staticmethod
    def _ema_module(
        target: nn.Module,
        online: nn.Module,
        *,
        momentum: float,
    ) -> None:
        target_parameters = dict(target.named_parameters())
        online_parameters = dict(online.named_parameters())
        if target_parameters.keys() != online_parameters.keys():
            raise ValueError("Target and online encoder parameters do not match")
        for name, target_parameter in target_parameters.items():
            online_parameter = (
                online_parameters[name]
                .detach()
                .to(
                    device=target_parameter.device,
                    dtype=target_parameter.dtype,
                )
            )
            target_parameter.mul_(momentum).add_(
                online_parameter,
                alpha=1.0 - momentum,
            )

        target_buffers = dict(target.named_buffers())
        online_buffers = dict(online.named_buffers())
        if target_buffers.keys() != online_buffers.keys():
            raise ValueError("Target and online encoder buffers do not match")
        for name, target_buffer in target_buffers.items():
            online_buffer = (
                online_buffers[name]
                .detach()
                .to(
                    device=target_buffer.device,
                    dtype=target_buffer.dtype,
                )
            )
            if torch.is_floating_point(target_buffer):
                target_buffer.mul_(momentum).add_(
                    online_buffer,
                    alpha=1.0 - momentum,
                )
            else:
                target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def update(
        self,
        *,
        semantic_encoder: SemanticStateEncoder,
        proprio_encoder: ProprioStateEncoder,
        momentum: float | None = None,
    ) -> None:
        """Apply equation (14) after one online optimizer step."""

        current_momentum = self.momentum if momentum is None else momentum
        if not 0.0 <= current_momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {current_momentum}")
        self._ema_module(
            self.semantic,
            semantic_encoder,
            momentum=current_momentum,
        )
        self._ema_module(
            self.proprio,
            proprio_encoder,
            momentum=current_momentum,
        )
        self._freeze_targets()


class CLaDObjective(nn.Module):
    """Compute equations (17)--(19) with stopped-gradient EMA targets."""

    def __init__(self, config: ForesightConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _validate_pair(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if prediction.ndim != 2:
            raise ValueError(f"{name} prediction must have shape [B, D]")
        if target.shape != prediction.shape:
            raise ValueError(
                f"{name} target shape {tuple(target.shape)} does not match "
                f"prediction shape {tuple(prediction.shape)}"
            )
        if prediction.device != target.device:
            raise ValueError(f"{name} prediction and target must share one device")
        return target.detach().to(dtype=prediction.dtype)

    @staticmethod
    def _squared_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().sum(dim=-1).mean()

    @staticmethod
    def _l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().sum(dim=-1).mean()

    def forward(
        self,
        *,
        foresight: LatentForesight,
        targets: ForesightTargets,
        reconstructions: ForesightReconstructions,
        proprio_future: torch.Tensor,
        semantic_visual_future: torch.Tensor,
    ) -> CLaDLossOutput:
        target_proprio = self._validate_pair(
            foresight.proprio,
            targets.proprio,
            name="proprio latent",
        )
        target_semantic = self._validate_pair(
            foresight.semantic,
            targets.semantic,
            name="semantic latent",
        )
        future_proprio = self._validate_pair(
            reconstructions.proprio,
            proprio_future,
            name="proprio reconstruction",
        )
        future_semantic = self._validate_pair(
            reconstructions.semantic_visual,
            semantic_visual_future,
            name="semantic reconstruction",
        )

        target_proprio = F.normalize(
            target_proprio,
            dim=-1,
            eps=self.config.normalization_eps,
        )
        target_semantic = F.normalize(
            target_semantic,
            dim=-1,
            eps=self.config.normalization_eps,
        )
        latent_proprio = self._squared_l2(
            foresight.proprio,
            target_proprio,
        )
        latent_semantic = self._squared_l2(
            foresight.semantic,
            target_semantic,
        )
        reconstruction_proprio = self._l1(
            reconstructions.proprio,
            future_proprio,
        )
        reconstruction_semantic = self._l1(
            reconstructions.semantic_visual,
            future_semantic,
        )

        latent = latent_proprio + latent_semantic
        reconstruction = reconstruction_proprio + reconstruction_semantic
        total = latent + self.config.reconstruction_weight * reconstruction
        return CLaDLossOutput(
            total=total,
            latent=latent,
            reconstruction=reconstruction,
            latent_proprio=latent_proprio,
            latent_semantic=latent_semantic,
            reconstruction_proprio=reconstruction_proprio,
            reconstruction_semantic=reconstruction_semantic,
        )
