"""Two-modality conditional action diffusion for CLaD and its baselines.

The model implements equation (22): a DDPM noise-prediction objective over
action chunks. CLaD uses observation-modulated proprioceptive and semantic
foresights; Policy-only uses learned current-observation embeddings while
sharing the same denoiser and diffusion process.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from clad.models.clad_stage2 import (
    CLaDHistoryBatch,
    Stage2ConditioningOutput,
)


@dataclass(frozen=True, slots=True)
class DiffusionPolicyConfig:
    """Architecture and DDPM settings for the Stage 2 action policy."""

    action_dim: int = 7
    horizon: int = 6
    condition_dim_per_modality: int = 1024
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (512, 1024, 1536)
    kernel_size: int = 5
    num_groups: int = 8
    condition_predict_scale: bool = True
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    cosine_s: float = 0.008
    max_beta: float = 0.999
    clip_sample: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "down_dims", tuple(self.down_dims))
        for name, value in {
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "condition_dim_per_modality": self.condition_dim_per_modality,
            "diffusion_step_embed_dim": self.diffusion_step_embed_dim,
            "kernel_size": self.kernel_size,
            "num_groups": self.num_groups,
            "num_train_timesteps": self.num_train_timesteps,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if len(self.down_dims) < 2 or any(dimension <= 0 for dimension in self.down_dims):
            raise ValueError("down_dims must contain at least two positive dimensions")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")
        if self.diffusion_step_embed_dim < 4 or self.diffusion_step_embed_dim % 2:
            raise ValueError("diffusion_step_embed_dim must be an even integer >= 4")
        invalid_group_dims = [
            dimension for dimension in self.down_dims if dimension % self.num_groups
        ]
        if invalid_group_dims:
            raise ValueError(
                "Every down dimension must be divisible by num_groups; "
                f"invalid={invalid_group_dims}, num_groups={self.num_groups}"
            )
        if self.num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2")
        if self.beta_schedule != "squaredcos_cap_v2":
            raise ValueError(
                f"Only beta_schedule='squaredcos_cap_v2' is implemented, got {self.beta_schedule!r}"
            )
        if self.cosine_s < 0.0:
            raise ValueError(f"cosine_s must be non-negative, got {self.cosine_s}")
        if not 0.0 < self.max_beta < 1.0:
            raise ValueError(f"max_beta must be in (0, 1), got {self.max_beta}")

    @property
    def global_condition_dim(self) -> int:
        return 2 * self.condition_dim_per_modality

    @property
    def temporal_multiple(self) -> int:
        """Temporal multiple required by the U-Net downsampling path."""

        return 2 ** (len(self.down_dims) - 1)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DiffusionPolicyConfig:
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown diffusion policy settings: {unknown}")
        return cls(**dict(values))


def _cosine_beta_schedule(
    num_train_timesteps: int,
    *,
    cosine_s: float,
    max_beta: float,
) -> torch.Tensor:
    """Return the discretized squared-cosine schedule used by DDPM."""

    def alpha_bar(time: float) -> float:
        angle = (time + cosine_s) / (1.0 + cosine_s) * math.pi / 2.0
        return math.cos(angle) ** 2

    betas = [
        min(
            1.0
            - alpha_bar((step + 1) / num_train_timesteps) / alpha_bar(step / num_train_timesteps),
            max_beta,
        )
        for step in range(num_train_timesteps)
    ]
    return torch.tensor(betas, dtype=torch.float32)


class DDPMSchedule(nn.Module):
    """Forward noising and fixed-small-variance reverse DDPM transitions."""

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        betas = _cosine_beta_schedule(
            config.num_train_timesteps,
            cosine_s=config.cosine_s,
            max_beta=config.max_beta,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _batch_timesteps(
        self,
        timesteps: torch.Tensor | int,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(timesteps, int):
            result = torch.full(
                (batch_size,),
                timesteps,
                device=device,
                dtype=torch.long,
            )
        elif isinstance(timesteps, torch.Tensor):
            result = timesteps.to(device=device, dtype=torch.long)
            if result.ndim == 0:
                result = result.expand(batch_size)
            elif result.ndim == 1 and result.shape[0] == 1:
                result = result.expand(batch_size)
            elif result.ndim != 1 or result.shape[0] != batch_size:
                raise ValueError(
                    f"timesteps must be scalar or shape [{batch_size}], got {tuple(result.shape)}"
                )
        else:
            raise TypeError("timesteps must be an int or Tensor")
        if torch.any(result < 0) or torch.any(result >= self.config.num_train_timesteps):
            raise ValueError(f"timesteps must be in [0, {self.config.num_train_timesteps})")
        return result

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        selected = values.gather(0, timesteps.to(device=values.device))
        shape = (timesteps.shape[0],) + (1,) * (reference.ndim - 1)
        return selected.reshape(shape).to(device=reference.device, dtype=reference.dtype)

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return torch.randint(
            0,
            self.config.num_train_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        )

    def add_noise(
        self,
        clean_actions: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor | int,
    ) -> torch.Tensor:
        """Apply equation (22)'s forward diffusion process."""

        if clean_actions.shape != noise.shape:
            raise ValueError(
                "clean_actions and noise must have the same shape, "
                f"got {tuple(clean_actions.shape)} and {tuple(noise.shape)}"
            )
        batch_timesteps = self._batch_timesteps(
            timesteps,
            batch_size=clean_actions.shape[0],
            device=clean_actions.device,
        )
        clean_scale = self._extract(
            self.sqrt_alphas_cumprod,
            batch_timesteps,
            clean_actions,
        )
        noise_scale = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            batch_timesteps,
            clean_actions,
        )
        return clean_scale * clean_actions + noise_scale * noise

    def step(
        self,
        predicted_noise: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``a_(k-1)`` from one DDPM reverse transition."""

        if predicted_noise.shape != sample.shape:
            raise ValueError("predicted_noise and sample must have matching shapes")
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() != 1:
                raise ValueError("Reverse diffusion timestep must be scalar")
            step = int(timestep.item())
        else:
            step = int(timestep)
        if not 0 <= step < self.config.num_train_timesteps:
            raise ValueError(
                f"timestep must be in [0, {self.config.num_train_timesteps}), got {step}"
            )

        batch_timesteps = torch.full(
            (sample.shape[0],),
            step,
            device=sample.device,
            dtype=torch.long,
        )
        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod,
            batch_timesteps,
            sample,
        )
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            batch_timesteps,
            sample,
        )
        predicted_clean = (sample - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
        if self.config.clip_sample:
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)

        mean = (
            self._extract(self.posterior_mean_coef1, batch_timesteps, sample) * predicted_clean
            + self._extract(self.posterior_mean_coef2, batch_timesteps, sample) * sample
        )
        if step == 0:
            return mean
        if noise is None:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
        elif noise.shape != sample.shape:
            raise ValueError("noise must have the same shape as sample")
        else:
            noise = noise.to(device=sample.device, dtype=sample.dtype)
        variance = self._extract(
            self.posterior_variance,
            batch_timesteps,
            sample,
        )
        return mean + torch.sqrt(variance) * noise


class SinusoidalTimestepEmbedding(nn.Module):
    """Encode integer diffusion steps without learned lookup limits."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2:
            raise ValueError("embedding_dim must be an even integer >= 4")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}")
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / (half_dim - 1)
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Conv1DBlock(nn.Module):
    """Same-length Conv1d followed by GroupNorm and Mish."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionalResidualBlock1D(nn.Module):
    """Residual temporal block with global scale-and-shift conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_dim: int,
        kernel_size: int,
        num_groups: int,
        condition_predict_scale: bool,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.condition_predict_scale = condition_predict_scale
        self.first = Conv1DBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
        self.second = Conv1DBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
        condition_channels = out_channels * (2 if condition_predict_scale else 1)
        self.condition_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dim, condition_channels),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.first(inputs)
        modulation = self.condition_encoder(condition).unsqueeze(-1)
        if self.condition_predict_scale:
            scale, shift = modulation.reshape(
                modulation.shape[0],
                2,
                self.out_channels,
                1,
            ).unbind(dim=1)
            outputs = scale * outputs + shift
        else:
            outputs = outputs + modulation
        return self.second(outputs) + self.residual(inputs)


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(inputs)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(inputs)


class ConditionalUnet1D(nn.Module):
    """Predict action noise from ``a_k``, timestep, ``g_p``, and ``g_s``."""

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        step_dim = config.diffusion_step_embed_dim
        self.timestep_encoder = nn.Sequential(
            SinusoidalTimestepEmbedding(step_dim),
            nn.Linear(step_dim, step_dim * 4),
            nn.Mish(),
            nn.Linear(step_dim * 4, step_dim),
        )
        residual_condition_dim = step_dim + config.global_condition_dim

        dimensions = (config.action_dim,) + config.down_dims
        in_out = tuple(zip(dimensions[:-1], dimensions[1:], strict=True))
        self.down_modules = nn.ModuleList()
        for level, (in_channels, out_channels) in enumerate(in_out):
            is_last = level == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    (
                        self._residual_block(
                            in_channels,
                            out_channels,
                            residual_condition_dim,
                        ),
                        self._residual_block(
                            out_channels,
                            out_channels,
                            residual_condition_dim,
                        ),
                        nn.Identity() if is_last else Downsample1D(out_channels),
                    )
                )
            )

        middle_channels = config.down_dims[-1]
        self.middle_modules = nn.ModuleList(
            (
                self._residual_block(
                    middle_channels,
                    middle_channels,
                    residual_condition_dim,
                ),
                self._residual_block(
                    middle_channels,
                    middle_channels,
                    residual_condition_dim,
                ),
            )
        )

        self.up_modules = nn.ModuleList()
        for out_channels, skip_channels in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    (
                        self._residual_block(
                            2 * skip_channels,
                            out_channels,
                            residual_condition_dim,
                        ),
                        self._residual_block(
                            out_channels,
                            out_channels,
                            residual_condition_dim,
                        ),
                        Upsample1D(out_channels),
                    )
                )
            )

        first_channels = config.down_dims[0]
        self.final = nn.Sequential(
            Conv1DBlock(
                first_channels,
                first_channels,
                kernel_size=config.kernel_size,
                num_groups=config.num_groups,
            ),
            nn.Conv1d(first_channels, config.action_dim, 1),
        )

    def _residual_block(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
    ) -> ConditionalResidualBlock1D:
        return ConditionalResidualBlock1D(
            in_channels,
            out_channels,
            condition_dim=condition_dim,
            kernel_size=self.config.kernel_size,
            num_groups=self.config.num_groups,
            condition_predict_scale=self.config.condition_predict_scale,
        )

    @staticmethod
    def _timesteps(
        timestep: torch.Tensor | int,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(timestep, int):
            return torch.full((batch_size,), timestep, dtype=torch.long, device=device)
        if not isinstance(timestep, torch.Tensor):
            raise TypeError("timestep must be an int or Tensor")
        timestep = timestep.to(device=device, dtype=torch.long)
        if timestep.ndim == 0 or (timestep.ndim == 1 and timestep.shape[0] == 1):
            return timestep.reshape(1).expand(batch_size)
        if timestep.ndim != 1 or timestep.shape[0] != batch_size:
            raise ValueError(
                f"timestep must be scalar or shape [{batch_size}], got {tuple(timestep.shape)}"
            )
        return timestep

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor | int,
        *,
        proprio_condition: torch.Tensor,
        semantic_condition: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.config.horizon, self.config.action_dim)
        if noisy_actions.ndim != 3 or tuple(noisy_actions.shape[1:]) != expected:
            raise ValueError(
                f"noisy_actions must have shape [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(noisy_actions.shape)}"
            )
        condition_shape = (
            noisy_actions.shape[0],
            self.config.condition_dim_per_modality,
        )
        for name, condition in {
            "proprio_condition": proprio_condition,
            "semantic_condition": semantic_condition,
        }.items():
            if condition.shape != condition_shape:
                raise ValueError(
                    f"{name} must have shape {condition_shape}, got {tuple(condition.shape)}"
                )

        timesteps = self._timesteps(
            timestep,
            batch_size=noisy_actions.shape[0],
            device=noisy_actions.device,
        )
        timestep_embedding = self.timestep_encoder(timesteps)
        global_condition = torch.cat(
            (
                timestep_embedding,
                proprio_condition.to(dtype=timestep_embedding.dtype),
                semantic_condition.to(dtype=timestep_embedding.dtype),
            ),
            dim=-1,
        )

        original_horizon = noisy_actions.shape[1]
        padding = (-original_horizon) % self.config.temporal_multiple
        temporal = noisy_actions.transpose(1, 2)
        if padding:
            temporal = F.pad(temporal, (0, padding))

        skips: list[torch.Tensor] = []
        for first, second, downsample in self.down_modules:
            temporal = first(temporal, global_condition)
            temporal = second(temporal, global_condition)
            skips.append(temporal)
            temporal = downsample(temporal)
        for middle in self.middle_modules:
            temporal = middle(temporal, global_condition)
        for first, second, upsample in self.up_modules:
            skip = skips.pop()
            if temporal.shape[-1] != skip.shape[-1]:
                raise RuntimeError(
                    f"U-Net temporal shapes diverged: {temporal.shape[-1]} != {skip.shape[-1]}"
                )
            temporal = torch.cat((temporal, skip), dim=1)
            temporal = first(temporal, global_condition)
            temporal = second(temporal, global_condition)
            temporal = upsample(temporal)

        predicted_noise = self.final(temporal).transpose(1, 2)
        return predicted_noise[:, :original_horizon]


class LinearActionNormalizer(nn.Module):
    """Per-action-dimension min/max mapping to the DDPM interval [-1, 1]."""

    def __init__(self, action_dim: int, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.register_buffer("minimum", torch.zeros(action_dim))
        self.register_buffer("maximum", torch.zeros(action_dim))
        self.register_buffer("scale", torch.ones(action_dim))
        self.register_buffer("bias", torch.zeros(action_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_from_bounds(
        self,
        minimum: torch.Tensor | Sequence[float],
        maximum: torch.Tensor | Sequence[float],
    ) -> None:
        minimum_tensor = torch.as_tensor(
            minimum,
            device=self.minimum.device,
            dtype=self.minimum.dtype,
        )
        maximum_tensor = torch.as_tensor(
            maximum,
            device=self.maximum.device,
            dtype=self.maximum.dtype,
        )
        expected = (self.action_dim,)
        if minimum_tensor.shape != expected or maximum_tensor.shape != expected:
            raise ValueError(
                f"Action bounds must both have shape {expected}, got "
                f"{tuple(minimum_tensor.shape)} and {tuple(maximum_tensor.shape)}"
            )
        if not torch.all(torch.isfinite(minimum_tensor)) or not torch.all(
            torch.isfinite(maximum_tensor)
        ):
            raise ValueError("Action bounds must be finite")
        ranges = maximum_tensor - minimum_tensor
        if torch.any(ranges <= self.epsilon):
            raise ValueError("Every action dimension must have a non-zero finite range")
        scale = 2.0 / ranges
        self.minimum.copy_(minimum_tensor)
        self.maximum.copy_(maximum_tensor)
        self.scale.copy_(scale)
        self.bias.copy_(-1.0 - minimum_tensor * scale)
        self.fitted.fill_(True)

    def _validate(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim < 1 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must end with dimension {self.action_dim}, got {tuple(actions.shape)}"
            )
        if not bool(self.fitted.item()):
            raise RuntimeError("Action normalizer has not been fitted")
        return actions.to(dtype=self.scale.dtype)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        actions = self._validate(actions)
        return actions * self.scale + self.bias

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        actions = self._validate(actions)
        return (actions - self.bias) / self.scale


@dataclass(frozen=True, slots=True)
class CLaDStage2Batch:
    """History and future action target consumed by equation (22)."""

    history: CLaDHistoryBatch
    target_actions: torch.Tensor

    @classmethod
    def from_mapping(cls, batch: Mapping[str, Any]) -> CLaDStage2Batch:
        target_actions = batch.get("target_actions")
        if not isinstance(target_actions, torch.Tensor):
            raise TypeError("batch field 'target_actions' must be a Tensor")
        return cls(
            history=CLaDHistoryBatch.from_mapping(batch),
            target_actions=target_actions,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CLaDStage2Batch:
        return CLaDStage2Batch(
            history=self.history.to(device, non_blocking=non_blocking),
            target_actions=self.target_actions.to(
                device=device,
                non_blocking=non_blocking,
            ),
        )


@dataclass(frozen=True, slots=True)
class DiffusionPolicyLossOutput:
    """Stage 2 loss and sampled diffusion variables for diagnostics."""

    total: torch.Tensor
    predicted_noise: torch.Tensor
    target_noise: torch.Tensor
    noisy_actions: torch.Tensor
    normalized_actions: torch.Tensor
    timesteps: torch.Tensor
    conditioning: Stage2ConditioningOutput


@dataclass(frozen=True, slots=True)
class DiffusionPolicySample:
    """Normalized and environment-scale action chunks from reverse DDPM."""

    actions: torch.Tensor
    normalized_actions: torch.Tensor
    conditioning: Stage2ConditioningOutput


class CLaDDiffusionPolicy(nn.Module):
    """Compose a two-modality conditioner and conditional action DDPM."""

    def __init__(
        self,
        *,
        conditioner: nn.Module,
        config: DiffusionPolicyConfig | None = None,
        action_normalizer: LinearActionNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.config = config or DiffusionPolicyConfig()
        if not hasattr(conditioner, "input_config") or not hasattr(
            conditioner, "policy_variant"
        ):
            raise TypeError("conditioner must expose input_config and policy_variant")
        conditioner_inputs = conditioner.input_config
        if conditioner_inputs.hidden_dim != self.config.condition_dim_per_modality:
            raise ValueError(
                "Conditioner hidden dimension and diffusion condition dimension must match: "
                f"{conditioner_inputs.hidden_dim} != "
                f"{self.config.condition_dim_per_modality}"
            )
        if conditioner_inputs.action_dim != self.config.action_dim:
            raise ValueError(
                "Conditioner and diffusion action dimensions must match: "
                f"{conditioner_inputs.action_dim} != {self.config.action_dim}"
            )
        if conditioner_inputs.horizon != self.config.horizon:
            raise ValueError(
                "Conditioner and diffusion horizons must match: "
                f"{conditioner_inputs.horizon} != {self.config.horizon}"
            )
        self.denoiser = ConditionalUnet1D(self.config)
        self.schedule = DDPMSchedule(self.config)
        self.action_normalizer = (
            action_normalizer
            if action_normalizer is not None
            else LinearActionNormalizer(self.config.action_dim)
        )

    @property
    def policy_variant(self) -> str:
        return str(self.conditioner.policy_variant)

    def forward(
        self,
        batch: CLaDStage2Batch | Mapping[str, Any],
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DiffusionPolicyLossOutput:
        if isinstance(batch, Mapping):
            batch = CLaDStage2Batch.from_mapping(batch)
        if not isinstance(batch, CLaDStage2Batch):
            raise TypeError("batch must be CLaDStage2Batch or a collated mapping")
        expected = (self.config.horizon, self.config.action_dim)
        if batch.target_actions.ndim != 3 or tuple(batch.target_actions.shape[1:]) != expected:
            raise ValueError(
                f"target_actions must have shape [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(batch.target_actions.shape)}"
            )

        normalized_actions = self.action_normalizer.normalize(batch.target_actions)
        if noise is None:
            noise = torch.randn(
                normalized_actions.shape,
                device=normalized_actions.device,
                dtype=normalized_actions.dtype,
                generator=generator,
            )
        elif noise.shape != normalized_actions.shape:
            raise ValueError("noise must have the same shape as target_actions")
        else:
            noise = noise.to(
                device=normalized_actions.device,
                dtype=normalized_actions.dtype,
            )
        if timesteps is None:
            timesteps = self.schedule.sample_timesteps(
                normalized_actions.shape[0],
                device=normalized_actions.device,
                generator=generator,
            )
        timesteps = self.schedule._batch_timesteps(
            timesteps,
            batch_size=normalized_actions.shape[0],
            device=normalized_actions.device,
        )
        noisy_actions = self.schedule.add_noise(
            normalized_actions,
            noise,
            timesteps,
        )
        conditioning = self.conditioner(batch.history)
        predicted_noise = self.denoiser(
            noisy_actions,
            timesteps,
            proprio_condition=conditioning.proprio,
            semantic_condition=conditioning.semantic,
        )
        total = F.mse_loss(predicted_noise, noise)
        return DiffusionPolicyLossOutput(
            total=total,
            predicted_noise=predicted_noise,
            target_noise=noise,
            noisy_actions=noisy_actions,
            normalized_actions=normalized_actions,
            timesteps=timesteps,
            conditioning=conditioning,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        history: CLaDHistoryBatch | Mapping[str, Any],
        *,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator] | None = None,
    ) -> DiffusionPolicySample:
        """Generate normalized action chunks with full DDPM sampling.

        ``generators`` gives every batch element an independent random stream.
        This is used by vectorized rollout evaluation so an episode's diffusion
        noise depends only on its seed, not on its position in a rollout batch.
        """

        conditioning = self.conditioner(history)
        parameter = next(self.denoiser.parameters())
        batch_size = conditioning.proprio.shape[0]
        sample_shape = (
            batch_size,
            self.config.horizon,
            self.config.action_dim,
        )
        if generator is not None and generators is not None:
            raise ValueError("Specify either generator or generators, not both")
        independent_generators: tuple[torch.Generator, ...] | None = None
        if generators is not None:
            independent_generators = tuple(generators)
            if len(independent_generators) != batch_size:
                raise ValueError(
                    f"generators must contain one entry per batch element: "
                    f"expected {batch_size}, got {len(independent_generators)}"
                )

        def sample_noise() -> torch.Tensor:
            if independent_generators is None:
                return torch.randn(
                    sample_shape,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    generator=generator,
                )
            return torch.cat(
                [
                    torch.randn(
                        (1, *sample_shape[1:]),
                        device=parameter.device,
                        dtype=parameter.dtype,
                        generator=episode_generator,
                    )
                    for episode_generator in independent_generators
                ],
                dim=0,
            )

        normalized_actions = sample_noise()
        for timestep in reversed(range(self.config.num_train_timesteps)):
            predicted_noise = self.denoiser(
                normalized_actions,
                timestep,
                proprio_condition=conditioning.proprio,
                semantic_condition=conditioning.semantic,
            )
            normalized_actions = self.schedule.step(
                predicted_noise,
                timestep,
                normalized_actions,
                generator=generator,
                noise=(sample_noise() if independent_generators is not None and timestep else None),
            )
        return DiffusionPolicySample(
            actions=self.action_normalizer.unnormalize(normalized_actions),
            normalized_actions=normalized_actions,
            conditioning=conditioning,
        )
