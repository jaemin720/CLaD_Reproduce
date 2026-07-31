"""Modality-specific transition encoders for Stage 1 CLaD.

The proprioceptive and semantic branches implement equations (7) and (8) from
the paper. Current state tokens are queries, while the concatenation of past
state tokens and encoded actions provides keys and values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CrossAttentionConfig:
    """Architecture settings shared by CLaD cross-attention stacks."""

    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 8
    ffn_multiplier: float = 4.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads: "
                f"{self.hidden_dim} % {self.num_heads} != 0"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.ffn_multiplier <= 0:
            raise ValueError(f"ffn_multiplier must be positive, got {self.ffn_multiplier}")
        for name, value in {
            "attention_dropout": self.attention_dropout,
            "residual_dropout": self.residual_dropout,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")

    @property
    def ffn_hidden_dim(self) -> int:
        return max(1, int(self.hidden_dim * self.ffn_multiplier))


@dataclass(frozen=True, slots=True)
class CrossAttentionOutput:
    """Transition tokens and optional per-layer, per-head attention maps."""

    tokens: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class CLaDTransitionOutput:
    """Semantic and proprioceptive transition representations."""

    proprio: CrossAttentionOutput
    semantic: CrossAttentionOutput


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a position-wise feed-forward MLP."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.query_norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        self.context_norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.attention_residual_dropout = nn.Dropout(config.residual_dropout)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.residual_dropout),
            nn.Linear(config.ffn_hidden_dim, config.hidden_dim),
        )
        self.ffn_residual_dropout = nn.Dropout(config.residual_dropout)

    def _validate_tokens(self, tokens: torch.Tensor, *, name: str) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if tokens.ndim != 3 or tokens.shape[1] == 0 or tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"{name} must have shape [B, N, {self.config.hidden_dim}] "
                f"with N > 0, got {tuple(tokens.shape)}"
            )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_tokens(query_tokens, name="query_tokens")
        self._validate_tokens(context_tokens, name="context_tokens")
        if query_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                "Query and context batch sizes must match: "
                f"{query_tokens.shape[0]} != {context_tokens.shape[0]}"
            )
        if query_tokens.device != context_tokens.device:
            raise ValueError(
                "Query and context must be on the same device: "
                f"{query_tokens.device} != {context_tokens.device}"
            )

        parameter = self.attention.in_proj_weight
        query_tokens = query_tokens.to(dtype=parameter.dtype)
        context_tokens = context_tokens.to(dtype=parameter.dtype)
        normalized_context = self.context_norm(context_tokens)
        attended, attention_weights = self.attention(
            query=self.query_norm(query_tokens),
            key=normalized_context,
            value=normalized_context,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query_tokens = query_tokens + self.attention_residual_dropout(attended)
        query_tokens = query_tokens + self.ffn_residual_dropout(
            self.ffn(self.ffn_norm(query_tokens))
        )
        return query_tokens, attention_weights


class CrossAttentionStack(nn.Module):
    """Apply an independent stack of cross-attention blocks."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(CrossAttentionBlock(config) for _ in range(config.num_layers))
        self.output_norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> CrossAttentionOutput:
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            query_tokens, weights = layer(
                query_tokens,
                context_tokens,
                return_attention=return_attention,
            )
            if weights is not None:
                attention_maps.append(weights)
        return CrossAttentionOutput(
            tokens=self.output_norm(query_tokens),
            attention_weights=tuple(attention_maps),
        )


class ModalityTransitionEncoder(nn.Module):
    """Compute one modality's transition from current, past, and action tokens."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.cross_attention = CrossAttentionStack(config)

    def _validate_context_inputs(
        self,
        current_tokens: torch.Tensor,
        past_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> None:
        expected_hidden = self.config.hidden_dim
        named_tokens = {
            "current_tokens": current_tokens,
            "past_tokens": past_tokens,
            "action_tokens": action_tokens,
        }
        for name, tokens in named_tokens.items():
            if not isinstance(tokens, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if tokens.ndim != 3 or tokens.shape[1] == 0 or tokens.shape[-1] != expected_hidden:
                raise ValueError(
                    f"{name} must have shape [B, N, {expected_hidden}] "
                    f"with N > 0, got {tuple(tokens.shape)}"
                )

        batch_sizes = {tokens.shape[0] for tokens in named_tokens.values()}
        if len(batch_sizes) != 1:
            raise ValueError(
                "Current, past, and action batch sizes must match, got "
                f"{[tokens.shape[0] for tokens in named_tokens.values()]}"
            )
        devices = {tokens.device for tokens in named_tokens.values()}
        if len(devices) != 1:
            raise ValueError("Current, past, and action tokens must share one device")

    def forward(
        self,
        current_tokens: torch.Tensor,
        past_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> CrossAttentionOutput:
        self._validate_context_inputs(current_tokens, past_tokens, action_tokens)
        context_tokens = torch.cat((past_tokens, action_tokens), dim=1)
        return self.cross_attention(
            current_tokens,
            context_tokens,
            return_attention=return_attention,
        )


class CLaDTransitionEncoders(nn.Module):
    """Independent semantic and proprioceptive transition branches."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.proprio = ModalityTransitionEncoder(config)
        self.semantic = ModalityTransitionEncoder(config)

    def forward(
        self,
        *,
        proprio_now: torch.Tensor,
        proprio_past: torch.Tensor,
        semantic_now: torch.Tensor,
        semantic_past: torch.Tensor,
        action_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> CLaDTransitionOutput:
        proprio = self.proprio(
            proprio_now,
            proprio_past,
            action_tokens,
            return_attention=return_attention,
        )
        semantic = self.semantic(
            semantic_now,
            semantic_past,
            action_tokens,
            return_attention=return_attention,
        )
        return CLaDTransitionOutput(proprio=proprio, semantic=semantic)
