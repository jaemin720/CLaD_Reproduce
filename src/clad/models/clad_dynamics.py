"""Asymmetric cross-modal dynamics and learnable pooling for CLaD.

This module implements equations (9) and (10). Proprioceptive transition
tokens query semantic transition tokens, then a Perceiver-style learnable
query pools the grounded transition sequence into the compact ``z_dyn``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from clad.models.clad_transition import (
    CrossAttentionBlock,
    CrossAttentionConfig,
    CrossAttentionStack,
)


@dataclass(frozen=True, slots=True)
class CrossModalDynamicsOutput:
    """Cross-modal transition tokens, pooled dynamics, and attention maps."""

    proprio_to_semantic: torch.Tensor
    z_dyn: torch.Tensor
    asymmetric_attention_weights: tuple[torch.Tensor, ...]
    pooling_attention_weights: torch.Tensor | None


class LearnableQueryPooler(nn.Module):
    """Pool a token sequence through one learned Perceiver-style query."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.output_query = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.cross_attention = CrossAttentionBlock(config)
        self.output_norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        nn.init.normal_(self.output_query, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a Tensor")
        if tokens.ndim != 3 or tokens.shape[1] == 0 or tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"tokens must have shape [B, N, {self.config.hidden_dim}] "
                f"with N > 0, got {tuple(tokens.shape)}"
            )
        if tokens.device != self.output_query.device:
            raise ValueError(
                "tokens and output_query must be on the same device: "
                f"{tokens.device} != {self.output_query.device}"
            )

        query = self.output_query.expand(tokens.shape[0], -1, -1)
        pooled_tokens, attention_weights = self.cross_attention(
            query,
            tokens,
            return_attention=return_attention,
        )
        z_dyn = self.output_norm(pooled_tokens[:, 0])
        return z_dyn, attention_weights


class CrossModalDynamicsEncoder(nn.Module):
    """Compute ``z_(p->s)`` and pool it into ``z_dyn``."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.proprio_queries_semantic = CrossAttentionStack(config)
        self.pooler = LearnableQueryPooler(config)

    def forward(
        self,
        proprio_transition: torch.Tensor,
        semantic_transition: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> CrossModalDynamicsOutput:
        asymmetric = self.proprio_queries_semantic(
            query_tokens=proprio_transition,
            context_tokens=semantic_transition,
            return_attention=return_attention,
        )
        z_dyn, pooling_weights = self.pooler(
            asymmetric.tokens,
            return_attention=return_attention,
        )
        return CrossModalDynamicsOutput(
            proprio_to_semantic=asymmetric.tokens,
            z_dyn=z_dyn,
            asymmetric_attention_weights=asymmetric.attention_weights,
            pooling_attention_weights=pooling_weights,
        )
