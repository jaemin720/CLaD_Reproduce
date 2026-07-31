from __future__ import annotations

import pytest
import torch

from clad.models import (
    CrossAttentionConfig,
    CrossModalDynamicsEncoder,
    LearnableQueryPooler,
)


def _config() -> CrossAttentionConfig:
    return CrossAttentionConfig(
        hidden_dim=12,
        num_heads=3,
        num_layers=2,
        ffn_multiplier=2.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def test_asymmetric_attention_preserves_proprioceptive_query_length() -> None:
    torch.manual_seed(0)
    encoder = CrossModalDynamicsEncoder(_config())
    proprio_transition = torch.randn(2, 2, 12)
    semantic_transition = torch.randn(2, 3, 12)

    output = encoder(
        proprio_transition,
        semantic_transition,
        return_attention=True,
    )

    assert output.proprio_to_semantic.shape == (2, 2, 12)
    assert output.z_dyn.shape == (2, 12)
    assert len(output.asymmetric_attention_weights) == 2
    for weights in output.asymmetric_attention_weights:
        assert weights.shape == (2, 3, 2, 3)
    assert output.pooling_attention_weights is not None
    assert output.pooling_attention_weights.shape == (2, 3, 1, 2)


def test_cross_modal_dynamics_backpropagates_to_both_modalities_and_query() -> None:
    torch.manual_seed(1)
    encoder = CrossModalDynamicsEncoder(_config())
    proprio_transition = torch.randn(2, 2, 12, requires_grad=True)
    semantic_transition = torch.randn(2, 3, 12, requires_grad=True)

    output = encoder(proprio_transition, semantic_transition)
    output.z_dyn.square().mean().backward()

    for inputs in (proprio_transition, semantic_transition):
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        assert torch.count_nonzero(inputs.grad) > 0
    assert encoder.pooler.output_query.grad is not None
    assert torch.count_nonzero(encoder.pooler.output_query.grad) > 0
    assert output.asymmetric_attention_weights == ()
    assert output.pooling_attention_weights is None


def test_semantic_transition_changes_grounded_dynamics() -> None:
    torch.manual_seed(2)
    encoder = CrossModalDynamicsEncoder(_config()).eval()
    proprio_transition = torch.randn(1, 2, 12)
    semantic_transition = torch.randn(1, 3, 12)
    feature_offset = torch.linspace(-1.0, 1.0, 12).reshape(1, 1, 12)

    original = encoder(proprio_transition, semantic_transition).z_dyn
    changed = encoder(
        proprio_transition,
        semantic_transition + feature_offset,
    ).z_dyn

    assert not torch.allclose(original, changed)


def test_learnable_query_pooling_is_invariant_to_token_order() -> None:
    torch.manual_seed(3)
    pooler = LearnableQueryPooler(_config()).eval()
    tokens = torch.randn(2, 5, 12)

    original, _ = pooler(tokens)
    permuted, _ = pooler(tokens[:, torch.tensor([2, 4, 0, 3, 1])])

    torch.testing.assert_close(original, permuted)


@pytest.mark.parametrize(
    "tokens",
    [
        torch.zeros(2, 12),
        torch.zeros(2, 0, 12),
        torch.zeros(2, 3, 10),
    ],
)
def test_pooler_rejects_invalid_tokens(tokens: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="tokens must have shape"):
        LearnableQueryPooler(_config())(tokens)


def test_asymmetric_attention_rejects_batch_mismatch() -> None:
    encoder = CrossModalDynamicsEncoder(_config())

    with pytest.raises(ValueError, match="batch sizes"):
        encoder(
            torch.zeros(2, 2, 12),
            torch.zeros(1, 3, 12),
        )
