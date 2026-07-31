from __future__ import annotations

import pytest
import torch

from clad.models import (
    CLaDTransitionEncoders,
    CrossAttentionConfig,
    CrossAttentionStack,
    ModalityTransitionEncoder,
)


def _config(**overrides: object) -> CrossAttentionConfig:
    values: dict[str, object] = {
        "hidden_dim": 12,
        "num_heads": 3,
        "num_layers": 2,
        "ffn_multiplier": 2.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
    }
    values.update(overrides)
    return CrossAttentionConfig(**values)


def test_default_attention_assumptions_match_stage1_config() -> None:
    config = CrossAttentionConfig()

    assert config.hidden_dim == 1024
    assert config.num_heads == 16
    assert config.num_layers == 8
    assert config.ffn_hidden_dim == 4096


def test_cross_attention_stack_returns_per_head_maps() -> None:
    torch.manual_seed(0)
    stack = CrossAttentionStack(_config())
    queries = torch.randn(2, 3, 12)
    context = torch.randn(2, 7, 12)

    output = stack(queries, context, return_attention=True)

    assert output.tokens.shape == (2, 3, 12)
    assert len(output.attention_weights) == 2
    for weights in output.attention_weights:
        assert weights.shape == (2, 3, 3, 7)
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2, 3, 3),
        )


def test_modality_transition_uses_past_and_action_as_context() -> None:
    torch.manual_seed(1)
    encoder = ModalityTransitionEncoder(_config())
    current = torch.randn(2, 4, 12, requires_grad=True)
    past = torch.randn(2, 4, 12, requires_grad=True)
    actions = torch.randn(2, 6, 12, requires_grad=True)

    output = encoder(
        current,
        past,
        actions,
        return_attention=True,
    )
    output.tokens.square().mean().backward()

    assert output.tokens.shape == (2, 4, 12)
    assert output.attention_weights[0].shape == (2, 3, 4, 10)
    for inputs in (current, past, actions):
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        assert torch.count_nonzero(inputs.grad) > 0


def test_semantic_and_proprioceptive_branches_are_independent() -> None:
    torch.manual_seed(2)
    encoders = CLaDTransitionEncoders(_config())
    proprio_now = torch.randn(2, 2, 12)
    proprio_past = torch.randn(2, 2, 12)
    semantic_now = torch.randn(2, 3, 12)
    semantic_past = torch.randn(2, 3, 12)
    actions = torch.randn(2, 4, 12)

    output = encoders(
        proprio_now=proprio_now,
        proprio_past=proprio_past,
        semantic_now=semantic_now,
        semantic_past=semantic_past,
        action_tokens=actions,
        return_attention=True,
    )

    assert output.proprio.tokens.shape == (2, 2, 12)
    assert output.semantic.tokens.shape == (2, 3, 12)
    assert output.proprio.attention_weights[0].shape[-1] == 6
    assert output.semantic.attention_weights[0].shape[-1] == 7
    assert (
        encoders.proprio.cross_attention.layers[0].attention.in_proj_weight.data_ptr()
        != encoders.semantic.cross_attention.layers[0].attention.in_proj_weight.data_ptr()
    )


def test_transition_changes_when_action_context_changes() -> None:
    torch.manual_seed(3)
    encoder = ModalityTransitionEncoder(_config())
    encoder.eval()
    current = torch.randn(1, 3, 12)
    past = torch.randn(1, 3, 12)
    actions = torch.randn(1, 4, 12)

    original = encoder(current, past, actions).tokens
    feature_offset = torch.linspace(-2.0, 2.0, 12).reshape(1, 1, 12)
    changed = encoder(current, past, actions + feature_offset).tokens

    assert not torch.allclose(original, changed)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_dim": 10}, "divisible"),
        ({"num_layers": 0}, "num_layers"),
        ({"attention_dropout": 1.0}, "attention_dropout"),
        ({"ffn_multiplier": 0.0}, "ffn_multiplier"),
    ],
)
def test_cross_attention_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_transition_rejects_incompatible_tokens() -> None:
    encoder = ModalityTransitionEncoder(_config())

    with pytest.raises(ValueError, match="action_tokens"):
        encoder(
            torch.zeros(2, 3, 12),
            torch.zeros(2, 3, 12),
            torch.zeros(2, 4, 10),
        )
    with pytest.raises(ValueError, match="batch sizes"):
        encoder(
            torch.zeros(2, 3, 12),
            torch.zeros(1, 3, 12),
            torch.zeros(2, 4, 12),
        )
