from __future__ import annotations

import pytest
import torch

from clad.models import (
    ActionSequenceEncoder,
    CLaDInputEncoderConfig,
    FeatureFiLM,
    ProprioStateEncoder,
    SemanticStateEncoder,
)


def _config(**overrides: object) -> CLaDInputEncoderConfig:
    values: dict[str, object] = {
        "vision_feature_dim": 8,
        "text_feature_dim": 6,
        "proprio_dim": 3,
        "action_dim": 2,
        "hidden_dim": 12,
        "tokenizer_mlp_hidden_dim": 16,
        "num_proprio_tokens": 2,
        "num_semantic_tokens": 3,
        "horizon": 4,
        "action_mask_ratio": 0.3,
    }
    values.update(overrides)
    return CLaDInputEncoderConfig(**values)


def test_paper_default_dimensions() -> None:
    config = CLaDInputEncoderConfig()

    assert config.hidden_dim == 1024
    assert config.num_proprio_tokens == 4
    assert config.num_semantic_tokens == 4
    assert config.horizon == 6
    assert config.action_mask_ratio == 0.3


def test_semantic_encoder_accepts_cached_dtype_and_expandable_views() -> None:
    torch.manual_seed(0)
    encoder = SemanticStateEncoder(_config())
    text = torch.randn(2, 6, dtype=torch.float16)
    agent = torch.randn(2, 8, dtype=torch.float16)
    wrist = torch.randn(2, 8, dtype=torch.float16)

    single_tokens = encoder({"agentview_rgb": agent}, text)
    stacked_tokens = encoder(torch.stack((agent, wrist), dim=1), text)
    mapped_tokens = encoder(
        {"agentview_rgb": agent, "eye_in_hand_rgb": wrist},
        text,
    )

    assert single_tokens.shape == (2, 3, 12)
    assert single_tokens.dtype == torch.float32
    torch.testing.assert_close(stacked_tokens, mapped_tokens)


def test_film_starts_as_identity_and_receives_gradients() -> None:
    torch.manual_seed(1)
    film = FeatureFiLM(feature_dim=5, condition_dim=4)
    vision = torch.randn(3, 5)
    text = torch.randn(3, 4)

    fused = film(vision, text)
    torch.testing.assert_close(fused, vision)

    fused.square().mean().backward()
    assert film.affine.weight.grad is not None
    assert torch.count_nonzero(film.affine.weight.grad) > 0


def test_proprio_encoder_tokenizes_and_backpropagates() -> None:
    encoder = ProprioStateEncoder(_config())
    proprio = torch.randn(5, 3, requires_grad=True)

    tokens = encoder(proprio)
    tokens.square().mean().backward()

    assert tokens.shape == (5, 2, 12)
    assert proprio.grad is not None
    assert torch.isfinite(proprio.grad).all()


def test_action_encoder_applies_explicit_mask_and_positions() -> None:
    torch.manual_seed(2)
    encoder = ActionSequenceEncoder(_config())
    actions = torch.randn(2, 4, 2, dtype=torch.float16)
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )

    output = encoder(actions, action_mask=mask)
    expected_mask_tokens = encoder.mask_token.expand(2, 4, -1) + encoder.position_embedding

    assert output.tokens.shape == (2, 4, 12)
    assert output.tokens.dtype == torch.float32
    assert torch.equal(output.mask, mask)
    torch.testing.assert_close(
        output.tokens[mask],
        expected_mask_tokens[mask],
    )

    output.tokens.sum().backward()
    assert encoder.mask_token.grad is not None
    assert torch.count_nonzero(encoder.mask_token.grad) > 0


def test_action_masking_defaults_to_train_only() -> None:
    encoder = ActionSequenceEncoder(_config(action_mask_ratio=1.0))
    actions = torch.zeros(2, 4, 2)

    encoder.train()
    train_output = encoder(actions)
    encoder.eval()
    eval_output = encoder(actions)

    assert train_output.mask.all()
    assert not eval_output.mask.any()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim must be positive"),
        ({"action_mask_ratio": 1.1}, "action_mask_ratio"),
        ({"view_fusion": "attention"}, "view_fusion"),
    ],
)
def test_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_encoders_reject_incompatible_shapes() -> None:
    config = _config()

    with pytest.raises(ValueError, match="text_features"):
        SemanticStateEncoder(config)(torch.zeros(2, 8), torch.zeros(2, 5))
    with pytest.raises(ValueError, match="Tokenizer inputs"):
        ProprioStateEncoder(config)(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="actions must have shape"):
        ActionSequenceEncoder(config)(torch.zeros(2, 3, 2))
