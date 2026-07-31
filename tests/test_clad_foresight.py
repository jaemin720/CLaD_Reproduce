from __future__ import annotations

import pytest
import torch

from clad.models import (
    CLaDInputEncoderConfig,
    CLaDInputEncoders,
    CLaDObjective,
    EMAStateEncoders,
    ForesightConfig,
    ForesightReconstructionHeads,
    ForesightReconstructions,
    ForesightTargets,
    GroundedForesightPredictor,
    LatentForesight,
)


def _input_config() -> CLaDInputEncoderConfig:
    return CLaDInputEncoderConfig(
        vision_feature_dim=8,
        text_feature_dim=6,
        proprio_dim=3,
        action_dim=2,
        hidden_dim=12,
        tokenizer_mlp_hidden_dim=16,
        num_proprio_tokens=2,
        num_semantic_tokens=3,
        horizon=4,
    )


def _foresight_config(**overrides: object) -> ForesightConfig:
    values: dict[str, object] = {
        "hidden_dim": 12,
        "predictor_hidden_dim": 16,
        "decoder_hidden_dim": 16,
        "proprio_dim": 3,
        "semantic_visual_dim": 8,
        "dropout": 0.0,
        "ema_momentum": 0.995,
        "reconstruction_weight": 0.1,
    }
    values.update(overrides)
    return ForesightConfig(**values)


def test_predictor_and_reconstruction_shapes() -> None:
    torch.manual_seed(0)
    config = _foresight_config()
    predictor = GroundedForesightPredictor(config)
    decoders = ForesightReconstructionHeads(config)
    z_dyn = torch.randn(4, 12, requires_grad=True)

    foresight = predictor(z_dyn)
    reconstructions = decoders(foresight)
    (
        foresight.combined.square().mean()
        + reconstructions.proprio.square().mean()
        + reconstructions.semantic_visual.square().mean()
    ).backward()

    assert foresight.proprio.shape == (4, 12)
    assert foresight.semantic.shape == (4, 12)
    assert foresight.combined.shape == (4, 24)
    assert reconstructions.proprio.shape == (4, 3)
    assert reconstructions.semantic_visual.shape == (4, 8)
    assert z_dyn.grad is not None
    assert torch.count_nonzero(z_dyn.grad) > 0


def test_ema_targets_start_equal_to_online_mean_pooled_tokens() -> None:
    torch.manual_seed(1)
    online = CLaDInputEncoders(_input_config()).eval()
    targets = EMAStateEncoders(
        semantic_encoder=online.semantic,
        proprio_encoder=online.proprio,
    )
    vision = torch.randn(2, 8, dtype=torch.float16)
    text = torch.randn(2, 6, dtype=torch.float16)
    proprio = torch.randn(2, 3)

    with torch.no_grad():
        expected_semantic = online.semantic(vision, text).mean(dim=1)
        expected_proprio = online.proprio(proprio).mean(dim=1)
        actual = targets(
            vision_future=vision,
            text_features=text,
            proprio_future=proprio,
        )

    torch.testing.assert_close(actual.semantic, expected_semantic)
    torch.testing.assert_close(actual.proprio, expected_proprio)
    assert all(not parameter.requires_grad for parameter in targets.parameters())
    targets.train()
    assert not targets.training
    assert not targets.semantic.training
    assert not targets.proprio.training


def test_ema_update_uses_requested_momentum() -> None:
    online = CLaDInputEncoders(_input_config())
    targets = EMAStateEncoders(
        semantic_encoder=online.semantic,
        proprio_encoder=online.proprio,
    )
    target_parameter = next(targets.semantic.parameters())
    online_parameter = next(online.semantic.parameters())
    before = target_parameter.detach().clone()

    with torch.no_grad():
        online_parameter.add_(2.0)
    targets.update(
        semantic_encoder=online.semantic,
        proprio_encoder=online.proprio,
        momentum=0.25,
    )

    torch.testing.assert_close(target_parameter, before + 1.5)
    assert not target_parameter.requires_grad


def test_objective_matches_equations_and_stops_target_gradients() -> None:
    config = ForesightConfig(
        hidden_dim=2,
        predictor_hidden_dim=2,
        decoder_hidden_dim=2,
        proprio_dim=2,
        semantic_visual_dim=1,
        reconstruction_weight=0.1,
    )
    objective = CLaDObjective(config)
    predicted_proprio = torch.zeros(1, 2, requires_grad=True)
    predicted_semantic = torch.zeros(1, 2, requires_grad=True)
    target_proprio = torch.tensor([[3.0, 4.0]], requires_grad=True)
    target_semantic = torch.tensor([[3.0, 4.0]], requires_grad=True)
    reconstructed_proprio = torch.zeros(1, 2, requires_grad=True)
    reconstructed_semantic = torch.zeros(1, 1, requires_grad=True)

    output = objective(
        foresight=LatentForesight(
            proprio=predicted_proprio,
            semantic=predicted_semantic,
        ),
        targets=ForesightTargets(
            proprio=target_proprio,
            semantic=target_semantic,
        ),
        reconstructions=ForesightReconstructions(
            proprio=reconstructed_proprio,
            semantic_visual=reconstructed_semantic,
        ),
        proprio_future=torch.tensor([[1.0, -2.0]]),
        semantic_visual_future=torch.tensor([[2.0]]),
    )
    output.total.backward()

    torch.testing.assert_close(output.latent, torch.tensor(2.0))
    torch.testing.assert_close(output.reconstruction, torch.tensor(5.0))
    torch.testing.assert_close(output.total, torch.tensor(2.5))
    assert predicted_proprio.grad is not None
    assert predicted_semantic.grad is not None
    assert reconstructed_proprio.grad is not None
    assert reconstructed_semantic.grad is not None
    assert target_proprio.grad is None
    assert target_semantic.grad is None


def test_small_grounded_foresight_pipeline_is_finite() -> None:
    torch.manual_seed(2)
    online = CLaDInputEncoders(_input_config())
    target_encoders = EMAStateEncoders(
        semantic_encoder=online.semantic,
        proprio_encoder=online.proprio,
    )
    config = _foresight_config()
    predictor = GroundedForesightPredictor(config)
    decoders = ForesightReconstructionHeads(config)
    objective = CLaDObjective(config)
    z_dyn = torch.randn(2, 12, requires_grad=True)
    vision_future = {
        "agentview_rgb": torch.randn(2, 8),
        "eye_in_hand_rgb": torch.randn(2, 8),
    }
    text = torch.randn(2, 6)
    proprio_future = torch.randn(2, 3)

    foresight = predictor(z_dyn)
    targets = target_encoders(
        vision_future=vision_future,
        text_features=text,
        proprio_future=proprio_future,
    )
    reconstructions = decoders(foresight)
    semantic_visual_future = online.semantic.film.fuse_views(vision_future)
    losses = objective(
        foresight=foresight,
        targets=targets,
        reconstructions=reconstructions,
        proprio_future=proprio_future,
        semantic_visual_future=semantic_visual_future,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert losses.total.item() > 0.0
    assert z_dyn.grad is not None
    assert all(parameter.grad is None for parameter in target_encoders.parameters())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"ema_momentum": 1.1}, "ema_momentum"),
        ({"reconstruction_weight": -0.1}, "reconstruction_weight"),
        ({"target_pooling": "max"}, "target_pooling"),
    ],
)
def test_foresight_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _foresight_config(**overrides)


def test_objective_rejects_mismatched_target_shape() -> None:
    config = _foresight_config()

    with pytest.raises(ValueError, match="proprio latent target shape"):
        CLaDObjective(config)(
            foresight=LatentForesight(
                proprio=torch.zeros(2, 12),
                semantic=torch.zeros(2, 12),
            ),
            targets=ForesightTargets(
                proprio=torch.zeros(2, 10),
                semantic=torch.zeros(2, 12),
            ),
            reconstructions=ForesightReconstructions(
                proprio=torch.zeros(2, 3),
                semantic_visual=torch.zeros(2, 8),
            ),
            proprio_future=torch.zeros(2, 3),
            semantic_visual_future=torch.zeros(2, 8),
        )
