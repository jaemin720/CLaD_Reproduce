from __future__ import annotations

import pytest
import torch

from clad.models import (
    CLaDDiffusionPolicy,
    CLaDForesightBackbone,
    CLaDHistoryBatch,
    CLaDInputEncoderConfig,
    CLaDStage1Config,
    CLaDStage1Model,
    CLaDStage2Batch,
    CLaDStage2Conditioner,
    ConditionalUnet1D,
    CrossAttentionConfig,
    DDPMSchedule,
    DiffusionPolicyConfig,
    ForesightConfig,
    LinearActionNormalizer,
    SinusoidalTimestepEmbedding,
)


def _diffusion_config(**overrides: object) -> DiffusionPolicyConfig:
    values: dict[str, object] = {
        "action_dim": 2,
        "horizon": 6,
        "condition_dim_per_modality": 12,
        "diffusion_step_embed_dim": 16,
        "down_dims": (16, 32, 64),
        "kernel_size": 3,
        "num_groups": 8,
        "num_train_timesteps": 4,
    }
    values.update(overrides)
    return DiffusionPolicyConfig(**values)


def _stage1_config(*, horizon: int = 2) -> CLaDStage1Config:
    return CLaDStage1Config(
        inputs=CLaDInputEncoderConfig(
            vision_feature_dim=8,
            text_feature_dim=6,
            proprio_dim=3,
            action_dim=2,
            hidden_dim=12,
            tokenizer_mlp_hidden_dim=16,
            num_proprio_tokens=2,
            num_semantic_tokens=3,
            horizon=horizon,
        ),
        attention=CrossAttentionConfig(
            hidden_dim=12,
            num_heads=3,
            num_layers=1,
            ffn_multiplier=2.0,
        ),
        foresight=ForesightConfig(
            hidden_dim=12,
            predictor_hidden_dim=16,
            decoder_hidden_dim=16,
            proprio_dim=3,
            semantic_visual_dim=8,
        ),
    )


def _backbone() -> CLaDForesightBackbone:
    stage1 = CLaDStage1Model(_stage1_config())
    backbone = CLaDForesightBackbone(stage1.config)
    selected = {
        name: value
        for name, value in stage1.state_dict().items()
        if name.startswith(("inputs.", "transitions.", "dynamics.", "foresight_predictor."))
    }
    backbone.load_state_dict(selected, strict=True)
    return backbone


def _stage2_batch(batch_size: int = 2) -> CLaDStage2Batch:
    history = CLaDHistoryBatch(
        vision_prev={"agentview_rgb": torch.randn(batch_size, 8)},
        vision_now={"agentview_rgb": torch.randn(batch_size, 8)},
        text_features=torch.randn(batch_size, 6),
        proprio_prev=torch.randn(batch_size, 3),
        proprio_now=torch.randn(batch_size, 3),
        past_actions=torch.randn(batch_size, 2, 2),
    )
    return CLaDStage2Batch(
        history=history,
        target_actions=torch.rand(batch_size, 2, 2) * 2.0 - 1.0,
    )


def test_ddpm_forward_process_matches_equation_22() -> None:
    config = _diffusion_config()
    schedule = DDPMSchedule(config)
    clean = torch.tensor(
        [
            [[0.25, -0.5]] * 6,
            [[-0.25, 0.5]] * 6,
        ]
    )
    noise = torch.ones_like(clean) * 0.3
    timesteps = torch.tensor([0, 3])

    actual = schedule.add_noise(clean, noise, timesteps)
    clean_scale = schedule.sqrt_alphas_cumprod[timesteps].reshape(2, 1, 1)
    noise_scale = schedule.sqrt_one_minus_alphas_cumprod[timesteps].reshape(2, 1, 1)
    expected = clean_scale * clean + noise_scale * noise

    torch.testing.assert_close(actual, expected)
    assert torch.all(schedule.betas > 0.0)
    assert torch.all(schedule.betas < 1.0)
    assert torch.all(schedule.alphas_cumprod[1:] < schedule.alphas_cumprod[:-1])


def test_ddpm_final_reverse_step_recovers_predicted_clean_sample() -> None:
    schedule = DDPMSchedule(_diffusion_config())
    clean = torch.rand(2, 6, 2) - 0.5
    noise = torch.randn_like(clean)
    noisy = schedule.add_noise(clean, noise, 0)

    recovered = schedule.step(noise, 0, noisy)

    torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=2e-6)


def test_ddpm_rejects_invalid_shapes_and_timesteps() -> None:
    schedule = DDPMSchedule(_diffusion_config())
    actions = torch.zeros(2, 6, 2)
    with pytest.raises(ValueError, match="same shape"):
        schedule.add_noise(actions, torch.zeros(2, 5, 2), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="timesteps must be"):
        schedule.add_noise(actions, actions, torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="must be in"):
        schedule.add_noise(actions, actions, torch.tensor([0, 4]))


def test_timestep_embedding_and_unet_support_six_step_horizon() -> None:
    torch.manual_seed(20)
    config = _diffusion_config()
    embedding = SinusoidalTimestepEmbedding(config.diffusion_step_embed_dim)
    denoiser = ConditionalUnet1D(config)
    actions = torch.randn(3, 6, 2, requires_grad=True)
    proprio = torch.randn(3, 12, requires_grad=True)
    semantic = torch.randn(3, 12, requires_grad=True)
    timesteps = torch.tensor([0, 2, 3])

    encoded = embedding(timesteps)
    predicted = denoiser(
        actions,
        timesteps,
        proprio_condition=proprio,
        semantic_condition=semantic,
    )
    predicted.square().mean().backward()

    assert encoded.shape == (3, 16)
    assert predicted.shape == actions.shape
    assert actions.grad is not None
    assert proprio.grad is not None
    assert semantic.grad is not None
    assert torch.isfinite(predicted).all()


def test_default_denoiser_matches_reported_policy_parameter_budget() -> None:
    with torch.device("meta"):
        denoiser = ConditionalUnet1D(DiffusionPolicyConfig())

    assert sum(parameter.numel() for parameter in denoiser.parameters()) == 227_412_743


def test_action_normalizer_maps_bounds_and_round_trips() -> None:
    normalizer = LinearActionNormalizer(2)
    actions = torch.tensor([[[-2.0, 1.0], [2.0, 5.0]]])
    with pytest.raises(RuntimeError, match="not been fitted"):
        normalizer.normalize(actions)

    normalizer.fit_from_bounds([-2.0, 1.0], [2.0, 5.0])
    normalized = normalizer.normalize(actions)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]]),
    )
    torch.testing.assert_close(normalizer.unnormalize(normalized), actions)
    assert bool(normalizer.fitted)


def test_complete_policy_loss_trains_film_and_denoiser_only() -> None:
    torch.manual_seed(21)
    conditioner = CLaDStage2Conditioner(backbone=_backbone())
    config = _diffusion_config(
        horizon=2,
        down_dims=(16, 32),
        num_groups=4,
    )
    policy = CLaDDiffusionPolicy(conditioner=conditioner, config=config)
    policy.action_normalizer.fit_from_bounds([-1.0, -1.0], [1.0, 1.0])
    batch = _stage2_batch()
    noise = torch.randn_like(batch.target_actions)
    timesteps = torch.tensor([0, 3])

    output = policy(batch, noise=noise, timesteps=timesteps)
    output.total.backward()

    assert output.predicted_noise.shape == (2, 2, 2)
    assert output.noisy_actions.shape == (2, 2, 2)
    assert output.timesteps.tolist() == [0, 3]
    assert torch.isfinite(output.total)
    assert all(parameter.grad is None for parameter in policy.conditioner.backbone.parameters())
    assert policy.conditioner.proprio_film.affine.weight.grad is not None
    assert policy.conditioner.semantic_film.affine.weight.grad is not None
    assert any(parameter.grad is not None for parameter in policy.denoiser.parameters())


def test_complete_policy_samples_environment_scale_action_chunk() -> None:
    torch.manual_seed(22)
    conditioner = CLaDStage2Conditioner(backbone=_backbone())
    config = _diffusion_config(
        horizon=2,
        down_dims=(16, 32),
        num_groups=4,
    )
    policy = CLaDDiffusionPolicy(conditioner=conditioner, config=config).eval()
    policy.action_normalizer.fit_from_bounds([-2.0, -4.0], [2.0, 4.0])
    generator = torch.Generator().manual_seed(123)

    sample = policy.sample_actions(_stage2_batch().history, generator=generator)

    assert sample.actions.shape == (2, 2, 2)
    assert sample.normalized_actions.shape == (2, 2, 2)
    assert torch.isfinite(sample.actions).all()
    assert torch.all(sample.normalized_actions >= -1.0)
    assert torch.all(sample.normalized_actions <= 1.0)
    torch.testing.assert_close(
        policy.action_normalizer.normalize(sample.actions),
        sample.normalized_actions,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"down_dims": (16,)}, "down_dims"),
        ({"kernel_size": 4}, "kernel_size"),
        ({"num_groups": 3}, "divisible"),
        ({"diffusion_step_embed_dim": 15}, "even"),
        ({"num_train_timesteps": 1}, "at least 2"),
        ({"beta_schedule": "linear"}, "beta_schedule"),
    ],
)
def test_diffusion_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _diffusion_config(**overrides)


def test_diffusion_config_accepts_yaml_list_and_rejects_unknown_keys() -> None:
    config = DiffusionPolicyConfig.from_mapping({"down_dims": [16, 32]})
    assert config.down_dims == (16, 32)
    with pytest.raises(ValueError, match="Unknown"):
        DiffusionPolicyConfig.from_mapping({"channels": [16, 32]})


def test_stage2_batch_mapping_requires_target_actions_but_not_future_observations() -> None:
    raw = {
        "vision_features": {
            "agentview_rgb": {
                "prev": torch.randn(2, 8),
                "now": torch.randn(2, 8),
            }
        },
        "text_feature": torch.randn(2, 6),
        "proprio_prev": torch.randn(2, 3),
        "proprio_now": torch.randn(2, 3),
        "past_actions": torch.randn(2, 2, 2),
        "target_actions": torch.randn(2, 2, 2),
    }

    batch = CLaDStage2Batch.from_mapping(raw).to("cpu")

    assert batch.target_actions.shape == (2, 2, 2)
    assert set(batch.history.vision_now) == {"agentview_rgb"}


def test_policy_rejects_mismatched_stage1_dimensions() -> None:
    conditioner = CLaDStage2Conditioner(backbone=_backbone())
    with pytest.raises(ValueError, match="condition dimension"):
        CLaDDiffusionPolicy(
            conditioner=conditioner,
            config=_diffusion_config(
                horizon=2,
                condition_dim_per_modality=16,
                down_dims=(16, 32),
                num_groups=4,
            ),
        )
