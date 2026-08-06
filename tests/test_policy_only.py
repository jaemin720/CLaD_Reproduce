from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from clad.evaluation import load_stage2_policy
from clad.models import (
    CLaDDiffusionPolicy,
    CLaDHistoryBatch,
    CLaDStage2Batch,
    DiffusionPolicyConfig,
    PolicyOnlyConditioner,
    PolicyOnlyConditionerConfig,
)
from clad.proprioception import LEGACY_ROBOT_STATE, LIBERO_JOINT_GRIPPER
from clad.training import Stage2Trainer, Stage2TrainerConfig, build_stage2_dataloader
from clad.training.stage2_trainer import STAGE2_CHECKPOINT_SCHEMA_VERSION


def _conditioner_config() -> PolicyOnlyConditionerConfig:
    return PolicyOnlyConditionerConfig(
        vision_feature_dim=4,
        text_feature_dim=4,
        proprio_dim=3,
        action_dim=2,
        hidden_dim=12,
        horizon=2,
        mlp_hidden_dim=16,
    )


def _diffusion_config() -> DiffusionPolicyConfig:
    return DiffusionPolicyConfig(
        action_dim=2,
        horizon=2,
        condition_dim_per_modality=12,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        kernel_size=3,
        num_groups=4,
        num_train_timesteps=4,
    )


def _history() -> CLaDHistoryBatch:
    return CLaDHistoryBatch(
        vision_prev={"agentview_rgb": torch.randn(2, 4)},
        vision_now={"agentview_rgb": torch.randn(2, 4)},
        text_features=torch.randn(2, 4),
        proprio_prev=torch.randn(2, 3),
        proprio_now=torch.randn(2, 3),
        past_actions=torch.randn(2, 2, 2),
    )


def _policy(
    proprioception: str = LIBERO_JOINT_GRIPPER,
) -> CLaDDiffusionPolicy:
    conditioner_values = asdict(_conditioner_config())
    conditioner_values["proprioception"] = proprioception
    model = CLaDDiffusionPolicy(
        conditioner=PolicyOnlyConditioner(
            PolicyOnlyConditionerConfig(**conditioner_values)
        ),
        config=_diffusion_config(),
    )
    model.action_normalizer.fit_from_bounds([-1.0, -1.0], [1.0, 1.0])
    return model


class _Dataset(Dataset[dict[str, torch.Tensor | dict[str, object]]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        value = float(index) / 10.0
        return {
            "vision_features": {
                "agentview_rgb": {
                    "prev": torch.full((4,), value),
                    "now": torch.full((4,), value + 0.1),
                }
            },
            "text_feature": torch.full((4,), value + 0.2),
            "proprio_prev": torch.full((3,), value),
            "proprio_now": torch.full((3,), value + 0.1),
            "past_actions": torch.full((2, 2), value),
            "target_actions": torch.full((2, 2), value + 0.1),
        }


def test_policy_only_conditioner_uses_current_observation_only() -> None:
    torch.manual_seed(1)
    conditioner = PolicyOnlyConditioner(_conditioner_config())
    history = _history()
    changed_history = CLaDHistoryBatch(
        vision_prev={"agentview_rgb": torch.randn(2, 4) * 100.0},
        vision_now=history.vision_now,
        text_features=history.text_features,
        proprio_prev=torch.randn(2, 3) * 100.0,
        proprio_now=history.proprio_now,
        past_actions=torch.randn(2, 2, 2) * 100.0,
    )

    first = conditioner(history)
    second = conditioner(changed_history)

    assert conditioner.policy_variant == "policy_only"
    assert conditioner.config.proprioception == LIBERO_JOINT_GRIPPER
    assert first.foresight is None
    assert first.proprio.shape == (2, 12)
    assert first.semantic.shape == (2, 12)
    torch.testing.assert_close(first.proprio, second.proprio)
    torch.testing.assert_close(first.semantic, second.semantic)


def test_policy_only_two_view_conditioner_uses_and_validates_both_views() -> None:
    config = PolicyOnlyConditionerConfig(
        vision_feature_dim=4,
        text_feature_dim=4,
        proprio_dim=3,
        action_dim=2,
        hidden_dim=12,
        horizon=2,
        mlp_hidden_dim=16,
        camera_views=("agentview_rgb", "eye_in_hand_rgb"),
    )
    conditioner = PolicyOnlyConditioner(config)
    history = _history()
    two_view_history = CLaDHistoryBatch(
        vision_prev={
            "agentview_rgb": history.vision_prev["agentview_rgb"],
            "eye_in_hand_rgb": torch.zeros(2, 4),
        },
        vision_now={
            "agentview_rgb": history.vision_now["agentview_rgb"],
            "eye_in_hand_rgb": torch.full((2, 4), 2.0),
        },
        text_features=history.text_features,
        proprio_prev=history.proprio_prev,
        proprio_now=history.proprio_now,
        past_actions=history.past_actions,
    )

    output = conditioner(two_view_history)
    expected_visual = (
        history.vision_now["agentview_rgb"] + two_view_history.vision_now["eye_in_hand_rgb"]
    ) / 2.0

    torch.testing.assert_close(
        conditioner.semantic_film.fuse_views(two_view_history.vision_now),
        expected_visual,
    )
    assert output.semantic.shape == (2, 12)
    with pytest.raises(ValueError, match="camera views do not match"):
        conditioner(history)


def test_policy_only_diffusion_trains_observation_encoders_and_denoiser() -> None:
    torch.manual_seed(2)
    policy = _policy()
    history = _history()
    batch = CLaDStage2Batch(
        history=history,
        target_actions=torch.randn(2, 2, 2),
    )

    output = policy(
        batch,
        noise=torch.randn(2, 2, 2),
        timesteps=torch.tensor([0, 2]),
    )
    output.total.backward()

    assert policy.policy_variant == "policy_only"
    assert output.conditioning.foresight is None
    assert policy.conditioner.proprio_encoder.network[0].weight.grad is not None
    assert policy.conditioner.semantic_encoder.network[0].weight.grad is not None
    assert policy.conditioner.semantic_film.affine.weight.grad is not None
    assert any(parameter.grad is not None for parameter in policy.denoiser.parameters())


def test_policy_only_trainer_checkpoint_has_no_stage1_dependency(tmp_path: Path) -> None:
    config = Stage2TrainerConfig(
        max_steps=2,
        batch_size=2,
        warmup_steps=0,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        log_interval=0,
        checkpoint_interval=1,
        amp_enabled=False,
    )
    trainer = Stage2Trainer(
        model=_policy(LEGACY_ROBOT_STATE),
        dataloader=build_stage2_dataloader(_Dataset(), config),
        config=config,
        device="cpu",
        output_dir=tmp_path,
        metric_callback=lambda _: None,
    )

    trainer.train(max_steps=1)
    payload = torch.load(trainer.checkpoint_path, map_location="cpu")

    assert payload["policy_variant"] == "policy_only"
    assert payload["foresight_checkpoint"] is None
    assert any(
        name.startswith("conditioner.proprio_encoder")
        for name in payload["model_trainable"]
    )
    assert all("backbone" not in name for name in payload["model_trainable"])

    # Checkpoints produced before camera_views became an explicit data contract
    # remain resumable as the historical single-agentview default.
    payload["conditioner_config"].pop("camera_views")
    payload["conditioner_config"].pop("proprioception")
    legacy_checkpoint = tmp_path / "legacy_policy_only.pt"
    torch.save(payload, legacy_checkpoint)
    official_resume = Stage2Trainer(
        model=_policy(),
        dataloader=build_stage2_dataloader(_Dataset(), config),
        config=config,
        device="cpu",
        output_dir=tmp_path / "official_resume",
        metric_callback=lambda _: None,
    )
    with pytest.raises(ValueError, match="conditioner config does not match"):
        official_resume.load_checkpoint(legacy_checkpoint)

    resumed = Stage2Trainer(
        model=_policy(LEGACY_ROBOT_STATE),
        dataloader=build_stage2_dataloader(_Dataset(), config),
        config=config,
        device="cpu",
        output_dir=tmp_path / "resumed",
        metric_callback=lambda _: None,
    )
    assert resumed.load_checkpoint(legacy_checkpoint) == 1


def test_policy_only_inference_loader_requires_no_foresight_file(tmp_path: Path) -> None:
    policy = _policy()
    raw = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    checkpoint = tmp_path / "policy_only.pt"
    torch.save(
        {
            "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
            "policy_variant": "policy_only",
            "global_step": 3,
            "attempt_step": 3,
            "model_trainable": raw,
            "action_normalizer": policy.action_normalizer.state_dict(),
            "ema": {
                "optimization_step": 3,
                "decay": 0.9,
                "shadow": {name: value + 0.1 for name, value in raw.items()},
            },
            "policy_config": asdict(policy.config),
            "conditioner_config": {
                name: value
                for name, value in asdict(policy.conditioner.config).items()
                if name not in {"camera_views", "proprioception"}
            },
            "foresight_checkpoint": None,
        },
        checkpoint,
    )

    loaded = load_stage2_policy(
        checkpoint,
        foresight_checkpoint=tmp_path / "does-not-exist.pt",
        weights="ema",
    )

    assert loaded.info.policy_variant == "policy_only"
    assert loaded.info.foresight_checkpoint is None
    assert loaded.model.policy_variant == "policy_only"
    assert loaded.model.conditioner.config.camera_views == ("agentview_rgb",)
    assert loaded.model.conditioner.config.proprioception == "robot_states"
    name, parameter = next(iter(raw.items()))
    torch.testing.assert_close(dict(loaded.model.named_parameters())[name], parameter + 0.1)
