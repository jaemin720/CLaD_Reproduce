from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from clad.evaluation import load_stage2_policy
from clad.models import (
    CLaDDiffusionPolicy,
    CLaDForesightBackbone,
    CLaDInputEncoderConfig,
    CLaDStage1Config,
    CLaDStage1Model,
    CLaDStage2Conditioner,
    CrossAttentionConfig,
    DiffusionPolicyConfig,
    ForesightConfig,
)
from clad.models.clad_stage2 import (
    FROZEN_FORESIGHT_ARTIFACT_TYPE,
    FROZEN_FORESIGHT_SCHEMA_VERSION,
)
from clad.training import ForesightCheckpointIdentity
from clad.training.stage2_trainer import STAGE2_CHECKPOINT_SCHEMA_VERSION


def _stage1_config() -> CLaDStage1Config:
    return CLaDStage1Config(
        inputs=CLaDInputEncoderConfig(
            vision_feature_dim=4,
            text_feature_dim=4,
            proprio_dim=3,
            action_dim=2,
            hidden_dim=12,
            tokenizer_mlp_hidden_dim=16,
            num_proprio_tokens=2,
            num_semantic_tokens=3,
            horizon=2,
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
            semantic_visual_dim=4,
        ),
    )


def _artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    stage1 = CLaDStage1Model(_stage1_config())
    selected = {
        name: value
        for name, value in stage1.state_dict().items()
        if name.startswith(("inputs.", "transitions.", "dynamics.", "foresight_predictor."))
    }
    foresight_path = tmp_path / "foresight.pt"
    torch.save(
        {
            "artifact_type": FROZEN_FORESIGHT_ARTIFACT_TYPE,
            "schema_version": FROZEN_FORESIGHT_SCHEMA_VERSION,
            "source_checkpoint": "test",
            "source_schema_version": 2,
            "global_step": 3,
            "model_config": asdict(stage1.config),
            "model": selected,
        },
        foresight_path,
    )
    backbone = CLaDForesightBackbone(stage1.config)
    backbone.load_state_dict(selected, strict=True)
    policy = CLaDDiffusionPolicy(
        conditioner=CLaDStage2Conditioner(backbone=backbone),
        config=DiffusionPolicyConfig(
            action_dim=2,
            horizon=2,
            condition_dim_per_modality=12,
            diffusion_step_embed_dim=16,
            down_dims=(16, 32),
            kernel_size=3,
            num_groups=4,
            num_train_timesteps=4,
        ),
    )
    policy.action_normalizer.fit_from_bounds([-2.0, -3.0], [2.0, 3.0])
    raw = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    ema = {name: value + 0.25 for name, value in raw.items()}
    checkpoint_path = tmp_path / "stage2.pt"
    torch.save(
        {
            "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
            "global_step": 7,
            "attempt_step": 8,
            "model_trainable": raw,
            "action_normalizer": policy.action_normalizer.state_dict(),
            "ema": {
                "optimization_step": 7,
                "decay": 0.9,
                "shadow": ema,
            },
            "policy_config": asdict(policy.config),
            "conditioner_config": asdict(policy.conditioner.config),
            "foresight_checkpoint": asdict(
                ForesightCheckpointIdentity.from_path(foresight_path)
            ),
        },
        checkpoint_path,
    )
    return checkpoint_path, foresight_path, raw


def test_inference_loader_selects_ema_and_freezes_policy(tmp_path: Path) -> None:
    checkpoint, foresight, raw = _artifacts(tmp_path)

    loaded = load_stage2_policy(
        checkpoint,
        foresight_checkpoint=foresight,
        weights="ema",
    )

    name, parameter = next(iter(loaded.model.named_parameters()))
    if "backbone" in name:
        name, parameter = next(
            (key, value)
            for key, value in loaded.model.named_parameters()
            if key in raw
        )
    torch.testing.assert_close(parameter, raw[name] + 0.25)
    assert loaded.info.global_step == 7
    assert loaded.info.attempt_step == 8
    assert loaded.info.ema_optimization_step == 7
    assert loaded.info.weights == "ema"
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    torch.testing.assert_close(
        loaded.model.action_normalizer.minimum,
        torch.tensor([-2.0, -3.0]),
    )


def test_inference_loader_can_select_raw_policy_weights(tmp_path: Path) -> None:
    checkpoint, foresight, raw = _artifacts(tmp_path)

    loaded = load_stage2_policy(
        checkpoint,
        foresight_checkpoint=foresight,
        weights="raw",
    )

    for name, parameter in loaded.model.named_parameters():
        if name in raw:
            torch.testing.assert_close(parameter, raw[name])
    assert loaded.info.ema_optimization_step is None


def test_inference_loader_rejects_wrong_foresight_and_invalid_weight_choice(
    tmp_path: Path,
) -> None:
    checkpoint, foresight, _ = _artifacts(tmp_path)
    different = tmp_path / "different.pt"
    different.write_bytes(b"different")

    with pytest.raises(ValueError, match="does not match"):
        load_stage2_policy(checkpoint, foresight_checkpoint=different)
    with pytest.raises(ValueError, match="weights"):
        load_stage2_policy(checkpoint, foresight_checkpoint=foresight, weights="online")
