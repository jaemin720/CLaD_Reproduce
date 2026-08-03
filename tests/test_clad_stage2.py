from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from clad.models import (
    CLaDForesightBackbone,
    CLaDHistoryBatch,
    CLaDInputEncoderConfig,
    CLaDStage1Batch,
    CLaDStage1Config,
    CLaDStage1Model,
    CLaDStage2Conditioner,
    CrossAttentionConfig,
    ForesightConfig,
    LatentFiLM,
    Stage2ConditionerConfig,
    export_frozen_foresight_checkpoint,
)


def _model_config() -> CLaDStage1Config:
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
            semantic_visual_dim=8,
        ),
    )


def _stage1_batch(batch_size: int = 2) -> CLaDStage1Batch:
    return CLaDStage1Batch(
        vision_prev={"agentview_rgb": torch.randn(batch_size, 8)},
        vision_now={"agentview_rgb": torch.randn(batch_size, 8)},
        vision_future={"agentview_rgb": torch.randn(batch_size, 8)},
        text_features=torch.randn(batch_size, 6),
        proprio_prev=torch.randn(batch_size, 3),
        proprio_now=torch.randn(batch_size, 3),
        proprio_future=torch.randn(batch_size, 3),
        past_actions=torch.randn(batch_size, 2, 2),
    )


def _history(stage1_batch: CLaDStage1Batch) -> CLaDHistoryBatch:
    return CLaDHistoryBatch(
        vision_prev=stage1_batch.vision_prev,
        vision_now=stage1_batch.vision_now,
        text_features=stage1_batch.text_features,
        proprio_prev=stage1_batch.proprio_prev,
        proprio_now=stage1_batch.proprio_now,
        past_actions=stage1_batch.past_actions,
    )


def _backbone_from_model(model: CLaDStage1Model) -> CLaDForesightBackbone:
    backbone = CLaDForesightBackbone(model.config)
    selected = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(("inputs.", "transitions.", "dynamics.", "foresight_predictor."))
    }
    backbone.load_state_dict(selected, strict=True)
    return backbone


def test_frozen_backbone_matches_stage1_history_path() -> None:
    torch.manual_seed(10)
    stage1 = CLaDStage1Model(_model_config()).eval()
    backbone = _backbone_from_model(stage1)
    batch = _stage1_batch()

    with torch.no_grad():
        expected = stage1(batch, mask_actions=False)
        actual = backbone(_history(batch), return_attention=True)

    torch.testing.assert_close(actual.foresight.proprio, expected.foresight.proprio)
    torch.testing.assert_close(actual.foresight.semantic, expected.foresight.semantic)
    torch.testing.assert_close(actual.dynamics.z_dyn, expected.dynamics.z_dyn)
    torch.testing.assert_close(actual.actions.tokens, expected.actions.tokens)
    assert not torch.any(actual.actions.mask)
    assert actual.transitions.proprio.attention_weights is not None
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_history_mapping_does_not_require_future_fields() -> None:
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
    }

    history = CLaDHistoryBatch.from_mapping(raw).to("cpu")

    assert set(history.vision_prev) == {"agentview_rgb"}
    assert history.past_actions.shape == (2, 2, 2)


def test_conditioner_starts_as_identity_and_only_film_trains() -> None:
    torch.manual_seed(11)
    backbone = _backbone_from_model(CLaDStage1Model(_model_config()))
    conditioner = CLaDStage2Conditioner(backbone=backbone)
    conditioner.train()

    output = conditioner(_history(_stage1_batch()))
    torch.testing.assert_close(output.proprio, output.foresight.proprio)
    torch.testing.assert_close(output.semantic, output.foresight.semantic)
    assert output.proprio_observation.shape == (2, 12)
    assert output.semantic_observation.shape == (2, 12)
    assert output.combined.shape == (2, 24)
    assert not conditioner.backbone.training

    output.combined.square().mean().backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert conditioner.proprio_film.affine.weight.grad is not None
    assert conditioner.semantic_film.affine.weight.grad is not None
    assert torch.count_nonzero(conditioner.proprio_film.affine.weight.grad) > 0
    assert torch.count_nonzero(conditioner.semantic_film.affine.weight.grad) > 0


def test_latent_film_validates_shapes() -> None:
    film = LatentFiLM(feature_dim=4, condition_dim=3)
    with pytest.raises(ValueError, match="features"):
        film(torch.zeros(2, 5), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="condition"):
        film(torch.zeros(2, 4), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="batch sizes"):
        film(torch.zeros(2, 4), torch.zeros(3, 3))


def test_conditioner_config_is_strict() -> None:
    assert Stage2ConditionerConfig.from_mapping({}).film_dropout == 0.0
    with pytest.raises(ValueError, match="Unknown"):
        Stage2ConditionerConfig.from_mapping({"hidden_dims": 12})
    with pytest.raises(ValueError, match="observation_pooling"):
        Stage2ConditionerConfig(observation_pooling="max")


def test_export_and_load_compact_frozen_foresight_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(12)
    stage1 = CLaDStage1Model(_model_config()).eval()
    source = tmp_path / "stage1_latest.pt"
    compact = tmp_path / "stage1_foresight.pt"
    torch.save(
        {
            "schema_version": 2,
            "global_step": 25_000,
            "model_config": asdict(stage1.config),
            "model": stage1.state_dict(),
            "optimizer": {"large_training_only_state": torch.ones(32)},
        },
        source,
    )

    info = export_frozen_foresight_checkpoint(source, compact)
    payload = torch.load(compact, map_location="cpu")
    loaded = CLaDForesightBackbone.from_checkpoint(compact)
    expected = _backbone_from_model(stage1)

    assert info.path == compact.resolve()
    assert info.global_step == 25_000
    assert info.tensor_count == len(expected.state_dict())
    assert payload["artifact_type"] == "clad_frozen_foresight"
    assert "optimizer" not in payload
    assert not any(name.startswith("target_encoders.") for name in payload["model"])
    assert not any(name.startswith("reconstruction_heads.") for name in payload["model"])
    assert not compact.with_suffix(".pt.tmp").exists()
    for name, expected_value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected_value)

    with pytest.raises(FileExistsError, match="already exists"):
        export_frozen_foresight_checkpoint(source, compact)


def test_backbone_can_load_full_stage1_checkpoint_directly(tmp_path: Path) -> None:
    stage1 = CLaDStage1Model(_model_config()).eval()
    source = tmp_path / "stage1.pt"
    torch.save(
        {
            "schema_version": 2,
            "global_step": 3,
            "model_config": asdict(stage1.config),
            "model": stage1.state_dict(),
        },
        source,
    )

    loaded = CLaDForesightBackbone.from_checkpoint(source, dtype=torch.float64)

    assert next(loaded.parameters()).dtype == torch.float64
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
