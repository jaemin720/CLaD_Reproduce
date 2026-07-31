from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import Dataset

from clad.models import (
    CLaDInputEncoderConfig,
    CLaDStage1Config,
    CLaDStage1Model,
    CrossAttentionConfig,
    ForesightConfig,
)
from clad.training import Stage1Trainer, Stage1TrainerConfig, build_stage1_dataloader


class TinyCachedDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, Any]:
        offset = float(index) / 10.0
        return {
            "text_feature": torch.tensor([0.1, 0.2, 0.3, 0.4]) + offset,
            "vision_features": {
                "agentview_rgb": {
                    "prev": torch.full((4,), offset),
                    "now": torch.full((4,), offset + 0.1),
                    "future": torch.full((4,), offset + 0.2),
                }
            },
            "proprio_prev": torch.tensor([offset, offset + 0.1, offset + 0.2]),
            "proprio_now": torch.tensor([offset + 0.1, offset + 0.2, offset + 0.3]),
            "proprio_future": torch.tensor([offset + 0.2, offset + 0.3, offset + 0.4]),
            "past_actions": torch.tensor([[offset, offset + 0.1], [offset + 0.2, offset + 0.3]]),
        }


def _model_config() -> CLaDStage1Config:
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
            ema_momentum=0.5,
        ),
    )


def _trainer_config() -> Stage1TrainerConfig:
    return Stage1TrainerConfig(
        max_steps=4,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        log_interval=1,
        checkpoint_interval=2,
        amp_enabled=False,
        seed=7,
    )


def _build_trainer(
    output_dir: Path,
    *,
    metrics: list[dict[str, float]],
) -> Stage1Trainer:
    config = _trainer_config()
    generator = Stage1Trainer.seed_everything(config.seed)
    model = CLaDStage1Model(_model_config())
    dataloader = build_stage1_dataloader(
        TinyCachedDataset(),
        config,
        generator=generator,
    )
    return Stage1Trainer(
        model=model,
        dataloader=dataloader,
        config=config,
        device="cpu",
        output_dir=output_dir,
        metric_callback=metrics.append,
    )


def test_trainer_optimizes_updates_ema_and_resumes(tmp_path: Path) -> None:
    logged_metrics: list[dict[str, float]] = []
    trainer = _build_trainer(tmp_path / "first", metrics=logged_metrics)
    online_before = next(trainer.model.inputs.proprio.parameters()).detach().clone()
    target_before = next(trainer.model.target_encoders.proprio.parameters()).detach().clone()

    first_result = trainer.train(max_steps=2)

    assert first_result.global_step == 2
    assert first_result.checkpoint_path == trainer.checkpoint_path
    assert trainer.checkpoint_path.is_file()
    assert not trainer.checkpoint_path.with_suffix(".pt.tmp").exists()
    assert len(logged_metrics) == 2
    assert all(math.isfinite(value) for value in first_result.latest_metrics.values())
    assert first_result.latest_metrics["optimizer_step_skipped"] == 0.0
    assert not torch.equal(
        next(trainer.model.inputs.proprio.parameters()).detach(),
        online_before,
    )
    assert not torch.equal(
        next(trainer.model.target_encoders.proprio.parameters()).detach(),
        target_before,
    )

    resumed_metrics: list[dict[str, float]] = []
    resumed = _build_trainer(tmp_path / "resumed", metrics=resumed_metrics)
    assert resumed.load_checkpoint(trainer.checkpoint_path) == 2
    for expected, actual in zip(
        trainer.model.parameters(),
        resumed.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    resumed_result = resumed.train()
    assert resumed_result.global_step == 4
    assert len(resumed_metrics) == 2
    assert resumed.scheduler.last_epoch == 4
    assert resumed_result.checkpoint_path == resumed.checkpoint_path

    uninterrupted = _build_trainer(tmp_path / "uninterrupted", metrics=[])
    uninterrupted.train()
    for expected, actual in zip(
        uninterrupted.model.parameters(),
        resumed.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_contains_complete_training_state(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, metrics=[])
    trainer.train(max_steps=1)

    payload = torch.load(trainer.checkpoint_path, map_location="cpu")

    assert payload["schema_version"] == 1
    assert payload["global_step"] == 1
    assert set(payload) >= {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "trainer_config",
        "model_config",
        "latest_metrics",
        "rng_state",
        "data_state",
    }
    assert payload["data_state"]["batch_sampler"]["batches_consumed"] == 2
    assert payload["trainer_config"]["max_steps"] == 4
    assert payload["model_config"]["inputs"]["hidden_dim"] == 12


def test_stage1_yaml_configs_construct_runtime_objects() -> None:
    root = Path(__file__).parents[1]
    model_values = yaml.safe_load((root / "configs/model/clad_stage1.yaml").read_text())
    trainer_values = yaml.safe_load((root / "configs/train/stage1.yaml").read_text())

    model_config = CLaDStage1Config.from_mapping(model_values)
    trainer_config = Stage1TrainerConfig.from_mapping(trainer_values)

    assert model_config.inputs.hidden_dim == 1024
    assert model_config.attention.num_layers == 8
    assert model_config.foresight.ema_momentum == pytest.approx(0.995)
    assert trainer_config.max_steps == 25_000
    assert trainer_config.batch_size == 128


def test_trainer_config_rejects_invalid_warmup() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        Stage1TrainerConfig(max_steps=2, warmup_steps=2)


def test_dataloader_rejects_incomplete_only_batch(tmp_path: Path) -> None:
    del tmp_path
    config = Stage1TrainerConfig(
        max_steps=1,
        batch_size=16,
        warmup_steps=0,
        num_workers=0,
    )
    dataloader = build_stage1_dataloader(TinyCachedDataset(), config)

    with pytest.raises(ValueError, match="no complete batches"):
        Stage1Trainer(
            model=CLaDStage1Model(_model_config()),
            dataloader=dataloader,
            config=config,
            device="cpu",
            output_dir="unused",
        )
