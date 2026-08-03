from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import Dataset

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
from clad.training import (
    ForesightCheckpointIdentity,
    Stage2Trainer,
    Stage2TrainerConfig,
    TrainableParameterEMA,
    build_stage2_dataloader,
)


class TinyStage2Dataset(Dataset[dict[str, Any]]):
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
                }
            },
            "proprio_prev": torch.tensor([offset, offset + 0.1, offset + 0.2]),
            "proprio_now": torch.tensor([offset + 0.1, offset + 0.2, offset + 0.3]),
            "past_actions": torch.tensor([[offset, offset + 0.1], [offset + 0.2, offset + 0.3]]),
            "target_actions": torch.tensor(
                [[offset + 0.1, offset + 0.2], [offset + 0.3, offset + 0.4]]
            ),
        }


class NonFiniteStage2Dataset(TinyStage2Dataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["target_actions"][0, 0] = math.nan
        return sample


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


def _policy() -> CLaDDiffusionPolicy:
    stage1 = CLaDStage1Model(_stage1_config())
    backbone = CLaDForesightBackbone(stage1.config)
    selected = {
        name: value
        for name, value in stage1.state_dict().items()
        if name.startswith(("inputs.", "transitions.", "dynamics.", "foresight_predictor."))
    }
    backbone.load_state_dict(selected, strict=True)
    conditioner = CLaDStage2Conditioner(backbone=backbone)
    policy = CLaDDiffusionPolicy(
        conditioner=conditioner,
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
    policy.action_normalizer.fit_from_bounds([-1.0, -1.0], [1.0, 1.0])
    return policy


def _trainer_config() -> Stage2TrainerConfig:
    return Stage2TrainerConfig(
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
        seed=17,
    )


def _identity(
    tmp_path: Path, *, contents: bytes = b"frozen-foresight"
) -> ForesightCheckpointIdentity:
    path = tmp_path / "stage1_foresight.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return ForesightCheckpointIdentity.from_path(path)


def _build_trainer(
    output_dir: Path,
    *,
    metrics: list[dict[str, float]],
    config: Stage2TrainerConfig | None = None,
    dataset: Dataset[dict[str, Any]] | None = None,
    identity: ForesightCheckpointIdentity | None = None,
) -> Stage2Trainer:
    config = config or _trainer_config()
    generator = Stage2Trainer.seed_everything(config.seed)
    model = _policy()
    dataloader = build_stage2_dataloader(
        dataset if dataset is not None else TinyStage2Dataset(),
        config,
        generator=generator,
    )
    identity = identity or _identity(output_dir / "source")
    return Stage2Trainer(
        model=model,
        dataloader=dataloader,
        config=config,
        device="cpu",
        output_dir=output_dir,
        foresight_checkpoint=identity,
        metric_callback=metrics.append,
    )


def _trainable_state(trainer: Stage2Trainer) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }


def test_trainer_optimizes_ema_and_resumes_exactly(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "shared")
    logged: list[dict[str, float]] = []
    trainer = _build_trainer(
        tmp_path / "first",
        metrics=logged,
        identity=identity,
    )
    trainable_before = _trainable_state(trainer)
    frozen_before = {
        name: value.detach().clone()
        for name, value in trainer.model.conditioner.backbone.named_parameters()
    }

    first_result = trainer.train(max_steps=2)

    assert first_result.global_step == 2
    assert first_result.attempt_step == 2
    assert first_result.skipped_optimizer_steps == 0
    assert first_result.checkpoint_path == trainer.checkpoint_path
    assert len(logged) == 2
    assert trainer.ema is not None
    assert trainer.ema.optimization_step == 2
    assert any(
        not torch.equal(value, trainable_before[name])
        for name, value in _trainable_state(trainer).items()
    )
    for name, value in trainer.model.conditioner.backbone.named_parameters():
        torch.testing.assert_close(value, frozen_before[name])

    resumed = _build_trainer(
        tmp_path / "resumed",
        metrics=[],
        identity=identity,
    )
    assert resumed.load_checkpoint(trainer.checkpoint_path) == 2
    resumed.train()

    uninterrupted = _build_trainer(
        tmp_path / "uninterrupted",
        metrics=[],
        identity=identity,
    )
    uninterrupted.train()
    for name, value in _trainable_state(uninterrupted).items():
        torch.testing.assert_close(_trainable_state(resumed)[name], value)
    assert resumed.ema is not None
    assert uninterrupted.ema is not None
    for name, value in uninterrupted.ema.shadow.items():
        torch.testing.assert_close(resumed.ema.shadow[name], value)


def test_checkpoint_excludes_frozen_clad_and_contains_complete_stage2_state(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path, metrics=[])
    trainer.train(max_steps=1)

    payload = torch.load(trainer.checkpoint_path, map_location="cpu")

    assert payload["schema_version"] == 1
    assert payload["global_step"] == 1
    assert set(payload) >= {
        "model_trainable",
        "action_normalizer",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "trainer_config",
        "policy_config",
        "conditioner_config",
        "foresight_checkpoint",
        "rng_state",
        "data_state",
    }
    assert payload["data_state"]["batch_sampler"]["batches_consumed"] == 2
    assert all("backbone" not in name for name in payload["model_trainable"])
    assert all(
        name.startswith(("conditioner.proprio_film", "conditioner.semantic_film", "denoiser"))
        for name in payload["model_trainable"]
    )
    assert bool(payload["action_normalizer"]["fitted"])
    assert payload["ema"]["optimization_step"] == 1


def test_resume_rejects_different_frozen_foresight(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path / "source", metrics=[])
    trainer.train(max_steps=1)
    mismatched = _identity(tmp_path / "different", contents=b"different")
    resumed = _build_trainer(
        tmp_path / "resumed",
        metrics=[],
        identity=mismatched,
    )

    with pytest.raises(ValueError, match="Frozen foresight checkpoint"):
        resumed.load_checkpoint(trainer.checkpoint_path)


def test_foresight_identity_allows_relocation_of_identical_bytes(tmp_path: Path) -> None:
    first = _identity(tmp_path / "first")
    second = _identity(tmp_path / "second")

    assert first.path != second.path
    assert first.matches(second)


def test_trainable_parameter_ema_warmup_and_copy() -> None:
    model = torch.nn.Linear(2, 1)
    ema = TrainableParameterEMA(
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_decay=0.0,
        max_decay=0.9999,
    )
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    first_shadow = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.add_(3.0)
    ema.update(model)

    assert ema.get_decay(0) == 0.0
    assert ema.get_decay(1) == 0.0
    assert 0.0 < ema.decay < 0.9999
    assert not torch.equal(ema.shadow["weight"], first_shadow)
    ema.copy_to(model)
    torch.testing.assert_close(model.weight, ema.shadow["weight"])


def test_nonfinite_gradient_without_scaler_skips_optimizer_and_ema(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(
        tmp_path,
        metrics=[],
        dataset=NonFiniteStage2Dataset(),
    )
    trainable_before = _trainable_state(trainer)

    result = trainer._train_step()

    assert not result.optimizer_ran
    assert math.isnan(result.metrics["gradient_norm"])
    assert trainer.ema is not None
    assert trainer.ema.optimization_step == 0
    for name, value in _trainable_state(trainer).items():
        torch.testing.assert_close(value, trainable_before[name])


def test_skipped_attempt_does_not_consume_stage2_step_budget(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, metrics=[])
    real_train_step = trainer._train_step
    first = True

    def skip_once() -> Any:
        nonlocal first
        if first:
            first = False
            return SimpleNamespace(
                optimizer_ran=False,
                metrics={
                    "gradient_norm": math.nan,
                    "amp_scale": 1024.0,
                    "optimizer_step_skipped": 1.0,
                    "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                    "ema_decay": 0.0,
                },
            )
        return real_train_step()

    trainer._train_step = skip_once  # type: ignore[method-assign]
    result = trainer.train(max_steps=1)

    assert result.global_step == 1
    assert result.attempt_step == 2
    assert result.skipped_optimizer_steps == 1
    assert trainer.scheduler.last_epoch == 1
    assert trainer.ema is not None
    assert trainer.ema.optimization_step == 1


def test_trainer_requires_fitted_action_normalizer(tmp_path: Path) -> None:
    config = _trainer_config()
    generator = Stage2Trainer.seed_everything(config.seed)
    policy = _policy()
    policy.action_normalizer.fitted.fill_(False)
    dataloader = build_stage2_dataloader(
        TinyStage2Dataset(),
        config,
        generator=generator,
    )

    with pytest.raises(ValueError, match="normalizer must be fitted"):
        Stage2Trainer(
            model=policy,
            dataloader=dataloader,
            config=config,
            device="cpu",
            output_dir=tmp_path,
            foresight_checkpoint=_identity(tmp_path / "source"),
        )


def test_stage2_yaml_config_constructs_paper_scale_trainer() -> None:
    root = Path(__file__).parents[1]
    values = yaml.safe_load((root / "configs/train/stage2.yaml").read_text())

    config = Stage2TrainerConfig.from_mapping(values)

    assert config.max_steps == 200_000
    assert config.batch_size == 128
    assert config.beta1 == pytest.approx(0.95)
    assert config.ema_power == pytest.approx(0.75)
    assert config.ema_max_decay == pytest.approx(0.9999)


def test_stage2_config_validation_and_empty_loader(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        Stage2TrainerConfig(max_steps=2, warmup_steps=2)
    with pytest.raises(ValueError, match="EMA decays"):
        Stage2TrainerConfig(ema_min_decay=0.9, ema_max_decay=0.8)

    config = replace(
        _trainer_config(),
        max_steps=1,
        batch_size=16,
        warmup_steps=0,
    )
    dataloader = build_stage2_dataloader(TinyStage2Dataset(), config)
    with pytest.raises(ValueError, match="no complete batches"):
        Stage2Trainer(
            model=_policy(),
            dataloader=dataloader,
            config=config,
            device="cpu",
            output_dir=tmp_path,
            foresight_checkpoint=_identity(tmp_path / "source"),
        )
