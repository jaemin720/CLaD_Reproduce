from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from clad.data import (
    CachedLiberoWindowDataset,
    DecisionNCEFeatureCache,
    DecisionNCEFeatureCacheBuilder,
    FeatureCacheSpec,
    LiberoDatasetConfig,
    LiberoWindowDataset,
    compute_libero_action_bounds,
)
from clad.models import (
    CLaDInputEncoderConfig,
    CLaDStage1Batch,
    CLaDStage1Config,
    CLaDStage1Model,
    CrossAttentionConfig,
    DecisionNCEAdapter,
    DecisionNCEAdapterConfig,
    ForesightConfig,
)
from tests.fakes import FakeDecisionNCEBackend


def _create_task_file(root: Path) -> None:
    path = root / "SYNTHETIC_CACHED_SCENE_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["num_demos"] = 1
        data.attrs["problem_info"] = json.dumps({"language_instruction": "move the cached object"})
        demo = data.create_group("demo_0")
        steps = np.arange(5, dtype=np.float32)
        demo.create_dataset(
            "actions",
            data=np.stack((steps, steps + 0.5), axis=-1),
        )
        demo.create_dataset(
            "robot_states",
            data=np.stack((steps, steps + 1.0, steps + 2.0), axis=-1),
        )
        observations = demo.create_group("obs")
        agent = np.zeros((5, 4, 4, 3), dtype=np.uint8)
        wrist = np.zeros((5, 3, 3, 3), dtype=np.uint8)
        for step in range(5):
            agent[step] = step
            wrist[step] = step + 10
        observations.create_dataset("agentview_rgb", data=agent)
        observations.create_dataset("eye_in_hand_rgb", data=wrist)


def _build_cache(dataset_dir: Path, cache_dir: Path) -> None:
    adapter_config = DecisionNCEAdapterConfig(
        model_name="DecisionNCE-T",
        device="cpu",
        source_revision="test-revision",
        checkpoint_sha256="test-checkpoint",
    )
    adapter = DecisionNCEAdapter(
        backend=FakeDecisionNCEBackend(),
        config=adapter_config,
    )
    builder = DecisionNCEFeatureCacheBuilder(
        adapter=adapter,
        spec=FeatureCacheSpec(
            model_name="DecisionNCE-T",
            source_revision="test-revision",
            checkpoint_sha256="test-checkpoint",
            camera_keys=("obs/agentview_rgb", "obs/eye_in_hand_rgb"),
        ),
        batch_size=2,
    )
    builder.build(dataset_dir=dataset_dir, cache_dir=cache_dir)


def _cached_dataset(
    tmp_path: Path,
    *,
    include_future_features: bool = True,
) -> CachedLiberoWindowDataset:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    dataset_dir.mkdir()
    _create_task_file(dataset_dir)
    _build_cache(dataset_dir, cache_dir)
    base = LiberoWindowDataset(
        LiberoDatasetConfig(
            dataset_dir=dataset_dir,
            camera_keys=("obs/agentview_rgb", "obs/eye_in_hand_rgb"),
            horizon=1,
            include_images=False,
        )
    )
    return CachedLiberoWindowDataset(
        base_dataset=base,
        feature_cache=DecisionNCEFeatureCache(cache_dir),
        include_future_features=include_future_features,
    )


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
            horizon=1,
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


def test_cached_dataset_aligns_features_to_window(tmp_path: Path) -> None:
    dataset = _cached_dataset(tmp_path)
    try:
        sample = dataset[0]

        assert sample["anchor_step"] == 1
        assert "images" not in sample
        assert sample["text_feature"].shape == (4,)
        assert set(sample["vision_features"]) == {
            "agentview_rgb",
            "eye_in_hand_rgb",
        }
        torch.testing.assert_close(
            sample["vision_features"]["agentview_rgb"]["prev"],
            torch.zeros(4, dtype=torch.float16),
        )
        torch.testing.assert_close(
            sample["vision_features"]["agentview_rgb"]["future"],
            torch.full((4,), 2.0, dtype=torch.float16),
        )
        torch.testing.assert_close(
            sample["vision_features"]["eye_in_hand_rgb"]["now"],
            torch.full((4,), 11.0, dtype=torch.float16),
        )
    finally:
        dataset.close()


def test_stage2_cache_view_skips_future_feature_and_computes_action_bounds(
    tmp_path: Path,
) -> None:
    dataset = _cached_dataset(tmp_path, include_future_features=False)
    try:
        sample = dataset[0]
        bounds = compute_libero_action_bounds(
            dataset.base_dataset,
            expected_action_dim=2,
        )

        assert set(sample["vision_features"]["agentview_rgb"]) == {"prev", "now"}
        torch.testing.assert_close(bounds.minimum, torch.tensor([0.0, 0.5]))
        torch.testing.assert_close(bounds.maximum, torch.tensor([4.0, 4.5]))
        assert bounds.count == 5
    finally:
        dataset.close()


def test_cached_batch_runs_composed_stage1_and_updates_ema(tmp_path: Path) -> None:
    dataset = _cached_dataset(tmp_path)
    try:
        collated = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        batch = CLaDStage1Batch.from_mapping(collated).to("cpu")
        model = CLaDStage1Model(_model_config())
        model.train()
        action_mask = torch.tensor([[True], [False]])

        output = model(
            batch,
            action_mask=action_mask,
            return_attention=True,
        )
        output.losses.total.backward()

        assert output.dynamics.z_dyn.shape == (2, 12)
        assert output.foresight.combined.shape == (2, 24)
        assert output.targets.proprio.shape == (2, 12)
        assert output.reconstructions.proprio.shape == (2, 3)
        assert output.reconstructions.semantic_visual.shape == (2, 4)
        assert torch.equal(output.actions.mask, action_mask)
        assert output.dynamics.asymmetric_attention_weights[0].shape == (
            2,
            3,
            2,
            3,
        )
        assert torch.isfinite(output.losses.total)
        assert all(parameter.grad is None for parameter in model.target_encoders.parameters())
        assert any(parameter.grad is not None for parameter in model.inputs.semantic.parameters())
        assert not model.target_encoders.training

        target_parameter = next(model.target_encoders.semantic.parameters())
        online_parameter = next(model.inputs.semantic.parameters())
        before = target_parameter.detach().clone()
        with torch.no_grad():
            online_parameter.add_(1.0)
        model.update_ema(momentum=0.5)
        torch.testing.assert_close(target_parameter, before + 0.5)
    finally:
        dataset.close()


def test_cached_dataset_requires_complete_camera_coverage(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    dataset_dir.mkdir()
    _create_task_file(dataset_dir)
    _build_cache(dataset_dir, cache_dir)
    base = LiberoWindowDataset(
        LiberoDatasetConfig(
            dataset_dir=dataset_dir,
            camera_keys=("obs/unseen_camera_rgb",),
            horizon=1,
            include_images=False,
        )
    )
    cache = DecisionNCEFeatureCache(cache_dir)

    try:
        with pytest.raises(ValueError, match="requested cameras"):
            CachedLiberoWindowDataset(
                base_dataset=base,
                feature_cache=cache,
            )
    finally:
        base.close()
        cache.close()


def test_stage1_config_rejects_dimension_mismatch() -> None:
    config = _model_config()

    with pytest.raises(ValueError, match="hidden dimensions"):
        CLaDStage1Config(
            inputs=config.inputs,
            attention=CrossAttentionConfig(
                hidden_dim=16,
                num_heads=4,
                num_layers=1,
            ),
            foresight=config.foresight,
        )


def test_stage1_batch_requires_feature_mapping() -> None:
    with pytest.raises(ValueError, match="vision_features"):
        CLaDStage1Batch.from_mapping({})
