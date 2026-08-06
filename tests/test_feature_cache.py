from __future__ import annotations

import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from clad.data import (
    DecisionNCEFeatureCache,
    DecisionNCEFeatureCacheBuilder,
    FeatureCacheSpec,
)
from clad.models import DecisionNCEAdapter, DecisionNCEAdapterConfig
from tests.fakes import FakeDecisionNCEBackend


def _create_task_file(root: Path) -> Path:
    path = root / "SYNTHETIC_CACHE_SCENE_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["num_demos"] = 1
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": "cache the object"}
        )
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((5, 2), dtype=np.float32))
        demo.create_dataset("robot_states", data=np.zeros((5, 3), dtype=np.float32))
        obs = demo.create_group("obs")
        agent_images = np.zeros((5, 4, 3, 3), dtype=np.uint8)
        wrist_images = np.zeros((5, 2, 2, 3), dtype=np.uint8)
        for step in range(5):
            agent_images[step] = step
            wrist_images[step] = step + 10
        obs.create_dataset("agentview_rgb", data=agent_images)
        obs.create_dataset("eye_in_hand_rgb", data=wrist_images)
    return path


def _builder() -> DecisionNCEFeatureCacheBuilder:
    config = DecisionNCEAdapterConfig(
        model_name="DecisionNCE-T",
        device="cpu",
        source_revision="test-revision",
        checkpoint_sha256="test-checkpoint",
    )
    adapter = DecisionNCEAdapter(
        backend=FakeDecisionNCEBackend(),
        config=config,
    )
    spec = FeatureCacheSpec(
        model_name="DecisionNCE-T",
        source_revision="test-revision",
        checkpoint_sha256="test-checkpoint",
        camera_keys=("obs/agentview_rgb", "obs/eye_in_hand_rgb"),
        feature_dtype="float16",
    )
    return DecisionNCEFeatureCacheBuilder(adapter=adapter, spec=spec, batch_size=2)


def test_builder_writes_and_reuses_versioned_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    dataset_dir.mkdir()
    _create_task_file(dataset_dir)
    builder = _builder()

    first_result = builder.build(dataset_dir=dataset_dir, cache_dir=cache_dir)
    second_result = builder.build(dataset_dir=dataset_dir, cache_dir=cache_dir)

    assert first_result.built_tasks == ("SYNTHETIC_CACHE_SCENE",)
    assert first_result.skipped_tasks == ()
    assert second_result.built_tasks == ()
    assert second_result.skipped_tasks == ("SYNTHETIC_CACHE_SCENE",)
    assert first_result.manifest_path.is_file()

    task_cache = cache_dir / "SYNTHETIC_CACHE_SCENE.hdf5"
    with h5py.File(task_cache, "r") as handle:
        assert handle.attrs["schema_version"] == 1
        assert handle.attrs["model_name"] == "DecisionNCE-T"
        assert handle.attrs["image_feature_dim"] == 4
        assert handle["text_feature"].shape == (4,)
        assert handle["text_feature"].dtype == np.float16
        assert handle["data/demo_0/images/agentview_rgb"].shape == (5, 4)
        assert handle["data/demo_0/images/eye_in_hand_rgb"].shape == (5, 4)


def test_cache_reader_returns_features_and_reopens_after_pickle(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    dataset_dir.mkdir()
    _create_task_file(dataset_dir)
    _builder().build(dataset_dir=dataset_dir, cache_dir=cache_dir)
    cache = DecisionNCEFeatureCache(cache_dir)

    try:
        text = cache.text_feature("SYNTHETIC_CACHE_SCENE")
        agent = cache.image_feature(
            task_id="SYNTHETIC_CACHE_SCENE",
            demo_key="demo_0",
            view_name="agentview_rgb",
            index=slice(1, 3),
        )
        wrist = cache.image_feature(
            task_id="SYNTHETIC_CACHE_SCENE",
            demo_key="demo_0",
            view_name="eye_in_hand_rgb",
            index=2,
        )

        assert text.shape == (4,)
        assert text.dtype == torch.float16
        assert agent.shape == (2, 4)
        assert wrist.shape == (4,)
        assert agent[0, 0].item() == 1.0
        assert wrist[0].item() == 12.0

        restored = pickle.loads(pickle.dumps(cache))
        try:
            assert not restored._files
            restored_text = restored.text_feature("SYNTHETIC_CACHE_SCENE")
            torch.testing.assert_close(restored_text, text)
        finally:
            restored.close()
    finally:
        cache.close()


def test_cache_propagates_common_rerender_metadata(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    dataset_dir.mkdir()
    task_path = _create_task_file(dataset_dir)
    with h5py.File(task_path, "r+") as handle:
        data = handle["data"]
        data.attrs["clad_rerender_schema_version"] = 1
        data.attrs["clad_render_height"] = 256
        data.attrs["clad_render_width"] = 256
        data.attrs["clad_image_transform"] = "rotate_180"
        data.attrs["clad_filter_noops"] = True
        data.attrs["clad_noop_threshold"] = 1e-4
        data.attrs["clad_settle_steps"] = 10
        data.attrs["clad_keep_only_successes"] = True

    _builder().build(dataset_dir=dataset_dir, cache_dir=cache_dir)
    cache = DecisionNCEFeatureCache(cache_dir)
    try:
        assert cache.dataset_metadata["clad_render_height"] == 256
        assert cache.dataset_metadata["clad_render_width"] == 256
        assert cache.dataset_metadata["clad_image_transform"] == "rotate_180"
    finally:
        cache.close()
