from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from clad.data import LiberoDatasetConfig, LiberoWindowDataset, discover_libero_tasks


def _create_demo(data_group: h5py.Group, demo_index: int, length: int) -> None:
    demo = data_group.create_group(f"demo_{demo_index}")
    offset = demo_index * 1_000
    steps = np.arange(length, dtype=np.float32) + offset

    demo.create_dataset("actions", data=np.stack([steps, -steps], axis=-1))
    demo.create_dataset("robot_states", data=np.stack([steps, steps + 0.5], axis=-1))
    obs = demo.create_group("obs")
    images = np.zeros((length, 4, 5, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = np.arange(length, dtype=np.uint8)
    obs.create_dataset("agentview_rgb", data=images)


def _create_task_file(root: Path) -> Path:
    path = root / "SYNTHETIC_SCENE_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["num_demos"] = 2
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": "move the synthetic object"}
        )
        _create_demo(data, demo_index=0, length=8)
        _create_demo(data, demo_index=1, length=7)
    return path


def test_task_discovery_reads_instruction_and_demo_count(tmp_path: Path) -> None:
    path = _create_task_file(tmp_path)

    tasks = discover_libero_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].path == path
    assert tasks[0].task_id == "SYNTHETIC_SCENE"
    assert tasks[0].instruction == "move the synthetic object"
    assert tasks[0].num_demos == 2


def test_dataset_builds_episode_safe_windows(tmp_path: Path) -> None:
    _create_task_file(tmp_path)
    dataset = LiberoWindowDataset(
        LiberoDatasetConfig(dataset_dir=tmp_path, horizon=2)
    )

    try:
        # Episode lengths 8 and 7 yield 4 and 3 anchors respectively.
        assert len(dataset) == 7

        first = dataset[0]
        assert first["episode_id"] == "demo_0"
        assert first["anchor_step"] == 2
        torch.testing.assert_close(first["proprio_prev"], torch.tensor([0.0, 0.5]))
        torch.testing.assert_close(first["proprio_now"], torch.tensor([2.0, 2.5]))
        torch.testing.assert_close(first["proprio_future"], torch.tensor([4.0, 4.5]))
        torch.testing.assert_close(
            first["past_actions"],
            torch.tensor([[0.0, 0.0], [1.0, -1.0]]),
        )
        torch.testing.assert_close(
            first["target_actions"],
            torch.tensor([[2.0, -2.0], [3.0, -3.0]]),
        )
        assert first["image_now"].shape == (4, 5, 3)
        assert first["image_now"].dtype == torch.uint8

        last = dataset[-1]
        assert last["episode_id"] == "demo_1"
        assert last["anchor_step"] == 4
        assert last["proprio_prev"][0].item() == 1002.0
        assert last["proprio_future"][0].item() == 1006.0
    finally:
        dataset.close()


def test_images_can_be_excluded_for_feature_only_loading(tmp_path: Path) -> None:
    _create_task_file(tmp_path)
    dataset = LiberoWindowDataset(
        LiberoDatasetConfig(
            dataset_dir=tmp_path,
            horizon=2,
            include_images=False,
        )
    )

    try:
        sample = dataset[0]
        assert not any(key.startswith("image_") for key in sample)
    finally:
        dataset.close()


def test_open_dataset_can_be_pickled_for_dataloader_workers(tmp_path: Path) -> None:
    import pickle

    _create_task_file(tmp_path)
    dataset = LiberoWindowDataset(
        LiberoDatasetConfig(dataset_dir=tmp_path, horizon=2)
    )

    try:
        _ = dataset[0]
        assert dataset._files

        restored = pickle.loads(pickle.dumps(dataset))
        try:
            assert not restored._files
            assert restored[0]["anchor_step"] == 2
        finally:
            restored.close()
    finally:
        dataset.close()
