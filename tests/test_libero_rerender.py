from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clad.data.image_transform import transform_rgb_image
from clad.data.libero_dataset import LiberoDatasetConfig, LiberoWindowDataset
from clad.data.libero_rerender import (
    LiberoRerenderConfig,
    is_noop_action,
    replay_demonstration,
    rerender_task_file,
)


class _FakeState:
    def __init__(self, value: int) -> None:
        self.value = value

    def flatten(self) -> np.ndarray:
        return np.array([self.value, self.value + 0.5], dtype=np.float64)


class _FakeSim:
    def __init__(self, environment: _FakeEnvironment) -> None:
        self.environment = environment

    def get_state(self) -> _FakeState:
        return _FakeState(self.environment.control_steps)


class _FakeEnvironment:
    def __init__(self, *, height: int = 2, width: int = 3) -> None:
        self.height = height
        self.width = width
        self.control_steps = 0
        self._settle_remaining = 0
        self.sim = _FakeSim(self)

    def _observation(self) -> dict[str, np.ndarray]:
        pixels = np.arange(self.height * self.width * 3, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        pixels = pixels + self.control_steps
        return {
            "agentview_image": pixels,
            "robot0_eye_in_hand_image": pixels + 20,
            "robot0_joint_pos": np.arange(7, dtype=np.float32) + self.control_steps,
            "robot0_joint_vel": np.arange(7, dtype=np.float32) + 0.25,
            "robot0_gripper_qpos": np.array([0.03, -0.03], dtype=np.float32),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    def reset(self) -> dict[str, np.ndarray]:
        self.control_steps = 0
        return self._observation()

    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        assert state.shape == (2,)
        return self._observation()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        del action
        if self._settle_remaining > 0:
            self._settle_remaining -= 1
        else:
            self.control_steps += 1
        success = self.control_steps >= 2
        return self._observation(), float(success), success, {}

    def check_success(self) -> bool:
        return self.control_steps >= 2


def test_named_rgb_transforms_are_spatial_and_contiguous() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    np.testing.assert_array_equal(transform_rgb_image(image, "none"), image)
    np.testing.assert_array_equal(
        transform_rgb_image(image, "flip_vertical"), image[::-1]
    )
    rotated = transform_rgb_image(image, "rotate_180")
    np.testing.assert_array_equal(rotated, image[::-1, ::-1])
    assert rotated.flags.c_contiguous


def test_openvla_noop_predicate_preserves_gripper_transitions() -> None:
    previous = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    same_gripper = previous.copy()
    changed_gripper = previous.copy()
    changed_gripper[-1] = 1
    moving = previous.copy()
    moving[0] = 0.01

    assert is_noop_action(same_gripper, None, threshold=1e-4)
    assert is_noop_action(same_gripper, previous, threshold=1e-4)
    assert not is_noop_action(changed_gripper, previous, threshold=1e-4)
    assert not is_noop_action(moving, previous, threshold=1e-4)


def test_replay_filters_noops_and_records_both_native_views() -> None:
    environment = _FakeEnvironment()
    actions = np.array(
        [
            [0, 0, 0, 0, 0, 0, -1],
            [0.1, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    config = LiberoRerenderConfig(
        render_height=2,
        render_width=3,
        settle_steps=0,
        image_transform="rotate_180",
        compression="none",
    )

    result = replay_demonstration(
        environment=environment,
        initial_state=np.zeros(2),
        actions=actions,
        config=config,
    )

    assert result.success
    assert result.retained_steps == 2
    assert result.removed_noops == 2
    np.testing.assert_array_equal(result.arrays["source_action_indices"], [1, 3])
    assert result.arrays["obs/agentview_rgb"].shape == (2, 2, 3, 3)
    assert result.arrays["obs/eye_in_hand_rgb"].shape == (2, 2, 3, 3)
    assert result.arrays["obs/joint_velocities"].shape == (2, 7)
    assert result.arrays["obs/ee_ori"].shape == (2, 3)
    assert result.arrays["obs/ee_states"].shape == (2, 6)
    expected_first = environment._observation()["agentview_image"] - 2
    np.testing.assert_array_equal(
        result.arrays["obs/agentview_rgb"][0], expected_first[::-1, ::-1]
    )


def _source_task(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["num_demos"] = 1
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": "move the object"}
        )
        data.attrs["env_args"] = json.dumps(
            {"env_kwargs": {"camera_heights": 128, "camera_widths": 128}}
        )
        demo = data.create_group("demo_0")
        demo.create_dataset("states", data=np.zeros((8, 2), dtype=np.float64))
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, 0] = 0.1
        demo.create_dataset("actions", data=actions)


def test_task_rerender_is_atomic_and_current_loader_compatible(tmp_path: Path) -> None:
    source = tmp_path / "source" / "SYNTHETIC_SCENE_demo.hdf5"
    destination = tmp_path / "output" / source.name
    source.parent.mkdir()
    _source_task(source)
    config = LiberoRerenderConfig(
        render_height=2,
        render_width=3,
        settle_steps=0,
        image_transform="none",
        compression="lzf",
    )

    result = rerender_task_file(
        source_path=source,
        destination_path=destination,
        environment=_FakeEnvironment(),
        config=config,
        log_interval=1,
    )

    assert result.written_demos == 1
    assert result.output_steps == 8
    assert destination.is_file()
    assert not destination.with_suffix(".hdf5.tmp").exists()
    with h5py.File(destination, "r") as handle:
        data = handle["data"]
        assert data.attrs["num_demos"] == 1
        assert data.attrs["clad_render_height"] == 2
        assert data.attrs["clad_render_width"] == 3
        assert data.attrs["clad_image_transform"] == "none"
        assert data.attrs["clad_environment_seed"] == 0
        env_args = json.loads(data.attrs["env_args"])
        assert env_args["env_kwargs"]["camera_heights"] == 2
        assert env_args["env_kwargs"]["camera_widths"] == 3
        assert handle["data/demo_0/obs/agentview_rgb"].shape == (8, 2, 3, 3)
        assert handle["data/demo_0/obs/eye_in_hand_rgb"].shape == (8, 2, 3, 3)
        assert handle["data/demo_0/obs/joint_velocities"].shape == (8, 7)
        assert handle["data/demo_0/obs/ee_ori"].shape == (8, 3)
        assert handle["data/demo_0/obs/ee_states"].shape == (8, 6)

    dataset = LiberoWindowDataset(
        LiberoDatasetConfig(
            dataset_dir=destination.parent,
            camera_keys=("obs/agentview_rgb", "obs/eye_in_hand_rgb"),
            horizon=2,
        )
    )
    try:
        assert len(dataset) == 4
        sample = dataset[0]
        assert tuple(sample["images"]) == ("agentview_rgb", "eye_in_hand_rgb")
        assert sample["proprio_now"].shape == (9,)
    finally:
        dataset.close()
