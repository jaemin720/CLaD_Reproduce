from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from clad.evaluation import (
    EncodedObservation,
    EpisodeResult,
    EvaluationRecorder,
    LiberoRolloutConfig,
    OnlineDecisionNCEEncoder,
    OnlineHistoryBuffer,
    PolicyPlan,
    libero_proprioception,
    rollout_episode,
)
from clad.evaluation.libero_rollout import _libero_video_frame


def _encoded(value: float) -> EncodedObservation:
    return EncodedObservation(
        vision_features={"agentview_rgb": torch.full((4,), value)},
        proprioception=torch.tensor([value, value + 1.0, value + 2.0]),
    )


def test_libero_proprioception_matches_training_robot_state_layout() -> None:
    observation = {
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
        "robot0_eef_pos": np.array([-0.2, 0.1, 1.1]),
        "robot0_eef_quat": np.array([1.0, 0.01, 0.02, 0.03]),
    }

    actual = libero_proprioception(observation)

    np.testing.assert_allclose(
        actual,
        [0.03, -0.03, -0.2, 0.1, 1.1, 1.0, 0.01, 0.02, 0.03],
    )
    assert actual.dtype == np.float32


def test_libero_video_frame_flips_only_the_display_copy() -> None:
    raw_frame = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    original = raw_frame.copy()

    video_frame = _libero_video_frame({"agentview_image": raw_frame}, "agentview_image")

    np.testing.assert_array_equal(video_frame, original[::-1])
    np.testing.assert_array_equal(raw_frame, original)
    assert video_frame.flags.c_contiguous


def test_online_history_repeat_zero_padding_and_sliding_window() -> None:
    buffer = OnlineHistoryBuffer(horizon=2, action_dim=2)
    buffer.reset(_encoded(0.0))

    padded = buffer.history(torch.arange(4.0))

    torch.testing.assert_close(padded.vision_prev["agentview_rgb"], torch.zeros(1, 4))
    torch.testing.assert_close(padded.vision_now["agentview_rgb"], torch.zeros(1, 4))
    torch.testing.assert_close(padded.past_actions, torch.zeros(1, 2, 2))
    buffer.append(np.array([1.0, 2.0]), _encoded(1.0))
    buffer.append(np.array([3.0, 4.0]), _encoded(2.0))
    history = buffer.history(torch.arange(4.0))
    torch.testing.assert_close(history.vision_prev["agentview_rgb"], torch.zeros(1, 4))
    torch.testing.assert_close(history.vision_now["agentview_rgb"], torch.full((1, 4), 2.0))
    torch.testing.assert_close(
        history.past_actions,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    )


class _FakeAdapter:
    device = torch.device("cpu")

    def encode_views(self, images: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: image.float().mean(dim=(1, 2)).mean(dim=-1, keepdim=True).repeat(1, 4)
            for name, image in images.items()
        }


class _FakeFeatureCache:
    camera_keys = ("obs/agentview_rgb",)
    manifest = {
        "tasks": [
            {
                "task_id": "task_name",
                "source": {"instruction": "do the task"},
            }
        ]
    }

    @staticmethod
    def text_feature(task_id: str) -> torch.Tensor:
        assert task_id == "task_name"
        return torch.arange(4.0)


def test_online_encoder_maps_live_camera_and_reuses_cached_text() -> None:
    encoder = OnlineDecisionNCEEncoder(
        adapter=_FakeAdapter(),  # type: ignore[arg-type]
        feature_cache=_FakeFeatureCache(),  # type: ignore[arg-type]
        camera_observation_keys={"agentview_rgb": "agentview_image"},
    )
    observation = {
        "agentview_image": np.full((3, 4, 3), 10, dtype=np.uint8),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
        "robot0_eef_pos": np.array([-0.2, 0.1, 1.1]),
        "robot0_eef_quat": np.array([1.0, 0.01, 0.02, 0.03]),
    }

    encoded = encoder.encode_observation(observation)

    torch.testing.assert_close(encoded.vision_features["agentview_rgb"], torch.full((4,), 10.0))
    torch.testing.assert_close(
        encoder.text_feature("task_name", "do the task"),
        torch.arange(4.0),
    )
    assert encoded.proprioception.shape == (9,)
    with pytest.raises(ValueError, match="does not match"):
        encoder.text_feature("task_name", "different instruction")


class _FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.seed_value: int | None = None

    @staticmethod
    def _observation() -> dict[str, np.ndarray]:
        return {"frame": np.zeros((2, 2, 3), dtype=np.uint8)}

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self) -> Mapping[str, Any]:
        self.actions.clear()
        return self._observation()

    def set_init_state(self, state: np.ndarray) -> Mapping[str, Any]:
        assert state.shape == (2,)
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[Mapping[str, Any], float, bool, dict[str, Any]]:
        self.actions.append(np.asarray(action).copy())
        success = len(self.actions) >= 5
        return self._observation(), float(success), success, {}

    def check_success(self) -> bool:
        return len(self.actions) >= 5


class _FakePolicy:
    execution_steps = 2

    def __init__(self) -> None:
        self.observed: list[np.ndarray] = []
        self.plan_calls = 0
        self.reset_values: tuple[str, str, int] | None = None

    def reset(
        self,
        *,
        task_id: str,
        instruction: str,
        observation: Mapping[str, Any],
        seed: int,
    ) -> None:
        assert "frame" in observation
        self.reset_values = (task_id, instruction, seed)

    def observe(self, action: np.ndarray, observation: Mapping[str, Any]) -> None:
        assert "frame" in observation
        self.observed.append(np.asarray(action).copy())

    def plan(self) -> PolicyPlan:
        self.plan_calls += 1
        return PolicyPlan(
            actions=np.array([[2.0, -2.0], [0.25, -0.25]], dtype=np.float32),
            inference_seconds=0.1,
        )


def test_rollout_executes_chunks_clips_actions_and_stops_on_success() -> None:
    environment = _FakeEnvironment()
    policy = _FakePolicy()

    result = rollout_episode(
        environment=environment,
        policy=policy,
        initial_state=np.array([1.0, 2.0]),
        task_id=3,
        task_name="task_name",
        instruction="do the task",
        rollout_id=4,
        init_state_id=1,
        seed=123,
        max_steps=10,
        warmup_steps=2,
        action_dim=2,
        clip_actions=True,
    )

    assert result.success
    assert result.steps == 3
    assert result.policy_calls == 2
    assert result.inference_seconds == pytest.approx(0.2)
    assert policy.reset_values == ("task_name", "do the task", 123)
    assert len(policy.observed) == 5
    np.testing.assert_allclose(environment.actions[2], [1.0, -1.0])


def _result(*, rollout_id: int, success: bool) -> EpisodeResult:
    return EpisodeResult(
        task_id=0,
        task_name="task",
        instruction="instruction",
        rollout_id=rollout_id,
        init_state_id=rollout_id,
        seed=rollout_id,
        success=success,
        steps=5,
        total_reward=float(success),
        policy_calls=2,
        inference_seconds=0.4,
    )


def test_evaluation_recorder_is_resumable_and_writes_success_summary(tmp_path: Path) -> None:
    identity = {"checkpoint": "abc", "down_dims": (16, 32)}
    recorder = EvaluationRecorder(tmp_path, run_identity=identity, resume=True)
    recorder.record(_result(rollout_id=0, success=True))
    recorder.record(_result(rollout_id=1, success=False))

    resumed = EvaluationRecorder(tmp_path, run_identity=identity, resume=True)
    summary = resumed.write_summary()

    assert resumed.completed(0, 0)
    assert summary["completed_rollouts"] == 2
    assert summary["macro_task_success_rate"] == pytest.approx(0.5)
    assert summary["tasks"]["0"]["mean_inference_seconds_per_policy_call"] == pytest.approx(0.2)
    assert len((tmp_path / "episode_results.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with pytest.raises(ValueError, match="different run identity"):
        EvaluationRecorder(tmp_path, run_identity={"checkpoint": "other"}, resume=True)


def test_rollout_config_accepts_yaml_lists_and_rejects_invalid_values() -> None:
    config = LiberoRolloutConfig.from_mapping({"task_ids": [1, 2], "execution_steps": 3})
    assert config.task_ids == (1, 2)
    with pytest.raises(ValueError, match="positive"):
        LiberoRolloutConfig(max_steps=0)
    with pytest.raises(ValueError, match="Unknown"):
        LiberoRolloutConfig.from_mapping({"episodes": 20})
