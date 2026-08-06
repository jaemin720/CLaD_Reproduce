from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from clad.evaluation import (
    BatchedPolicyPlan,
    CLaDOnlinePolicy,
    EncodedObservation,
    EpisodeResult,
    EvaluationRecorder,
    LiberoRolloutConfig,
    OnlineDecisionNCEEncoder,
    OnlineHistoryBuffer,
    PolicyPlan,
    libero_proprioception,
    rollout_episode,
    rollout_episode_batch,
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

    actual = libero_proprioception(observation, contract="robot_states")

    np.testing.assert_allclose(
        actual,
        [0.03, -0.03, -0.2, 0.1, 1.1, 1.0, 0.01, 0.02, 0.03],
    )
    assert actual.dtype == np.float32


def test_libero_proprioception_matches_official_joint_gripper_layout() -> None:
    observation = {
        "robot0_joint_pos": np.arange(7, dtype=np.float32) + 0.5,
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
    }

    actual = libero_proprioception(observation)

    np.testing.assert_allclose(
        actual,
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.03, -0.03],
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


def test_batched_history_preserves_camera_view_order() -> None:
    observation = EncodedObservation(
        vision_features={
            "eye_in_hand_rgb": torch.ones(4),
            "agentview_rgb": torch.zeros(4),
        },
        proprioception=torch.zeros(3),
    )
    buffers = [OnlineHistoryBuffer(horizon=2, action_dim=2) for _ in range(2)]
    for buffer in buffers:
        buffer.reset(observation)

    batched = CLaDOnlinePolicy._stack_histories(
        [buffer.history(torch.arange(4.0)) for buffer in buffers]
    )

    assert tuple(batched.vision_prev) == ("eye_in_hand_rgb", "agentview_rgb")
    assert tuple(batched.vision_now) == ("eye_in_hand_rgb", "agentview_rgb")


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


class _CapturingAdapter(_FakeAdapter):
    def __init__(self) -> None:
        self.images: dict[str, torch.Tensor] = {}

    def encode_views(self, images: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.images = {name: image.clone() for name, image in images.items()}
        return super().encode_views(images)


class _RerenderedFeatureCache(_FakeFeatureCache):
    dataset_metadata = {
        "clad_render_height": 2,
        "clad_render_width": 3,
        "clad_image_transform": "rotate_180",
    }


def test_online_encoder_maps_live_camera_and_reuses_cached_text() -> None:
    encoder = OnlineDecisionNCEEncoder(
        adapter=_FakeAdapter(),  # type: ignore[arg-type]
        feature_cache=_FakeFeatureCache(),  # type: ignore[arg-type]
        camera_observation_keys={"agentview_rgb": "agentview_image"},
    )
    observation = {
        "agentview_image": np.full((3, 4, 3), 10, dtype=np.uint8),
        "robot0_joint_pos": np.arange(7, dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
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


def test_online_encoder_batches_live_frames_per_view() -> None:
    encoder = OnlineDecisionNCEEncoder(
        adapter=_FakeAdapter(),  # type: ignore[arg-type]
        feature_cache=_FakeFeatureCache(),  # type: ignore[arg-type]
        camera_observation_keys={"agentview_rgb": "agentview_image"},
    )
    observations = [
        {
            "agentview_image": np.full((3, 4, 3), value, dtype=np.uint8),
            "robot0_joint_pos": np.arange(7, dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.03, -0.03]),
        }
        for value in (10, 20)
    ]

    encoded = encoder.encode_observations(observations)

    assert len(encoded) == 2
    torch.testing.assert_close(encoded[0].vision_features["agentview_rgb"], torch.full((4,), 10.0))
    torch.testing.assert_close(encoded[1].vision_features["agentview_rgb"], torch.full((4,), 20.0))


def test_online_encoder_replays_cached_image_geometry() -> None:
    adapter = _CapturingAdapter()
    encoder = OnlineDecisionNCEEncoder(
        adapter=adapter,  # type: ignore[arg-type]
        feature_cache=_RerenderedFeatureCache(),  # type: ignore[arg-type]
        camera_observation_keys={"agentview_rgb": "agentview_image"},
    )
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    observation = {
        "agentview_image": image,
        "robot0_joint_pos": np.arange(7, dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
    }

    encoder.encode_observation(observation)

    np.testing.assert_array_equal(
        adapter.images["agentview_rgb"][0].numpy(), image[::-1, ::-1]
    )
    assert encoder.source_image_size == (2, 3)
    assert encoder.image_transform == "rotate_180"


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
        environment_seed=0,
        max_steps=10,
        warmup_steps=2,
        warmup_gripper_action=-1.0,
        action_dim=2,
        clip_actions=True,
    )

    assert result.success
    assert result.steps == 3
    assert result.policy_calls == 2
    assert result.inference_seconds == pytest.approx(0.2)
    assert policy.reset_values == ("task_name", "do the task", 123)
    assert environment.seed_value == 0
    assert len(policy.observed) == 5
    np.testing.assert_allclose(environment.actions[0], [0.0, -1.0])
    np.testing.assert_allclose(environment.actions[1], [0.0, -1.0])
    np.testing.assert_allclose(environment.actions[2], [1.0, -1.0])


class _FakeVectorEnvironment:
    def __init__(self, success_after: tuple[int, ...]) -> None:
        self.success_after = success_after
        self.counts = [0] * len(success_after)
        self.seed_values: list[int | None] = [None] * len(success_after)
        self.actions: list[tuple[int, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self.counts)

    @staticmethod
    def _observation(slot_id: int) -> dict[str, np.ndarray]:
        return {"frame": np.full((2, 2, 3), slot_id, dtype=np.uint8)}

    def seed(self, seeds: list[int | None]) -> None:
        self.seed_values = list(seeds)

    def reset(self, id: list[int]) -> list[Mapping[str, Any]]:
        for slot_id in id:
            self.counts[slot_id] = 0
        return [self._observation(slot_id) for slot_id in id]

    def set_init_state(
        self,
        states: np.ndarray,
        id: list[int],
    ) -> list[Mapping[str, Any]]:
        assert len(states) == len(id)
        return [self._observation(slot_id) for slot_id in id]

    def step(
        self,
        actions: np.ndarray,
        id: list[int],
    ) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        rewards: list[float] = []
        dones: list[bool] = []
        for slot_id, action in zip(id, actions, strict=True):
            self.counts[slot_id] += 1
            self.actions.append((slot_id, np.asarray(action).copy()))
            success = self.counts[slot_id] >= self.success_after[slot_id]
            rewards.append(float(success))
            dones.append(success)
        return (
            [self._observation(slot_id) for slot_id in id],
            np.asarray(rewards),
            np.asarray(dones),
            [{} for _ in id],
        )

    def check_success(self) -> list[bool]:
        return [
            count >= threshold
            for count, threshold in zip(self.counts, self.success_after, strict=True)
        ]


class _FakeVectorPolicy:
    execution_steps = 2

    def __init__(self) -> None:
        self.seeds: tuple[int, ...] = ()
        self.observed_slots: list[tuple[int, ...]] = []

    def reset_batch(
        self,
        *,
        slot_ids: tuple[int, ...],
        task_ids: tuple[str, ...],
        instructions: tuple[str, ...],
        observations: list[Mapping[str, Any]],
        seeds: tuple[int, ...],
    ) -> None:
        assert len(slot_ids) == len(task_ids) == len(instructions) == len(observations)
        self.seeds = seeds

    def observe_batch(
        self,
        *,
        slot_ids: list[int],
        actions: np.ndarray,
        observations: list[Mapping[str, Any]],
    ) -> None:
        assert len(slot_ids) == len(actions) == len(observations)
        self.observed_slots.append(tuple(slot_ids))

    def plan_batch(self, slot_ids: list[int]) -> BatchedPolicyPlan:
        actions = np.tile(
            np.array([[2.0, -2.0], [0.25, -0.25]], dtype=np.float32),
            (len(slot_ids), 1, 1),
        )
        return BatchedPolicyPlan(
            slot_ids=tuple(slot_ids),
            actions=actions,
            inference_seconds=0.2,
        )


def test_vector_rollout_keeps_episode_state_independent_and_drops_done_slots() -> None:
    environment = _FakeVectorEnvironment(success_after=(3, 5))
    policy = _FakeVectorPolicy()

    results = rollout_episode_batch(
        environment=environment,
        policy=policy,
        slot_ids=(0, 1),
        initial_states=(np.array([1.0]), np.array([2.0])),
        task_id=3,
        task_name="task_name",
        instruction="do the task",
        rollout_ids=(4, 5),
        init_state_ids=(0, 1),
        seeds=(104, 105),
        environment_seed=0,
        max_steps=10,
        warmup_steps=1,
        warmup_gripper_action=-1.0,
        action_dim=2,
        clip_actions=True,
    )

    assert [result.success for result in results] == [True, True]
    assert [result.steps for result in results] == [2, 4]
    assert [result.policy_calls for result in results] == [1, 2]
    assert [result.inference_seconds for result in results] == pytest.approx([0.2, 0.4])
    assert policy.seeds == (104, 105)
    assert environment.seed_values == [0, 0]
    assert [result.environment_seed for result in results] == [0, 0]
    np.testing.assert_allclose(environment.actions[0][1], [0.0, -1.0])
    np.testing.assert_allclose(environment.actions[1][1], [0.0, -1.0])
    assert policy.observed_slots[-1] == (1,)
    assert max(np.abs(action).max() for _, action in environment.actions) <= 1.0


def _result(*, rollout_id: int, success: bool) -> EpisodeResult:
    return EpisodeResult(
        task_id=0,
        task_name="task",
        instruction="instruction",
        rollout_id=rollout_id,
        init_state_id=rollout_id,
        seed=rollout_id,
        environment_seed=0,
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
    assert config.num_envs == 4
    assert config.warmup_steps == 10
    assert config.warmup_gripper_action == -1.0
    assert config.environment_seed == 0
    with pytest.raises(ValueError, match="positive"):
        LiberoRolloutConfig(max_steps=0)
    with pytest.raises(ValueError, match="num_envs"):
        LiberoRolloutConfig(num_envs=0)
    with pytest.raises(ValueError, match="warmup_gripper_action"):
        LiberoRolloutConfig(warmup_gripper_action=-2.0)
    with pytest.raises(ValueError, match="environment_seed"):
        LiberoRolloutConfig(environment_seed=-1)
    with pytest.raises(ValueError, match="Unknown"):
        LiberoRolloutConfig.from_mapping({"episodes": 20})
