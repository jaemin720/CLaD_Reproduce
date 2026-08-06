"""Fixed-initial-state rollout evaluation on official LIBERO vector envs."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from clad.evaluation.online_policy import BatchedPolicyPlan, CLaDOnlinePolicy, PolicyPlan


@dataclass(frozen=True, slots=True)
class LiberoRolloutConfig:
    """Environment and protocol settings for LIBERO success evaluation."""

    suite_name: str = "libero_10"
    task_order_index: int = 0
    task_ids: tuple[int, ...] = ()
    rollouts_per_task: int = 50
    num_envs: int = 4
    max_steps: int = 600
    warmup_steps: int = 10
    warmup_gripper_action: float = -1.0
    execution_steps: int = 6
    camera_height: int = 128
    camera_width: int = 128
    environment_seed: int = 0
    seed: int = 42
    clip_actions: bool = True
    save_videos: bool = False
    video_fps: int = 20
    video_observation_key: str = "agentview_image"
    render_gpu_device_id: int = -1
    resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        if not self.suite_name:
            raise ValueError("suite_name cannot be empty")
        if self.task_order_index < 0:
            raise ValueError("task_order_index must be non-negative")
        if any(task_id < 0 for task_id in self.task_ids):
            raise ValueError("task_ids must be non-negative")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids cannot contain duplicates")
        for name, value in {
            "rollouts_per_task": self.rollouts_per_task,
            "num_envs": self.num_envs,
            "max_steps": self.max_steps,
            "execution_steps": self.execution_steps,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "video_fps": self.video_fps,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not -1.0 <= self.warmup_gripper_action <= 1.0:
            raise ValueError("warmup_gripper_action must be in [-1, 1]")
        if self.environment_seed < 0:
            raise ValueError("environment_seed must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.video_observation_key:
            raise ValueError("video_observation_key cannot be empty")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LiberoRolloutConfig:
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown LIBERO rollout settings: {unknown}")
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One fixed-initial-state rollout result."""

    task_id: int
    task_name: str
    instruction: str
    rollout_id: int
    init_state_id: int
    seed: int
    environment_seed: int
    success: bool
    steps: int
    total_reward: float
    policy_calls: int
    inference_seconds: float


class _RolloutPolicy(Protocol):
    execution_steps: int

    def reset(
        self,
        *,
        task_id: str,
        instruction: str,
        observation: Mapping[str, Any],
        seed: int,
    ) -> None: ...

    def observe(self, action: np.ndarray, observation: Mapping[str, Any]) -> None: ...

    def plan(self) -> PolicyPlan: ...


class _VectorRolloutPolicy(Protocol):
    execution_steps: int

    def reset_batch(
        self,
        *,
        slot_ids: Sequence[int],
        task_ids: Sequence[str],
        instructions: Sequence[str],
        observations: Sequence[Mapping[str, Any]],
        seeds: Sequence[int],
    ) -> None: ...

    def observe_batch(
        self,
        *,
        slot_ids: Sequence[int],
        actions: np.ndarray,
        observations: Sequence[Mapping[str, Any]],
    ) -> None: ...

    def plan_batch(self, slot_ids: Sequence[int]) -> BatchedPolicyPlan: ...


class _VideoSink(Protocol):
    def append(self, observation: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class _NullVideoSink:
    def append(self, observation: Mapping[str, Any]) -> None:
        del observation

    def close(self) -> None:
        pass


def _libero_video_frame(observation: Mapping[str, Any], observation_key: str) -> np.ndarray:
    if observation_key not in observation:
        raise KeyError(f"LIBERO observation is missing video field {observation_key!r}")
    frame = np.asarray(observation[observation_key], dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"LIBERO video field {observation_key!r} must have shape [H,W,3], got {frame.shape}"
        )
    # robosuite 1.4 exposes camera observations in OpenGL row order. Match
    # LIBERO's official VideoWriter by flipping only the display artifact;
    # policy inputs retain the raw train/eval-compatible observation.
    return np.ascontiguousarray(frame[::-1])


class _ImageioVideoSink:
    def __init__(self, path: Path, *, fps: int, observation_key: str) -> None:
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as error:  # pragma: no cover - optional runtime
            raise ModuleNotFoundError(
                "Video export requires imageio and imageio-ffmpeg. "
                "Install this project with `pip install -e '.[eval]'`."
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(path, fps=fps)
        self._observation_key = observation_key

    def append(self, observation: Mapping[str, Any]) -> None:
        self._writer.append_data(_libero_video_frame(observation, self._observation_key))

    def close(self) -> None:
        self._writer.close()


def _is_success(environment: Any, reward: float) -> bool:
    if reward > 0.0:
        return True
    check_success = getattr(environment, "check_success", None)
    return bool(check_success()) if check_success is not None else False


def _warmup_action(action_dim: int, gripper_action: float) -> np.ndarray:
    """Build the LIBERO no-op used to settle physics with an open gripper."""

    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if not -1.0 <= gripper_action <= 1.0:
        raise ValueError("gripper_action must be in [-1, 1]")
    action = np.zeros(action_dim, dtype=np.float32)
    action[-1] = gripper_action
    return action


def rollout_episode(
    *,
    environment: Any,
    policy: _RolloutPolicy,
    initial_state: Any,
    task_id: int,
    task_name: str,
    instruction: str,
    rollout_id: int,
    init_state_id: int,
    seed: int,
    environment_seed: int,
    max_steps: int,
    warmup_steps: int,
    warmup_gripper_action: float,
    action_dim: int,
    clip_actions: bool,
    video: _VideoSink | None = None,
) -> EpisodeResult:
    """Run one episode while keeping the CLaD history synchronized with actions."""

    sink = video or _NullVideoSink()
    environment.seed(environment_seed)
    environment.reset()
    observation = environment.set_init_state(np.asarray(initial_state))
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO set_init_state() must return an observation mapping")
    policy.reset(
        task_id=task_name,
        instruction=instruction,
        observation=observation,
        seed=seed,
    )
    sink.append(observation)

    total_reward = 0.0
    success = _is_success(environment, 0.0)
    terminated = False
    warmup_action = _warmup_action(action_dim, warmup_gripper_action)
    for _ in range(warmup_steps):
        observation, reward, done, _ = environment.step(warmup_action)
        reward_value = float(reward)
        total_reward += reward_value
        policy.observe(warmup_action, observation)
        sink.append(observation)
        success = _is_success(environment, reward_value)
        terminated = bool(done) and not success
        if success or terminated:
            break

    steps = 0
    policy_calls = 0
    inference_seconds = 0.0
    while not success and not terminated and steps < max_steps:
        plan = policy.plan()
        policy_calls += 1
        inference_seconds += plan.inference_seconds
        if plan.actions.ndim != 2 or plan.actions.shape[1] != action_dim:
            raise ValueError(
                f"Policy plan must have shape [K,{action_dim}], got {plan.actions.shape}"
            )
        if plan.actions.shape[0] == 0:
            raise ValueError("Policy plan cannot be empty")
        for sampled_action in plan.actions[: max_steps - steps]:
            action = np.asarray(sampled_action, dtype=np.float32)
            if clip_actions:
                action = np.clip(action, -1.0, 1.0)
            observation, reward, done, _ = environment.step(action)
            reward_value = float(reward)
            total_reward += reward_value
            steps += 1
            policy.observe(action, observation)
            sink.append(observation)
            success = _is_success(environment, reward_value)
            terminated = bool(done) and not success
            if success or terminated or steps >= max_steps:
                break

    return EpisodeResult(
        task_id=task_id,
        task_name=task_name,
        instruction=instruction,
        rollout_id=rollout_id,
        init_state_id=init_state_id,
        seed=seed,
        environment_seed=environment_seed,
        success=success,
        steps=steps,
        total_reward=total_reward,
        policy_calls=policy_calls,
        inference_seconds=inference_seconds,
    )


@dataclass(slots=True)
class _EpisodeProgress:
    task_id: int
    task_name: str
    instruction: str
    rollout_id: int
    init_state_id: int
    seed: int
    success: bool = False
    terminated: bool = False
    steps: int = 0
    total_reward: float = 0.0
    policy_calls: int = 0
    inference_seconds: float = 0.0


def _observation_batch(values: Any, *, expected: int) -> list[Mapping[str, Any]]:
    if isinstance(values, Mapping):
        observations: list[Any] = [values]
    else:
        observations = list(values)
    if len(observations) != expected:
        raise ValueError(
            f"Vector environment returned {len(observations)} observations, expected {expected}"
        )
    if any(not isinstance(value, Mapping) for value in observations):
        raise TypeError("LIBERO vector observations must contain mappings")
    return observations


def _vector_step(
    environment: Any,
    actions: np.ndarray,
    slot_ids: Sequence[int],
) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray]:
    result = environment.step(actions, id=list(slot_ids))
    if len(result) == 4:
        observations, rewards, dones, _ = result
    elif len(result) == 5:
        observations, rewards, terminated, truncated, _ = result
        dones = np.logical_or(terminated, truncated)
    else:
        raise ValueError(f"Unexpected vector environment step result length: {len(result)}")
    count = len(slot_ids)
    reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    done_values = np.asarray(dones, dtype=bool).reshape(-1)
    if reward_values.shape != (count,) or done_values.shape != (count,):
        raise ValueError("Vector environment rewards and dones must match active slots")
    return (
        _observation_batch(observations, expected=count),
        reward_values,
        done_values,
    )


def _success_by_slot(environment: Any, slot_ids: Sequence[int]) -> dict[int, bool]:
    successes = np.asarray(environment.check_success(), dtype=bool).reshape(-1)
    if any(slot_id < 0 or slot_id >= len(successes) for slot_id in slot_ids):
        raise ValueError("Vector environment check_success result does not cover active slots")
    return {slot_id: bool(successes[slot_id]) for slot_id in slot_ids}


def rollout_episode_batch(
    *,
    environment: Any,
    policy: _VectorRolloutPolicy,
    slot_ids: Sequence[int],
    initial_states: Sequence[Any],
    task_id: int,
    task_name: str,
    instruction: str,
    rollout_ids: Sequence[int],
    init_state_ids: Sequence[int],
    seeds: Sequence[int],
    environment_seed: int,
    max_steps: int,
    warmup_steps: int,
    warmup_gripper_action: float,
    action_dim: int,
    clip_actions: bool,
    videos: Sequence[_VideoSink] | None = None,
) -> list[EpisodeResult]:
    """Run one synchronous wave with independent history and RNG per env."""

    slots = tuple(int(value) for value in slot_ids)
    states = tuple(initial_states)
    rollout_values = tuple(int(value) for value in rollout_ids)
    init_state_values = tuple(int(value) for value in init_state_ids)
    episode_seeds = tuple(int(value) for value in seeds)
    batch_size = len(slots)
    if not batch_size or len(set(slots)) != batch_size:
        raise ValueError("slot_ids must be non-empty and unique")
    if any(
        len(values) != batch_size
        for values in (states, rollout_values, init_state_values, episode_seeds)
    ):
        raise ValueError("All batched episode fields must match slot_ids")
    sinks = tuple(videos) if videos is not None else tuple(_NullVideoSink() for _ in slots)
    if len(sinks) != batch_size:
        raise ValueError("videos must contain one sink per episode")
    sink_by_slot = dict(zip(slots, sinks, strict=True))

    vector_size = len(environment)
    seed_batch: list[int | None] = [None] * vector_size
    for slot_id in slots:
        if not 0 <= slot_id < vector_size:
            raise ValueError(f"slot_id {slot_id} is outside vector environment")
        seed_batch[slot_id] = environment_seed
    environment.seed(seed_batch)
    environment.reset(id=list(slots))
    raw_observations = environment.set_init_state(
        np.stack([np.asarray(value) for value in states]),
        id=list(slots),
    )
    observations = _observation_batch(raw_observations, expected=batch_size)
    policy.reset_batch(
        slot_ids=slots,
        task_ids=(task_name,) * batch_size,
        instructions=(instruction,) * batch_size,
        observations=observations,
        seeds=episode_seeds,
    )
    for slot_id, observation in zip(slots, observations, strict=True):
        sink_by_slot[slot_id].append(observation)

    progress = {
        slot_id: _EpisodeProgress(
            task_id=task_id,
            task_name=task_name,
            instruction=instruction,
            rollout_id=rollout_id,
            init_state_id=init_state_id,
            seed=seed,
        )
        for slot_id, rollout_id, init_state_id, seed in zip(
            slots,
            rollout_values,
            init_state_values,
            episode_seeds,
            strict=True,
        )
    }
    initial_success = _success_by_slot(environment, slots)
    for slot_id in slots:
        progress[slot_id].success = initial_success[slot_id]

    active = [slot_id for slot_id in slots if not progress[slot_id].success]
    for _ in range(warmup_steps):
        if not active:
            break
        actions = np.repeat(
            _warmup_action(action_dim, warmup_gripper_action)[None, :],
            len(active),
            axis=0,
        )
        observations, rewards, dones = _vector_step(environment, actions, active)
        policy.observe_batch(slot_ids=active, actions=actions, observations=observations)
        success_flags = _success_by_slot(environment, active)
        for index, (slot_id, observation) in enumerate(zip(active, observations, strict=True)):
            episode = progress[slot_id]
            episode.total_reward += float(rewards[index])
            sink_by_slot[slot_id].append(observation)
            episode.success = bool(rewards[index] > 0.0 or success_flags[slot_id])
            episode.terminated = bool(dones[index]) and not episode.success
        active = [
            slot_id
            for slot_id in active
            if not progress[slot_id].success and not progress[slot_id].terminated
        ]

    while active:
        plan = policy.plan_batch(active)
        if plan.slot_ids != tuple(active):
            raise ValueError("Batched policy plan slot order does not match active env slots")
        if plan.actions.ndim != 3 or plan.actions.shape[:1] != (len(active),):
            raise ValueError(
                f"Batched policy actions must have shape [B,K,A], got {plan.actions.shape}"
            )
        if plan.actions.shape[1] == 0 or plan.actions.shape[2] != action_dim:
            raise ValueError(
                f"Batched policy actions must have non-empty K and A={action_dim}, "
                f"got {plan.actions.shape}"
            )
        plan_index = {slot_id: index for index, slot_id in enumerate(active)}
        for slot_id in active:
            progress[slot_id].policy_calls += 1
            progress[slot_id].inference_seconds += plan.inference_seconds

        for chunk_index in range(plan.actions.shape[1]):
            step_slots = [slot_id for slot_id in active if progress[slot_id].steps < max_steps]
            if not step_slots:
                active = []
                break
            actions = np.stack(
                [plan.actions[plan_index[slot_id], chunk_index] for slot_id in step_slots]
            ).astype(np.float32, copy=False)
            if clip_actions:
                actions = np.clip(actions, -1.0, 1.0)
            observations, rewards, dones = _vector_step(environment, actions, step_slots)
            policy.observe_batch(
                slot_ids=step_slots,
                actions=actions,
                observations=observations,
            )
            success_flags = _success_by_slot(environment, step_slots)
            for index, (slot_id, observation) in enumerate(
                zip(step_slots, observations, strict=True)
            ):
                episode = progress[slot_id]
                episode.total_reward += float(rewards[index])
                episode.steps += 1
                sink_by_slot[slot_id].append(observation)
                episode.success = bool(rewards[index] > 0.0 or success_flags[slot_id])
                episode.terminated = bool(dones[index]) and not episode.success
            active = [
                slot_id
                for slot_id in active
                if not progress[slot_id].success
                and not progress[slot_id].terminated
                and progress[slot_id].steps < max_steps
            ]
            if not active:
                break

    return [
        EpisodeResult(
            task_id=progress[slot_id].task_id,
            task_name=progress[slot_id].task_name,
            instruction=progress[slot_id].instruction,
            rollout_id=progress[slot_id].rollout_id,
            init_state_id=progress[slot_id].init_state_id,
            seed=progress[slot_id].seed,
            environment_seed=environment_seed,
            success=progress[slot_id].success,
            steps=progress[slot_id].steps,
            total_reward=progress[slot_id].total_reward,
            policy_calls=progress[slot_id].policy_calls,
            inference_seconds=progress[slot_id].inference_seconds,
        )
        for slot_id in slots
    ]


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class EvaluationRecorder:
    """Append-only episode recorder with strict run-identity resume checks."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_identity: Mapping[str, Any],
        resume: bool,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.output_dir / "run_identity.json"
        self.results_path = self.output_dir / "episode_results.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        # Canonicalize tuples, Paths supplied by callers, and mapping order to
        # exactly the JSON representation used on a later resumed process.
        identity = json.loads(json.dumps(dict(run_identity), sort_keys=True, default=str))
        if self.identity_path.exists():
            existing = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if existing != identity:
                raise ValueError(
                    "Evaluation output directory belongs to a different run identity: "
                    f"{self.identity_path}"
                )
            if not resume:
                raise FileExistsError(
                    f"Evaluation output already exists: {self.output_dir}; enable resume "
                    "or choose another output directory"
                )
        else:
            _atomic_write_json(self.identity_path, identity)

        self._results: dict[tuple[int, int], EpisodeResult] = {}
        if self.results_path.exists():
            for line_number, line in enumerate(
                self.results_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                try:
                    values = json.loads(line)
                    result = EpisodeResult(**values)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid evaluation record at {self.results_path}:{line_number}"
                    ) from error
                key = (result.task_id, result.rollout_id)
                if key in self._results:
                    raise ValueError(f"Duplicate evaluation episode record: {key}")
                self._results[key] = result

    def completed(self, task_id: int, rollout_id: int) -> bool:
        return (task_id, rollout_id) in self._results

    def record(self, result: EpisodeResult) -> None:
        key = (result.task_id, result.rollout_id)
        if key in self._results:
            raise ValueError(f"Evaluation episode was already recorded: {key}")
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._results[key] = result
        self.write_summary()

    def write_summary(self) -> dict[str, Any]:
        grouped: dict[int, list[EpisodeResult]] = defaultdict(list)
        for result in self._results.values():
            grouped[result.task_id].append(result)
        tasks: dict[str, Any] = {}
        task_success_rates: list[float] = []
        for task_id, results in sorted(grouped.items()):
            successes = sum(result.success for result in results)
            rate = successes / len(results)
            task_success_rates.append(rate)
            tasks[str(task_id)] = {
                "task_name": results[0].task_name,
                "instruction": results[0].instruction,
                "completed_rollouts": len(results),
                "successes": successes,
                "success_rate": rate,
                "mean_steps": sum(result.steps for result in results) / len(results),
                "mean_inference_seconds_per_policy_call": (
                    sum(result.inference_seconds for result in results)
                    / max(1, sum(result.policy_calls for result in results))
                ),
            }
        total = len(self._results)
        total_successes = sum(result.success for result in self._results.values())
        summary = {
            "completed_rollouts": total,
            "successes": total_successes,
            "episode_weighted_success_rate": total_successes / total if total else None,
            "macro_task_success_rate": (
                sum(task_success_rates) / len(task_success_rates) if task_success_rates else None
            ),
            "tasks": tasks,
        }
        _atomic_write_json(self.summary_path, summary)
        return summary


def require_libero_runtime() -> tuple[Any, Any]:
    """Import LIBERO lazily so training and unit tests do not require MuJoCo."""

    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as error:  # pragma: no cover - optional runtime
        raise ModuleNotFoundError(
            "LIBERO rollout dependencies are not installed. Install the official "
            "Lifelong-Robot-Learning/LIBERO repository and its robosuite/bddl "
            "dependencies in the clad environment."
        ) from error
    return benchmark, OffScreenRenderEnv


def evaluate_libero(
    *,
    policy: CLaDOnlinePolicy,
    config: LiberoRolloutConfig,
    recorder: EvaluationRecorder,
) -> dict[str, Any]:
    """Evaluate selected tasks and update a resumable summary after every episode."""

    benchmark, environment_class = require_libero_runtime()
    from libero.libero.envs import DummyVectorEnv, SubprocVectorEnv

    benchmark_types = benchmark.get_benchmark_dict()
    if config.suite_name not in benchmark_types:
        raise ValueError(
            f"Unknown LIBERO suite {config.suite_name!r}; available={sorted(benchmark_types)}"
        )
    suite = benchmark_types[config.suite_name](config.task_order_index)
    task_ids: Sequence[int] = config.task_ids or tuple(range(suite.n_tasks))
    invalid = [task_id for task_id in task_ids if not 0 <= task_id < suite.n_tasks]
    if invalid:
        raise ValueError(f"Task ids outside suite range [0, {suite.n_tasks}): {invalid}")
    if config.execution_steps != policy.execution_steps:
        raise ValueError(
            "Rollout and online-policy execution_steps differ: "
            f"{config.execution_steps} != {policy.execution_steps}"
        )

    videos_dir = recorder.output_dir / "videos"
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) == 0:
            raise ValueError(f"LIBERO task {task_id} has no fixed initial states")
        pending_rollouts = [
            rollout_id
            for rollout_id in range(config.rollouts_per_task)
            if not recorder.completed(task_id, rollout_id)
        ]
        if pending_rollouts:
            environment_count = min(config.num_envs, len(pending_rollouts))
            environment_factory = partial(
                environment_class,
                bddl_file_name=suite.get_task_bddl_file_path(task_id),
                camera_heights=config.camera_height,
                camera_widths=config.camera_width,
                render_gpu_device_id=config.render_gpu_device_id,
                horizon=config.max_steps + config.warmup_steps + 1,
            )
            vector_class = DummyVectorEnv if environment_count == 1 else SubprocVectorEnv
            environment = vector_class([environment_factory for _ in range(environment_count)])
            try:
                for offset in range(0, len(pending_rollouts), environment_count):
                    rollout_batch = pending_rollouts[offset : offset + environment_count]
                    slot_ids = tuple(range(len(rollout_batch)))
                    init_state_ids = tuple(
                        rollout_id % len(initial_states) for rollout_id in rollout_batch
                    )
                    episode_seeds = tuple(
                        config.seed + task_id * 100_000 + rollout_id for rollout_id in rollout_batch
                    )
                    videos: list[_VideoSink] = []
                    for rollout_id in rollout_batch:
                        if config.save_videos:
                            videos.append(
                                _ImageioVideoSink(
                                    videos_dir / f"task{task_id:02d}_rollout{rollout_id:03d}.mp4",
                                    fps=config.video_fps,
                                    observation_key=config.video_observation_key,
                                )
                            )
                        else:
                            videos.append(_NullVideoSink())
                    try:
                        results = rollout_episode_batch(
                            environment=environment,
                            policy=policy,
                            slot_ids=slot_ids,
                            initial_states=[initial_states[index] for index in init_state_ids],
                            task_id=task_id,
                            task_name=task.name,
                            instruction=task.language,
                            rollout_ids=rollout_batch,
                            init_state_ids=init_state_ids,
                            seeds=episode_seeds,
                            environment_seed=config.environment_seed,
                            max_steps=config.max_steps,
                            warmup_steps=config.warmup_steps,
                            warmup_gripper_action=config.warmup_gripper_action,
                            action_dim=policy.model.config.action_dim,
                            clip_actions=config.clip_actions,
                            videos=videos,
                        )
                    finally:
                        for video in videos:
                            video.close()
                    for result in results:
                        recorder.record(result)
                        status = "success" if result.success else "failure"
                        print(
                            f"[Eval] task={task_id:02d} | rollout="
                            f"{result.rollout_id + 1:02d}/{config.rollouts_per_task} | "
                            f"{status} | steps={result.steps} | "
                            f"policy_calls={result.policy_calls} | "
                            f"inference={result.inference_seconds:.2f}s",
                            flush=True,
                        )
            finally:
                environment.close()
        task_summary = recorder.write_summary()["tasks"].get(str(task_id))
        if task_summary is not None:
            print(
                f"[Eval] task={task_id:02d} complete | "
                f"success_rate={100.0 * task_summary['success_rate']:.1f}% "
                f"({task_summary['successes']}/{task_summary['completed_rollouts']})",
                flush=True,
            )
    return recorder.write_summary()
