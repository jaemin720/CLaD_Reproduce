"""Sequential fixed-initial-state rollout evaluation on official LIBERO."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from clad.evaluation.online_policy import CLaDOnlinePolicy, PolicyPlan


@dataclass(frozen=True, slots=True)
class LiberoRolloutConfig:
    """Environment and protocol settings for LIBERO success evaluation."""

    suite_name: str = "libero_10"
    task_order_index: int = 0
    task_ids: tuple[int, ...] = ()
    rollouts_per_task: int = 50
    max_steps: int = 600
    warmup_steps: int = 5
    execution_steps: int = 6
    camera_height: int = 128
    camera_width: int = 128
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
            f"LIBERO video field {observation_key!r} must have shape [H,W,3], "
            f"got {frame.shape}"
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
    max_steps: int,
    warmup_steps: int,
    action_dim: int,
    clip_actions: bool,
    video: _VideoSink | None = None,
) -> EpisodeResult:
    """Run one episode while keeping the CLaD history synchronized with actions."""

    sink = video or _NullVideoSink()
    environment.seed(seed)
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
    zero_action = np.zeros(action_dim, dtype=np.float32)
    for _ in range(warmup_steps):
        observation, reward, done, _ = environment.step(zero_action)
        reward_value = float(reward)
        total_reward += reward_value
        policy.observe(zero_action, observation)
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
        success=success,
        steps=steps,
        total_reward=total_reward,
        policy_calls=policy_calls,
        inference_seconds=inference_seconds,
    )


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
                sum(task_success_rates) / len(task_success_rates)
                if task_success_rates
                else None
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
    benchmark_types = benchmark.get_benchmark_dict()
    if config.suite_name not in benchmark_types:
        raise ValueError(
            f"Unknown LIBERO suite {config.suite_name!r}; "
            f"available={sorted(benchmark_types)}"
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
        environment = environment_class(
            bddl_file_name=suite.get_task_bddl_file_path(task_id),
            camera_heights=config.camera_height,
            camera_widths=config.camera_width,
            render_gpu_device_id=config.render_gpu_device_id,
            horizon=config.max_steps + config.warmup_steps + 1,
        )
        try:
            for rollout_id in range(config.rollouts_per_task):
                if recorder.completed(task_id, rollout_id):
                    continue
                init_state_id = rollout_id % len(initial_states)
                episode_seed = config.seed + task_id * 100_000 + rollout_id
                video: _VideoSink
                if config.save_videos:
                    video = _ImageioVideoSink(
                        videos_dir / f"task{task_id:02d}_rollout{rollout_id:03d}.mp4",
                        fps=config.video_fps,
                        observation_key=config.video_observation_key,
                    )
                else:
                    video = _NullVideoSink()
                try:
                    result = rollout_episode(
                        environment=environment,
                        policy=policy,
                        initial_state=initial_states[init_state_id],
                        task_id=task_id,
                        task_name=task.name,
                        instruction=task.language,
                        rollout_id=rollout_id,
                        init_state_id=init_state_id,
                        seed=episode_seed,
                        max_steps=config.max_steps,
                        warmup_steps=config.warmup_steps,
                        action_dim=policy.model.config.action_dim,
                        clip_actions=config.clip_actions,
                        video=video,
                    )
                finally:
                    video.close()
                recorder.record(result)
                status = "success" if result.success else "failure"
                print(
                    f"[Eval] task={task_id:02d} | rollout="
                    f"{rollout_id + 1:02d}/{config.rollouts_per_task} | "
                    f"{status} | steps={result.steps} | policy_calls={result.policy_calls} | "
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
