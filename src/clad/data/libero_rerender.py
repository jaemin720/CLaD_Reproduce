"""Replay LIBERO demonstrations and write native-resolution HDF5 datasets.

The implementation follows the data-quality operations documented by OpenVLA:
replay simulator states/actions, render at the target resolution, discard
no-op actions, and retain only demonstrations that still solve the task.  The
output deliberately remains in LIBERO's HDF5 layout so the CLaD loader does
not depend on RLDS or LeRobot.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clad.data.image_transform import IMAGE_TRANSFORMS, transform_rgb_image
from clad.data.task_registry import list_demo_keys

RERENDER_SCHEMA_VERSION = 1
RERENDER_METADATA_KEYS = (
    "clad_rerender_schema_version",
    "clad_render_height",
    "clad_render_width",
    "clad_image_transform",
    "clad_environment_seed",
    "clad_filter_noops",
    "clad_noop_threshold",
    "clad_settle_steps",
    "clad_keep_only_successes",
)


@dataclass(frozen=True, slots=True)
class LiberoRerenderConfig:
    """Protocol settings that affect regenerated demonstrations."""

    render_height: int = 256
    render_width: int = 256
    settle_steps: int = 10
    environment_seed: int = 0
    filter_noops: bool = True
    noop_threshold: float = 1e-4
    keep_only_successes: bool = True
    image_transform: str = "rotate_180"
    compression: str = "lzf"

    def __post_init__(self) -> None:
        if self.render_height <= 0 or self.render_width <= 0:
            raise ValueError("render_height and render_width must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if self.environment_seed < 0:
            raise ValueError("environment_seed must be non-negative")
        if self.noop_threshold < 0.0:
            raise ValueError("noop_threshold must be non-negative")
        if self.image_transform not in IMAGE_TRANSFORMS:
            raise ValueError(
                f"image_transform must be one of {IMAGE_TRANSFORMS}, "
                f"got {self.image_transform!r}"
            )
        if self.compression not in {"none", "lzf", "gzip"}:
            raise ValueError("compression must be one of ('none', 'lzf', 'gzip')")


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """A replayed trajectory and the filtering decisions that produced it."""

    arrays: dict[str, np.ndarray]
    success: bool
    source_steps: int
    retained_steps: int
    removed_noops: int


@dataclass(frozen=True, slots=True)
class TaskRerenderResult:
    """Summary for one regenerated task file."""

    destination: Path
    source_demos: int
    written_demos: int
    failed_replays: int
    empty_replays: int
    removed_noops: int
    output_steps: int
    elapsed_seconds: float
    fingerprint: str


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_noop_action(
    action: np.ndarray,
    previous_action: np.ndarray | None,
    *,
    threshold: float,
) -> bool:
    """Return OpenVLA's LIBERO no-op predicate for a 7D delta action."""

    current = np.asarray(action, dtype=np.float32).reshape(-1)
    if current.size < 2:
        raise ValueError(f"LIBERO action must have at least 2 values, got {current.size}")
    if np.linalg.norm(current[:-1]) >= threshold:
        return False
    if previous_action is None:
        return True
    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous.shape != current.shape:
        raise ValueError(
            f"Previous action shape {previous.shape} differs from {current.shape}"
        )
    return bool(np.isclose(current[-1], previous[-1]))


def _sim_state(environment: Any) -> np.ndarray:
    simulator = getattr(environment, "sim", None)
    if simulator is None:
        wrapped = getattr(environment, "env", None)
        simulator = getattr(wrapped, "sim", None)
    if simulator is None:
        raise AttributeError("LIBERO environment does not expose sim or env.sim")
    state = simulator.get_state()
    flatten = getattr(state, "flatten", None)
    return np.asarray(flatten() if flatten is not None else state, dtype=np.float64)


def _observation_array(
    observation: Mapping[str, Any],
    key: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    if key not in observation:
        raise KeyError(f"LIBERO observation is missing {key!r}")
    value = np.asarray(observation[key], dtype=np.float32).reshape(-1)
    if expected_size is not None and value.size != expected_size:
        raise ValueError(
            f"LIBERO observation {key!r} must contain {expected_size} values, "
            f"got {value.size}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"LIBERO observation {key!r} contains NaN or Inf")
    return value


def _image_array(
    observation: Mapping[str, Any],
    key: str,
    *,
    config: LiberoRerenderConfig,
) -> np.ndarray:
    if key not in observation:
        raise KeyError(f"LIBERO observation is missing camera field {key!r}")
    image = np.asarray(observation[key], dtype=np.uint8)
    expected = (config.render_height, config.render_width, 3)
    if image.shape != expected:
        raise ValueError(f"Camera {key!r} must have shape {expected}, got {image.shape}")
    return transform_rgb_image(image, config.image_transform)


def _check_success(environment: Any, final_reward: float) -> bool:
    check_success = getattr(environment, "check_success", None)
    if check_success is not None:
        return bool(check_success())
    wrapped = getattr(environment, "env", None)
    check_success = getattr(wrapped, "_check_success", None)
    if check_success is not None:
        return bool(check_success())
    return final_reward > 0.0


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert robosuite's XYZW quaternion to a three-value rotation vector."""

    quat = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("End-effector quaternion has zero norm")
    quat /= norm
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if denominator < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * np.arccos(quat[3]) / denominator)


def replay_demonstration(
    *,
    environment: Any,
    initial_state: np.ndarray,
    actions: np.ndarray,
    config: LiberoRerenderConfig,
) -> ReplayResult:
    """Replay one demonstration and capture pre-action simulator observations."""

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[1] != 7:
        raise ValueError(f"LIBERO actions must have shape [T,7], got {action_array.shape}")
    if action_array.shape[0] == 0:
        return ReplayResult({}, False, 0, 0, 0)

    environment.reset()
    observation = environment.set_init_state(np.asarray(initial_state))
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO set_init_state() must return an observation mapping")

    dummy_action = np.zeros(7, dtype=np.float32)
    dummy_action[-1] = -1.0
    for _ in range(config.settle_steps):
        observation, _, _, _ = environment.step(dummy_action)

    records: dict[str, list[np.ndarray | float | bool | int]] = {
        "actions": [],
        "states": [],
        "robot_states": [],
        "rewards": [],
        "dones": [],
        "source_action_indices": [],
        "obs/agentview_rgb": [],
        "obs/eye_in_hand_rgb": [],
        "obs/joint_states": [],
        "obs/joint_velocities": [],
        "obs/gripper_states": [],
        "obs/ee_pos": [],
        "obs/ee_ori": [],
        "obs/ee_states": [],
    }
    previous_action: np.ndarray | None = None
    removed_noops = 0
    final_reward = 0.0

    for source_index, action in enumerate(action_array):
        if config.filter_noops and is_noop_action(
            action,
            previous_action,
            threshold=config.noop_threshold,
        ):
            removed_noops += 1
            continue

        joint = _observation_array(observation, "robot0_joint_pos", expected_size=7)
        joint_velocity = _observation_array(
            observation, "robot0_joint_vel", expected_size=7
        )
        gripper = _observation_array(
            observation, "robot0_gripper_qpos", expected_size=2
        )
        eef_position = _observation_array(
            observation, "robot0_eef_pos", expected_size=3
        )
        eef_orientation = _observation_array(
            observation, "robot0_eef_quat", expected_size=4
        )
        eef_axis_angle = _quaternion_to_axis_angle(eef_orientation)

        records["actions"].append(action.copy())
        records["states"].append(_sim_state(environment))
        records["robot_states"].append(
            np.concatenate((gripper, eef_position, eef_orientation))
        )
        records["source_action_indices"].append(source_index)
        records["obs/agentview_rgb"].append(
            _image_array(observation, "agentview_image", config=config)
        )
        records["obs/eye_in_hand_rgb"].append(
            _image_array(observation, "robot0_eye_in_hand_image", config=config)
        )
        records["obs/joint_states"].append(joint)
        records["obs/joint_velocities"].append(joint_velocity)
        records["obs/gripper_states"].append(gripper)
        records["obs/ee_pos"].append(eef_position)
        records["obs/ee_ori"].append(eef_axis_angle)
        records["obs/ee_states"].append(
            np.concatenate((eef_position, eef_axis_angle))
        )

        observation, reward, done, _ = environment.step(action)
        final_reward = float(reward)
        records["rewards"].append(final_reward)
        records["dones"].append(bool(done))
        previous_action = action

    retained_steps = len(records["actions"])
    if retained_steps == 0:
        return ReplayResult(
            arrays={},
            success=False,
            source_steps=int(action_array.shape[0]),
            retained_steps=0,
            removed_noops=removed_noops,
        )

    success = _check_success(environment, final_reward)
    records["dones"][-1] = success
    if success:
        records["rewards"][-1] = max(float(records["rewards"][-1]), 1.0)

    arrays = {
        key: np.asarray(values)
        for key, values in records.items()
    }
    return ReplayResult(
        arrays=arrays,
        success=success,
        source_steps=int(action_array.shape[0]),
        retained_steps=retained_steps,
        removed_noops=removed_noops,
    )


def _dataset_options(config: LiberoRerenderConfig) -> dict[str, Any]:
    if config.compression == "none":
        return {}
    if config.compression == "gzip":
        return {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def _write_replay(
    destination: h5py.Group,
    replay: ReplayResult,
    *,
    source_demo_key: str,
    config: LiberoRerenderConfig,
) -> None:
    options = _dataset_options(config)
    destination.attrs["num_samples"] = replay.retained_steps
    destination.attrs["source_demo_key"] = source_demo_key
    destination.attrs["source_num_samples"] = replay.source_steps
    destination.attrs["removed_noops"] = replay.removed_noops
    destination.attrs["replay_success"] = replay.success
    for key, array in replay.arrays.items():
        group = destination
        parts = key.split("/")
        for part in parts[:-1]:
            group = group.require_group(part)
        group.create_dataset(parts[-1], data=array, **options)


def _copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def _update_environment_metadata(
    data: h5py.Group,
    *,
    config: LiberoRerenderConfig,
) -> None:
    raw = data.attrs.get("env_args")
    if raw is None:
        return
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        env_args = json.loads(str(raw))
    except json.JSONDecodeError:
        return
    env_kwargs = env_args.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        return
    env_kwargs["camera_heights"] = config.render_height
    env_kwargs["camera_widths"] = config.render_width
    data.attrs["env_args"] = json.dumps(env_args)


def task_fingerprint(
    source_path: str | Path,
    *,
    source_sha256: str,
    config: LiberoRerenderConfig,
    max_demos: int | None = None,
) -> str:
    source = Path(source_path).expanduser().resolve()
    stat = source.stat()
    payload = {
        "schema_version": RERENDER_SCHEMA_VERSION,
        "source_path": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": source_sha256,
        "config": asdict(config),
        "max_demos": max_demos,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matching_rerender(path: str | Path, fingerprint: str) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            return (
                int(data.attrs.get("clad_rerender_schema_version", -1))
                == RERENDER_SCHEMA_VERSION
                and str(data.attrs.get("clad_rerender_fingerprint", "")) == fingerprint
            )
    except (OSError, KeyError):
        return False


def rerender_task_file(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    environment: Any,
    config: LiberoRerenderConfig,
    max_demos: int | None = None,
    source_sha256: str | None = None,
    log_interval: int = 25,
) -> TaskRerenderResult:
    """Regenerate one LIBERO task into an atomic, loader-compatible HDF5 file."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if source == destination:
        raise ValueError("Source and destination HDF5 paths must differ")
    if max_demos is not None and max_demos <= 0:
        raise ValueError("max_demos must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"Source LIBERO task does not exist: {source}")

    checksum = source_sha256 or sha256_file(source)
    fingerprint = task_fingerprint(
        source,
        source_sha256=checksum,
        config=config,
        max_demos=max_demos,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    started = time.monotonic()
    written = failed = empty = removed = output_steps = 0

    try:
        with h5py.File(source, "r") as source_handle:
            source_data = source_handle["data"]
            demo_keys = list_demo_keys(source_data)
            if max_demos is not None:
                demo_keys = demo_keys[:max_demos]

            with h5py.File(temporary, "w") as output_handle:
                _copy_attrs(source_handle.attrs, output_handle.attrs)
                output_data = output_handle.create_group("data")
                _copy_attrs(source_data.attrs, output_data.attrs)
                _update_environment_metadata(output_data, config=config)
                output_data.attrs["clad_rerender_schema_version"] = RERENDER_SCHEMA_VERSION
                output_data.attrs["clad_rerender_fingerprint"] = fingerprint
                output_data.attrs["clad_source_path"] = str(source)
                output_data.attrs["clad_source_sha256"] = checksum
                output_data.attrs["clad_render_height"] = config.render_height
                output_data.attrs["clad_render_width"] = config.render_width
                output_data.attrs["clad_image_transform"] = config.image_transform
                output_data.attrs["clad_environment_seed"] = config.environment_seed
                output_data.attrs["clad_filter_noops"] = config.filter_noops
                output_data.attrs["clad_noop_threshold"] = config.noop_threshold
                output_data.attrs["clad_settle_steps"] = config.settle_steps
                output_data.attrs["clad_keep_only_successes"] = (
                    config.keep_only_successes
                )
                output_data.attrs["clad_compression"] = config.compression

                for index, demo_key in enumerate(demo_keys, start=1):
                    source_demo = source_data[demo_key]
                    if "actions" not in source_demo or "states" not in source_demo:
                        raise KeyError(
                            f"{source}:{demo_key} must contain actions and states"
                        )
                    source_states = np.asarray(source_demo["states"])
                    if source_states.ndim != 2 or source_states.shape[0] == 0:
                        raise ValueError(
                            f"{source}:{demo_key}/states must have shape [T,D]"
                        )
                    replay = replay_demonstration(
                        environment=environment,
                        initial_state=source_states[0],
                        actions=np.asarray(source_demo["actions"]),
                        config=config,
                    )
                    removed += replay.removed_noops
                    if replay.retained_steps == 0:
                        empty += 1
                    elif config.keep_only_successes and not replay.success:
                        failed += 1
                    else:
                        output_demo = output_data.create_group(f"demo_{written}")
                        output_demo.attrs["init_state"] = source_states[0]
                        _write_replay(
                            output_demo,
                            replay,
                            source_demo_key=demo_key,
                            config=config,
                        )
                        written += 1
                        output_steps += replay.retained_steps

                    if index % log_interval == 0 or index == len(demo_keys):
                        print(
                            f"  demos={index}/{len(demo_keys)} | kept={written} | "
                            f"failed={failed} | empty={empty} | noops={removed}",
                            flush=True,
                        )

                if written == 0:
                    raise RuntimeError(
                        f"No demonstrations survived replay for {source.name}; "
                        "the temporary output was not installed"
                    )
                output_data.attrs["num_demos"] = written
                output_data.attrs["total"] = output_steps
                output_data.attrs["clad_source_num_demos"] = len(demo_keys)
                output_data.attrs["clad_failed_replays"] = failed
                output_data.attrs["clad_empty_replays"] = empty
                output_data.attrs["clad_removed_noops"] = removed

        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return TaskRerenderResult(
        destination=destination,
        source_demos=len(demo_keys),
        written_demos=written,
        failed_replays=failed,
        empty_replays=empty,
        removed_noops=removed,
        output_steps=output_steps,
        elapsed_seconds=time.monotonic() - started,
        fingerprint=fingerprint,
    )
