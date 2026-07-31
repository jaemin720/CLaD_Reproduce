"""PyTorch dataset for CLaD's past/current/future LIBERO windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from clad.data.sequence_sampler import WindowIndex, build_window_indices
from clad.data.task_registry import LiberoTask, discover_libero_tasks, list_demo_keys


@dataclass(frozen=True, slots=True)
class LiberoDatasetConfig:
    dataset_dir: str | Path
    file_pattern: str = "*_demo.hdf5"
    camera_keys: tuple[str, ...] = ("obs/agentview_rgb",)
    proprio_key: str = "robot_states"
    action_key: str = "actions"
    horizon: int = 6
    include_images: bool = True
    strict: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.camera_keys, str):
            raise TypeError(
                "camera_keys must be a sequence of HDF5 paths, not a single string"
            )

        # Hydra/OmegaConf commonly constructs this field from a YAML list.
        # Normalize it once so the rest of the data pipeline has an immutable
        # and hashable camera specification.
        camera_keys = tuple(self.camera_keys)
        object.__setattr__(self, "camera_keys", camera_keys)

        if self.include_images and not camera_keys:
            raise ValueError("camera_keys cannot be empty when include_images=True")
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError(f"camera_keys contains duplicate paths: {camera_keys}")

        view_names = tuple(_camera_view_name(key) for key in camera_keys)
        if len(set(view_names)) != len(view_names):
            raise ValueError(
                "Each camera path must have a unique final component; "
                f"got view names {view_names}"
            )


def _camera_view_name(camera_key: str) -> str:
    """Convert ``obs/agentview_rgb`` into the stable view id ``agentview_rgb``."""

    normalized = camera_key.rstrip("/")
    if not normalized:
        raise ValueError("camera keys cannot be empty")
    return normalized.rsplit("/", maxsplit=1)[-1]


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    task_index: int
    demo_key: str
    length: int


class LiberoWindowDataset(Dataset[dict[str, Any]]):
    """Read complete temporal windows without crossing episode boundaries.

    HDF5 handles are opened lazily per process. This keeps the dataset safe to
    construct in the parent process and then use with PyTorch DataLoader
    workers.
    """

    def __init__(self, config: LiberoDatasetConfig) -> None:
        super().__init__()
        if config.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {config.horizon}")

        self.config = config
        self.tasks: list[LiberoTask] = discover_libero_tasks(
            config.dataset_dir,
            file_pattern=config.file_pattern,
        )
        self.episodes: list[EpisodeRecord] = []
        self.indices: list[WindowIndex] = []
        self._files: dict[Path, h5py.File] = {}
        self._index_dataset()

    def _index_dataset(self) -> None:
        required_keys = [self.config.proprio_key, self.config.action_key]
        if self.config.include_images:
            required_keys.extend(self.config.camera_keys)

        for task_index, task in enumerate(self.tasks):
            with h5py.File(task.path, "r") as handle:
                data_group = handle["data"]
                for demo_key in list_demo_keys(data_group):
                    demo_group = data_group[demo_key]
                    missing = [key for key in required_keys if key not in demo_group]
                    if missing:
                        raise ValueError(
                            f"Missing datasets {missing} in {task.path}:{demo_key}"
                        )

                    action_dataset = demo_group[self.config.action_key]
                    length = int(action_dataset.shape[0])
                    if action_dataset.ndim != 2:
                        raise ValueError(
                            f"Actions must have shape [T, Da], got "
                            f"{action_dataset.shape} in {task.path}:{demo_key}"
                        )

                    for key in required_keys:
                        dataset = demo_group[key]
                        if not isinstance(dataset, h5py.Dataset):
                            raise ValueError(
                                f"Expected a dataset at {task.path}:{demo_key}/{key}"
                            )
                        if int(dataset.shape[0]) != length:
                            raise ValueError(
                                f"Temporal length mismatch at {task.path}:{demo_key}/{key}: "
                                f"{dataset.shape[0]} != {length}"
                            )

                    proprio_dataset = demo_group[self.config.proprio_key]
                    if proprio_dataset.ndim != 2:
                        raise ValueError(
                            f"Proprioception must have shape [T, Dp], got "
                            f"{proprio_dataset.shape} in {task.path}:{demo_key}"
                        )

                    if self.config.include_images:
                        for camera_key in self.config.camera_keys:
                            image_dataset = demo_group[camera_key]
                            if image_dataset.ndim != 4 or image_dataset.shape[-1] != 3:
                                raise ValueError(
                                    f"Camera {camera_key!r} must have shape [T, H, W, 3], "
                                    f"got {image_dataset.shape} in {task.path}:{demo_key}"
                                )

                    episode_indices = build_window_indices(
                        task_index=task_index,
                        demo_key=demo_key,
                        episode_length=length,
                        horizon=self.config.horizon,
                    )
                    if self.config.strict and not episode_indices:
                        raise ValueError(
                            f"Episode {task.path}:{demo_key} has length {length}, "
                            f"which is too short for horizon {self.config.horizon}"
                        )
                    self.episodes.append(
                        EpisodeRecord(
                            task_index=task_index,
                            demo_key=demo_key,
                            length=length,
                        )
                    )
                    self.indices.extend(episode_indices)

        if not self.indices:
            raise ValueError("The dataset contains no valid temporal windows")

    def __len__(self) -> int:
        return len(self.indices)

    def _get_file(self, path: Path) -> h5py.File:
        handle = self._files.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._files[path] = handle
        return handle

    @staticmethod
    def _float_tensor(dataset: h5py.Dataset, index: int | slice) -> torch.Tensor:
        array = np.asarray(dataset[index], dtype=np.float32)
        return torch.from_numpy(array)

    @staticmethod
    def _image_tensor(dataset: h5py.Dataset, index: int) -> torch.Tensor:
        array = np.asarray(dataset[index], dtype=np.uint8)
        return torch.from_numpy(array)

    def __getitem__(self, item: int) -> dict[str, Any]:
        window = self.indices[item]
        task = self.tasks[window.task_index]
        demo_group = self._get_file(task.path)["data"][window.demo_key]

        tau = self.config.horizon
        t = window.anchor_step
        proprio = demo_group[self.config.proprio_key]
        actions = demo_group[self.config.action_key]

        sample: dict[str, Any] = {
            "task_id": task.task_id,
            "episode_id": window.demo_key,
            "anchor_step": t,
            "instruction": task.instruction,
            "proprio_prev": self._float_tensor(proprio, t - tau),
            "proprio_now": self._float_tensor(proprio, t),
            "proprio_future": self._float_tensor(proprio, t + tau),
            "past_actions": self._float_tensor(actions, slice(t - tau, t)),
            "target_actions": self._float_tensor(actions, slice(t, t + tau)),
        }

        if self.config.include_images:
            sample["images"] = {
                _camera_view_name(camera_key): {
                    "prev": self._image_tensor(demo_group[camera_key], t - tau),
                    "now": self._image_tensor(demo_group[camera_key], t),
                    "future": self._image_tensor(demo_group[camera_key], t + tau),
                }
                for camera_key in self.config.camera_keys
            }

        return sample

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # h5py handles cannot be pickled. DataLoader workers reopen them lazily.
        state["_files"] = {}
        return state

    def __del__(self) -> None:
        files = getattr(self, "_files", None)
        if files:
            self.close()
