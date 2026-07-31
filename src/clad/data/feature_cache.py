"""Persistent DecisionNCE features for LIBERO trajectories.

The VLM is frozen in CLaD, so every image and task instruction can be encoded
once. Cache files mirror task/demo/view hierarchy, while a fingerprint guards
against silently mixing datasets, checkpoints, camera selections, or source
revisions.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from clad.data.camera import camera_view_name, normalize_camera_keys
from clad.data.task_registry import LiberoTask, discover_libero_tasks, list_demo_keys
from clad.models.decisionnce_adapter import DecisionNCEAdapter

CACHE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class FeatureCacheSpec:
    """Fields that uniquely identify cached VLM features."""

    model_name: str
    source_revision: str
    checkpoint_sha256: str
    camera_keys: tuple[str, ...] = ("obs/agentview_rgb",)
    feature_dtype: str = "float16"

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_keys", normalize_camera_keys(self.camera_keys))
        if self.feature_dtype not in {"float16", "float32"}:
            raise ValueError(
                "feature_dtype must be 'float16' or 'float32', "
                f"got {self.feature_dtype!r}"
            )


@dataclass(frozen=True, slots=True)
class CacheBuildResult:
    built_tasks: tuple[str, ...]
    skipped_tasks: tuple[str, ...]
    manifest_path: Path


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a checkpoint identity without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(task: LiberoTask) -> dict[str, Any]:
    stat = task.path.stat()
    return {
        "path": str(task.path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "instruction": task.instruction,
        "num_demos": task.num_demos,
    }


def _task_fingerprint(task: LiberoTask, spec: FeatureCacheSpec) -> str:
    return _stable_digest(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source": _source_identity(task),
            "spec": asdict(spec),
        }
    )


def _numpy_dtype(name: str) -> np.dtype[Any]:
    return np.dtype(np.float16 if name == "float16" else np.float32)


class DecisionNCEFeatureCacheBuilder:
    """Encode LIBERO task files incrementally into per-task HDF5 caches."""

    def __init__(
        self,
        *,
        adapter: DecisionNCEAdapter,
        spec: FeatureCacheSpec,
        batch_size: int = 256,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if adapter.config.model_name != spec.model_name:
            raise ValueError(
                f"Adapter model {adapter.config.model_name!r} does not match "
                f"cache model {spec.model_name!r}"
            )
        if adapter.config.source_revision != spec.source_revision:
            raise ValueError(
                "Adapter and cache source revisions differ: "
                f"{adapter.config.source_revision!r} != {spec.source_revision!r}"
            )
        if (
            adapter.config.checkpoint_sha256
            and adapter.config.checkpoint_sha256 != spec.checkpoint_sha256
        ):
            raise ValueError("Adapter and cache checkpoint SHA-256 values differ")

        self.adapter = adapter
        self.spec = spec
        self.batch_size = batch_size

    @staticmethod
    def task_cache_path(cache_dir: str | Path, task_id: str) -> Path:
        return Path(cache_dir) / f"{task_id}.hdf5"

    @staticmethod
    def _cache_matches(path: Path, fingerprint: str) -> bool:
        try:
            with h5py.File(path, "r") as handle:
                return (
                    int(handle.attrs.get("schema_version", -1)) == CACHE_SCHEMA_VERSION
                    and str(handle.attrs.get("fingerprint", "")) == fingerprint
                )
        except OSError:
            return False

    def _encode_image_dataset(
        self,
        *,
        source: h5py.Dataset,
        output_group: h5py.Group,
        view_name: str,
    ) -> int:
        length = int(source.shape[0])
        destination: h5py.Dataset | None = None
        feature_dim = -1
        output_dtype = _numpy_dtype(self.spec.feature_dtype)

        for start in range(0, length, self.batch_size):
            stop = min(start + self.batch_size, length)
            images = torch.from_numpy(np.asarray(source[start:stop], dtype=np.uint8))
            features = self.adapter.encode_images(images).to(device="cpu")
            feature_array = features.numpy().astype(output_dtype, copy=False)

            if destination is None:
                feature_dim = int(feature_array.shape[1])
                destination = output_group.create_dataset(
                    view_name,
                    shape=(length, feature_dim),
                    dtype=output_dtype,
                    chunks=(min(self.batch_size, length), feature_dim),
                )
            elif feature_array.shape[1] != feature_dim:
                raise ValueError(
                    f"Feature dimension changed inside one trajectory: "
                    f"{feature_array.shape[1]} != {feature_dim}"
                )
            destination[start:stop] = feature_array

        if destination is None:
            raise ValueError(f"Cannot cache an empty image trajectory at {source.name}")
        return feature_dim

    def _write_task(
        self,
        *,
        task: LiberoTask,
        destination: Path,
        fingerprint: str,
    ) -> None:
        temporary = destination.with_suffix(".tmp.hdf5")
        temporary.unlink(missing_ok=True)
        observed_feature_dim: int | None = None

        try:
            with (
                h5py.File(task.path, "r") as source_handle,
                h5py.File(temporary, "w") as output_handle,
            ):
                output_handle.attrs["schema_version"] = CACHE_SCHEMA_VERSION
                output_handle.attrs["fingerprint"] = fingerprint
                output_handle.attrs["task_id"] = task.task_id
                output_handle.attrs["instruction"] = task.instruction
                output_handle.attrs["source_path"] = str(task.path)
                output_handle.attrs["model_name"] = self.spec.model_name
                output_handle.attrs["source_revision"] = self.spec.source_revision
                output_handle.attrs["checkpoint_sha256"] = self.spec.checkpoint_sha256
                output_handle.attrs["camera_keys"] = json.dumps(self.spec.camera_keys)
                output_handle.attrs["feature_dtype"] = self.spec.feature_dtype

                text_feature = (
                    self.adapter.encode_texts([task.instruction])
                    .to(device="cpu")
                    .numpy()
                    .astype(_numpy_dtype(self.spec.feature_dtype), copy=False)
                )
                output_handle.create_dataset("text_feature", data=text_feature[0])

                source_data = source_handle["data"]
                output_data = output_handle.create_group("data")
                for demo_key in list_demo_keys(source_data):
                    source_demo = source_data[demo_key]
                    output_views = output_data.create_group(demo_key).create_group("images")
                    for camera_key in self.spec.camera_keys:
                        current_dim = self._encode_image_dataset(
                            source=source_demo[camera_key],
                            output_group=output_views,
                            view_name=camera_view_name(camera_key),
                        )
                        if observed_feature_dim is None:
                            observed_feature_dim = current_dim
                        elif current_dim != observed_feature_dim:
                            raise ValueError(
                                f"DecisionNCE feature dimension changed from "
                                f"{observed_feature_dim} to {current_dim}"
                            )

                if observed_feature_dim is None:
                    raise ValueError(f"No image features were written for {task.task_id}")
                output_handle.attrs["image_feature_dim"] = observed_feature_dim
                output_handle.attrs["text_feature_dim"] = int(text_feature.shape[1])

            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def build(
        self,
        *,
        dataset_dir: str | Path,
        cache_dir: str | Path,
        file_pattern: str = "*_demo.hdf5",
        overwrite: bool = False,
        max_tasks: int | None = None,
    ) -> CacheBuildResult:
        """Build missing task caches and write an atomic manifest."""

        if max_tasks is not None and max_tasks <= 0:
            raise ValueError(f"max_tasks must be positive, got {max_tasks}")

        tasks = discover_libero_tasks(dataset_dir, file_pattern=file_pattern)
        if max_tasks is not None:
            tasks = tasks[:max_tasks]

        cache_root = Path(cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        built: list[str] = []
        skipped: list[str] = []
        task_entries: list[dict[str, Any]] = []

        for task in tasks:
            fingerprint = _task_fingerprint(task, self.spec)
            destination = self.task_cache_path(cache_root, task.task_id)
            if destination.exists() and self._cache_matches(destination, fingerprint):
                skipped.append(task.task_id)
            elif destination.exists() and not overwrite:
                raise FileExistsError(
                    f"Stale or incompatible cache exists at {destination}. "
                    "Pass overwrite=True to replace it."
                )
            else:
                self._write_task(
                    task=task,
                    destination=destination,
                    fingerprint=fingerprint,
                )
                built.append(task.task_id)

            task_entries.append(
                {
                    "task_id": task.task_id,
                    "cache_file": destination.name,
                    "fingerprint": fingerprint,
                    "source": _source_identity(task),
                }
            )

        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "spec": asdict(self.spec),
            "tasks": task_entries,
        }
        manifest_path = cache_root / MANIFEST_NAME
        temporary_manifest = cache_root / f"{MANIFEST_NAME}.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)

        return CacheBuildResult(
            built_tasks=tuple(built),
            skipped_tasks=tuple(skipped),
            manifest_path=manifest_path,
        )


class DecisionNCEFeatureCache:
    """Lazy, DataLoader-pickle-safe reader for cached DecisionNCE features."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Feature cache manifest does not exist: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported feature cache schema: "
                f"{self.manifest.get('schema_version')!r}"
            )
        self._task_paths = {
            entry["task_id"]: self.cache_dir / entry["cache_file"]
            for entry in self.manifest["tasks"]
        }
        self._files: dict[str, h5py.File] = {}

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Task identifiers available in this cache, in manifest order."""

        return tuple(entry["task_id"] for entry in self.manifest["tasks"])

    @property
    def camera_keys(self) -> tuple[str, ...]:
        """Source HDF5 camera keys encoded into this cache."""

        return normalize_camera_keys(self.manifest["spec"]["camera_keys"])

    def _get_file(self, task_id: str) -> h5py.File:
        if task_id not in self._task_paths:
            raise KeyError(f"Task {task_id!r} is not present in the feature cache")
        handle = self._files.get(task_id)
        if handle is None:
            handle = h5py.File(self._task_paths[task_id], "r")
            self._files[task_id] = handle
        return handle

    @staticmethod
    def _tensor(dataset: h5py.Dataset, index: int | slice | None = None) -> torch.Tensor:
        value = dataset[()] if index is None else dataset[index]
        return torch.from_numpy(np.asarray(value))

    def text_feature(self, task_id: str) -> torch.Tensor:
        return self._tensor(self._get_file(task_id)["text_feature"])

    def image_feature(
        self,
        *,
        task_id: str,
        demo_key: str,
        view_name: str,
        index: int | slice,
    ) -> torch.Tensor:
        dataset = self._get_file(task_id)["data"][demo_key]["images"][view_name]
        return self._tensor(dataset, index)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self) -> None:
        files = getattr(self, "_files", None)
        if files:
            self.close()
