"""Join LIBERO temporal windows with cached DecisionNCE features."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from clad.data.camera import camera_view_name, normalize_camera_keys
from clad.data.feature_cache import DecisionNCEFeatureCache
from clad.data.libero_dataset import LiberoWindowDataset


class CachedLiberoWindowDataset(Dataset[dict[str, Any]]):
    """Add past/current/future VLM features to a raw-state LIBERO dataset.

    The wrapped base dataset must disable raw image loading. Each returned
    sample keeps the original proprioception/action fields and adds:

    ``text_feature``
        One cached language feature for the task.
    ``vision_features[view][prev|now|future]``
        Cached image features aligned to the same episode-safe window. Stage 2
        may disable the future entry because it has no future-state target.
    """

    def __init__(
        self,
        *,
        base_dataset: LiberoWindowDataset,
        feature_cache: DecisionNCEFeatureCache,
        include_future_features: bool = True,
    ) -> None:
        super().__init__()
        if base_dataset.config.include_images:
            raise ValueError(
                "base_dataset must use include_images=False when cached features are enabled"
            )

        self.base_dataset = base_dataset
        self.feature_cache = feature_cache
        self.include_future_features = include_future_features
        self.camera_keys = normalize_camera_keys(base_dataset.config.camera_keys)
        self.view_names = tuple(camera_view_name(key) for key in self.camera_keys)
        self._validate_coverage()

    def _validate_coverage(self) -> None:
        dataset_tasks = {task.task_id for task in self.base_dataset.tasks}
        cached_tasks = set(self.feature_cache.task_ids)
        missing_tasks = sorted(dataset_tasks - cached_tasks)
        if missing_tasks:
            raise ValueError(f"Feature cache does not cover all dataset tasks: {missing_tasks}")

        cached_cameras = set(self.feature_cache.camera_keys)
        missing_cameras = [
            camera_key for camera_key in self.camera_keys if camera_key not in cached_cameras
        ]
        if missing_cameras:
            raise ValueError(f"Feature cache does not contain requested cameras: {missing_cameras}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.base_dataset[item]
        task_id = sample["task_id"]
        demo_key = sample["episode_id"]
        anchor = int(sample["anchor_step"])
        horizon = self.base_dataset.config.horizon

        sample["text_feature"] = self.feature_cache.text_feature(task_id)
        sample["vision_features"] = {}
        for view_name in self.view_names:
            timeline = {
                "prev": self.feature_cache.image_feature(
                    task_id=task_id,
                    demo_key=demo_key,
                    view_name=view_name,
                    index=anchor - horizon,
                ),
                "now": self.feature_cache.image_feature(
                    task_id=task_id,
                    demo_key=demo_key,
                    view_name=view_name,
                    index=anchor,
                ),
            }
            if self.include_future_features:
                timeline["future"] = self.feature_cache.image_feature(
                    task_id=task_id,
                    demo_key=demo_key,
                    view_name=view_name,
                    index=anchor + horizon,
                )
            sample["vision_features"][view_name] = timeline
        return sample

    def close(self) -> None:
        self.base_dataset.close()
        self.feature_cache.close()

    def __del__(self) -> None:
        base_dataset = getattr(self, "base_dataset", None)
        feature_cache = getattr(self, "feature_cache", None)
        if base_dataset is not None:
            base_dataset.close()
        if feature_cache is not None:
            feature_cache.close()
