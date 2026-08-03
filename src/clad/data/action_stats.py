"""Training-split action statistics for diffusion normalization."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch

from clad.data.libero_dataset import LiberoWindowDataset
from clad.data.task_registry import list_demo_keys


@dataclass(frozen=True, slots=True)
class ActionBounds:
    """Per-dimension extrema and the number of source actions inspected."""

    minimum: torch.Tensor
    maximum: torch.Tensor
    count: int


def compute_libero_action_bounds(
    dataset: LiberoWindowDataset,
    *,
    expected_action_dim: int | None = None,
) -> ActionBounds:
    """Scan each source action exactly once, without repeated window samples."""

    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    count = 0
    for task in dataset.tasks:
        with h5py.File(task.path, "r") as handle:
            data = handle["data"]
            for demo_key in list_demo_keys(data):
                actions = np.asarray(
                    data[demo_key][dataset.config.action_key],
                    dtype=np.float32,
                )
                if actions.ndim != 2:
                    raise ValueError(
                        f"Actions must have shape [T, Da], got {actions.shape} "
                        f"in {task.path}:{demo_key}"
                    )
                if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
                    raise ValueError(
                        "Dataset action dimension does not match the policy: "
                        f"{actions.shape[1]} != {expected_action_dim} at "
                        f"{task.path}:{demo_key}"
                    )
                if actions.shape[0] == 0:
                    continue
                if not np.isfinite(actions).all():
                    raise ValueError(f"Non-finite action found in {task.path}:{demo_key}")
                demo_minimum = actions.min(axis=0)
                demo_maximum = actions.max(axis=0)
                minimum = demo_minimum if minimum is None else np.minimum(minimum, demo_minimum)
                maximum = demo_maximum if maximum is None else np.maximum(maximum, demo_maximum)
                count += int(actions.shape[0])

    if minimum is None or maximum is None or count == 0:
        raise ValueError("Dataset contains no actions for normalization")
    return ActionBounds(
        minimum=torch.from_numpy(minimum.copy()),
        maximum=torch.from_numpy(maximum.copy()),
        count=count,
    )
