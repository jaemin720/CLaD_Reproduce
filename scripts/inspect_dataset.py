#!/usr/bin/env python3
"""Inspect LIBERO metadata and the first CLaD temporal window."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from clad.data import LiberoDatasetConfig, LiberoWindowDataset


def _shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}"
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--camera-key", default="obs/agentview_rgb")
    parser.add_argument("--proprio-key", default="robot_states")
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LiberoWindowDataset(
        LiberoDatasetConfig(
            dataset_dir=args.dataset_dir,
            horizon=args.horizon,
            camera_key=args.camera_key,
            proprio_key=args.proprio_key,
            include_images=not args.no_images,
        )
    )

    try:
        print(f"tasks: {len(dataset.tasks)}")
        print(f"episodes: {len(dataset.episodes)}")
        print(f"windows: {len(dataset):,}")
        for task in dataset.tasks:
            print(f"- {task.task_id}: demos={task.num_demos}, instruction={task.instruction!r}")

        print("\nfirst window")
        for key, value in dataset[0].items():
            print(f"- {key}: {_shape(value)}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()

