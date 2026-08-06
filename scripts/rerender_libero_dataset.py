#!/usr/bin/env python3
"""Regenerate LIBERO HDF5 demonstrations at native training resolution."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py

from clad.data.image_transform import IMAGE_TRANSFORMS
from clad.data.libero_rerender import (
    LiberoRerenderConfig,
    matching_rerender,
    rerender_task_file,
    sha256_file,
    task_fingerprint,
)
from clad.data.task_registry import discover_libero_tasks
from clad.evaluation.libero_rollout import require_libero_runtime

MANIFEST_NAME = "rerender_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay LIBERO demonstrations and write native-resolution HDF5 files. "
            "The source dataset is never modified."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-name", default="libero_10")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        help="LIBERO suite task indices; defaults to every task in the suite.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--environment-seed",
        type=int,
        default=0,
        help="LIBERO environment seed; OpenVLA uses 0 for every task.",
    )
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument("--keep-noops", action="store_true")
    parser.add_argument("--keep-failed-replays", action="store_true")
    parser.add_argument(
        "--image-transform",
        choices=IMAGE_TRANSFORMS,
        default="rotate_180",
        help="Stored-image orientation. rotate_180 matches OpenVLA's final LIBERO data.",
    )
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--max-demos-per-task", type=int)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _existing_task_summary(path: Path, *, fingerprint: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        return {
            "destination": str(path),
            "fingerprint": fingerprint,
            "source_demos": int(data.attrs.get("clad_source_num_demos", 0)),
            "written_demos": int(data.attrs["num_demos"]),
            "failed_replays": int(data.attrs.get("clad_failed_replays", 0)),
            "empty_replays": int(data.attrs.get("clad_empty_replays", 0)),
            "removed_noops": int(data.attrs.get("clad_removed_noops", 0)),
            "output_steps": int(data.attrs.get("total", 0)),
            "status": "skipped_matching",
        }


def main() -> None:
    args = parse_args()
    source_root = args.source_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("--output-dir must differ from --source-dir")
    if args.task_order_index < 0:
        raise ValueError("--task-order-index must be non-negative")

    protocol = LiberoRerenderConfig(
        render_height=args.resolution,
        render_width=args.resolution,
        settle_steps=args.settle_steps,
        environment_seed=args.environment_seed,
        filter_noops=not args.keep_noops,
        noop_threshold=args.noop_threshold,
        keep_only_successes=not args.keep_failed_replays,
        image_transform=args.image_transform,
        compression=args.compression,
    )
    source_tasks = {task.task_id: task for task in discover_libero_tasks(source_root)}
    benchmark, environment_class = require_libero_runtime()
    benchmark_types = benchmark.get_benchmark_dict()
    if args.suite_name not in benchmark_types:
        raise ValueError(
            f"Unknown LIBERO suite {args.suite_name!r}; "
            f"available={sorted(benchmark_types)}"
        )
    suite = benchmark_types[args.suite_name](args.task_order_index)
    task_ids = tuple(args.task_ids) if args.task_ids else tuple(range(suite.n_tasks))
    invalid = [task_id for task_id in task_ids if not 0 <= task_id < suite.n_tasks]
    if invalid:
        raise ValueError(f"Task ids outside [0, {suite.n_tasks}): {invalid}")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("--task-ids cannot contain duplicates")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_NAME
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite_name": args.suite_name,
        "task_order_index": args.task_order_index,
        "source_dir": str(source_root),
        "output_dir": str(output_root),
        "protocol": asdict(protocol),
        "max_demos_per_task": args.max_demos_per_task,
        "tasks": {},
    }

    print("OpenVLA-style LIBERO HDF5 regeneration")
    print(
        f"  suite={args.suite_name} | tasks={list(task_ids)} | "
        f"resolution={args.resolution}x{args.resolution}"
    )
    noop_mode = "filter" if protocol.filter_noops else "keep"
    failure_mode = "drop" if protocol.keep_only_successes else "keep"
    print(
        f"  transform={protocol.image_transform} | noops={noop_mode} | "
        f"failed={failure_mode}"
    )
    print(f"  environment_seed={protocol.environment_seed} (same for every task)")
    print(f"  source={source_root}")
    print(f"  output={output_root}")

    total_started = time.monotonic()
    for sequence, task_id in enumerate(task_ids, start=1):
        task = suite.get_task(task_id)
        task_name = str(task.name)
        source_task = source_tasks.get(task_name)
        if source_task is None:
            raise FileNotFoundError(
                f"Suite task {task_id} ({task_name}) has no matching "
                f"{task_name}_demo.hdf5 under {source_root}"
            )
        destination = output_root / source_task.path.name
        print(
            f"[Task {sequence}/{len(task_ids)}] id={task_id:02d} | {task_name}",
            flush=True,
        )
        print("  hashing source...", flush=True)
        checksum = sha256_file(source_task.path)
        fingerprint = task_fingerprint(
            source_task.path,
            source_sha256=checksum,
            config=protocol,
            max_demos=args.max_demos_per_task,
        )
        if destination.exists() and matching_rerender(destination, fingerprint):
            summary = _existing_task_summary(destination, fingerprint=fingerprint)
            print("  matching output exists; skipped", flush=True)
        elif destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Stale or incompatible output exists at {destination}. "
                "Use a new --output-dir or pass --overwrite explicitly."
            )
        else:
            environment = environment_class(
                bddl_file_name=suite.get_task_bddl_file_path(task_id),
                camera_heights=args.resolution,
                camera_widths=args.resolution,
                render_gpu_device_id=args.render_gpu_device_id,
                horizon=10_000,
            )
            try:
                # OpenVLA explicitly uses seed 0 for every task because the
                # seed can affect object placement even after set_init_state().
                environment.seed(protocol.environment_seed)
                result = rerender_task_file(
                    source_path=source_task.path,
                    destination_path=destination,
                    environment=environment,
                    config=protocol,
                    max_demos=args.max_demos_per_task,
                    source_sha256=checksum,
                    log_interval=args.log_interval,
                )
            finally:
                environment.close()
            summary = {
                **asdict(result),
                "destination": str(result.destination),
                "status": "built",
            }
            print(
                f"  complete | kept={result.written_demos}/{result.source_demos} | "
                f"steps={result.output_steps:,} | {result.elapsed_seconds:.1f}s",
                flush=True,
            )

        summary["task_id"] = task_id
        summary["task_name"] = task_name
        summary["instruction"] = str(task.language)
        summary["source_file"] = str(source_task.path)
        summary["source_sha256"] = checksum
        manifest["tasks"][str(task_id)] = summary
        _atomic_json(manifest_path, manifest)

    print(
        f"Regeneration finished | tasks={len(task_ids)} | "
        f"elapsed={time.monotonic() - total_started:.1f}s | manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
