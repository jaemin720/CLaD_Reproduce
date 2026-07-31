"""Discovery and metadata parsing for LIBERO per-task HDF5 files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py


@dataclass(frozen=True, slots=True)
class LiberoTask:
    """Metadata needed to address one LIBERO task file."""

    task_id: str
    instruction: str
    path: Path
    num_demos: int


def _decode_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_instruction(data_group: h5py.Group, path: Path) -> str:
    raw_problem_info = _decode_attribute(data_group.attrs.get("problem_info", ""))
    if raw_problem_info:
        try:
            problem_info = json.loads(raw_problem_info)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid problem_info JSON in {path}: {error}") from error
        instruction = str(problem_info.get("language_instruction", "")).strip()
        if instruction:
            return instruction

    # Keep discovery usable for synthetic or converted datasets that only
    # carry a direct language_instruction attribute.
    direct = _decode_attribute(data_group.attrs.get("language_instruction", ""))
    if direct:
        return str(direct).strip()

    raise ValueError(f"No language instruction found in HDF5 metadata: {path}")


def _demo_sort_key(name: str) -> tuple[int, str]:
    prefix, separator, suffix = name.rpartition("_")
    if separator and prefix == "demo" and suffix.isdigit():
        return int(suffix), name
    return 2**31 - 1, name


def list_demo_keys(data_group: h5py.Group) -> list[str]:
    """Return numerically sorted ``demo_N`` group names."""

    return sorted(
        (
            name
            for name, value in data_group.items()
            if isinstance(value, h5py.Group)
            and name.startswith("demo_")
            and name.removeprefix("demo_").isdigit()
        ),
        key=_demo_sort_key,
    )


def discover_libero_tasks(
    dataset_dir: str | Path,
    *,
    file_pattern: str = "*_demo.hdf5",
) -> list[LiberoTask]:
    """Discover task files and parse their language instructions."""

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO dataset directory does not exist: {root}")

    paths = sorted(root.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No LIBERO task files matching {file_pattern!r} were found in {root}"
        )

    tasks: list[LiberoTask] = []
    seen_ids: set[str] = set()
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "data" not in handle or not isinstance(handle["data"], h5py.Group):
                raise ValueError(f"Missing HDF5 group 'data' in {path}")
            data_group = handle["data"]
            demo_keys = list_demo_keys(data_group)
            if not demo_keys:
                raise ValueError(f"No demonstration groups found in {path}")
            instruction = _read_instruction(data_group, path)
            declared_num_demos = int(data_group.attrs.get("num_demos", len(demo_keys)))
            if declared_num_demos != len(demo_keys):
                raise ValueError(
                    f"num_demos={declared_num_demos} but found {len(demo_keys)} groups in {path}"
                )

        task_id = path.stem.removesuffix("_demo")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id {task_id!r} under {root}")
        seen_ids.add(task_id)
        tasks.append(
            LiberoTask(
                task_id=task_id,
                instruction=instruction,
                path=path,
                num_demos=len(demo_keys),
            )
        )

    return tasks
