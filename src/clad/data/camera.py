"""Camera-key utilities shared by raw and cached LIBERO datasets."""

from __future__ import annotations

from collections.abc import Sequence


def camera_view_name(camera_key: str) -> str:
    """Convert ``obs/agentview_rgb`` into the stable view id ``agentview_rgb``."""

    normalized = camera_key.rstrip("/")
    if not normalized:
        raise ValueError("camera keys cannot be empty")
    return normalized.rsplit("/", maxsplit=1)[-1]


def normalize_camera_keys(camera_keys: Sequence[str]) -> tuple[str, ...]:
    """Validate camera paths and return an immutable representation."""

    if isinstance(camera_keys, str):
        raise TypeError(
            "camera_keys must be a sequence of HDF5 paths, not a single string"
        )

    normalized = tuple(camera_keys)
    if not normalized:
        raise ValueError("camera_keys cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"camera_keys contains duplicate paths: {normalized}")

    view_names = tuple(camera_view_name(key) for key in normalized)
    if len(set(view_names)) != len(view_names):
        raise ValueError(
            "Each camera path must have a unique final component; "
            f"got view names {view_names}"
        )
    return normalized

