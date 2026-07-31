"""Episode-safe temporal windows used by both CLaD training stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowIndex:
    """Location of one CLaD window inside one demonstration.

    For an anchor ``t`` and horizon ``tau`` the sample contains states at
    ``t - tau``, ``t``, and ``t + tau``. Past and target actions use Python
    half-open intervals ``[t - tau, t)`` and ``[t, t + tau)``.
    """

    task_index: int
    demo_key: str
    anchor_step: int
    episode_length: int


def valid_anchor_steps(episode_length: int, horizon: int) -> range:
    """Return anchors with complete past/future state and action context."""

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if episode_length < 0:
        raise ValueError(f"episode_length must be non-negative, got {episode_length}")

    # t + horizon is read as a future observation, hence it must be strictly
    # smaller than episode_length. The stop value of range is exclusive.
    return range(horizon, episode_length - horizon)


def build_window_indices(
    *,
    task_index: int,
    demo_key: str,
    episode_length: int,
    horizon: int,
) -> list[WindowIndex]:
    """Build all valid windows for one episode without boundary padding."""

    return [
        WindowIndex(
            task_index=task_index,
            demo_key=demo_key,
            anchor_step=anchor,
            episode_length=episode_length,
        )
        for anchor in valid_anchor_steps(episode_length, horizon)
    ]

