import pytest

from clad.data.sequence_sampler import build_window_indices, valid_anchor_steps


def test_valid_anchor_steps_require_all_three_states() -> None:
    assert list(valid_anchor_steps(10, 2)) == [2, 3, 4, 5, 6, 7]
    assert list(valid_anchor_steps(5, 2)) == [2]
    assert list(valid_anchor_steps(4, 2)) == []


def test_build_window_indices_preserves_episode_metadata() -> None:
    windows = build_window_indices(
        task_index=3,
        demo_key="demo_7",
        episode_length=8,
        horizon=2,
    )

    assert [window.anchor_step for window in windows] == [2, 3, 4, 5]
    assert all(window.task_index == 3 for window in windows)
    assert all(window.demo_key == "demo_7" for window in windows)
    assert all(window.episode_length == 8 for window in windows)


@pytest.mark.parametrize("horizon", [0, -1])
def test_horizon_must_be_positive(horizon: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        valid_anchor_steps(10, horizon)

