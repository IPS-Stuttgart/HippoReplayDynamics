from __future__ import annotations

import numpy as np

from hipporeplayimm.position_validation import _subsample_position_windows


def _windows(count: int) -> list[dict[str, float]]:
    return [
        {
            "start_time": float(index),
            "end_time": float(index + 1),
            "center_time": float(index) + 0.5,
            "true_x": float(index),
            "true_y": 0.0,
        }
        for index in range(count)
    ]


def test_capped_position_windows_are_seeded_and_not_an_early_prefix() -> None:
    windows = _windows(20)

    selected = _subsample_position_windows(windows, 5, np.random.default_rng(17))
    repeated = _subsample_position_windows(windows, 5, np.random.default_rng(17))

    starts = [window["start_time"] for window in selected]
    assert starts == [window["start_time"] for window in repeated]
    assert starts == sorted(starts)
    assert starts != [window["start_time"] for window in windows[:5]]


def test_inactive_window_cap_does_not_advance_fold_rng() -> None:
    windows = _windows(6)
    rng = np.random.default_rng(23)

    selected = _subsample_position_windows(windows, len(windows), rng)
    observed_fold_order = rng.permutation(len(windows))
    expected_fold_order = np.random.default_rng(23).permutation(len(windows))

    assert selected is windows
    np.testing.assert_array_equal(observed_fold_order, expected_fold_order)
