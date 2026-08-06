from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.clusterless_cv_exclusion import _merge_half_open_intervals


@pytest.mark.parametrize(
    "intervals",
    [
        np.array([[0.0, np.nan]]),
        np.array([[0.0, np.inf]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 1.0]]),
        np.array([0.0, 1.0, 2.0]),
        np.array([[False, True]]),
    ],
)
def test_clusterless_cv_rejects_invalid_excluded_intervals(
    intervals: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="excluded_intervals"):
        _merge_half_open_intervals(intervals)


def test_clusterless_cv_rejects_complex_excluded_intervals_without_warning() -> None:
    intervals = np.array([[0.0 + 1.0j, 1.0 + 0.0j]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite real bounds"):
            _merge_half_open_intervals(intervals)


def test_clusterless_cv_merges_valid_flat_interval_pairs() -> None:
    intervals = np.array([2.0, 3.0, 0.0, 1.0, 1.0, 2.0])

    merged = _merge_half_open_intervals(intervals)

    np.testing.assert_array_equal(merged, np.array([[0.0, 3.0]]))
