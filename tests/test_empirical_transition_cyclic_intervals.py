from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.empirical_transition import _validated_run_intervals


def test_empirical_transition_rejects_cyclic_run_interval_wrappers() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    intervals = np.empty((1, 2), dtype=object)
    intervals[0, 0] = cyclic
    intervals[0, 1] = 1.0

    with pytest.raises(
        ValueError,
        match="session.run_times must be a finite real array",
    ):
        _validated_run_intervals(intervals)


def test_empirical_transition_accepts_nested_real_run_interval_wrappers() -> None:
    nested_start = np.empty((), dtype=object)
    nested_start[()] = np.array(0.25)
    nested_end = np.empty((), dtype=object)
    nested_end[()] = np.array(np.float64(1.5))
    intervals = np.empty((1, 2), dtype=object)
    intervals[0, 0] = nested_start
    intervals[0, 1] = nested_end

    np.testing.assert_array_equal(
        _validated_run_intervals(intervals),
        np.array([[0.25, 1.5]]),
    )
