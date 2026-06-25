from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import _as_intervals


def test_interval_arrays_reject_nonfinite_endpoints() -> None:
    with pytest.raises(ValueError, match="finite"):
        _as_intervals(np.array([[0.0, np.nan]], dtype=float))

    with pytest.raises(ValueError, match="finite"):
        _as_intervals(np.array([[0.0, np.inf]], dtype=float))


def test_interval_arrays_reject_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="end times"):
        _as_intervals(np.array([[2.0, 1.0]], dtype=float))


def test_interval_arrays_accept_zero_duration_bounds() -> None:
    intervals = _as_intervals(np.array([1.0, 1.0], dtype=float))

    np.testing.assert_allclose(intervals, np.array([[1.0, 1.0]], dtype=float))
