from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _time_bin_edges


def test_time_bin_edges_rejects_duration_overflow() -> None:
    limit = np.finfo(float).max

    with pytest.raises(
        ValueError,
        match="ripple duration exceeds floating-point range",
    ):
        _time_bin_edges(-limit, limit, 1.0)


def test_time_bin_edges_rejects_unrepresentable_bin_count() -> None:
    with pytest.raises(
        ValueError,
        match="ripple time-bin count exceeds platform index range",
    ):
        _time_bin_edges(0.0, np.finfo(float).max, np.finfo(float).tiny)


def test_time_bin_edges_keeps_partial_final_bin() -> None:
    edges = _time_bin_edges(0.0, 0.035, 0.02)

    np.testing.assert_allclose(edges, np.array([0.0, 0.02, 0.035]))
