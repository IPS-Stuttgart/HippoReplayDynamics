from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.first_order_imm_time_weighting import (
    _duration_weighted_mode_summary,
)


def test_duration_weighted_summary_avoids_finite_duration_sum_overflow() -> None:
    mode_posterior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    durations = np.asarray([1.2e308, 6.0e307], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = _duration_weighted_mode_summary(mode_posterior, durations)

    assert summary["event_probability"] == pytest.approx(
        [2.0 / 3.0, 1.0 / 3.0, 0.0]
    )
    assert summary["fraction_time_map_stationary"] == pytest.approx(2.0 / 3.0)
    assert summary["fraction_time_map_nonstationary"] == pytest.approx(1.0 / 3.0)
    assert summary["longest_nonstationary_bout_s"] == pytest.approx(6.0e307)
    assert summary["mean_mode_entropy"] == pytest.approx(0.0)
