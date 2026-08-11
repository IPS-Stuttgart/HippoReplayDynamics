from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_improvement_extensions import _apply_replay_gains


def test_event_gain_uses_physical_bin_durations() -> None:
    rates_hz = np.array([[2.0, 10.0]], dtype=float)
    counts = np.array([[1], [3]], dtype=int)
    bin_durations = np.array([0.75, 0.25], dtype=float)

    calibrated, metadata = _apply_replay_gains(
        rates_hz,
        counts,
        bin_durations,
        mode="event",
        prior_count=0.0,
        max_gain=10.0,
    )

    np.testing.assert_allclose(
        calibrated,
        np.maximum(rates_hz, np.finfo(float).tiny),
    )
    assert metadata["replay_event_gain"] == pytest.approx(1.0)


def test_cell_gains_use_physical_bin_durations() -> None:
    rates_hz = np.array(
        [
            [2.0, 10.0],
            [8.0, 0.0],
        ],
        dtype=float,
    )
    counts = np.array(
        [
            [2, 3],
            [2, 3],
        ],
        dtype=int,
    )
    bin_durations = np.array([0.75, 0.25], dtype=float)

    calibrated, metadata = _apply_replay_gains(
        rates_hz,
        counts,
        bin_durations,
        mode="cell",
        prior_count=0.0,
        max_gain=10.0,
    )

    np.testing.assert_allclose(
        calibrated,
        np.maximum(rates_hz, np.finfo(float).tiny),
    )
    assert metadata["replay_cell_gain_min"] == pytest.approx(1.0)
    assert metadata["replay_cell_gain_max"] == pytest.approx(1.0)
    assert metadata["replay_cell_gain_geomean"] == pytest.approx(1.0)
