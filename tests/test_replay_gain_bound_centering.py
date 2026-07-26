from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_improvement_extensions import _apply_replay_gains


def test_cell_gain_centering_preserves_reciprocal_bounds() -> None:
    rates_hz = np.ones((3, 1), dtype=float)
    counts = np.array([[100, 100, 0]], dtype=int)

    calibrated, metadata = _apply_replay_gains(
        rates_hz,
        counts,
        np.array([1.0], dtype=float),
        mode="cell",
        prior_count=0.0,
        max_gain=4.0,
    )

    gains = calibrated[:, 0] / rates_hz[:, 0]
    np.testing.assert_allclose(gains, np.array([2.0, 2.0, 0.25]))
    assert float(np.min(gains)) >= 0.25
    assert float(np.max(gains)) <= 4.0
    assert float(np.exp(np.mean(np.log(gains)))) == pytest.approx(1.0)
    assert metadata["replay_cell_gain_min"] == pytest.approx(0.25)
    assert metadata["replay_cell_gain_max"] == pytest.approx(2.0)
    assert metadata["replay_cell_gain_geomean"] == pytest.approx(1.0)


def test_cell_gain_centering_with_unit_bound_returns_unit_gains() -> None:
    rates_hz = np.array([[0.5], [2.0]], dtype=float)
    counts = np.array([[20, 0]], dtype=int)

    calibrated, metadata = _apply_replay_gains(
        rates_hz,
        counts,
        np.array([1.0], dtype=float),
        mode="cell",
        prior_count=0.0,
        max_gain=1.0,
    )

    np.testing.assert_allclose(calibrated, rates_hz)
    assert metadata["replay_cell_gain_min"] == pytest.approx(1.0)
    assert metadata["replay_cell_gain_max"] == pytest.approx(1.0)
    assert metadata["replay_cell_gain_geomean"] == pytest.approx(1.0)
