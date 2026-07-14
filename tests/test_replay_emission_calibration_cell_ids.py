from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.replay_emission_calibration import (
    ReplayEmissionCalibration,
    apply_replay_cell_gains,
)


def _two_cell_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(),
    )


def _calibration(cell_ids: object, gains: object) -> ReplayEmissionCalibration:
    gain_values = np.asarray(gains)
    return ReplayEmissionCalibration(
        cell_ids=np.asarray(cell_ids),
        gains=gain_values,
        observed_spikes=np.zeros(gain_values.shape, dtype=float),
        expected_spikes=np.zeros(gain_values.shape, dtype=float),
        prior_count=0.0,
        prior_gain=1.0,
        event_count=0,
    )


@pytest.mark.parametrize(
    ("cell_ids", "message"),
    [
        (np.array([1.5, 2.0]), "integer-valued"),
        (np.array([True, False]), "boolean"),
        (np.array([1, 1]), "unique"),
    ],
    ids=["fractional", "boolean", "duplicate"],
)
def test_apply_replay_cell_gains_rejects_invalid_calibration_cell_ids(
    cell_ids: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_replay_cell_gains(
            _two_cell_encoding(),
            _calibration(cell_ids, np.array([2.0, 3.0])),
        )


def test_apply_replay_cell_gains_rejects_calibration_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="match cell_ids shape"):
        apply_replay_cell_gains(
            _two_cell_encoding(),
            _calibration(np.array([1, 2]), np.array([[2.0], [3.0]])),
        )


def test_apply_replay_cell_gains_preserves_integral_float_cell_ids() -> None:
    calibrated = apply_replay_cell_gains(
        _two_cell_encoding(),
        _calibration(np.array([1.0, 2.0]), np.array([2.0, 3.0])),
    )

    np.testing.assert_allclose(calibrated.rates_hz, np.array([[2.0], [3.0]]))
