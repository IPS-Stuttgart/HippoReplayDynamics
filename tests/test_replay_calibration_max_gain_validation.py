from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, EncodingModel
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)


def _single_ripple_session() -> ReplaySession:
    times = np.linspace(0.0, 2.0, 21)
    position = np.column_stack(
        [
            times,
            np.linspace(0.0, 20.0, times.shape[0]),
            np.linspace(0.0, 5.0, times.shape[0]),
        ]
    )
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array([[0.20, 0.30, 0.25, 1.0, 0.0, 0.0]], dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _single_bin_encoding() -> EncodingModel:
    cell_ids = np.array([1], dtype=int)
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((cell_ids.shape[0], 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=cell_ids,
        config=EncodingConfig(),
    )


def test_replay_calibrated_emissions_reject_max_gain_below_one() -> None:
    with pytest.raises(ValueError, match="max_gain"):
        build_sorted_emissions_with_replay_calibration(
            _single_ripple_session(),
            _single_bin_encoding(),
            np.int64(0),
            EmissionConfig(time_bin_s=0.05),
            ReplayEmissionCalibration(gain_mode="event-cell", max_gain=0.5),
        )
