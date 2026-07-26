from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, EncodingModel
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)


def _session() -> ReplaySession:
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=Path("."),
        position=np.array([[0.0, 0.0, 0.0], [0.02, 1.0, 0.0]], dtype=float),
        spikes=np.array([[0.01, 1.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array([[0.0, 0.02, 0.01, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _encoding(rates_hz: np.ndarray, cell_ids: np.ndarray) -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.asarray(rates_hz, dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.asarray(cell_ids, dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize("emission_model", ["poisson", "negative-binomial"])
def test_replay_calibration_preserves_impossible_zero_rate_bins(emission_model: str) -> None:
    hipporeplayimm.apply_runtime_patches()
    emissions = build_sorted_emissions_with_replay_calibration(
        _session(),
        _encoding(np.array([[0.0, 5.0]]), np.array([1])),
        0,
        EmissionConfig(time_bin_s=0.02),
        ReplayEmissionCalibration(gain_mode="none", emission_model=emission_model),
    )

    assert emissions.log_likelihood.shape == (1, 2)
    assert np.isneginf(emissions.log_likelihood[0, 0])
    assert np.isfinite(emissions.log_likelihood[0, 1])


def test_zero_weight_cells_do_not_make_replay_bins_impossible() -> None:
    hipporeplayimm.apply_runtime_patches()
    session = _session()
    encoding = _encoding(
        np.array(
            [
                [0.0, 5.0],
                [5.0, 5.0],
            ]
        ),
        np.array([1, 2]),
    )
    emissions = build_sorted_emissions_with_replay_calibration(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=0.02, cell_weights=np.array([0.0, 1.0])),
        ReplayEmissionCalibration(gain_mode="none", emission_model="poisson"),
    )

    assert np.all(np.isfinite(emissions.log_likelihood))
