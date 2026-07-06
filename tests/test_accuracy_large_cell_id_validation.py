from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.accuracy_upgrades import (
    ContinuousTimeEmissionConfig,
    ReplayGainConfig,
    build_continuous_time_emissions,
    estimate_replay_cell_gains,
)
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


LARGE_A = 9007199254740993
LARGE_B = 9007199254740995


def _large_id_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenTest",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.array(
            [
                [0.10, LARGE_A],
                [0.20, LARGE_B],
                [0.25, LARGE_A],
            ],
            dtype=object,
        ),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array([[0.0, 0.30, 0.15, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _large_id_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([LARGE_B, LARGE_A], dtype=object),
        config=EncodingConfig(),
    )


def test_estimate_replay_cell_gains_preserves_large_integer_cell_ids() -> None:
    gains = estimate_replay_cell_gains(
        _large_id_session(),
        _large_id_encoding(),
        [0],
        ReplayGainConfig(
            prior_observed_spikes=0.0,
            prior_expected_spikes=1.0,
            min_gain=0.0,
            max_gain=10.0,
        ),
    )

    np.testing.assert_allclose(gains, [1.0 / 1.3, 2.0 / 1.3])


def test_continuous_time_emissions_preserves_large_integer_cell_ids() -> None:
    emissions = build_continuous_time_emissions(
        _large_id_session(),
        _large_id_encoding(),
        0,
        ContinuousTimeEmissionConfig(include_terminal_no_spike_interval=False),
    )

    assert emissions.cell_ids.tolist() == [LARGE_B, LARGE_A]
    np.testing.assert_array_equal(emissions.spike_counts.sum(axis=0), np.array([1, 2]))
