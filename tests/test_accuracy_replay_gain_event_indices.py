from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.accuracy_upgrades import (
    ContinuousTimeEmissionConfig,
    build_continuous_time_emissions,
    estimate_replay_cell_gains,
)
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel



def _session_with_two_ripples() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenTest",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array(
            [
                [0.0, 0.30, 0.15, 0.0, 0.0, 0.0],
                [0.40, 0.70, 0.55, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )



def _two_cell_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize("event_index", [True, False, np.bool_(True), np.bool_(False)])
def test_estimate_replay_cell_gains_rejects_boolean_event_indices(event_index) -> None:
    with pytest.raises(TypeError, match="event index must be an integer, not boolean"):
        estimate_replay_cell_gains(_session_with_two_ripples(), _two_cell_encoding(), [event_index])


@pytest.mark.parametrize("event_index", [0.0, 1.5, np.float64(1.0), "0"])
def test_estimate_replay_cell_gains_rejects_non_integer_event_indices(event_index) -> None:
    with pytest.raises(TypeError, match="event index must be an integer"):
        estimate_replay_cell_gains(_session_with_two_ripples(), _two_cell_encoding(), [event_index])


def test_continuous_time_emissions_ignore_out_of_window_malformed_cell_ids() -> None:
    session = _session_with_two_ripples()
    session.spikes = np.array(
        [
            [0.10, 1],
            [0.35, "bad-outside-window"],
        ],
        dtype=object,
    )

    emissions = build_continuous_time_emissions(
        session,
        _two_cell_encoding(),
        0,
        ContinuousTimeEmissionConfig(min_interval_s=1e-6),
    )

    assert emissions.n_spikes == 1
    assert int(emissions.spike_counts.sum()) == 1
    assert emissions.spike_counts.shape[1] == 2
