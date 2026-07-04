from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, EmissionConfig, build_emissions


def _minimal_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[1.0]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


def _minimal_session_with_spikes(spikes: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=None,  # type: ignore[arg-type]
        position=np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=float),
        spikes=np.asarray(spikes, dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array([[0.0, 0.02, 0.01, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.array([[0.0, 0.02]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


@pytest.mark.parametrize(
    "bad_cell_id, match",
    [
        (1.5, "integer-valued"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_build_emissions_rejects_malformed_spike_cell_ids(bad_cell_id: float, match: str) -> None:
    session = _minimal_session_with_spikes(np.array([[0.01, bad_cell_id]], dtype=float))

    with pytest.raises(ValueError, match=match):
        build_emissions(session, _minimal_encoding(), 0, EmissionConfig(time_bin_s=0.02))


def test_build_emissions_still_accepts_integral_float_spike_cell_ids() -> None:
    session = _minimal_session_with_spikes(np.array([[0.01, 1.0]], dtype=float))

    emissions = build_emissions(session, _minimal_encoding(), 0, EmissionConfig(time_bin_s=0.02))

    assert emissions.n_spikes == 1
    np.testing.assert_array_equal(emissions.spike_counts, np.array([[1]], dtype=int))
