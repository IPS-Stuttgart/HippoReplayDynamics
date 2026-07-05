from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, EmissionConfig, LogEmissionTensor, _cell_id_row_indices, build_emissions


def test_cell_id_row_indices_rejects_string_encoding_ids() -> None:
    with pytest.raises(ValueError, match="encoding.cell_ids.*numeric"):
        _cell_id_row_indices(np.array(["1"], dtype=object), np.array([1.0]))


def test_cell_id_row_indices_rejects_string_spike_ids() -> None:
    with pytest.raises(ValueError, match="spike cell IDs.*numeric"):
        _cell_id_row_indices(np.array([1.0]), np.array(["1"], dtype=object))


def test_build_emissions_rejects_string_ripple_spike_cell_ids() -> None:
    with pytest.raises(ValueError, match="spike cell IDs.*numeric"):
        build_emissions(
            _single_ripple_session(spike_cell_id="1"),
            _single_cell_encoding(),
            0,
            EmissionConfig(time_bin_s=1.0),
        )


def test_build_emissions_rejects_string_encoding_cell_ids() -> None:
    with pytest.raises(ValueError, match="encoding.cell_ids.*numeric"):
        build_emissions(
            _single_ripple_session(spike_cell_id=1.0),
            _single_cell_encoding(cell_ids=np.array(["1"], dtype=object)),
            0,
            EmissionConfig(time_bin_s=1.0),
        )


def test_log_emission_tensor_rejects_string_cell_ids() -> None:
    with pytest.raises(ValueError, match="cell_ids.*numeric"):
        LogEmissionTensor(
            log_likelihood=np.zeros((1, 1), dtype=float),
            spike_counts=np.zeros((1, 1), dtype=int),
            times=np.array([0.0]),
            dt=1.0,
            cell_ids=np.array(["1"], dtype=object),
            n_spikes=0,
        )


def test_log_emission_tensor_still_accepts_integer_valued_float_cell_ids() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 1), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1.0]),
        n_spikes=0,
    )

    np.testing.assert_array_equal(emissions.cell_ids, np.array([1], dtype=int))


def _single_ripple_session(*, spike_cell_id: object) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,  # type: ignore[arg-type]
        position=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        spikes=np.array([[0.5, spike_cell_id]], dtype=object),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 1.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def _single_cell_encoding(*, cell_ids: np.ndarray | None = None) -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.array([[2.0, 4.0]]),
        occupancy_s=np.ones(2),
        cell_ids=np.array([1]) if cell_ids is None else cell_ids,
        config=EncodingConfig(),
    )
