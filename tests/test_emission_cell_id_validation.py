import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import (
    EncodingConfig,
    EncodingModel,
    EmissionConfig,
    _cell_id_row_indices,
    build_emissions,
)


def test_cell_id_row_indices_rejects_fractional_encoding_ids():
    with pytest.raises(ValueError, match="encoding.cell_ids.*integer"):
        _cell_id_row_indices(np.array([1.5]), np.array([1.0]))


def test_cell_id_row_indices_rejects_fractional_spike_ids():
    with pytest.raises(ValueError, match="spike cell IDs.*integer"):
        _cell_id_row_indices(np.array([1.0]), np.array([1.5]))


def test_build_emissions_rejects_fractional_ripple_spike_cell_ids():
    session = _single_ripple_session(spike_cell_id=1.5)

    with pytest.raises(ValueError, match="spike cell IDs.*integer"):
        build_emissions(
            session,
            _single_cell_encoding(),
            0,
            EmissionConfig(time_bin_s=1.0),
        )


def test_build_emissions_rejects_fractional_encoding_cell_ids():
    with pytest.raises(ValueError, match="encoding.cell_ids.*integer"):
        build_emissions(
            _single_ripple_session(spike_cell_id=1.0),
            _single_cell_encoding(cell_ids=np.array([1.5])),
            0,
            EmissionConfig(time_bin_s=1.0),
        )


def _single_ripple_session(*, spike_cell_id: float) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        spikes=np.array([[0.5, spike_cell_id]], dtype=float),
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
