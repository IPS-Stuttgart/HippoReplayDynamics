from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession, _mark_group_ids_from_tetrode_cell_ids


def _session_with_ids(spike_ids, excitatory_ids=()):
    spike_ids = np.asarray(spike_ids, dtype=float)
    spikes = np.column_stack([np.arange(spike_ids.shape[0], dtype=float), spike_ids])
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=np.empty((0, 3), dtype=float),
        spikes=spikes,
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.asarray(excitatory_ids, dtype=float),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _session_with_raw_spikes(spikes) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=np.empty((0, 3), dtype=float),
        spikes=np.asarray(spikes, dtype=object),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def test_replay_session_cell_ids_accept_integral_float_ids():
    session = _session_with_ids([2.0, 1.0, 2.0])

    np.testing.assert_array_equal(session.cell_ids, np.array([1, 2], dtype=int))


def test_replay_session_cell_ids_reject_fractional_spike_ids():
    session = _session_with_ids([1.0, 2.5])

    with pytest.raises(ValueError, match="spike cell IDs"):
        _ = session.cell_ids


def test_replay_session_cell_ids_reject_boolean_spike_ids():
    session = _session_with_raw_spikes([[0.0, True]])

    with pytest.raises(ValueError, match="boolean"):
        _ = session.cell_ids


def test_replay_session_cell_ids_reject_out_of_range_spike_ids():
    session = _session_with_ids([1.0, 1e20])

    with pytest.raises(ValueError, match="integer identifier range"):
        _ = session.cell_ids


def test_replay_session_excitatory_spikes_reject_fractional_excitatory_ids():
    session = _session_with_ids([1.0, 2.0], excitatory_ids=[1.5])

    with pytest.raises(ValueError, match="excitatory neuron IDs"):
        session.excitatory_spikes()


def test_mark_group_ids_reject_fractional_tetrode_mapping_ids():
    with pytest.raises(ValueError, match="tetrode/cell IDs"):
        _mark_group_ids_from_tetrode_cell_ids(
            np.array([1, 2], dtype=int),
            np.array([[7.0, 1.0], [8.5, 2.0]], dtype=float),
        )
