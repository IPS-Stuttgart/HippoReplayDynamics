from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.emission_cell_id_validation import _cell_id_row_indices


def test_large_integer_log_emission_cell_ids_remain_distinct() -> None:
    first = 2**53
    second = first + 1

    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 2), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=np.array([first, second], dtype=object),
        n_spikes=0,
    )

    assert emissions.cell_ids.tolist() == [first, second]


def test_large_integer_emission_row_lookup_remains_distinct() -> None:
    first = 2**53
    second = first + 1

    rows = _cell_id_row_indices(
        np.array([first, second], dtype=object),
        np.array([second], dtype=object),
    )

    assert rows.tolist() == [1]


def test_large_text_integer_emission_row_lookup_remains_distinct() -> None:
    first = 2**53
    second = first + 1
    cases = (
        ([str(first), str(second)], [str(second)]),
        ([f"{first}.0", f"{second}.0"], [f"{second}.0"]),
        ([str(first).encode(), str(second).encode()], [str(second).encode()]),
    )

    for available, requested in cases:
        rows = _cell_id_row_indices(
            np.array(available, dtype=object),
            np.array(requested, dtype=object),
        )

        assert rows.tolist() == [1]


def test_large_integer_replay_session_cell_ids_remain_distinct() -> None:
    first = 2**53
    second = first + 1
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.array([[0.1, first], [0.2, second]], dtype=object),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([second], dtype=object),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )

    assert session.cell_ids.tolist() == [first, second]
    assert session.excitatory_spikes()[:, 1].tolist() == [second]
