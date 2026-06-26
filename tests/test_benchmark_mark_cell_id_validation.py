from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.benchmarks import _session_with_mark_cell_subset
from hipporeplayimm.data import ReplaySession, SpikeMarkData


def _marked_session(
    mark_cell_ids,
    mark_group_ids=None,
) -> ReplaySession:
    times = np.array([0.10, 0.20], dtype=float)
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("synthetic"),
        position=np.column_stack([times, np.zeros_like(times), np.zeros_like(times)]),
        spikes=np.array([[0.10, 1.0], [0.20, 2.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([1, 2], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=times,
            marks=np.array([[1.0], [2.0]], dtype=float),
            source_file="synthetic.mat",
            source_variable="Marks",
            feature_names=("amplitude",),
            cell_ids=np.asarray(mark_cell_ids),
            group_ids=None if mark_group_ids is None else np.asarray(mark_group_ids),
        ),
    )


def test_clusterless_mark_subset_rejects_fractional_requested_cell_ids() -> None:
    session = _marked_session([1, 2])

    with pytest.raises(ValueError, match="train cell IDs.*integer-valued"):
        _session_with_mark_cell_subset(session, np.array([1.9], dtype=float), role="train")


def test_clusterless_mark_subset_rejects_fractional_mark_cell_ids() -> None:
    session = _marked_session([1.5, 2.0])

    with pytest.raises(ValueError, match="spike-mark cell IDs.*integer-valued"):
        _session_with_mark_cell_subset(session, np.array([1], dtype=int), role="train")


def test_clusterless_mark_subset_rejects_boolean_mark_group_ids() -> None:
    session = _marked_session([1, 2], [True, 20])

    with pytest.raises(ValueError, match="spike-mark group IDs.*boolean"):
        _session_with_mark_cell_subset(session, np.array([1], dtype=int), role="train")


def test_clusterless_mark_subset_accepts_integral_float_ids_and_preserves_groups() -> None:
    session = _marked_session([1.0, 2.0], [10.0, 20.0])

    subset = _session_with_mark_cell_subset(session, np.array([2.0], dtype=float), role="test")

    assert subset.spike_marks is not None
    np.testing.assert_array_equal(subset.spike_marks.times, np.array([0.20], dtype=float))
    np.testing.assert_array_equal(subset.spike_marks.marks, np.array([[2.0]], dtype=float))
    np.testing.assert_array_equal(subset.spike_marks.cell_ids, np.array([2], dtype=int))
    np.testing.assert_array_equal(subset.spike_marks.group_ids, np.array([20], dtype=int))
