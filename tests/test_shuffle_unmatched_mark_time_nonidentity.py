from __future__ import annotations

from pathlib import Path

import numpy as np

import hipporeplayimm.shuffle_spike_time_order as shuffle_patch
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.result_improvements import shuffle_spike_times_session


def test_shuffle_spike_times_retries_identity_draw_for_unmatched_mark_rows(
    monkeypatch,
) -> None:
    class ControlledGenerator:
        def __init__(self) -> None:
            self.permutation_calls = 0

        def permutation(self, values):
            if np.ndim(values) == 0:
                array = np.arange(int(values), dtype=int)
            else:
                array = np.asarray(values).copy()
            self.permutation_calls += 1
            if self.permutation_calls == 2:
                return array.copy()
            return array[::-1].copy()

    generator = ControlledGenerator()
    monkeypatch.setattr(
        shuffle_patch.np.random,
        "default_rng",
        lambda _seed: generator,
    )

    spikes = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=float,
    )
    mark_times = np.array([0.25, 0.75], dtype=float)
    mark_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=spikes,
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=mark_values,
            source_file="synthetic",
            source_variable="marks",
            feature_names=("feature_a", "feature_b"),
            cell_ids=np.array([10, 20], dtype=int),
            group_ids=np.array([1, 2], dtype=int),
        ),
    )

    shuffled = shuffle_spike_times_session(session, random_seed=2)

    assert shuffled.spike_marks is not None
    assert generator.permutation_calls == 3
    np.testing.assert_array_equal(shuffled.spike_marks.times, mark_times[::-1])
    np.testing.assert_array_equal(shuffled.spike_marks.marks, mark_values)
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, [10, 20])
    np.testing.assert_array_equal(shuffled.spike_marks.group_ids, [1, 2])
