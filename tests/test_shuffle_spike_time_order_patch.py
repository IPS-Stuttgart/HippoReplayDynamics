from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm.data import ReplaySession, SpikeMarkData
import hipporeplayimm.result_improvements as result_improvements


def test_runtime_patch_refreshes_replaced_shuffle_spike_time_helper(monkeypatch) -> None:
    def stale_shuffle_helper(session, random_seed: int = 1):
        return session

    monkeypatch.setattr(
        result_improvements,
        "_shuffle_spike_time_order_patch_applied",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        result_improvements,
        "shuffle_spike_times_session",
        stale_shuffle_helper,
    )

    apply_runtime_patches()

    assert result_improvements.shuffle_spike_times_session is not stale_shuffle_helper

    shuffled = result_improvements.shuffle_spike_times_session(_marked_session(), random_seed=4)

    assert np.all(np.diff(shuffled.spikes[:, 0]) >= 0.0)
    assert shuffled.spike_marks is not None
    np.testing.assert_allclose(shuffled.spike_marks.times, shuffled.spikes[:, 0])
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, shuffled.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shuffled.spike_marks.marks[:, 0], shuffled.spikes[:, 1])


def _marked_session() -> ReplaySession:
    spikes = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=float,
    )
    spike_marks = SpikeMarkData(
        times=spikes[:, 0].copy(),
        marks=np.array(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
                [40.0, 4.0],
            ],
            dtype=float,
        ),
        source_file="synthetic",
        source_variable="marks",
        feature_names=("cell_id_proxy", "feature"),
        cell_ids=spikes[:, 1].astype(int),
        group_ids=np.array([1, 2, 3, 4], dtype=int),
    )
    return ReplaySession(
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
        spike_marks=spike_marks,
    )
