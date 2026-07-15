from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.result_improvements import shuffle_spike_times_session
from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_spike_times_returns_time_sorted_mark_aligned_session() -> None:
    session = _marked_session()

    shuffled = shuffle_spike_times_session(session, random_seed=4)

    assert shuffled.spike_marks is not None
    assert np.all(np.diff(shuffled.spikes[:, 0]) >= 0.0)
    np.testing.assert_allclose(shuffled.spike_marks.times, shuffled.spikes[:, 0])
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, shuffled.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shuffled.spike_marks.marks[:, 0], shuffled.spikes[:, 1])


@pytest.mark.parametrize("random_seed", [-1, 1.5, True, float("nan")])
def test_shuffle_spike_times_rejects_invalid_random_seed(random_seed) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        shuffle_spike_times_session(_marked_session(), random_seed=random_seed)


def test_shuffle_p_values_keep_large_integer_event_scopes_separate() -> None:
    first_event = 2**53
    second_event = first_event + 1
    real_scores = pd.DataFrame(
        {
            "session": ["RatX/OpenX", "RatX/OpenX"],
            "event_index": [first_event, second_event],
            "model": ["model", "model"],
            "log_evidence": [50.0, 50.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["RatX/OpenX", "RatX/OpenX"],
            "event_index": [first_event, second_event],
            "model": ["model", "model"],
            "log_evidence": [10.0, 100.0],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert out["shuffle_count"].tolist() == [1, 1]
    assert out["shuffle_log_evidence_median"].tolist() == [10.0, 100.0]
    np.testing.assert_allclose(out["shuffle_p_value"], [0.5, 1.0])


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
