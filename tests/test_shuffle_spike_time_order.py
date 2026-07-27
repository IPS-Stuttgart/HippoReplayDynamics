from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.shuffle_spike_time_order as shuffle_patch
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.result_improvements import (
    shuffle_mark_features_session,
    shuffle_spike_times_session,
)
from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_spike_times_returns_time_sorted_mark_aligned_session() -> None:
    session = _marked_session()

    shuffled = shuffle_spike_times_session(session, random_seed=4)

    assert shuffled.spike_marks is not None
    assert np.all(np.diff(shuffled.spikes[:, 0]) >= 0.0)
    np.testing.assert_allclose(shuffled.spike_marks.times, shuffled.spikes[:, 0])
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, shuffled.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shuffled.spike_marks.marks[:, 0], shuffled.spikes[:, 1])


def test_shuffle_spike_times_retries_identity_draw_when_change_is_possible() -> None:
    session = _marked_session()

    # NumPy's first four-element permutation for seed 1 is the identity.
    shuffled = shuffle_spike_times_session(session, random_seed=1)

    assert shuffled.spike_marks is not None
    assert not np.array_equal(shuffled.spikes, session.spikes)
    np.testing.assert_allclose(shuffled.spikes[:, 0], session.spikes[:, 0])
    np.testing.assert_allclose(shuffled.spike_marks.times, shuffled.spikes[:, 0])
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, shuffled.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shuffled.spike_marks.marks[:, 0], shuffled.spikes[:, 1])


def test_shuffle_spike_times_retries_duplicate_time_noop_draw() -> None:
    session = _marked_session()
    session.spikes[1, 0] = session.spikes[0, 0]
    assert session.spike_marks is not None
    session.spike_marks.times[1] = session.spike_marks.times[0]
    original_cell_order = session.spikes[:, 1].copy()

    # Seed 31 first swaps only the duplicate-time rows, which is observationally
    # unchanged and therefore must be retried.
    shuffled = shuffle_spike_times_session(session, random_seed=31)

    assert shuffled.spike_marks is not None
    assert not np.array_equal(shuffled.spikes[:, 1], original_cell_order)
    assert np.all(np.diff(shuffled.spikes[:, 0]) >= 0.0)
    np.testing.assert_allclose(shuffled.spike_marks.times, shuffled.spikes[:, 0])
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, shuffled.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shuffled.spike_marks.marks[:, 0], shuffled.spikes[:, 1])


def test_shuffle_mark_features_retries_identity_draw_when_change_is_possible() -> None:
    session = _marked_session()
    assert session.spike_marks is not None
    session.spike_marks = replace(
        session.spike_marks,
        marks=session.spike_marks.marks[:, :1].copy(),
        feature_names=("cell_id_proxy",),
    )

    # NumPy's first four-element permutation for seed 1 is the identity.
    shuffled = shuffle_mark_features_session(session, random_seed=1)

    assert shuffled.spike_marks is not None
    assert not np.array_equal(shuffled.spike_marks.marks, session.spike_marks.marks)


def test_shuffle_mark_features_retries_duplicate_value_noop_draw() -> None:
    session = _marked_session()
    assert session.spike_marks is not None
    session.spike_marks = replace(
        session.spike_marks,
        marks=np.array([[0.0], [0.0], [1.0], [2.0]]),
        feature_names=("feature",),
    )

    # Seed 31 first swaps only the equal-valued rows, which is observationally
    # unchanged and therefore must be retried.
    shuffled = shuffle_mark_features_session(session, random_seed=31)

    assert shuffled.spike_marks is not None
    assert not np.array_equal(shuffled.spike_marks.marks, session.spike_marks.marks)


@pytest.mark.parametrize("random_seed", [-1, 1.5, True, float("nan")])
def test_shuffle_mark_features_rejects_invalid_random_seed(random_seed) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        shuffle_mark_features_session(_marked_session(), random_seed=random_seed)


@pytest.mark.parametrize("random_seed", [-1, 1.5, True, float("nan")])
def test_shuffle_spike_times_rejects_invalid_random_seed(random_seed) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        shuffle_spike_times_session(_marked_session(), random_seed=random_seed)


@pytest.mark.parametrize("random_seed", [2**53 + 1, np.int64(2**53 + 1)])
def test_shuffle_spike_times_preserves_exact_large_integer_seed(monkeypatch, random_seed) -> None:
    captured_seeds: list[int] = []

    class SeedCaptureGenerator:
        @staticmethod
        def permutation(values):
            if np.ndim(values) == 0:
                return np.arange(int(values) - 1, -1, -1, dtype=int)
            return np.asarray(values).copy()

    def capture_seed(seed: int) -> SeedCaptureGenerator:
        captured_seeds.append(seed)
        return SeedCaptureGenerator()

    monkeypatch.setattr(shuffle_patch.np.random, "default_rng", capture_seed)

    shuffle_spike_times_session(_marked_session(), random_seed=random_seed)

    assert captured_seeds == [2**53 + 1]


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
