from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hipporeplayimm.result_improvements import shuffle_mark_features_session


@dataclass(frozen=True)
class _Marks:
    marks: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.marks.shape[1]) if self.marks.ndim == 2 else 0


@dataclass(frozen=True)
class _Session:
    spike_marks: _Marks | None


def _session(values: list[list[float]]) -> _Session:
    return _Session(spike_marks=_Marks(marks=np.asarray(values, dtype=float)))


def test_mark_feature_shuffle_rejects_identity_draw() -> None:
    session = _session([[1.0], [2.0]])

    shuffled = shuffle_mark_features_session(session, random_seed=0)

    np.testing.assert_array_equal(
        shuffled.spike_marks.marks,
        np.array([[2.0], [1.0]]),
    )


def test_mark_feature_shuffle_changes_duplicate_column_when_possible() -> None:
    session = _session([[1.0], [1.0], [2.0]])

    shuffled = shuffle_mark_features_session(session, random_seed=0)

    assert not np.array_equal(
        shuffled.spike_marks.marks,
        session.spike_marks.marks,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.sort(shuffled.spike_marks.marks[:, 0]),
        np.sort(session.spike_marks.marks[:, 0]),
    )


def test_mark_feature_shuffle_preserves_unavoidable_constant_column() -> None:
    session = _session([[3.0], [3.0]])

    shuffled = shuffle_mark_features_session(session, random_seed=0)

    np.testing.assert_array_equal(
        shuffled.spike_marks.marks,
        session.spike_marks.marks,
    )


def test_mark_feature_shuffle_validates_seed_before_noop() -> None:
    session = _Session(spike_marks=None)

    with pytest.raises(ValueError, match="random_seed"):
        shuffle_mark_features_session(session, random_seed=True)
