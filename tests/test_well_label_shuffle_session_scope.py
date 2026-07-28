from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import hipporeplayimm.well_label_shuffle_patch as well_label_patch
from hipporeplayimm.well_label_shuffle_patch import shuffle_well_labels


_LABEL_COLUMNS = ["true_well_id", "true_well_x", "true_well_y"]


class _DuplicateNoopThenChangeGenerator:
    """Return a duplicate-only swap before a genuinely changed permutation."""

    def __init__(self) -> None:
        self._permutations = iter(
            (
                np.array([1, 0, 2], dtype=int),
                np.array([2, 1, 0], dtype=int),
            )
        )
        self.calls = 0

    def permutation(self, size: int) -> np.ndarray:
        assert size == 3
        self.calls += 1
        return next(self._permutations).copy()


def _label_tuples(frame: pd.DataFrame, session: str) -> Counter[tuple[object, ...]]:
    values = frame.loc[frame["session"].eq(session), _LABEL_COLUMNS].to_numpy()
    return Counter(tuple(row) for row in values)


def test_shuffle_well_labels_preserves_session_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat2/Open1", "Rat2/Open1"],
            "event": [0, 1, 0, 1],
            "true_well_id": ["A1", "A2", "B1", "B2"],
            "true_well_x": [1.0, 2.0, 101.0, 102.0],
            "true_well_y": [10.0, 20.0, 110.0, 120.0],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    for session in frame["session"].unique():
        assert _label_tuples(shuffled, session) == _label_tuples(frame, session)
        assert shuffled.loc[
            shuffled["session"].eq(session), _LABEL_COLUMNS
        ].to_numpy().tolist() != frame.loc[
            frame["session"].eq(session), _LABEL_COLUMNS
        ].to_numpy().tolist()
    pd.testing.assert_frame_equal(
        shuffled[["session", "event"]],
        frame[["session", "event"]],
    )


def test_shuffle_well_labels_rejects_identity_draw_without_session() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1],
            "true_well_id": ["A1", "A2"],
            "true_well_x": [1.0, 2.0],
            "true_well_y": [10.0, 20.0],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=0)

    assert shuffled[_LABEL_COLUMNS].to_numpy().tolist() == frame.loc[
        [1, 0], _LABEL_COLUMNS
    ].to_numpy().tolist()
    pd.testing.assert_series_equal(shuffled["event"], frame["event"])


def test_shuffle_well_labels_retries_duplicate_row_noop(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_id": ["A1", "A1", "A2"],
            "true_well_x": [1.0, 1.0, 2.0],
            "true_well_y": [10.0, 10.0, 20.0],
        }
    )
    generator = _DuplicateNoopThenChangeGenerator()
    monkeypatch.setattr(
        well_label_patch.np.random,
        "default_rng",
        lambda _seed: generator,
    )

    shuffled = shuffle_well_labels(frame, random_seed=0)

    assert generator.calls == 2
    assert shuffled[_LABEL_COLUMNS].to_numpy().tolist() == frame.loc[
        [2, 1, 0], _LABEL_COLUMNS
    ].to_numpy().tolist()
    pd.testing.assert_series_equal(shuffled["event"], frame["event"])
