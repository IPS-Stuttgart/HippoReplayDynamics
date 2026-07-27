from __future__ import annotations

from collections import Counter

import pandas as pd

from hipporeplayimm.well_label_shuffle_patch import shuffle_well_labels


_LABEL_COLUMNS = ["true_well_id", "true_well_x", "true_well_y"]


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
    assert shuffled["true_well_id"].tolist() != frame["true_well_id"].tolist()
    pd.testing.assert_frame_equal(
        shuffled[["session", "event"]],
        frame[["session", "event"]],
    )
