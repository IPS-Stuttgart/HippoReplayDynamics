from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.well_label_shuffle_patch import (
    _coordinate_well_rows,
    _labelled_well_rows,
    shuffle_well_labels,
)


def test_labelled_well_rows_rejects_boolean_ids_without_rejecting_integers() -> None:
    values = pd.Series(
        [True, False, np.bool_(True), np.asarray(False), 0, 1, "home"],
        dtype=object,
    )

    actual = _labelled_well_rows(values)

    expected = pd.Series(
        [False, False, False, False, True, True, True],
        index=values.index,
    )
    pd.testing.assert_series_equal(actual, expected)


def test_coordinate_well_rows_rejects_boolean_coordinates() -> None:
    frame = pd.DataFrame(
        {
            "true_well_x": [True, np.asarray(False), 1.0, 1.0],
            "true_well_y": [0.0, 2.0, 2.0, np.bool_(True)],
        }
    )

    actual = _coordinate_well_rows(frame)

    expected = pd.Series([False, False, True, False], index=frame.index)
    pd.testing.assert_series_equal(actual, expected)


def test_shuffle_well_labels_leaves_boolean_only_rows_out_of_permutation() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2, 3],
            "true_well_id": pd.Series([True, np.bool_(False), "A", "B"], dtype=object),
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled["true_well_id"].tolist() == [True, np.bool_(False), "B", "A"]
    pd.testing.assert_series_equal(shuffled["event"], frame["event"])
