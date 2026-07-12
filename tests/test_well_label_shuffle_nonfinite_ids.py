from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.well_label_shuffle_patch import (
    _labelled_well_rows,
    shuffle_well_labels,
)


def test_labelled_well_rows_rejects_nonfinite_numeric_ids() -> None:
    values = pd.Series(
        [np.inf, -np.inf, 1.0, "home", "inf", "-inf", None, "nan"],
        dtype=object,
    )

    actual = _labelled_well_rows(values)

    expected = pd.Series(
        [False, False, True, True, False, False, False, False],
        index=values.index,
    )
    pd.testing.assert_series_equal(actual, expected)


def test_shuffle_well_labels_keeps_nonfinite_ids_out_of_permutation() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_id": [np.inf, "A", "B"],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=0)

    assert np.isposinf(shuffled.loc[0, "true_well_id"])
    assert set(shuffled.loc[[1, 2], "true_well_id"]) == {"A", "B"}
    pd.testing.assert_series_equal(shuffled["event"], frame["event"])
