from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.well_label_shuffle_patch import (
    _coordinate_well_rows,
    _labelled_well_rows,
    shuffle_well_labels,
)


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "value",
    [
        3.0 + 4.0j,
        np.complex128(3.0 + 0.0j),
        _nested_scalar(np.complex128(3.0 + 4.0j)),
    ],
)
def test_labelled_well_rows_rejects_complex_scalars_without_lossy_cast(
    value: object,
) -> None:
    values = pd.Series([value, "A"], dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labelled = _labelled_well_rows(values)

    assert labelled.tolist() == [False, True]


def test_coordinate_well_rows_rejects_complex_scalars_without_lossy_cast() -> None:
    frame = pd.DataFrame(
        {
            "true_well_x": [1.0 + 2.0j, 3.0, 4.0],
            "true_well_y": [5.0, np.complex128(6.0 + 0.0j), 7.0],
        },
        dtype=object,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labelled = _coordinate_well_rows(frame)

    assert labelled.tolist() == [False, False, True]


def test_shuffle_well_labels_leaves_complex_ids_out_of_permutation() -> None:
    malformed = np.complex128(1.0 + 2.0j)
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_id": pd.Series([malformed, "A", "B"], dtype=object),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shuffled = shuffle_well_labels(frame, random_seed=0)

    assert shuffled.loc[0, "true_well_id"] == malformed
    assert shuffled.loc[[1, 2], "true_well_id"].tolist() == ["B", "A"]
    pd.testing.assert_series_equal(shuffled["event"], frame["event"])


def test_labelled_well_rows_rejects_cyclic_scalar_wrapper() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    values = pd.Series([cyclic, "A"], dtype=object)

    labelled = _labelled_well_rows(values)

    assert labelled.tolist() == [False, True]


def test_shuffle_well_labels_excludes_rows_with_malformed_sibling_labels() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2, 3],
            "true_well_id": pd.Series(
                [np.complex128(1.0 + 2.0j), "A", "B", "C"],
                dtype=object,
            ),
            "true_well_x": [1.0, np.complex128(2.0 + 1.0j), 3.0, 4.0],
            "true_well_y": [1.0, 2.0, 3.0, 4.0],
        },
        dtype=object,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shuffled = shuffle_well_labels(frame, random_seed=0)

    pd.testing.assert_series_equal(shuffled.loc[0], frame.loc[0])
    pd.testing.assert_series_equal(shuffled.loc[1], frame.loc[1])
    assert shuffled.loc[2, "true_well_id"] == "C"
    assert shuffled.loc[3, "true_well_id"] == "B"
    assert shuffled.loc[2, "true_well_x"] == 4.0
    assert shuffled.loc[3, "true_well_x"] == 3.0
