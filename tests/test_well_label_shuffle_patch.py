from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvements import shuffle_well_labels


def test_shuffle_well_labels_shuffles_ids_when_coordinates_are_missing() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_id": ["A", "B", None],
            "true_well_x": [np.nan, np.nan, np.nan],
            "true_well_y": [np.nan, np.nan, np.nan],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled.loc[:1, "true_well_id"].tolist() == ["B", "A"]
    assert shuffled.loc[:1, ["true_well_x", "true_well_y"]].isna().all().all()
    assert shuffled.loc[2, ["true_well_id", "true_well_x", "true_well_y"]].isna().all()
    assert shuffled["event"].tolist() == frame["event"].tolist()


def test_shuffle_well_labels_leaves_textual_missing_well_ids_unlabelled() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2, 3],
            "true_well_id": ["A", "", "nan", "B"],
            "true_well_x": [1.0, np.nan, np.nan, 2.0],
            "true_well_y": [3.0, np.nan, np.nan, 4.0],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled.loc[[0, 3], "true_well_id"].tolist() == ["B", "A"]
    assert shuffled.loc[1, "true_well_id"] == ""
    assert shuffled.loc[2, "true_well_id"] == "nan"
    assert shuffled.loc[[1, 2], ["true_well_x", "true_well_y"]].isna().all().all()
    assert shuffled["event"].tolist() == frame["event"].tolist()


def test_shuffle_well_labels_shuffles_coordinate_only_tables() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_x": [1.0, 2.0, np.nan],
            "true_well_y": [10.0, 20.0, np.nan],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled.loc[:1, ["true_well_x", "true_well_y"]].to_numpy().tolist() == [
        [2.0, 20.0],
        [1.0, 10.0],
    ]
    assert shuffled.loc[2, ["true_well_x", "true_well_y"]].isna().all()
    assert shuffled["event"].tolist() == frame["event"].tolist()


def test_shuffle_well_labels_uses_coordinates_when_well_id_column_is_missing() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_id": ["", None, "nan"],
            "true_well_x": [1.0, 2.0, np.nan],
            "true_well_y": [10.0, 20.0, np.nan],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled.loc[:1, ["true_well_x", "true_well_y"]].to_numpy().tolist() == [
        [2.0, 20.0],
        [1.0, 10.0],
    ]
    assert shuffled.loc[2, "true_well_id"] == "nan"
    assert shuffled.loc[2, ["true_well_x", "true_well_y"]].isna().all()
    assert shuffled["event"].tolist() == frame["event"].tolist()


def test_shuffle_well_labels_does_not_treat_partial_coordinates_as_labels() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2, 3],
            "true_well_id": ["", "", "", ""],
            "true_well_x": [1.0, 2.0, 3.0, np.nan],
            "true_well_y": [np.nan, 20.0, 30.0, np.nan],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled.loc[0, "true_well_x"] == 1.0
    assert pd.isna(shuffled.loc[0, "true_well_y"])
    assert shuffled.loc[[1, 2], ["true_well_x", "true_well_y"]].to_numpy().tolist() == [
        [3.0, 30.0],
        [2.0, 20.0],
    ]
    assert shuffled.loc[3, ["true_well_x", "true_well_y"]].isna().all()
    assert shuffled["event"].tolist() == frame["event"].tolist()


def test_shuffle_well_labels_requires_complete_coordinate_pair_for_coordinate_only_rows() -> None:
    x_only = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_x": [1.0, 2.0, np.nan],
        }
    )
    y_only = pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_y": [10.0, 20.0, np.nan],
        }
    )

    pd.testing.assert_frame_equal(shuffle_well_labels(x_only, random_seed=3), x_only)
    pd.testing.assert_frame_equal(shuffle_well_labels(y_only, random_seed=3), y_only)
