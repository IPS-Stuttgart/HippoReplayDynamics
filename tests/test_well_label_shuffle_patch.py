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
