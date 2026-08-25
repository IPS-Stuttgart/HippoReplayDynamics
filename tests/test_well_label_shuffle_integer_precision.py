from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvements import shuffle_well_labels


def test_shuffle_well_labels_preserves_large_integer_ids_exactly() -> None:
    first_id = 2**53
    second_id = first_id + 1
    frame = pd.DataFrame(
        {
            "true_well_id": pd.Series([first_id, second_id], dtype=np.int64),
            "true_well_x": [1.0, 2.0],
            "true_well_y": [10.0, 20.0],
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled["true_well_id"].tolist() == [second_id, first_id]
    assert shuffled["true_well_id"].dtype == frame["true_well_id"].dtype
    assert shuffled[["true_well_x", "true_well_y"]].to_numpy().tolist() == [
        [2.0, 20.0],
        [1.0, 10.0],
    ]
