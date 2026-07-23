from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvements import shuffle_well_labels


def test_shuffle_well_labels_keeps_byte_buffer_missing_markers_unlabelled() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2, 3],
            "true_well_id": [
                b"A",
                bytearray(b"NA"),
                memoryview(b"missing"),
                b"B",
            ],
            "true_well_x": [np.nan] * 4,
            "true_well_y": [np.nan] * 4,
        }
    )

    shuffled = shuffle_well_labels(frame, random_seed=3)

    assert shuffled.loc[0, "true_well_id"] == b"B"
    assert shuffled.loc[3, "true_well_id"] == b"A"
    assert bytes(shuffled.loc[1, "true_well_id"]) == b"NA"
    assert isinstance(shuffled.loc[1, "true_well_id"], bytearray)
    assert bytes(shuffled.loc[2, "true_well_id"]) == b"missing"
    assert isinstance(shuffled.loc[2, "true_well_id"], memoryview)
    assert shuffled["event"].tolist() == frame["event"].tolist()
