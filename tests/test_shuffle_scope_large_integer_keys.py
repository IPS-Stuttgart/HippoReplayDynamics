from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_keep_large_integer_event_ids_distinct() -> None:
    first_event = 2**53
    second_event = first_event + 1
    model = "sorted-spike-state-space-imm"
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": np.array([first_event, second_event], dtype=np.int64),
            "model": [model, model],
            "log_evidence": [5.0, 5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": np.array(
                [first_event, first_event, second_event, second_event],
                dtype=np.int64,
            ),
            "model": [model] * 4,
            "log_evidence": [0.0, 10.0, 6.0, 7.0],
        }
    )

    scored = add_shuffle_p_values(real_scores, control_scores)

    assert scored["shuffle_count"].tolist() == [2, 2]
    assert scored["shuffle_log_evidence_median"].tolist() == pytest.approx([5.0, 6.5])
    assert scored["shuffle_p_value"].tolist() == pytest.approx([2.0 / 3.0, 1.0])
