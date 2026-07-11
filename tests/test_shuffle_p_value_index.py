from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_add_shuffle_p_values_preserves_named_duplicate_score_index() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 4],
            "model": ["model-a", "model-a"],
            "log_evidence": [10.0, 5.0],
        },
        index=pd.Index(["score-row", "score-row"], name="score_index"),
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [3, 3, 4, 4],
            "model": ["model-a"] * 4,
            "log_evidence": [9.0, 11.0, 4.0, 6.0],
        }
    )

    result = add_shuffle_p_values(real_scores, control_scores)

    pd.testing.assert_index_equal(result.index, real_scores.index)
    np.testing.assert_allclose(result["shuffle_p_value"], [2.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(result["shuffle_log_evidence_median"], [10.0, 5.0])
    np.testing.assert_array_equal(result["shuffle_count"], [2, 2])
