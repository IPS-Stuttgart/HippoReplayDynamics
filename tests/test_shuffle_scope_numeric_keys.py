from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_match_integral_numeric_scope_keys_after_csv_roundtrip() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [7.0],
            "model": ["sorted-spike-state-space-imm"],
            "window_index": [1.0],
            "log_evidence": [5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": np.array([7, 7], dtype=int),
            "model": ["sorted-spike-state-space-imm", "sorted-spike-state-space-imm"],
            "window_index": np.array([1, 1], dtype=int),
            "log_evidence": [0.0, 10.0],
        }
    )

    scored = add_shuffle_p_values(real_scores, control_scores)

    assert scored.loc[0, "shuffle_count"] == 2
    assert scored.loc[0, "shuffle_log_evidence_median"] == pytest.approx(5.0)
    assert scored.loc[0, "shuffle_p_value"] == pytest.approx(2.0 / 3.0)
