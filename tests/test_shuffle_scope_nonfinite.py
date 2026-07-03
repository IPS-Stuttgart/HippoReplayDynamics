from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_add_shuffle_p_values_handles_nonfinite_numeric_scope_values() -> None:
    hipporeplayimm.apply_runtime_patches()
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "window_start_s": [np.inf],
            "log_evidence": [10.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["sorted-spike-state-space-first-order-imm", "sorted-spike-state-space-first-order-imm"],
            "window_start_s": [np.inf, np.inf],
            "log_evidence": [8.0, 12.0],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert np.isclose(out.loc[0, "shuffle_p_value"], 2.0 / 3.0)
    assert out.loc[0, "shuffle_count"] == 2
    assert out.loc[0, "shuffle_log_evidence_median"] == 10.0
