from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_match_mapping_scope_regardless_of_order() -> None:
    hipporeplayimm.apply_runtime_patches()
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [10.0],
            "benchmark_cell_split_strata": [{"rat": "R1", "fold": 2}],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["sorted-spike-state-space-first-order-imm"] * 2,
            "log_evidence": [11.0, 8.0],
            "benchmark_cell_split_strata": [
                {"fold": 2, "rat": "R1"},
                {"fold": 2, "rat": "R1"},
            ],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert np.isclose(out.loc[0, "shuffle_p_value"], 2.0 / 3.0)
    assert out.loc[0, "shuffle_count"] == 2
    assert out.loc[0, "shuffle_log_evidence_median"] == 9.5
