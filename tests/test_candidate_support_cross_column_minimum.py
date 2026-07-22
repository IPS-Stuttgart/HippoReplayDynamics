from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvements import add_candidate_support_quality_columns


def test_candidate_support_quality_uses_worst_mass_across_diagnostic_columns() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "state-space-imm",
                "evidence_support": "truncated_full_grid",
                "min_candidate_log_mass": -0.005,
                "diagnostic_state_space_imm_min_candidate_log_mass": np.asarray(
                    [-0.02, -1.0]
                ),
            }
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_min_log_mass"] == -1.0
    assert labelled.loc[0, "candidate_support_quality"] == "conservative_poor"
