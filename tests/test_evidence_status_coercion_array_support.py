import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.evidence_reporting as reporting


def test_status_coercion_patch_flattens_array_like_support_labels() -> None:
    hipporeplayimm.apply_runtime_patches()
    rows = pd.DataFrame(
        [
            {
                "status": np.nan,
                "model": "state-space-imm",
                "log_evidence": 0.0,
                "diagnostic_state_space_imm_evidence_support": np.asarray(
                    [reporting.EXACT_EVIDENCE_SUPPORT, reporting.TRUNCATED_EVIDENCE_SUPPORT],
                    dtype=object,
                ),
            }
        ]
    )

    scored = reporting.ensure_evidence_support_columns(rows)

    assert scored.loc[0, "status"] == "success"
    assert scored.loc[0, "evidence_support"] == reporting.TRUNCATED_EVIDENCE_SUPPORT
    assert scored.loc[0, "evidence_comparison"] == reporting.EVIDENCE_COMPARISON_LOWER_BOUND
    assert not bool(scored.loc[0, "evidence_comparable"])
