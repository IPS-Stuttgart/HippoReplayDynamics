import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)


def test_array_like_evidence_support_diagnostic_does_not_crash_and_keeps_nonexact_priority():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "state-space-imm",
                "log_evidence": 0.0,
                "diagnostic_state_space_imm_evidence_support": np.asarray(
                    [EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT],
                    dtype=object,
                ),
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_array_like_missing_evidence_support_diagnostic_is_ignored():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "state-space-diffusion",
                "log_evidence": 0.0,
                "diagnostic_state_space_imm_evidence_support": np.asarray(
                    [np.nan, None],
                    dtype=object,
                ),
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert bool(scored.loc[0, "evidence_comparable"])
