import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_LOWER_BOUND,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)


def test_missing_evidence_support_strings_fall_back_to_diagnostics():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "sorted-spike-state-space-imm",
                "log_evidence": 0.0,
                "evidence_support": "nan",
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "model": "sorted-spike-state-space-displacement-imm",
                "log_evidence": 0.0,
                "evidence_support": "None",
                "diagnostic_state_space_displacement_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored["evidence_support"].tolist() == [
        TRUNCATED_EVIDENCE_SUPPORT,
        TRUNCATED_EVIDENCE_SUPPORT,
    ]
    assert scored["evidence_comparison"].tolist() == [
        EVIDENCE_COMPARISON_LOWER_BOUND,
        EVIDENCE_COMPARISON_LOWER_BOUND,
    ]
    assert not scored["evidence_comparable"].any()
