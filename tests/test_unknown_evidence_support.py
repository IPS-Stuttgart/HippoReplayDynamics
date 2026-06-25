import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_UNKNOWN,
    EXACT_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)


def test_unknown_diagnostic_evidence_support_is_not_comparable():
    scored = ensure_evidence_support_columns(
        pd.DataFrame(
            [
                {
                    "status": "success",
                    "model": "state-space-imm",
                    "log_evidence": 0.0,
                    "diagnostic_state_space_imm_evidence_support": "future_support_label",
                }
            ]
        )
    )

    assert scored.loc[0, "evidence_support"] == EVIDENCE_COMPARISON_UNKNOWN
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_UNKNOWN
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_missing_diagnostic_evidence_support_keeps_legacy_exact_default():
    scored = ensure_evidence_support_columns(
        pd.DataFrame(
            [
                {
                    "status": "success",
                    "model": "stationary",
                    "log_evidence": 0.0,
                }
            ]
        )
    )

    assert scored.loc[0, "evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert bool(scored.loc[0, "evidence_comparable"])
