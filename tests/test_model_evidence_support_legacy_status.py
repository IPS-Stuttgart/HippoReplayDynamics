from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
import audit_model_evidence_support as paired_audit  # noqa: E402
import model_evidence_support_audit as table_audit  # noqa: E402


def _legacy_status_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "state-space-diffusion",
                "model_family": "trajectory",
                "status": np.nan,
                "log_evidence": -10.0,
                "evidence_support": paired_audit.EXACT_EVIDENCE_SUPPORT,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "state-space-momentum",
                "model_family": "trajectory",
                "status": "",
                "log_evidence": -9.0,
                "evidence_support": paired_audit.TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "state-space-imm",
                "model_family": "trajectory",
                "status": "failed",
                "log_evidence": -8.0,
                "evidence_support": paired_audit.EXACT_EVIDENCE_SUPPORT,
            },
        ]
    )


def test_pairwise_support_audit_keeps_blank_status_rows_as_legacy_success():
    scores = _legacy_status_scores()

    summary = paired_audit.evidence_support_summary(scores)
    paired = paired_audit.paired_delta_summary(scores)

    assert set(summary["model"]) == {"state-space-diffusion", "state-space-momentum"}
    assert "state-space-imm" not in set(summary["model"])
    assert int(summary["events"].sum()) == 2
    assert paired.loc[0, "comparison"] == "state-space-momentum_minus_state-space-diffusion"
    assert paired.loc[0, "events"] == 1


def test_table_support_audit_keeps_blank_status_rows_as_legacy_success():
    scores = _legacy_status_scores()

    event_audit = table_audit.event_support_audit(scores)
    pairwise = table_audit.pairwise_support_audit(scores)

    assert event_audit.loc[0, "models"] == "state-space-diffusion,state-space-momentum"
    assert event_audit.loc[0, "exact_rows"] == 1
    assert event_audit.loc[0, "truncated_rows"] == 1
    assert event_audit.loc[0, "comparable_rows"] == 1
    assert pairwise.loc[0, "events"] == 1
    assert pairwise.loc[0, "mixes_exact_and_truncated"]
