from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from audit_model_evidence_support import (  # noqa: E402
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
)
import model_evidence_support_audit as support_audit_tables  # noqa: E402


def _scores_with_nonfinite_success_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": 1.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "momentum",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": float("inf"),
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "imm",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": "-inf",
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
        ]
    )


def test_support_audit_summary_ignores_nonfinite_success_rows() -> None:
    summary = support_audit_tables.evidence_support_summary(_scores_with_nonfinite_success_row())

    assert summary["model"].tolist() == ["diffusion"]
    assert summary.loc[0, "rows"] == 1
    assert summary.loc[0, "events"] == 1
    assert summary.loc[0, "mean_log_evidence"] == 1.0


def test_support_audit_pairs_ignore_nonfinite_success_rows() -> None:
    scores = _scores_with_nonfinite_success_row()

    event_audit = support_audit_tables.event_support_audit(scores)
    pairwise = support_audit_tables.pairwise_support_audit(scores)

    assert event_audit["models"].tolist() == ["diffusion"]
    assert event_audit.loc[0, "truncated_rows"] == 0
    assert not bool(event_audit.loc[0, "has_mixed_exact_truncated"])
    assert pairwise.empty
