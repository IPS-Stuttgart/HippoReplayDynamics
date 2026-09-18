from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from audit_model_evidence_support import (  # noqa: E402
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
)
import model_evidence_support_audit as support_audit_tables  # noqa: E402


def _scores_with_infinite_success_rows() -> pd.DataFrame:
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
                "log_evidence": "-inf",
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "imm",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": float("inf"),
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
        ]
    )


def test_support_audit_keeps_negative_infinite_and_rejects_positive_infinite_rows() -> None:
    summary = support_audit_tables.evidence_support_summary(
        _scores_with_infinite_success_rows()
    ).set_index("model")

    assert set(summary.index) == {"diffusion", "momentum"}
    assert summary.loc["diffusion", "rows"] == 1
    assert summary.loc["momentum", "rows"] == 1
    assert summary.loc["momentum", "events"] == 1
    assert summary.loc["diffusion", "mean_log_evidence"] == 1.0
    assert np.isneginf(summary.loc["momentum", "mean_log_evidence"])


def test_support_audit_negative_infinity_still_exposes_mixed_support() -> None:
    scores = _scores_with_infinite_success_rows()

    event_audit = support_audit_tables.event_support_audit(scores)
    pairwise = support_audit_tables.pairwise_support_audit(scores)

    assert len(event_audit) == 1
    assert set(event_audit.loc[0, "models"].split(",")) == {"diffusion", "momentum"}
    assert event_audit.loc[0, "exact_rows"] == 1
    assert event_audit.loc[0, "truncated_rows"] == 1
    assert bool(event_audit.loc[0, "has_mixed_exact_truncated"])
    assert len(pairwise) == 1
    assert bool(pairwise.loc[0, "mixes_exact_and_truncated"])
    assert pairwise.loc[0, "support_a"] == EXACT_EVIDENCE_SUPPORT
    assert pairwise.loc[0, "support_b"] == TRUNCATED_EVIDENCE_SUPPORT
