from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm.evidence_reporting as reporting
import hipporeplayimm.evidence_status_coercion as status_coercion
import hipporeplayimm.recovery_diagnostics as diagnostics


def test_status_coercion_retries_recovery_diagnostics_after_reporting_patch(monkeypatch) -> None:
    """A later idempotent patch call must still refresh recovery diagnostics."""

    def legacy_successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
        status_ok = (
            group["status"].astype(str).eq("success")
            if "status" in group
            else pd.Series(True, index=group.index)
        )
        finite = pd.Series(
            np.isfinite(pd.to_numeric(group["log_evidence"], errors="coerce")),
            index=group.index,
        )
        return group[status_ok & finite].copy()

    monkeypatch.setattr(reporting, status_coercion._PATCHED_FLAG, True, raising=False)
    monkeypatch.delattr(
        diagnostics,
        status_coercion._RECOVERY_DIAGNOSTICS_PATCHED_FLAG,
        raising=False,
    )
    monkeypatch.setattr(
        diagnostics,
        "_successful_finite_scores",
        legacy_successful_finite_scores,
    )

    scores = pd.DataFrame(
        {
            "status": [np.nan, "failure"],
            "log_evidence": [1.0, 2.0],
        }
    )
    assert diagnostics._successful_finite_scores(scores).empty

    status_coercion.apply_evidence_status_coercion_patch()

    filtered = diagnostics._successful_finite_scores(scores)
    assert len(filtered) == 1
    assert pd.isna(filtered["status"].iloc[0])
