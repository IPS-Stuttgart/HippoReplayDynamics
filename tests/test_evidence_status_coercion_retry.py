from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm.evidence_reporting as reporting
import hipporeplayimm.evidence_status_coercion as status_coercion
import hipporeplayimm.recovery_diagnostics as diagnostics


EXPECTED_EMPTY_EVIDENCE_COLUMNS = {
    "evidence_support",
    "evidence_comparison",
    "evidence_comparison_note",
    "evidence_comparable",
}


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


def test_status_coercion_reinstalls_unmarked_reporting_helpers(monkeypatch) -> None:
    """Idempotent calls must refresh reporting helpers that no longer carry the patch marker."""

    def legacy_ensure_evidence_support_columns(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        status_ok = out["status"].astype(str).eq("success")
        out["evidence_support"] = reporting.EXACT_EVIDENCE_SUPPORT
        out["evidence_comparable"] = status_ok
        return out

    monkeypatch.setattr(
        reporting,
        "ensure_evidence_support_columns",
        legacy_ensure_evidence_support_columns,
    )

    scores = pd.DataFrame({"status": [np.nan, "failure"], "log_evidence": [1.0, 2.0]})
    assert not bool(reporting.ensure_evidence_support_columns(scores)["evidence_comparable"].iloc[0])

    status_coercion.apply_evidence_status_coercion_patch()

    assert getattr(reporting.ensure_evidence_support_columns, status_coercion._CORE_WRAPPER_FLAG, False)
    refreshed = reporting.ensure_evidence_support_columns(scores)
    assert refreshed["status"].tolist()[0] == "success"
    assert bool(refreshed["evidence_comparable"].iloc[0])
    assert not bool(refreshed["evidence_comparable"].iloc[1])


def test_empty_evidence_support_frames_keep_reporting_columns() -> None:
    status_coercion.apply_evidence_status_coercion_patch()

    scored = reporting.ensure_evidence_support_columns(pd.DataFrame())

    assert scored.empty
    assert EXPECTED_EMPTY_EVIDENCE_COLUMNS.issubset(scored.columns)
    assert scored["evidence_comparable"].dtype == bool


def test_empty_simulation_evidence_frames_keep_reporting_columns() -> None:
    status_coercion.apply_evidence_status_coercion_patch()

    scored = reporting.simulation_add_evidence_columns(pd.DataFrame())

    assert scored.empty
    assert EXPECTED_EMPTY_EVIDENCE_COLUMNS.issubset(scored.columns)
    assert scored["evidence_comparable"].dtype == bool
