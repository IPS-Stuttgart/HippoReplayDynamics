from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.evidence_reporting as reporting
import hipporeplayimm.evidence_status_coercion as status_coercion
import hipporeplayimm.recovery_diagnostics as diagnostics
import hipporeplayimm.simulation_recovery as recovery


def test_runtime_patches_repatch_stale_recovery_diagnostics_helper(monkeypatch) -> None:
    def legacy_successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
        status_ok = group["status"].astype(str).eq("success")
        finite = pd.Series(
            np.isfinite(pd.to_numeric(group["log_evidence"], errors="coerce")),
            index=group.index,
        )
        return group[status_ok & finite].copy()

    monkeypatch.setattr(reporting, status_coercion._PATCHED_FLAG, True, raising=False)
    monkeypatch.setattr(diagnostics, status_coercion._RECOVERY_DIAGNOSTICS_PATCHED_FLAG, True, raising=False)
    monkeypatch.setattr(diagnostics, "_successful_finite_scores", legacy_successful_finite_scores)

    scores = pd.DataFrame({"status": [np.nan, "error"], "log_evidence": [1.0, 2.0]})
    assert diagnostics._successful_finite_scores(scores).empty

    hipporeplayimm.apply_runtime_patches()

    filtered = diagnostics._successful_finite_scores(scores)
    assert len(filtered) == 1
    assert pd.isna(filtered["status"].iloc[0])


def test_runtime_patches_repatch_stale_certified_recovery_helpers(monkeypatch) -> None:
    def legacy_certified_vs_exact_event_recovery(event_scores: pd.DataFrame) -> pd.DataFrame:
        status_ok = event_scores["status"].astype(str).eq("success")
        if not status_ok.any():
            return pd.DataFrame()
        return pd.DataFrame({"legacy_successful_rows": [int(status_ok.sum())]})

    def legacy_certified_vs_exact_recovery_summary(event_scores: pd.DataFrame) -> pd.DataFrame:
        events = legacy_certified_vs_exact_event_recovery(event_scores)
        return pd.DataFrame({"legacy_events": [int(len(events))]})

    monkeypatch.setattr(recovery, status_coercion._CERTIFIED_RECOVERY_PATCHED_FLAG, True, raising=False)
    monkeypatch.setattr(recovery, "certified_vs_exact_event_recovery", legacy_certified_vs_exact_event_recovery)
    monkeypatch.setattr(recovery, "certified_vs_exact_recovery_summary", legacy_certified_vs_exact_recovery_summary)

    scores = pd.DataFrame({"status": [pd.NA, "error"], "log_evidence": [1.0, 2.0]})
    assert recovery.certified_vs_exact_event_recovery(scores).empty

    hipporeplayimm.apply_runtime_patches()

    events = recovery.certified_vs_exact_event_recovery(scores)
    summary = recovery.certified_vs_exact_recovery_summary(scores)
    assert getattr(recovery.certified_vs_exact_event_recovery, status_coercion._CERTIFIED_EVENT_WRAPPER_FLAG, False)
    assert getattr(recovery.certified_vs_exact_recovery_summary, status_coercion._CERTIFIED_SUMMARY_WRAPPER_FLAG, False)
    assert events.loc[0, "legacy_successful_rows"] == 1
    assert summary.loc[0, "legacy_events"] == 1
