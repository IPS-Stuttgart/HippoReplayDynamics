"""Count simulation-recovery events by their full session/event identity.

Synthetic recovery event indices restart for each session.  Summary helpers that are
applied to concatenated multi-session score tables must therefore count distinct
``(session, event_index)`` pairs instead of only unique integer event indices.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

_PATCHED_FLAG = "_simulation_recovery_session_event_count_patch_applied"


def apply_simulation_recovery_event_count_patch() -> None:
    """Install session-aware event counting for simulation-recovery summaries."""

    import hipporeplayimm.simulation_recovery as recovery

    if getattr(recovery, _PATCHED_FLAG, False):
        return

    original_recovery_summary = recovery.recovery_summary
    original_certified_summary = recovery.certified_vs_exact_recovery_summary

    @wraps(original_recovery_summary)
    def recovery_summary_with_session_event_counts(event_scores: pd.DataFrame) -> pd.DataFrame:
        summary = original_recovery_summary(event_scores)
        if summary.empty or "simulated_events" not in summary.columns:
            return summary
        best = recovery._event_best_rows(event_scores)
        return _replace_simulated_event_counts(summary, best)

    @wraps(original_certified_summary)
    def certified_vs_exact_recovery_summary_with_session_event_counts(
        event_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        summary = original_certified_summary(event_scores)
        if summary.empty or "simulated_events" not in summary.columns:
            return summary
        events = recovery.certified_vs_exact_event_recovery(event_scores)
        return _replace_simulated_event_counts(summary, events)

    recovery.recovery_summary = recovery_summary_with_session_event_counts
    recovery.certified_vs_exact_recovery_summary = (
        certified_vs_exact_recovery_summary_with_session_event_counts
    )
    setattr(recovery, _PATCHED_FLAG, True)


def _replace_simulated_event_counts(
    summary: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    out = summary.copy()
    if events.empty or "true_model" not in events.columns:
        return out
    for index, row in out.iterrows():
        label = str(row.get("true_model", ""))
        scoped = events if label == "overall" else events[events["true_model"].astype(str) == label]
        out.at[index, "simulated_events"] = _distinct_event_count(scoped)
    return out


def _distinct_event_count(events: pd.DataFrame) -> int:
    """Return the number of unique simulated events in a score/event table."""

    if events.empty:
        return 0
    if {"session", "event_index"}.issubset(events.columns):
        return int(events[["session", "event_index"]].drop_duplicates().shape[0])
    if "event_index" in events.columns:
        return int(events["event_index"].nunique())
    return int(len(events))


__all__ = ["apply_simulation_recovery_event_count_patch"]
