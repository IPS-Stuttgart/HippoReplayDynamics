"""Count simulation-recovery events by their full available event identity.

Synthetic recovery event indices restart for each session and for independent
random-seed/run sweeps. Summary helpers that are applied to concatenated score
tables must therefore count distinct full event keys rather than only unique
integer event indices.

The same summaries may be rebuilt from CSV artifacts. Pandas can then expose
boolean flags as strings such as ``"True"``/``"False"``; normalize those columns
before delegating to the original summary helpers so string false values are not
treated as truthy or rejected by reductions.
"""

from __future__ import annotations

from functools import wraps

import pandas as pd

from .evidence_reporting import _coerce_bool_series

_PATCHED_FLAG = "_simulation_recovery_session_event_count_patch_applied"
_SOURCE_SCORE_FILE_COLUMN = "source_recovery_score_file"
_SUMMARY_BOOL_COLUMNS = (
    "recovered_expected_model",
    "exact_surrogate_recovered_expected_model",
    "evidence_comparable",
)
_EVENT_SCOPE_COLUMNS = (
    "session",
    _SOURCE_SCORE_FILE_COLUMN,
    "simulation_random_seed",
    "random_seed",
    "benchmark_random_seed",
    "simulation_event_index",
    "event_index",
    "event_id",
    "window_index",
    "benchmark_cell_split_index",
    "event_window_variant",
)


def apply_simulation_recovery_event_count_patch() -> None:
    """Install full-scope, CSV-tolerant event counting for recovery summaries."""

    import hipporeplayimm.simulation_best_row_flags as best_row_flags
    import hipporeplayimm.simulation_recovery as recovery

    _extend_best_row_event_scope(best_row_flags)

    if getattr(recovery, _PATCHED_FLAG, False):
        return

    original_recovery_summary = recovery.recovery_summary
    original_certified_summary = recovery.certified_vs_exact_recovery_summary

    @wraps(original_recovery_summary)
    def recovery_summary_with_session_event_counts(event_scores: pd.DataFrame) -> pd.DataFrame:
        normalized_scores = _normalize_summary_bool_columns(event_scores)
        summary = original_recovery_summary(normalized_scores)
        if summary.empty or "simulated_events" not in summary.columns:
            return summary
        best = recovery._event_best_rows(normalized_scores)
        return _replace_simulated_event_counts(summary, best)

    @wraps(original_certified_summary)
    def certified_vs_exact_recovery_summary_with_session_event_counts(
        event_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        normalized_scores = _normalize_summary_bool_columns(event_scores)
        summary = original_certified_summary(normalized_scores)
        if summary.empty or "simulated_events" not in summary.columns:
            return summary
        events = recovery.certified_vs_exact_event_recovery(normalized_scores)
        return _replace_simulated_event_counts(summary, events)

    recovery.recovery_summary = recovery_summary_with_session_event_counts
    recovery.certified_vs_exact_recovery_summary = (
        certified_vs_exact_recovery_summary_with_session_event_counts
    )
    setattr(recovery, _PATCHED_FLAG, True)


def _extend_best_row_event_scope(best_row_flags) -> None:
    """Keep score-file provenance in the primary event grouping helpers."""

    columns = tuple(getattr(best_row_flags, "_GROUP_COLUMNS", ()))
    if _SOURCE_SCORE_FILE_COLUMN in columns:
        return
    if "session" in columns:
        index = columns.index("session") + 1
    else:
        index = 0
    best_row_flags._GROUP_COLUMNS = (
        *columns[:index],
        _SOURCE_SCORE_FILE_COLUMN,
        *columns[index:],
    )


def _normalize_summary_bool_columns(event_scores: pd.DataFrame) -> pd.DataFrame:
    """Return score rows with CSV-round-tripped boolean columns restored."""

    if event_scores.empty:
        return event_scores
    out = event_scores.copy()
    for column in _SUMMARY_BOOL_COLUMNS:
        if column in out.columns:
            out[column] = _coerce_bool_series(out[column])
    return out


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
    event_columns = [column for column in _EVENT_SCOPE_COLUMNS if column in events.columns]
    if event_columns:
        return int(events[event_columns].drop_duplicates().shape[0])
    return int(len(events))


__all__ = ["apply_simulation_recovery_event_count_patch"]
