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
from typing import Any

import numpy as np
import pandas as pd

from .evidence_reporting import _coerce_bool_series

_PATCHED_FLAG = "_simulation_recovery_session_event_count_patch_applied"
_CERTIFIED_EVENT_PATCHED_FLAG = "_simulation_recovery_certified_event_duplicate_model_patch_applied"
_SOURCE_SCORE_FILE_COLUMN = "source_recovery_score_file"
_EVENT_ID_COLUMN = "event_id"
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
    _EVENT_ID_COLUMN,
    "window_index",
    "benchmark_cell_split_index",
    "event_window_variant",
)


def apply_simulation_recovery_event_count_patch() -> None:
    """Install full-scope, CSV-tolerant event counting for recovery summaries."""

    import hipporeplayimm.evidence_reporting as reporting
    import hipporeplayimm.simulation_best_row_flags as best_row_flags
    import hipporeplayimm.simulation_recovery as recovery

    _extend_best_row_event_scope(best_row_flags)
    _patch_certified_event_recovery(reporting, recovery, best_row_flags)

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


def _patch_certified_event_recovery(reporting: Any, recovery: Any, best_row_flags: Any) -> None:
    """Keep certified-vs-exact event diagnostics tied to the winning row.

    Concatenated score tables can contain duplicate rows for the same model
    label within one simulated event.  When the best comparable row is an
    acceptable model, report that row directly instead of selecting the first
    row with the same model label.
    """

    if getattr(recovery, _CERTIFIED_EVENT_PATCHED_FLAG, False):
        return

    original_certified_events = recovery.certified_vs_exact_event_recovery

    @wraps(original_certified_events)
    def certified_vs_exact_event_recovery(event_scores: pd.DataFrame) -> pd.DataFrame:
        return _certified_vs_exact_event_recovery(
            event_scores,
            reporting,
            recovery,
            best_row_flags,
        )

    recovery.certified_vs_exact_event_recovery = certified_vs_exact_event_recovery
    setattr(recovery, _CERTIFIED_EVENT_PATCHED_FLAG, True)
    _sync_recovery_diagnostics(recovery)


def _certified_vs_exact_event_recovery(
    event_scores: pd.DataFrame,
    reporting: Any,
    recovery: Any,
    best_row_flags: Any,
) -> pd.DataFrame:
    if event_scores.empty:
        return pd.DataFrame()

    event_scores = reporting._coerce_log_evidence_column(
        reporting.ensure_evidence_support_columns(event_scores)
    )
    rows: list[dict[str, object]] = []
    for _, group in best_row_flags._iter_event_groups(event_scores):
        group = group.copy()
        first = group.iloc[0]
        expected_model = str(first.get("expected_model", ""))
        base: dict[str, object] = {
            "session": best_row_flags._event_scalar(group, "session"),
            "event_index": best_row_flags._event_scalar(group, "event_index"),
            "true_model": str(first.get("true_model", "")),
            "expected_model": expected_model,
            "n_time": best_row_flags._event_scalar(group, "n_time"),
            "n_spikes": best_row_flags._event_scalar(group, "n_spikes"),
        }
        for column in best_row_flags._event_group_columns(group):
            if column not in base:
                base[column] = best_row_flags._event_scalar(group, column)

        scored = group[reporting._status_success_series(group)].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_successful_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        finite = np.isfinite(scored["log_evidence"].to_numpy(float))
        scored = scored.loc[finite].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_finite_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        comparable_mask = recovery._comparable_mask(scored)
        comparable_rows = scored.loc[comparable_mask].copy()
        best_comparable_row: pd.Series | None = None
        best_comparable_model = ""
        best_comparable_log_evidence = np.nan
        if not comparable_rows.empty:
            best_comparable_row = best_row_flags._best_log_evidence_row(comparable_rows)
            best_comparable_model = str(best_comparable_row["model"])
            best_comparable_log_evidence = float(best_comparable_row["log_evidence"])

        acceptable_models = recovery._event_acceptable_recovery_models(group)
        acceptable_rows = scored[scored["model"].astype(str).isin(acceptable_models)].copy()
        expected_rows = scored[scored["model"].astype(str) == expected_model]
        if acceptable_rows.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "expected_model_not_scored",
                    "certified_reference_model": "",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": best_comparable_model,
                    "best_comparable_log_evidence": best_comparable_log_evidence,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        if best_comparable_row is not None and best_comparable_model in acceptable_models:
            expected = best_comparable_row
        elif not expected_rows.empty:
            expected = best_row_flags._best_log_evidence_row(expected_rows)
        else:
            expected = best_row_flags._best_log_evidence_row(acceptable_rows)

        certified_reference_model = str(expected["model"])
        expected_log_evidence = float(expected["log_evidence"])
        expected_support = str(expected.get("evidence_support", ""))
        raw_expected_comparable = expected.get(
            "evidence_comparable",
            recovery._evidence_is_comparable(expected_support),
        )
        expected_comparable = (
            recovery._evidence_is_comparable(expected_support)
            if pd.isna(raw_expected_comparable)
            else bool(reporting._coerce_bool_series(pd.Series([raw_expected_comparable])).iloc[0])
        )
        margin = expected_log_evidence - best_comparable_log_evidence

        if best_comparable_row is not None and best_comparable_model in acceptable_models:
            recovered = True
            reason = (
                "expected_comparable_best"
                if best_comparable_model == expected_model
                else "exact_surrogate_comparable_best"
            )
        elif expected_comparable:
            recovered = False
            reason = (
                "expected_comparable_not_best"
                if certified_reference_model == expected_model
                else "exact_surrogate_comparable_not_best"
            )
        elif expected_support != "truncated_full_grid":
            recovered = False
            reason = "expected_noncomparable_not_certified"
        elif not np.isfinite(best_comparable_log_evidence):
            recovered = False
            reason = "no_comparable_exact_reference"
        else:
            recovered = bool(margin > 0.0)
            reason = (
                "expected_lower_bound_beats_best_comparable"
                if recovered
                else "expected_lower_bound_not_above_best_comparable"
            )

        rows.append(
            {
                **base,
                "certified_vs_exact_recovered_expected_model": recovered,
                "certified_vs_exact_reason": reason,
                "certified_reference_model": certified_reference_model,
                "expected_model_log_evidence": expected_log_evidence,
                "expected_model_evidence_support": expected_support,
                "expected_model_evidence_comparable": expected_comparable,
                "best_comparable_model": best_comparable_model,
                "best_comparable_log_evidence": best_comparable_log_evidence,
                "expected_minus_best_comparable_log_evidence": float(margin),
            }
        )
    return pd.DataFrame(rows)


def _extend_best_row_event_scope(best_row_flags: Any) -> None:
    """Keep score-file provenance and explicit event IDs in event grouping helpers."""

    columns = list(getattr(best_row_flags, "_GROUP_COLUMNS", ()))
    insert_after = "session"
    for column in (_SOURCE_SCORE_FILE_COLUMN, _EVENT_ID_COLUMN):
        if column in columns:
            insert_after = column
            continue
        if insert_after in columns:
            index = columns.index(insert_after) + 1
        elif "session" in columns:
            index = columns.index("session") + 1
        else:
            index = 0
        columns.insert(index, column)
        insert_after = column
    best_row_flags._GROUP_COLUMNS = tuple(columns)


def _sync_recovery_diagnostics(recovery: Any) -> None:
    try:
        import hipporeplayimm.recovery_diagnostics as diagnostics
    except ImportError:
        return
    diagnostics.certified_vs_exact_event_recovery = recovery.certified_vs_exact_event_recovery
    diagnostics.certified_vs_exact_recovery_summary = recovery.certified_vs_exact_recovery_summary


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
