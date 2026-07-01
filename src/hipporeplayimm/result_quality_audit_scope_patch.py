"""Scope result-quality audit event groups by independent score-window metadata."""

from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

_PATCHED_FLAG = "_result_quality_audit_scope_patch_applied"
_EVENT_COUNT_PATCHED_FLAG = "_result_quality_audit_event_count_scope_patch_applied"
_HELDOUT_INFLUENCE_PATCHED_FLAG = "_result_quality_audit_heldout_influence_patch_applied"
_EVENT_GROUP_SESSION_COLUMNS = ("session",)
_EVENT_GROUP_EVENT_COLUMNS = ("event_index", "event_id")
_EVENT_GROUP_SCOPE_COLUMNS = (
    "window_role",
    "window_index",
    "event_window_variant",
    "window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "random_seed",
    "null_random_seed",
    "cell_split_index",
    "cell_split_seed",
    "cell_split_count",
    "cell_split_shard_index",
    "cell_split_shard_count",
    "split_shard_index",
    "split_shard_count",
    "test_cell_fraction",
    "train_cell_count",
    "test_cell_count",
    "train_cell_ids",
    "test_cell_ids",
    "benchmark_random_seed",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
    "benchmark_event_subset_base_seed",
    "benchmark_test_cell_fraction",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
)
_INFLUENCE_COLUMNS = (
    "model",
    "full_mean",
    "leave_one_mean",
    "left_out_group_col",
    "left_out_group",
    "influence_delta",
)
_INFLUENCE_VALUE_COLUMNS = (
    "relative_log_evidence",
    "log_evidence",
    "heldout_log_likelihood",
)


def _column_has_observed_metadata(scores: Any, column: str) -> bool:
    """Return True when an optional grouping key is populated for at least one row."""

    try:
        values = scores[column]
    except (KeyError, TypeError):
        return False
    if bool(getattr(values, "empty", False)):
        return False
    return bool(values.notna().any())


def _scoped_event_group_columns(scores: Any) -> list[str]:
    """Return columns identifying one independent model-comparison unit."""

    frame_columns = getattr(scores, "columns", ())
    columns = [column for column in _EVENT_GROUP_SESSION_COLUMNS if column in frame_columns]
    for event_column in _EVENT_GROUP_EVENT_COLUMNS:
        if event_column in frame_columns:
            columns.append(event_column)
            break
    for optional in _EVENT_GROUP_SCOPE_COLUMNS:
        if optional in frame_columns and optional not in columns and _column_has_observed_metadata(scores, optional):
            columns.append(optional)
    return columns


def _first_influence_value_column(scores: Any) -> str | None:
    frame_columns = getattr(scores, "columns", ())
    for column in _INFLUENCE_VALUE_COLUMNS:
        if column in frame_columns:
            return column
    return None


def _heldout_aware_influence_summary(audit_module: Any, scores: pd.DataFrame) -> pd.DataFrame:
    """Return influence rows for regular and held-out evidence schemas."""

    value_col = _first_influence_value_column(scores)
    if value_col is None or "session" not in scores.columns:
        return pd.DataFrame(columns=list(_INFLUENCE_COLUMNS))

    frames = [audit_module.leave_one_group_influence(scores, group_col="session", value_col=value_col)]
    rat_scores = scores.copy()
    rat_scores["rat"] = rat_scores["session"].map(audit_module.rat_from_session)
    frames.append(audit_module.leave_one_group_influence(rat_scores, group_col="rat", value_col=value_col))

    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=list(_INFLUENCE_COLUMNS))
    out = pd.concat(nonempty, ignore_index=True)
    return out if not out.empty else pd.DataFrame(columns=list(_INFLUENCE_COLUMNS))


def apply_result_quality_audit_scope_patch() -> None:
    """Install scoped grouping for result-quality audit summaries."""

    from . import advanced_result_evidence_margin_duplicates
    from . import result_quality_audit as audit_module

    advanced_result_evidence_margin_duplicates.apply_evidence_margin_distinct_model_patch()

    current_group_columns = audit_module.event_group_columns
    if not getattr(current_group_columns, _PATCHED_FLAG, False):

        @wraps(current_group_columns)
        def event_group_columns(scores):
            return _scoped_event_group_columns(scores)

        setattr(event_group_columns, _PATCHED_FLAG, True)
        audit_module.event_group_columns = event_group_columns

    current_event_count = audit_module._event_count
    if not getattr(current_event_count, _EVENT_COUNT_PATCHED_FLAG, False):

        @wraps(current_event_count)
        def _event_count(scores):
            group_columns = _scoped_event_group_columns(scores)
            if not group_columns:
                return "unknown"
            return int(scores[group_columns].drop_duplicates().shape[0])

        setattr(_event_count, _EVENT_COUNT_PATCHED_FLAG, True)
        audit_module._event_count = _event_count

    current_influence_summary = audit_module._influence_summary
    if not getattr(current_influence_summary, _HELDOUT_INFLUENCE_PATCHED_FLAG, False):

        @wraps(current_influence_summary)
        def _influence_summary(scores):
            return _heldout_aware_influence_summary(audit_module, scores)

        setattr(_influence_summary, _HELDOUT_INFLUENCE_PATCHED_FLAG, True)
        audit_module._influence_summary = _influence_summary


__all__ = ["apply_result_quality_audit_scope_patch"]
