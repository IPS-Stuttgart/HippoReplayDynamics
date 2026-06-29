"""Scope result-quality audit event groups by independent score-window metadata."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCHED_FLAG = "_result_quality_audit_scope_patch_applied"
_EVENT_COUNT_PATCHED_FLAG = "_result_quality_audit_event_count_scope_patch_applied"
_EVENT_GROUP_BASE_COLUMNS = ("session", "event_index")
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


def _column_has_complete_metadata(scores: Any, column: str) -> bool:
    """Return True only when an optional grouping key is populated for every row."""

    try:
        values = scores[column]
    except (KeyError, TypeError):
        return False
    if bool(getattr(values, "empty", False)):
        return False
    return bool(values.notna().all())


def _scoped_event_group_columns(scores: Any) -> list[str]:
    """Return columns identifying one independent model-comparison unit."""

    frame_columns = getattr(scores, "columns", ())
    columns = [column for column in _EVENT_GROUP_BASE_COLUMNS if column in frame_columns]
    for optional in _EVENT_GROUP_SCOPE_COLUMNS:
        if optional in frame_columns and optional not in columns and _column_has_complete_metadata(scores, optional):
            columns.append(optional)
    return columns


def apply_result_quality_audit_scope_patch() -> None:
    """Install scoped grouping for result-quality audit summaries."""

    from . import result_quality_audit as audit_module

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


__all__ = ["apply_result_quality_audit_scope_patch"]
