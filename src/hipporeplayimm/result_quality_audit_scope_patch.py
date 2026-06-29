"""Scope result-quality audit event groups by independent score-window metadata."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCHED_FLAG = "_result_quality_audit_scope_patch_applied"
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
    "benchmark_random_seed",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
    "benchmark_event_subset_base_seed",
    "benchmark_test_cell_fraction",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
)


def _scoped_event_group_columns(scores: Any) -> list[str]:
    """Return columns identifying one independent model-comparison unit."""

    frame_columns = getattr(scores, "columns", ())
    columns = [column for column in _EVENT_GROUP_BASE_COLUMNS if column in frame_columns]
    for optional in _EVENT_GROUP_SCOPE_COLUMNS:
        if optional in frame_columns and optional not in columns:
            columns.append(optional)
    return columns


def apply_result_quality_audit_scope_patch() -> None:
    """Install scoped grouping for result-quality audit summaries."""

    from . import result_quality_audit as audit_module

    current = audit_module.event_group_columns
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def event_group_columns(scores):
        return _scoped_event_group_columns(scores)

    setattr(event_group_columns, _PATCHED_FLAG, True)
    audit_module.event_group_columns = event_group_columns


__all__ = ["apply_result_quality_audit_scope_patch"]
