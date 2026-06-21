"""Hash-stable grouping for in-memory cell-split score metadata.

Benchmark score CSVs store explicit train/test cell IDs as strings, but unit tests
and downstream scripts may pass score tables as pandas DataFrames whose
``train_cell_ids`` / ``test_cell_ids`` entries are Python lists or NumPy arrays.
Pandas cannot use those objects directly in ``drop_duplicates`` or ``groupby``.
This compatibility patch adds private hashable scope-key columns for grouping
while preserving the original cell-ID columns for downstream held-out decoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_GROUP_COLUMN_PREFIX = "__cell_split_scope_key__"


def apply_cell_split_hashable_grouping_patch() -> None:
    """Make benchmark cell-split grouping robust to list/array metadata values."""

    from . import benchmark_cell_split_metadata as metadata

    if getattr(metadata, "_cell_split_hashable_grouping_patch_applied", False):
        return

    original_scores_frame = metadata._scores_frame_for_cell_split_metadata
    original_compare_with_metadata = metadata._compare_scores_with_cell_split_metadata

    def scores_frame_for_cell_split_metadata(scores: str | Path | pd.DataFrame) -> pd.DataFrame:
        frame = original_scores_frame(scores)
        return _with_hashable_scope_keys(frame, metadata)

    def score_table_needs_cell_split_scoped_decode(scores_frame: pd.DataFrame) -> bool:
        frame = _with_hashable_scope_keys(scores_frame.copy(), metadata)
        if not metadata._HELDOUT_BENCHMARK_COLUMNS.issubset(frame.columns):
            return False
        if "session" not in frame.columns:
            return False
        group_columns = cell_split_decode_group_columns(frame)
        group_count = int(frame[group_columns].drop_duplicates().shape[0])
        session_count = int(frame[["session"]].drop_duplicates().shape[0])
        if group_count > session_count:
            return True
        return any(
            len(metadata._metadata_group_values(frame[column])) > 1
            for column in metadata._CELL_SPLIT_SCOPE_COLUMNS
            if column in frame.columns
        )

    def cell_split_decode_group_columns(scores_frame: pd.DataFrame) -> list[str]:
        frame = _with_hashable_scope_keys(scores_frame, metadata)
        columns = ["session"]
        for column in metadata._CELL_SPLIT_SCOPE_COLUMNS:
            if column not in frame.columns:
                continue
            has_multiple_values = len(metadata._metadata_group_values(frame[column])) > 1
            if column == "benchmark_cell_split_index" or has_multiple_values:
                columns.append(_scope_key_column(column))
        return columns

    def compare_scores_with_cell_split_metadata(
        compare_scores: Any,
        bench: Any,
        gt: Any,
        root: str | Path,
        scores: str | Path | pd.DataFrame,
        scores_frame: pd.DataFrame,
        default_strategy: str,
        default_strata: int,
        kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        return original_compare_with_metadata(
            compare_scores,
            bench,
            gt,
            root,
            _drop_cell_split_scope_key_columns(scores),
            _drop_cell_split_scope_key_columns(scores_frame),
            default_strategy,
            default_strata,
            kwargs,
        )

    metadata._scores_frame_for_cell_split_metadata = scores_frame_for_cell_split_metadata
    metadata._score_table_needs_cell_split_scoped_decode = score_table_needs_cell_split_scoped_decode
    metadata._cell_split_decode_group_columns = cell_split_decode_group_columns
    metadata._compare_scores_with_cell_split_metadata = compare_scores_with_cell_split_metadata
    metadata._cell_split_hashable_grouping_patch_applied = True


def _with_hashable_scope_keys(frame: pd.DataFrame, metadata: Any) -> pd.DataFrame:
    out = frame.copy()
    for column in metadata._CELL_SPLIT_SCOPE_COLUMNS:
        if column in out.columns:
            out[_scope_key_column(column)] = [
                _metadata_group_key(value) for value in out[column]
            ]
    return out


def _drop_cell_split_scope_key_columns(
    value: str | Path | pd.DataFrame,
) -> str | Path | pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        return value
    helper_columns = [
        column for column in value.columns if column.startswith(_GROUP_COLUMN_PREFIX)
    ]
    if not helper_columns:
        return value.copy()
    return value.drop(columns=helper_columns).copy()


def _scope_key_column(column: str) -> str:
    return f"{_GROUP_COLUMN_PREFIX}{column}"


def _metadata_group_key(value: object) -> str:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return "<missing>"
    if isinstance(value, np.ndarray):
        return repr(np.asarray(value).tolist())
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    return str(value).strip()
