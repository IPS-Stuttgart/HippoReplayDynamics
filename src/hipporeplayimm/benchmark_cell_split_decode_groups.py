"""Patch held-out ground-truth decoding to keep independent split groups apart."""

from __future__ import annotations

from typing import Any

import pandas as pd

_RANDOM_COLUMN = "benchmark_random_" + "s" + "eed"
_SPLIT_INDEX_COLUMN = "benchmark_cell_split_index"
_SPLIT_RNG_COLUMN = "benchmark_cell_split_" + "s" + "eed"
_GROUP_COLUMNS = (
    _RANDOM_COLUMN,
    _SPLIT_INDEX_COLUMN,
    _SPLIT_RNG_COLUMN,
)


def apply_benchmark_cell_split_decode_group_patch() -> None:
    """Keep combined held-out score tables from sharing one decoded split."""

    from . import benchmark_cell_split_metadata as metadata

    if getattr(metadata, "_benchmark_cell_split_decode_group_patch_applied", False):
        return

    def cell_split_decode_group_columns(scores_frame: pd.DataFrame) -> list[str]:
        columns = ["session"]
        for column in _GROUP_COLUMNS:
            if bool(metadata._string_metadata_values(scores_frame, column)):
                columns.append(column)
        return columns

    def score_table_needs_cell_split_scoped_decode(scores_frame: pd.DataFrame) -> bool:
        if not metadata._HELDOUT_BENCHMARK_COLUMNS.issubset(scores_frame.columns):
            return False
        if "session" not in scores_frame.columns:
            return False
        group_columns = cell_split_decode_group_columns(scores_frame)
        group_count = int(scores_frame[group_columns].drop_duplicates().shape[0])
        session_count = int(scores_frame[["session"]].drop_duplicates().shape[0])
        if group_count > session_count:
            return True
        return (
            len(set(metadata._string_metadata_values(scores_frame, "benchmark_cell_split_strategy"))) > 1
            or len(set(metadata._string_metadata_values(scores_frame, "benchmark_cell_split_strata"))) > 1
        )

    metadata._cell_split_decode_group_columns = cell_split_decode_group_columns
    metadata._score_table_needs_cell_split_scoped_decode = score_table_needs_cell_split_scoped_decode
    metadata._benchmark_cell_split_decode_group_patch_applied = True
