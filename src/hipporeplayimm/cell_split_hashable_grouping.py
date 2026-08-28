"""Hash-stable grouping for in-memory cell-split and shuffle scope metadata.

Benchmark score CSVs store explicit train/test cell IDs as strings, but unit tests
and downstream scripts may pass score tables as pandas DataFrames whose
``train_cell_ids`` / ``test_cell_ids`` entries are Python lists or NumPy arrays.
Pandas cannot use those objects directly in ``drop_duplicates`` or ``groupby``.
This compatibility patch adds private hashable scope-key columns for grouping
while preserving the original cell-ID columns for downstream held-out decoding.
It also keeps integral shuffle-scope identifiers exact instead of routing them
through binary64 and rejects ambiguous shuffle scopes when a discriminator is
missing from one side of a real/control comparison.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_GROUP_COLUMN_PREFIX = "__cell_split_scope_key__"
_HASHABLE_GROUPING_PATCH_FLAG = "_cell_split_hashable_grouping_patch_applied"
_SHUFFLE_SCOPE_INTEGER_PATCH_FLAG = "_shuffle_scope_exact_integer_patch_applied"
_SHUFFLE_SCOPE_ALIGNMENT_PATCH_FLAG = "_shuffle_scope_alignment_patch_applied"


def _function_is_from_this_patch(function: object) -> bool:
    """Return whether ``function`` is one of this module's live wrappers."""

    return getattr(function, "__module__", None) == __name__


def _metadata_grouping_patch_is_current(metadata: Any) -> bool:
    """Return whether all hashable-grouping wrappers are still installed."""

    if not getattr(metadata, _HASHABLE_GROUPING_PATCH_FLAG, False):
        return False
    return all(
        _function_is_from_this_patch(getattr(metadata, name, None))
        for name in (
            "_scores_frame_for_cell_split_metadata",
            "_score_table_needs_cell_split_scoped_decode",
            "_cell_split_decode_group_columns",
            "_compare_scores_with_cell_split_metadata",
        )
    )


def _shuffle_scope_integer_patch_is_current(shuffle_controls: Any) -> bool:
    """Return whether both shuffle-scope wrappers are still live."""

    if not getattr(shuffle_controls, _SHUFFLE_SCOPE_INTEGER_PATCH_FLAG, False):
        return False
    if not getattr(shuffle_controls, _SHUFFLE_SCOPE_ALIGNMENT_PATCH_FLAG, False):
        return False
    return all(
        _function_is_from_this_patch(getattr(shuffle_controls, name, None))
        for name in ("_numeric_scope_label", "_shuffle_p_value_group_columns")
    )


def apply_cell_split_hashable_grouping_patch() -> None:
    """Make grouping robust to unhashable metadata and large integer scope IDs."""

    _apply_shuffle_scope_exact_integer_patch()

    from . import benchmark_cell_split_metadata as metadata

    if _metadata_grouping_patch_is_current(metadata):
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
            _scope_key_has_multiple_values(frame, column)
            for column in metadata._CELL_SPLIT_SCOPE_COLUMNS
            if column in frame.columns
        )

    def cell_split_decode_group_columns(scores_frame: pd.DataFrame) -> list[str]:
        frame = _with_hashable_scope_keys(scores_frame, metadata)
        columns = ["session"]
        for column in metadata._CELL_SPLIT_SCOPE_COLUMNS:
            if column not in frame.columns:
                continue
            has_multiple_values = _scope_key_has_multiple_values(frame, column)
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
    setattr(metadata, _HASHABLE_GROUPING_PATCH_FLAG, True)


def _apply_shuffle_scope_exact_integer_patch() -> None:
    """Preserve exact shuffle identities and reject ambiguous scope alignment."""

    from . import shuffle_controls

    if _shuffle_scope_integer_patch_is_current(shuffle_controls):
        return

    current_numeric_scope_label = shuffle_controls._numeric_scope_label
    if _function_is_from_this_patch(current_numeric_scope_label):
        numeric_scope_label = current_numeric_scope_label
    else:
        original_numeric_scope_label = current_numeric_scope_label

        def numeric_scope_label(value: object) -> str | None:
            if isinstance(value, (bool, np.bool_)):
                return None
            try:
                integer = operator.index(value)
            except TypeError:
                return original_numeric_scope_label(value)
            return str(int(integer))

    current_group_columns = shuffle_controls._shuffle_p_value_group_columns
    if _function_is_from_this_patch(current_group_columns):
        shuffle_p_value_group_columns = current_group_columns
    else:
        original_group_columns = current_group_columns

        def shuffle_p_value_group_columns(
            real_scores: pd.DataFrame,
            control_scores: pd.DataFrame,
        ) -> list[str]:
            columns = original_group_columns(real_scores, control_scores)
            for column in shuffle_controls._SHUFFLE_P_VALUE_SCOPE_COLUMNS:
                in_real = column in real_scores.columns
                in_control = column in control_scores.columns
                if in_real == in_control:
                    continue
                source = real_scores if in_real else control_scores
                source_name = "real_scores" if in_real else "control_scores"
                missing_name = "control_scores" if in_real else "real_scores"
                if _shuffle_scope_column_varies_within_groups(
                    source,
                    column,
                    columns,
                    shuffle_controls,
                ):
                    raise ValueError(
                        f"{source_name} contains multiple {column} values within one "
                        f"shared shuffle scope, but {missing_name} is missing {column}; "
                        f"add {column} to both score tables or compute each scope separately"
                    )
            return columns

    shuffle_controls._numeric_scope_label = numeric_scope_label
    shuffle_controls._shuffle_p_value_group_columns = shuffle_p_value_group_columns
    setattr(shuffle_controls, _SHUFFLE_SCOPE_INTEGER_PATCH_FLAG, True)
    setattr(shuffle_controls, _SHUFFLE_SCOPE_ALIGNMENT_PATCH_FLAG, True)


def _shuffle_scope_column_varies_within_groups(
    frame: pd.DataFrame,
    column: str,
    group_columns: list[str],
    shuffle_controls: Any,
) -> bool:
    """Return whether an omitted discriminator varies inside a shared scope."""

    if frame.empty:
        return False
    shared_keys = shuffle_controls._scope_keys(frame, group_columns)
    scope_labels = frame[column].map(shuffle_controls._scope_label)
    probe = pd.DataFrame(
        {
            "shared_key": list(shared_keys),
            "scope_label": list(scope_labels),
        }
    )
    counts = probe.groupby("shared_key", sort=False, dropna=False)["scope_label"].nunique(
        dropna=False
    )
    return bool((counts > 1).any())


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


def _scope_key_has_multiple_values(frame: pd.DataFrame, column: str) -> bool:
    """Return whether a scope column has multiple canonical metadata identities."""

    key_column = _scope_key_column(column)
    return int(frame[key_column].nunique(dropna=False)) > 1


def _metadata_group_key(value: object) -> str:
    """Return a recursively canonical, hashable representation of metadata."""

    return repr(_canonical_metadata_value(value))


def _canonical_metadata_value(value: object) -> tuple[object, ...]:
    """Normalize nested containers without losing array shape or scalar identity."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ("missing", None)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=object)
        return (
            "array",
            tuple(arr.shape),
            tuple(_canonical_metadata_value(item) for item in arr.reshape(-1)),
        )
    if isinstance(value, (list, tuple)):
        return (
            "sequence",
            tuple(_canonical_metadata_value(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        items = sorted(
            (_canonical_metadata_value(item) for item in value),
            key=repr,
        )
        return ("set", tuple(items))
    if isinstance(value, dict):
        items = sorted(
            (
                (_canonical_metadata_value(key), _canonical_metadata_value(item))
                for key, item in value.items()
            ),
            key=lambda pair: repr(pair[0]),
        )
        return ("mapping", tuple(items))
    return ("scalar", str(value).strip())
