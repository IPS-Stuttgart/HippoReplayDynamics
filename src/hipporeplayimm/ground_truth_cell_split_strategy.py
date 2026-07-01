"""Ground-truth held-out decoding cell-split compatibility.

Held-out benchmark scoring can use stratified cell splits through
``BenchmarkConfig.cell_split_strategy``.  Post-hoc ground-truth decoding must
reconstruct the same train/test split whenever score rows do not already contain
explicit ``train_cell_ids``/``test_cell_ids`` metadata.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any

from .benchmarks import _split_cells_from_encoding


_DEFAULT_CELL_SPLIT_STRATEGY = "random"
_DEFAULT_CELL_SPLIT_STRATA = 4
_DECODE_GROUP_COLUMNS_PATCH_ATTR = "_random_seed_decode_grouping_wrapped"
_DECODE_GROUP_METADATA_COLUMNS = (
    "cell_split_index",
    "benchmark_random_seed",
    "benchmark_cell_split_seed",
    "cell_split_seed",
)


def apply_ground_truth_cell_split_strategy_patch() -> None:
    """Make ground-truth split reconstruction honor benchmark split strategy."""

    from . import ground_truth as gt

    base_cell_split_for_score_rows = gt._cell_split_for_score_rows
    if not getattr(base_cell_split_for_score_rows, "_cell_split_strategy_wrapped", False):

        def cell_split_for_score_rows_with_strategy(session_scores, encoding, config):
            train_cells = gt._cell_ids_from_score_column(session_scores, "train_cell_ids")
            test_cells = gt._cell_ids_from_score_column(session_scores, "test_cell_ids")
            if train_cells is not None or test_cells is not None:
                return base_cell_split_for_score_rows(session_scores, encoding, config)

            test_cell_fraction = gt._unique_float_from_column(
                session_scores,
                "benchmark_test_cell_fraction",
                getattr(config, "test_cell_fraction", 0.25),
            )
            benchmark_random_seed = gt._unique_int_from_column(
                session_scores,
                "benchmark_random_seed",
                getattr(config, "random_seed", 1),
            )
            random_seed = gt._unique_int_from_column(
                session_scores,
                "benchmark_cell_split_seed",
                benchmark_random_seed,
            )
            split_config = _config_with_overrides(
                config,
                test_cell_fraction=float(test_cell_fraction),
                cell_split_strategy=_cell_split_strategy_from_score_rows(
                    gt,
                    session_scores,
                    config,
                ),
                cell_split_strata=_cell_split_strata_from_score_rows(
                    gt,
                    session_scores,
                    config,
                ),
            )
            return _split_cells_from_encoding(encoding, split_config, int(random_seed))

        cell_split_for_score_rows_with_strategy._cell_split_strategy_wrapped = True  # type: ignore[attr-defined]
        gt._cell_split_for_score_rows = cell_split_for_score_rows_with_strategy

    _patch_benchmark_decode_group_columns(gt)


def _patch_benchmark_decode_group_columns(gt: Any) -> None:
    base_decode_group_columns = gt._decode_group_columns
    if getattr(base_decode_group_columns, _DECODE_GROUP_COLUMNS_PATCH_ATTR, False):
        return

    def decode_group_columns_with_random_seed(scores_frame, benchmark_decode: bool) -> list[str]:
        columns = list(base_decode_group_columns(scores_frame, benchmark_decode))
        if not benchmark_decode:
            return columns
        insert_at = 1 if columns and columns[0] == "session" else len(columns)
        for column in _DECODE_GROUP_METADATA_COLUMNS:
            if _score_column_has_values(scores_frame, column) and column not in columns:
                columns.insert(insert_at, column)
                insert_at += 1
        return columns

    decode_group_columns_with_random_seed._random_seed_decode_grouping_wrapped = True  # type: ignore[attr-defined]
    gt._decode_group_columns = decode_group_columns_with_random_seed


def _score_column_has_values(frame: Any, column: str) -> bool:
    if column not in getattr(frame, "columns", ()):
        return False
    return bool(frame[column].notna().any())


def _cell_split_strategy_from_score_rows(gt: Any, session_scores: Any, config: Any) -> str:
    return gt._unique_string_from_column(
        session_scores,
        "benchmark_cell_split_strategy",
        getattr(config, "cell_split_strategy", _DEFAULT_CELL_SPLIT_STRATEGY),
    )


def _cell_split_strata_from_score_rows(gt: Any, session_scores: Any, config: Any) -> int:
    return gt._unique_int_from_column(
        session_scores,
        "benchmark_cell_split_strata",
        getattr(config, "cell_split_strata", _DEFAULT_CELL_SPLIT_STRATA),
    )


def _config_with_overrides(config: Any, **overrides: Any) -> SimpleNamespace:
    values = _config_values(config)
    values.update(overrides)
    return SimpleNamespace(**values)


def _config_values(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: getattr(config, field.name) for field in fields(config)}
    return dict(getattr(config, "__dict__", {}))
