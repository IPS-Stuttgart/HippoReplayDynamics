"""Compatibility patch for benchmark cell-split options.

The score-table metadata compatibility layer replaces ``benchmarks.BenchmarkConfig``
with a local dataclass so post-hoc decoding can reconstruct old score tables.
When new benchmark fields are added, that replacement class has to stay in sync
with the canonical benchmark configuration.  This patch keeps the stratified
cell-split knobs available even when the compatibility layer is active and uses
recorded split metadata when held-out score tables need fallback re-decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any


_DEFAULT_CELL_SPLIT_STRATEGY = "random"
_DEFAULT_CELL_SPLIT_STRATA = 4


def apply_benchmark_cell_split_metadata_patch() -> None:
    """Preserve benchmark cell-split options after metadata monkey-patching."""

    from . import benchmarks as bench
    from . import ground_truth as gt

    benchmark_config = bench.BenchmarkConfig
    field_names = _dataclass_field_names(benchmark_config)
    if (
        "cell_split_strategy" not in field_names
        or "cell_split_strata" not in field_names
    ):

        @dataclass(frozen=True)
        class BenchmarkConfigWithCellSplit(benchmark_config):  # type: ignore[misc, valid-type]
            cell_split_strategy: str = _DEFAULT_CELL_SPLIT_STRATEGY
            cell_split_strata: int = _DEFAULT_CELL_SPLIT_STRATA

        bench.BenchmarkConfig = BenchmarkConfigWithCellSplit
        gt.BenchmarkConfig = BenchmarkConfigWithCellSplit
    else:
        # Keep ground_truth's imported alias synchronized with benchmarks after
        # score_metadata.apply_model_hyperparam_patch() replaces both modules.
        gt.BenchmarkConfig = benchmark_config

    metadata = bench._benchmark_config_metadata
    if not getattr(metadata, "_cell_split_metadata_wrapped", False):

        def benchmark_config_metadata_with_cell_split(config: Any) -> dict[str, object]:
            out = dict(metadata(config))
            out["benchmark_cell_split_strategy"] = str(
                getattr(config, "cell_split_strategy", _DEFAULT_CELL_SPLIT_STRATEGY)
            )
            out["benchmark_cell_split_strata"] = int(
                getattr(config, "cell_split_strata", _DEFAULT_CELL_SPLIT_STRATA)
            )
            return out

        benchmark_config_metadata_with_cell_split._cell_split_metadata_wrapped = True  # type: ignore[attr-defined]
        bench._benchmark_config_metadata = benchmark_config_metadata_with_cell_split

    cell_split_for_score_rows = gt._cell_split_for_score_rows
    if not getattr(cell_split_for_score_rows, "_cell_split_metadata_wrapped", False):

        def cell_split_for_score_rows_with_metadata(
            session_scores,
            encoding,
            config,
        ):
            train_cells = gt._cell_ids_from_score_column(session_scores, "train_cell_ids")
            test_cells = gt._cell_ids_from_score_column(session_scores, "test_cell_ids")
            if train_cells is not None or test_cells is not None:
                return cell_split_for_score_rows(session_scores, encoding, config)

            test_cell_fraction = gt._unique_float_from_column(
                session_scores,
                "benchmark_test_cell_fraction",
                config.test_cell_fraction,
            )
            benchmark_random_seed = gt._unique_int_from_column(
                session_scores,
                "benchmark_random_seed",
                config.random_seed,
            )
            random_seed = gt._unique_int_from_column(
                session_scores,
                "benchmark_cell_split_seed",
                benchmark_random_seed,
            )
            strategy = gt._unique_string_from_column(
                session_scores,
                "benchmark_cell_split_strategy",
                getattr(config, "cell_split_strategy", _DEFAULT_CELL_SPLIT_STRATEGY),
            )
            n_strata = gt._unique_int_from_column(
                session_scores,
                "benchmark_cell_split_strata",
                getattr(config, "cell_split_strata", _DEFAULT_CELL_SPLIT_STRATA),
            )
            if strategy.strip().lower() in {"random", "shuffle"}:
                return bench._split_cells(encoding.cell_ids, test_cell_fraction, random_seed)
            scores = bench._cell_split_scores_from_encoding(encoding, strategy)
            return bench.stratified_cell_split(
                encoding.cell_ids,
                scores,
                test_cell_fraction,
                random_seed,
                n_strata=n_strata,
            )

        cell_split_for_score_rows_with_metadata._cell_split_metadata_wrapped = True  # type: ignore[attr-defined]
        gt._cell_split_for_score_rows = cell_split_for_score_rows_with_metadata

    bench._benchmark_cell_split_metadata_patch_applied = True


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {field.name for field in fields(cls)}
