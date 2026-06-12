"""Compatibility patch for benchmark cell-split options.

The score-table metadata compatibility layer replaces ``benchmarks.BenchmarkConfig``
with a local dataclass so post-hoc decoding can reconstruct old score tables.
When new benchmark fields are added, that replacement class has to stay in sync
with the canonical benchmark configuration.  This patch keeps the stratified
cell-split knobs available even when the compatibility layer is active.
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

    bench._benchmark_cell_split_metadata_patch_applied = True


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {field.name for field in fields(cls)}
