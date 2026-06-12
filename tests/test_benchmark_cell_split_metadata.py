from __future__ import annotations

import hipporeplayimm
import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth


def test_metadata_patch_preserves_cell_split_config_options() -> None:
    config = benchmarks.BenchmarkConfig(
        cell_split_strategy="mean-rate",
        cell_split_strata=6,
    )

    assert config.cell_split_strategy == "mean-rate"
    assert config.cell_split_strata == 6
    assert ground_truth.BenchmarkConfig is benchmarks.BenchmarkConfig
    assert hipporeplayimm.BenchmarkConfig is benchmarks.BenchmarkConfig


def test_benchmark_metadata_records_cell_split_config_options() -> None:
    config = benchmarks.BenchmarkConfig(
        cell_split_strategy="peak-rate",
        cell_split_strata=5,
    )

    metadata = benchmarks._benchmark_config_metadata(config)

    assert metadata["benchmark_cell_split_strategy"] == "peak-rate"
    assert metadata["benchmark_cell_split_strata"] == 5
