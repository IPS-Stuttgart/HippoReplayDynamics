from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth


class _EncodingStub:
    cell_ids = np.array([1, 2, 3, 4], dtype=int)
    rates_hz = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [20.0, 1.0],
        ],
        dtype=float,
    )
    occupancy_s = np.array([1.0, 1.0], dtype=float)


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


def test_ground_truth_fallback_split_uses_recorded_cell_split_metadata() -> None:
    encoding = _EncodingStub()
    rows = pd.DataFrame(
        {
            "benchmark_test_cell_fraction": [0.5],
            "benchmark_random_seed": [3],
            "benchmark_cell_split_seed": [3],
            "benchmark_cell_split_strategy": ["peak-rate"],
            "benchmark_cell_split_strata": [2],
        }
    )

    train, test = ground_truth._cell_split_for_score_rows(
        rows,
        encoding,
        benchmarks.BenchmarkConfig(),
    )
    expected_train, expected_test = benchmarks.stratified_cell_split(
        encoding.cell_ids,
        benchmarks._cell_split_scores_from_encoding(encoding, "peak-rate"),
        test_fraction=0.5,
        random_seed=3,
        n_strata=2,
    )

    np.testing.assert_array_equal(train, expected_train)
    np.testing.assert_array_equal(test, expected_test)
