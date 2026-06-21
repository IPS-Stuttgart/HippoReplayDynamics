from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.benchmarks as benchmarks


def test_relative_metrics_do_not_mix_na_scope_rows_with_current_sweep_rows() -> None:
    rows = [
        {
            "session": "RatX/Open1",
            "event_index": 0,
            "model": "stationary",
            "heldout_log_likelihood": 100.0,
            "test_spikes": 1,
            "benchmark_random_seed": 7,
            "benchmark_cell_split_index": 0,
            "benchmark_test_cell_fraction": np.nan,
            "benchmark_cell_split_seed": 7,
            "benchmark_cell_split_strategy": "random",
            "benchmark_cell_split_strata": 4,
        },
        {
            "session": "RatX/Open1",
            "event_index": 0,
            "model": "stationary",
            "heldout_log_likelihood": 10.0,
            "test_spikes": 1,
            "benchmark_random_seed": 7,
            "benchmark_cell_split_index": 0,
            "benchmark_test_cell_fraction": 0.50,
            "benchmark_cell_split_seed": 7,
            "benchmark_cell_split_strategy": "random",
            "benchmark_cell_split_strata": 4,
        },
        {
            "session": "RatX/Open1",
            "event_index": 0,
            "model": "imm",
            "heldout_log_likelihood": 12.0,
            "test_spikes": 1,
            "benchmark_random_seed": 7,
            "benchmark_cell_split_index": 0,
            "benchmark_test_cell_fraction": 0.50,
            "benchmark_cell_split_seed": 7,
            "benchmark_cell_split_strategy": "random",
            "benchmark_cell_split_strata": 4,
        },
    ]

    metrics = benchmarks._add_relative_metrics(pd.DataFrame(rows))

    imm_delta = metrics.loc[
        metrics["model"].eq("imm")
        & np.isclose(metrics["benchmark_test_cell_fraction"].to_numpy(dtype=float), 0.50),
        "delta_vs_best_static",
    ].iloc[0]
    assert imm_delta == pytest.approx(2.0)

    na_delta = metrics.loc[
        metrics["model"].eq("stationary")
        & metrics["benchmark_test_cell_fraction"].isna(),
        "delta_vs_best_static",
    ].iloc[0]
    assert na_delta == pytest.approx(0.0)
