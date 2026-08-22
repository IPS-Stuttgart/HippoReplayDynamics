from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.benchmarks as benchmarks


def test_relative_metrics_preserve_missing_scope_column_dtypes() -> None:
    rows = pd.DataFrame(
        {
            "session": ["RatX/Open1"] * 4,
            "event_index": [0] * 4,
            "model": ["stationary", "imm", "stationary", "imm"],
            "heldout_log_likelihood": [1.0, 2.0, 10.0, 12.0],
            "test_spikes": [1] * 4,
            "evidence_support": ["exact_full_grid"] * 4,
            "benchmark_event_subset_seed": pd.Series(
                [pd.NA, pd.NA, 7, 7],
                dtype="Int64",
            ),
            "emission_spike_rate_scale": pd.Series(
                [np.nan, np.nan, 1.5, 1.5],
                dtype="float32",
            ),
            "encoding_use_excitatory": pd.Series(
                [pd.NA, pd.NA, True, True],
                dtype="boolean",
            ),
            "benchmark_event_epoch": pd.Series(
                [pd.NA, pd.NA, "run", "run"],
                dtype="string",
            ),
        }
    )
    expected_dtypes = rows.dtypes.to_dict()

    metrics = benchmarks._add_relative_metrics(rows)

    for column in (
        "benchmark_event_subset_seed",
        "emission_spike_rate_scale",
        "encoding_use_excitatory",
        "benchmark_event_epoch",
    ):
        assert metrics[column].dtype == expected_dtypes[column]

    missing_imm = metrics[
        metrics["benchmark_event_subset_seed"].isna()
        & metrics["model"].eq("imm")
    ].iloc[0]
    run_imm = metrics[
        metrics["benchmark_event_subset_seed"].eq(7)
        & metrics["model"].eq("imm")
    ].iloc[0]
    assert missing_imm["delta_vs_best_static"] == pytest.approx(1.0)
    assert run_imm["delta_vs_best_static"] == pytest.approx(2.0)