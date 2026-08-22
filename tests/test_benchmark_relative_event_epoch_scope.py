from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm.benchmarks as benchmarks


def test_relative_metrics_keep_event_epochs_separate() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "benchmark_event_epoch": "run",
                "model": "stationary",
                "heldout_log_likelihood": 100.0,
                "test_spikes": 1,
                "evidence_support": "exact_full_grid",
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "benchmark_event_epoch": "sleep",
                "model": "stationary",
                "heldout_log_likelihood": 10.0,
                "test_spikes": 1,
                "evidence_support": "exact_full_grid",
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "benchmark_event_epoch": "sleep",
                "model": "imm",
                "heldout_log_likelihood": 12.0,
                "test_spikes": 1,
                "evidence_support": "exact_full_grid",
            },
        ]
    )

    metrics = benchmarks._add_relative_metrics(rows)
    sleep_imm = metrics[
        metrics["benchmark_event_epoch"].eq("sleep") & metrics["model"].eq("imm")
    ].iloc[0]

    assert sleep_imm["best_static_heldout_log_likelihood"] == pytest.approx(10.0)
    assert sleep_imm["delta_vs_best_static"] == pytest.approx(2.0)