from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm.benchmarks as benchmarks
from hipporeplayimm.advanced_result_diagnostics import evidence_margin_table
from hipporeplayimm.result_quality_audit import event_group_columns


def test_relative_metrics_keep_parameter_matrices_separate() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "matrix_id": "slow",
                "model": "stationary",
                "heldout_log_likelihood": 100.0,
                "test_spikes": 1,
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "matrix_id": "fast",
                "model": "stationary",
                "heldout_log_likelihood": 10.0,
                "test_spikes": 1,
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "matrix_id": "fast",
                "model": "imm",
                "heldout_log_likelihood": 12.0,
                "test_spikes": 1,
            },
        ]
    )

    metrics = benchmarks._add_relative_metrics(rows)
    fast_imm = metrics[
        metrics["matrix_id"].eq("fast") & metrics["model"].eq("imm")
    ].iloc[0]

    assert fast_imm["best_static_heldout_log_likelihood"] == pytest.approx(10.0)
    assert fast_imm["delta_vs_best_static"] == pytest.approx(2.0)


def test_result_quality_margins_keep_matrix_epochs_separate() -> None:
    scores = pd.DataFrame(
        {
            "session": ["RatX/Open1"] * 4,
            "event_index": [0] * 4,
            "matrix_id": ["slow", "slow", "fast", "fast"],
            "benchmark_event_epoch": ["run", "run", "sleep", "sleep"],
            "model": ["diffusion", "stationary", "diffusion", "stationary"],
            "log_evidence": [10.0, 0.0, 1.0, 9.0],
            "status": ["success"] * 4,
            "evidence_comparable": [True] * 4,
        }
    )

    group_columns = event_group_columns(scores)
    margins = evidence_margin_table(scores, group_cols=group_columns).sort_values(
        ["matrix_id", "benchmark_event_epoch"]
    )

    assert "matrix_id" in group_columns
    assert "benchmark_event_epoch" in group_columns
    assert margins["best_model_by_evidence"].tolist() == ["stationary", "diffusion"]
    assert margins["evidence_margin_to_second_best"].tolist() == pytest.approx([8.0, 10.0])
