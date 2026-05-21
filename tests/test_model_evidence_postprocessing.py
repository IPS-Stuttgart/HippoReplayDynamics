from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_model_evidence import _postprocess_evidence_scores  # noqa: E402


def test_postprocess_evidence_scores_adds_candidate_quality_and_margins() -> None:
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "random",
                "model_family": "nontrajectory",
                "log_evidence": -10.0,
                "n_time": 5,
                "n_spikes": 7,
                "runtime_s": 0.01,
                "error": "",
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "stationary",
                "model_family": "nontrajectory",
                "log_evidence": -12.0,
                "n_time": 5,
                "n_spikes": 7,
                "runtime_s": 0.01,
                "error": "",
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "model_family": "trajectory",
                "log_evidence": -8.0,
                "n_time": 5,
                "n_spikes": 7,
                "runtime_s": 0.01,
                "error": "",
                "diagnostic_state_space_momentum_evidence_support": "truncated_full_grid",
                "diagnostic_min_candidate_log_mass": -0.005,
            },
        ]
    )

    processed = _postprocess_evidence_scores(rows)

    assert "candidate_support_quality" in processed.columns
    assert "candidate_support_quality_good" in processed.columns
    assert "evidence_margin_to_second_best" in processed.columns
    assert "evidence_margin_category" in processed.columns

    exact_rows = processed[processed["evidence_comparable"]]
    assert exact_rows["best_model_by_evidence"].unique().tolist() == ["random"]
    assert np.isclose(exact_rows["evidence_margin_to_second_best"].iloc[0], 2.0)
    assert exact_rows["evidence_margin_category"].iloc[0] == "weak"

    momentum = processed[
        processed["model"].eq("sorted-spike-state-space-momentum")
    ].iloc[0]
    assert momentum["evidence_support"] == "truncated_full_grid"
    assert momentum["candidate_support_quality"] == "conservative_good"
    assert bool(momentum["candidate_support_quality_good"])
