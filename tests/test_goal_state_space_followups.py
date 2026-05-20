from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_model_evidence_runs import compare_runs  # noqa: E402
from marginalize_state_space_sweep import marginalize_sweep  # noqa: E402


def test_marginalized_momentum_keeps_truncated_support(tmp_path: Path) -> None:
    input_csv = tmp_path / "sweep.csv"
    pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": -1.0,
                "state_space_momentum_sigma_cm_sqrt_s": 85.0,
                "state_space_momentum_initial_sigma_cm_sqrt_s": 85.0,
                "state_space_momentum_velocity_decay": 0.95,
                "state_space_momentum_candidate_top_k": 128,
                "n_time": 3,
                "n_spikes": 4,
            }
        ]
    ).to_csv(input_csv, index=False)

    tables = marginalize_sweep(input_csv, tmp_path / "out", models=("momentum",), prior="uniform")

    event_model_evidence = tables["event_model_evidence"]
    assert event_model_evidence.loc[0, "evidence_support"] == "truncated_full_grid"
    assert not bool(event_model_evidence.loc[0, "evidence_comparable"])


def test_compare_runs_exact_only_recomputes_relative_evidence(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    rows = [
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "sorted-spike-state-space-diffusion",
            "model_family": "trajectory",
            "log_evidence": -10.0,
            "relative_log_evidence": -5.0,
        },
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "sorted-spike-state-space-momentum",
            "model_family": "trajectory",
            "log_evidence": -5.0,
            "relative_log_evidence": 0.0,
            "diagnostic_state_space_momentum_evidence_support": "truncated_full_grid",
        },
    ]
    pd.DataFrame(rows).to_csv(left / "event_model_evidence.csv", index=False)
    pd.DataFrame(rows).to_csv(right / "event_model_evidence.csv", index=False)

    tables = compare_runs(left, right, left_label="left", right_label="right", output=tmp_path / "cmp", exact_only=True)

    comparison = tables["event_comparison"]
    assert comparison.loc[0, "left_best_model"] == "sorted-spike-state-space-diffusion"
    assert np.isclose(comparison.loc[0, "left_best_relative_log_evidence"], 0.0)
    support = tables["support_counts"]
    assert set(support["evidence_support"]) == {"exact_full_grid"}
