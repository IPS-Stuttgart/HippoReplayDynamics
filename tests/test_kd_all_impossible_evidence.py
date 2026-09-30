from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_kd_model_evidence as kd_benchmark  # noqa: E402


def test_kd_all_impossible_event_has_no_order_dependent_winner() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 7,
                "model": "stationary",
                "model_family": "nontrajectory",
                "log_evidence": -np.inf,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 7,
                "model": "diffusion",
                "model_family": "trajectory",
                "log_evidence": -np.inf,
            },
        ]
    )

    scored = kd_benchmark._add_evidence_columns(rows)

    assert scored["best_model"].eq("").all()
    assert not scored["is_best_model"].any()
    assert scored["relative_log_evidence"].isna().all()
    assert scored["model_probability"].isna().all()
    assert scored["best_trajectory_model"].eq("").all()
    assert scored["best_nontrajectory_model"].eq("").all()
    assert scored["delta_vs_trajectory_best"].isna().all()
    assert scored["delta_vs_nontrajectory_best"].isna().all()


def test_kd_all_impossible_family_stays_unresolved_with_finite_other_family() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 8,
                "model": "random",
                "model_family": "nontrajectory",
                "log_evidence": 0.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 8,
                "model": "diffusion",
                "model_family": "trajectory",
                "log_evidence": -np.inf,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 8,
                "model": "momentum",
                "model_family": "trajectory",
                "log_evidence": -np.inf,
            },
        ]
    )

    scored = kd_benchmark._add_evidence_columns(rows)
    by_model = scored.set_index("model")

    assert scored["best_model"].eq("random").all()
    assert bool(by_model.loc["random", "is_best_model"])
    assert np.isclose(float(by_model.loc["random", "model_probability"]), 1.0)
    assert float(by_model.loc["diffusion", "model_probability"]) == 0.0
    assert float(by_model.loc["momentum", "model_probability"]) == 0.0

    assert scored["best_trajectory_model"].eq("").all()
    assert scored["delta_vs_trajectory_best"].isna().all()
    assert scored["best_nontrajectory_model"].eq("random").all()
    assert float(by_model.loc["random", "delta_vs_nontrajectory_best"]) == 0.0
    assert np.isneginf(float(by_model.loc["diffusion", "delta_vs_nontrajectory_best"]))
    assert np.isneginf(float(by_model.loc["momentum", "delta_vs_nontrajectory_best"]))


def test_kd_mixed_finite_and_impossible_family_keeps_finite_winner() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 9,
                "model": "diffusion",
                "model_family": "trajectory",
                "log_evidence": -2.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 9,
                "model": "momentum",
                "model_family": "trajectory",
                "log_evidence": -np.inf,
            },
        ]
    )

    scored = kd_benchmark._add_evidence_columns(rows)
    by_model = scored.set_index("model")

    assert scored["best_model"].eq("diffusion").all()
    assert scored["best_trajectory_model"].eq("diffusion").all()
    assert np.isclose(float(by_model.loc["diffusion", "model_probability"]), 1.0)
    assert float(by_model.loc["momentum", "model_probability"]) == 0.0
    assert float(by_model.loc["diffusion", "delta_vs_trajectory_best"]) == 0.0
    assert np.isneginf(float(by_model.loc["momentum", "delta_vs_trajectory_best"]))
