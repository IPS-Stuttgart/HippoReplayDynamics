from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_model_evidence as benchmark  # noqa: E402


def _row(
    model: str,
    family: str,
    log_evidence: float,
    *,
    evidence_support: str = "exact_full_grid",
    evidence_comparable: bool = True,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 0,
        "model": model,
        "requested_model": model,
        "model_family": family,
        "log_evidence": log_evidence,
        "evidence_support": evidence_support,
        "evidence_comparable": evidence_comparable,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
    }


def test_primary_benchmark_all_impossible_exact_models_have_no_winner() -> None:
    rows = pd.DataFrame(
        [
            _row("random", "nontrajectory", -np.inf),
            _row("diffusion", "trajectory", -np.inf),
        ]
    )

    scored = benchmark._add_evidence_columns(rows)

    assert scored["best_model"].eq("").all()
    assert not scored["is_best_model"].any()
    assert scored["relative_log_evidence"].isna().all()
    assert scored["model_probability"].isna().all()
    assert scored["best_trajectory_model"].eq("").all()
    assert scored["best_nontrajectory_model"].eq("").all()
    assert scored["delta_vs_trajectory_best"].isna().all()
    assert scored["delta_vs_nontrajectory_best"].isna().all()


def test_primary_benchmark_all_impossible_family_stays_unresolved() -> None:
    rows = pd.DataFrame(
        [
            _row("random", "nontrajectory", 0.0),
            _row("diffusion", "trajectory", -np.inf),
            _row("momentum", "trajectory", -np.inf),
        ]
    )

    scored = benchmark._add_evidence_columns(rows)
    by_model = scored.set_index("model")

    assert scored["best_model"].eq("random").all()
    assert bool(by_model.loc["random", "is_best_model"])
    assert np.isclose(float(by_model.loc["random", "model_probability"]), 1.0)
    assert float(by_model.loc["diffusion", "model_probability"]) == 0.0
    assert float(by_model.loc["momentum", "model_probability"]) == 0.0
    assert scored["best_trajectory_model"].eq("").all()
    assert scored["delta_vs_trajectory_best"].isna().all()
    assert scored["best_nontrajectory_model"].eq("random").all()


def test_primary_benchmark_all_impossible_truncated_models_have_no_winner() -> None:
    rows = pd.DataFrame(
        [
            _row("random", "nontrajectory", 0.0),
            _row(
                "diffusion",
                "trajectory",
                -np.inf,
                evidence_support="truncated_full_grid",
                evidence_comparable=False,
            ),
            _row(
                "momentum",
                "trajectory",
                -np.inf,
                evidence_support="truncated_full_grid",
                evidence_comparable=False,
            ),
        ]
    )

    scored = benchmark._add_evidence_columns(rows)

    assert scored["best_model"].eq("random").all()
    assert scored["best_truncated_lower_bound_model"].eq("").all()
    assert not scored["is_best_truncated_lower_bound"].any()
    truncated = scored[scored["evidence_support"].eq("truncated_full_grid")]
    assert truncated["truncated_relative_log_evidence"].isna().all()
