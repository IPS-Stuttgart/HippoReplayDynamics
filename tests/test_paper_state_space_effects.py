from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from paper_state_space_effects import event_effect_table, summarize_paper_effects  # noqa: E402


def _write_scores(path: Path) -> None:
    rows = [
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "stationary",
            "model_family": "nontrajectory",
            "log_evidence": 0.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "sorted-spike-state-space-diffusion",
            "model_family": "trajectory",
            "log_evidence": 2.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "sorted-spike-state-space-momentum",
            "model_family": "trajectory",
            "log_evidence": 3.0,
            "evidence_support": "truncated_full_grid",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 1,
            "model": "stationary",
            "model_family": "nontrajectory",
            "log_evidence": 2.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 1,
            "model": "sorted-spike-state-space-diffusion",
            "model_family": "trajectory",
            "log_evidence": 5.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 1,
            "model": "sorted-spike-state-space-momentum",
            "model_family": "trajectory",
            "log_evidence": 4.0,
            "evidence_support": "truncated_full_grid",
        },
        {
            "session": "Rat2/Open1",
            "event_index": 0,
            "model": "stationary",
            "model_family": "nontrajectory",
            "log_evidence": 1.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat2/Open1",
            "event_index": 0,
            "model": "sorted-spike-state-space-diffusion",
            "model_family": "trajectory",
            "log_evidence": 3.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat2/Open1",
            "event_index": 0,
            "model": "sorted-spike-state-space-momentum",
            "model_family": "trajectory",
            "log_evidence": 1.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat2/Open1",
            "event_index": 1,
            "model": "stationary",
            "model_family": "nontrajectory",
            "log_evidence": 5.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat2/Open1",
            "event_index": 1,
            "model": "sorted-spike-state-space-diffusion",
            "model_family": "trajectory",
            "log_evidence": 4.0,
            "evidence_support": "exact_full_grid",
        },
        {
            "session": "Rat2/Open1",
            "event_index": 1,
            "model": "sorted-spike-state-space-momentum",
            "model_family": "trajectory",
            "log_evidence": 6.0,
            "evidence_support": "truncated_full_grid",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_summarize_paper_effects_separates_strict_and_certified_wins(tmp_path):
    scores = tmp_path / "event_model_evidence.csv"
    output = tmp_path / "paper"
    _write_scores(scores)

    tables = summarize_paper_effects(
        scores,
        output,
        bootstrap_replicates=100,
        random_seed=7,
    )

    summary = tables["summary"].iloc[0]
    assert summary["events"] == 4
    assert summary["trajectory_strict_events"] == 4
    assert summary["trajectory_strict_wins"] == 3
    assert summary["trajectory_certified_events"] == 4
    assert summary["trajectory_certified_wins"] == 4
    assert summary["momentum_diffusion_paired_events"] == 4
    assert summary["momentum_vs_diffusion_reported_wins"] == 2
    assert summary["momentum_vs_diffusion_certified_wins"] == 2
    assert summary["momentum_vs_diffusion_certified_losses"] == 1
    assert summary["momentum_vs_diffusion_inconclusive_lower_bound_nonwins"] == 1
    assert summary["mean_momentum_minus_diffusion_log_evidence"] == 0.0

    event_effects = tables["event_effects"].sort_values(["session", "event_index"])
    rat2_event1 = event_effects[(event_effects["session"] == "Rat2/Open1") & (event_effects["event_index"] == 1)].iloc[0]
    assert bool(rat2_event1["trajectory_strict_win"]) is False
    assert bool(rat2_event1["trajectory_certified_win"]) is True
    assert rat2_event1["trajectory_certification_reason"] == "trajectory_lower_bound_beats_exact_nontrajectory"

    session = tables["session_effects"].set_index("session")
    assert session.loc["Rat1/Open1", "session_interpretation"] == "mixed"
    assert session.loc["Rat2/Open1", "session_interpretation"] == "mixed"
    assert (output / "paper_state_space_effect_summary.csv").exists()
    assert (output / "paper_state_space_session_effects.csv").exists()
    assert (output / "paper_state_space_event_effects.csv").exists()
    assert "lower-bound row" in (output / "paper_state_space_claims.md").read_text(encoding="utf-8")


def test_exact_only_removes_lower_bound_momentum_from_paired_effect(tmp_path):
    scores = tmp_path / "event_model_evidence.csv"
    output = tmp_path / "paper-exact"
    _write_scores(scores)

    tables = summarize_paper_effects(
        scores,
        output,
        bootstrap_replicates=0,
        exact_only=True,
    )

    summary = tables["summary"].iloc[0]
    assert summary["evidence_policy"] == "exact_only"
    assert summary["events"] == 4
    assert summary["momentum_diffusion_paired_events"] == 1
    assert summary["momentum_vs_diffusion_reported_wins"] == 0
    assert summary["momentum_vs_diffusion_certified_losses"] == 1


def test_event_effect_table_treats_string_false_comparable_as_false():
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "stationary",
                "canonical_model": "stationary",
                "canonical_model_family": "nontrajectory",
                "log_evidence": 50.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": "True",
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "canonical_model": "diffusion",
                "canonical_model_family": "trajectory",
                "log_evidence": 10.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": "True",
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "canonical_model": "momentum",
                "canonical_model_family": "trajectory",
                "log_evidence": 100.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": "0.0",
            },
        ]
    )

    event = event_effect_table(scores).iloc[0]

    assert event["exact_model_rows"] == 2
    assert event["best_exact_trajectory_model"] == "sorted-spike-state-space-diffusion"
    assert bool(event["trajectory_strict_win"]) is False
    assert bool(event["momentum_evidence_comparable"]) is False
    assert event["momentum_vs_diffusion_certification"] == "noncomparable_pair"
