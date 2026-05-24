from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_model_evidence_runs import canonical_model_name, compare_runs  # noqa: E402


def _write_event_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "event_model_evidence.csv", index=False)


def test_compare_runs_handles_empty_successful_score_tables(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    failed_row = {
        "session": "Rat1/Open1",
        "event_index": 0,
        "model": "momentum",
        "log_evidence": 0.0,
        "status": "error",
    }
    _write_event_scores(left, [failed_row])
    _write_event_scores(right, [failed_row])

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 0
    assert summary["right_events"] == 0
    assert summary["matched_events"] == 0
    assert summary["canonical_best_agreements"] == 0
    assert pd.isna(summary["canonical_best_agreement_fraction"])
    assert tables["event_comparison"].empty
    assert tables["relative"].empty
    assert (output / "model_evidence_run_comparison_summary.csv").exists()


def test_canonical_model_name_maps_state_space_aliases():
    assert canonical_model_name("sorted-spike-state-space-momentum") == "momentum"
    assert canonical_model_name("sorted-spike-state-space-momentum-exact-sparse") == "momentum"
    assert canonical_model_name("clusterless-state-space-momentum") == "momentum"
    assert canonical_model_name("state-space-diffusion") == "diffusion"
    assert canonical_model_name("jump") == "fragmented"
    assert canonical_model_name("stationary-gaussian") == "stationary-gaussian"


def test_compare_runs_writes_best_model_and_relative_evidence_tables(tmp_path):
    left = tmp_path / "kd"
    right = tmp_path / "state"
    output = tmp_path / "comparison"
    _write_event_scores(
        left,
        [
            {"session": "Rat1/Open1", "event_index": 0, "model": "momentum", "log_evidence": -1.0},
            {"session": "Rat1/Open1", "event_index": 0, "model": "diffusion", "log_evidence": -3.0},
            {"session": "Rat1/Open1", "event_index": 1, "model": "momentum", "log_evidence": -4.0},
            {"session": "Rat1/Open1", "event_index": 1, "model": "diffusion", "log_evidence": -2.0},
        ],
    )
    _write_event_scores(
        right,
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": -2.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": -4.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            },
        ],
    )

    tables = compare_runs(left, right, left_label="kd", right_label="state", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["matched_events"] == 2
    assert summary["left_events"] == 2
    assert summary["right_events"] == 2
    assert summary["canonical_best_agreements"] == 1
    assert summary["canonical_best_agreement_fraction"] == 0.5

    counts = pd.read_csv(output / "best_model_counts_comparison.csv")
    assert counts.set_index(["run_label", "canonical_model"])["events"].to_dict() == {
        ("kd", "momentum"): 1,
        ("kd", "diffusion"): 1,
        ("state", "diffusion"): 2,
    }

    relative = pd.read_csv(output / "shared_model_relative_evidence_summary.csv")
    assert set(relative["canonical_model"]) == {"diffusion", "momentum"}
    assert set(relative["matched_events"]) == {2}

    session_comparison = pd.read_csv(output / "session_model_evidence_comparison.csv")
    assert session_comparison.to_dict(orient="records") == [
        {
            "session": "Rat1/Open1",
            "kd_diffusion_wins": 1,
            "state_diffusion_wins": 2,
            "kd_momentum_wins": 1,
            "state_momentum_wins": 0,
            "momentum_win_delta": -1,
            "canonical_best_agreement_fraction": 0.5,
            "mean_momentum_relative_evidence_delta": -1.0,
        }
    ]

    assert (output / "event_best_model_comparison.csv").exists()
    assert (output / "best_model_canonical_crosstab.csv").exists()
    assert (output / "shared_model_relative_evidence_comparison.csv").exists()
    assert (output / "session_model_evidence_comparison.csv").exists()
    assert (output / "model_evidence_run_comparison_summary.csv").exists()
