from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cell_split_heldout_control import (  # noqa: E402
    aggregate_cell_split_heldout_scores,
    cell_split_control_gate_summary,
    cell_split_family_margin_decisions,
    cell_split_family_margin_summary,
)


def test_cell_split_heldout_family_margin_tables_and_gates(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_split_rows("Rat1/Open1", 0, 0, stationary=0.0, trajectory=10.0),
            *_event_split_rows("Rat1/Open1", 0, 1, stationary=0.0, trajectory=12.0),
            *_event_split_rows("Rat2/Open1", 1, 0, stationary=1.0, trajectory=11.0),
            *_event_split_rows("Rat2/Open1", 1, 1, stationary=1.0, trajectory=13.0),
        ]
    )

    decisions = cell_split_family_margin_decisions(scores)
    summary = cell_split_family_margin_summary(decisions)
    rat_summary = cell_split_family_margin_summary(decisions, group_cols=("rat",))
    gates = cell_split_control_gate_summary(scores, decisions)

    assert decisions["trajectory_minus_nontrajectory_heldout_log_likelihood"].tolist() == [10.0, 12.0, 10.0, 12.0]
    assert decisions["trajectory_confident_claim"].tolist() == [True, True, True, True]
    assert summary.iloc[0]["split_event_rows"] == 4
    assert summary.iloc[0]["events"] == 2
    assert summary.iloc[0]["cell_splits"] == 2
    assert summary.iloc[0]["median_family_margin"] == 11.0
    assert rat_summary["median_family_margin"].tolist() == [11.0, 11.0]
    assert bool(gates.set_index("gate").loc["overall", "passed"])


def test_cell_split_heldout_aggregate_writes_primary_outputs(tmp_path):
    score_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            *_event_split_rows("Rat1/Open1", 0, 0, stationary=0.0, trajectory=10.0),
            *_event_split_rows("Rat2/Open1", 1, 0, stationary=1.0, trajectory=11.0),
        ]
    ).to_csv(score_path, index=False)
    out = tmp_path / "out"

    aggregate_cell_split_heldout_scores(str(score_path), out)

    summary = pd.read_csv(out / "cell_split_heldout_family_margin_summary.csv")
    rat_summary = pd.read_csv(out / "rat_cell_split_heldout_summary.csv")
    gates = pd.read_csv(out / "cell_split_control_gate_summary.csv")

    assert summary.iloc[0]["events"] == 2
    assert rat_summary["rat"].tolist() == ["Rat1", "Rat2"]
    assert bool(gates.set_index("gate").loc["overall", "passed"])
    for expected in (
        "cell_split_heldout_model_evidence.csv",
        "cell_split_heldout_family_margin_decisions.csv",
        "cell_split_heldout_family_margin_summary.csv",
        "rat_cell_split_heldout_summary.csv",
        "cell_split_control_gate_summary.csv",
    ):
        assert (out / expected).exists()


def test_cell_split_heldout_workflow_exposes_control_outputs():
    workflow = Path(".github/workflows/cell-split-heldout-control.yml").read_text(encoding="utf-8")

    assert "name: Cell-split held-out replay control" in workflow
    assert "n_splits:" in workflow
    assert 'default: "20"' in workflow
    assert "test_cell_fraction:" in workflow
    assert "scripts/cell_split_heldout_control.py score" in workflow
    assert "scripts/cell_split_heldout_control.py aggregate" in workflow
    for expected in (
        "cell_split_heldout_model_evidence.csv",
        "cell_split_heldout_family_margin_decisions.csv",
        "cell_split_heldout_family_margin_summary.csv",
        "rat_cell_split_heldout_summary.csv",
        "cell_split_control_gate_summary.csv",
    ):
        assert expected in workflow


def _event_split_rows(
    session: str,
    event_index: int,
    split_index: int,
    *,
    stationary: float,
    trajectory: float,
) -> list[dict[str, object]]:
    models = [
        ("sorted-spike-state-space-stationary", stationary, "nontrajectory"),
        ("sorted-spike-state-space-diffusion", trajectory - 4.0, "trajectory"),
        ("sorted-spike-state-space-fragmented", trajectory - 2.0, "trajectory"),
        ("sorted-spike-state-space-first-order-imm", trajectory, "trajectory"),
        ("sorted-spike-state-space-momentum-exact-sparse", trajectory - 1.0, "trajectory"),
    ]
    return [
        {
            "status": "success",
            "session": session,
            "rat": session.split("/")[0],
            "event_index": event_index,
            "cell_split_index": split_index,
            "cell_split_seed": 1 + split_index,
            "test_cell_count": 7,
            "train_cell_count": 7,
            "model": model,
            "requested_model": model,
            "model_family": family,
            "heldout_log_likelihood": heldout,
            "log_evidence": heldout,
            "test_spikes": 20,
            "n_time": 30,
            "evidence_comparable": True,
        }
        for model, heldout, family in models
    ]
