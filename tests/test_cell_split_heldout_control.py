from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cell_split_heldout_control import (  # noqa: E402
    aggregate_cell_split_heldout_scores,
    cell_split_control_gate_summary,
    cell_split_family_margin_decisions,
    cell_split_family_margin_summary,
    _flush_partial_outputs,
    _initialize_partial_outputs,
    _split_indices_for_shard,
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


def test_cell_split_boolean_string_false_rows_are_not_exact_comparable():
    rows = _event_split_rows("Rat1/Open1", 0, 0, stationary=0.0, trajectory=10.0)
    rows.append(
        {
            **_event_split_rows("Rat1/Open1", 0, 0, stationary=0.0, trajectory=1000.0)[3],
            "heldout_log_likelihood": 1000.0,
            "log_evidence": 1000.0,
            "evidence_support": "candidate_pruned_lower_bound",
            "evidence_comparable": "False",
        }
    )
    for row in rows[:-1]:
        row["evidence_support"] = "exact_full_grid"
        row["evidence_comparable"] = "True"
    scores = pd.DataFrame(rows)

    decisions = cell_split_family_margin_decisions(scores)
    decision = decisions.iloc[0]
    assert decision["best_trajectory_model"] == "sorted-spike-state-space-first-order-imm"
    assert decision["trajectory_minus_nontrajectory_heldout_log_likelihood"] == 10.0

    round_tripped = decisions.copy()
    for column in ("required_models_complete", "trajectory_raw_win", "trajectory_confident_claim", "nontrajectory_confident_claim"):
        round_tripped[column] = round_tripped[column].map(lambda value: "True" if value else "False")

    summary = cell_split_family_margin_summary(round_tripped).iloc[0]
    gates = cell_split_control_gate_summary(scores, round_tripped).set_index("gate")
    assert int(summary["nontrajectory_confident_claims"]) == 0
    assert bool(gates.loc["required_models_complete", "passed"])
    assert bool(gates.loc["nontrajectory_claims_near_zero", "passed"])


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
    assert "split_shard_count:" in workflow
    assert 'default: "4"' in workflow
    assert "--split-shard-index" in workflow
    assert "--split-shard-count" in workflow
    assert "split_shard_index" in workflow
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


def test_split_indices_for_shard_uses_modulo_batches():
    assert _split_indices_for_shard(20, split_shard_index=0, split_shard_count=4) == (
        0,
        4,
        8,
        12,
        16,
    )
    assert _split_indices_for_shard(20, split_shard_index=1, split_shard_count=4) == (
        1,
        5,
        9,
        13,
        17,
    )
    assert _split_indices_for_shard(3, split_shard_index=2, split_shard_count=4) == (2,)


def test_partial_outputs_are_initialized_and_flushed_after_split(tmp_path):
    args = Namespace(
        session="Rat1/Open1",
        event_shard_index=3,
        split_shard_index=1,
        split_shard_count=4,
    )
    out = tmp_path / "partial"

    manifest = _initialize_partial_outputs(args, out, (1, 5, 9))

    manifest_path = out / "cell_split_heldout_manifest.json"
    scores_path = out / "cell_split_heldout_model_evidence.csv"
    assert manifest_path.exists()
    assert scores_path.exists()
    initial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert initial_manifest["session"] == "Rat1/Open1"
    assert initial_manifest["event_shard_index"] == 3
    assert initial_manifest["requested_splits"] == [1, 5, 9]
    assert initial_manifest["completed_splits"] == []
    assert initial_manifest["last_completed_split"] is None
    assert initial_manifest["partial_result"] is True

    rows = [
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": 0,
            "event_shard_index": 3,
            "cell_split_index": 1,
            "split_shard_index": 1,
            "split_shard_count": 4,
            "requested_splits": "1,5,9",
            "model": "sorted-spike-state-space-stationary",
            "heldout_log_likelihood": 0.0,
            "log_evidence": 0.0,
        }
    ]

    _flush_partial_outputs(
        out,
        manifest,
        rows,
        completed_splits=[1],
        last_completed_split=1,
        status="running",
    )

    flushed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    flushed_scores = pd.read_csv(scores_path)
    assert flushed_manifest["completed_splits"] == [1]
    assert flushed_manifest["last_completed_split"] == 1
    assert flushed_manifest["partial_result"] is True
    assert flushed_scores["cell_split_index"].tolist() == [1]
    assert flushed_scores["completed_splits"].astype(str).tolist() == ["1"]
    assert flushed_scores["last_completed_split"].tolist() == [1]
    assert flushed_scores["partial_result"].tolist() == [True]


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
