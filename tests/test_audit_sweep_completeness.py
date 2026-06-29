import json

import pandas as pd

from scripts.audit_sweep_completeness import audit_sweep_completeness, _write_summary_json


def test_state_space_sweep_completeness_flags_missing_cells(tmp_path):
    root = tmp_path / "artifacts"
    plan = root / "state-space-evidence-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "a", "state_space_momentum_candidate_top_k": 128},
            {"id": "b", "state_space_momentum_candidate_top_k": 256},
        ]
    ).to_csv(plan / "matrix.csv", index=False)

    run = root / "state-space-evidence-sweep-a"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "status": "success",
                "matrix_id": "a",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            }
        ]
    ).to_csv(run / "event_model_evidence.csv", index=False)
    pd.DataFrame([{"matrix_id": "a", "model": "sorted-spike-state-space-diffusion"}]).to_csv(
        run / "model_evidence_summary.csv",
        index=False,
    )

    table = audit_sweep_completeness(
        artifact_root=root,
        output=tmp_path / "completeness.csv",
        mode="state-space-evidence",
    )

    complete = table.set_index("matrix_id")
    assert bool(complete.loc["a", "artifact_complete"])
    assert bool(complete.loc["a", "included_in_final_ranking"])
    assert not bool(complete.loc["b", "artifact_complete"])
    assert complete.loc["b", "completeness_reason"] == "missing-score-artifact;missing-summary-artifact"


def test_sweep_completeness_ignores_blank_plan_matrix_ids(tmp_path):
    root = tmp_path / "artifacts"
    plan = root / "state-space-evidence-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "a"},
            {"id": ""},
            {"id": None},
            {"id": "   "},
        ]
    ).to_csv(plan / "matrix.csv", index=False)

    run = root / "state-space-evidence-sweep-a"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "status": "success",
                "matrix_id": "a",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            }
        ]
    ).to_csv(run / "event_model_evidence.csv", index=False)
    pd.DataFrame([{"matrix_id": "a", "model": "overall"}]).to_csv(
        run / "model_evidence_summary.csv",
        index=False,
    )

    table = audit_sweep_completeness(
        artifact_root=root,
        output=tmp_path / "completeness.csv",
        mode="state-space-evidence",
    )

    assert table["matrix_id"].astype(str).tolist() == ["a"]
    assert bool(table.loc[0, "artifact_complete"])
    summary = json.loads((tmp_path / "completeness.summary.json").read_text(encoding="utf-8"))
    assert summary["planned_matrix_cells"] == 1


def test_recovery_sweep_completeness_treats_failures_as_not_final(tmp_path):
    root = tmp_path / "artifacts"
    plan = root / "simulation-recovery-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame([{"id": "a"}]).to_csv(plan / "matrix.csv", index=False)

    run = root / "simulation-recovery-sweep-a"
    run.mkdir()
    pd.DataFrame(
        [
            {"status": "success", "matrix_id": "a", "event_index": 0},
            {"status": "failure", "matrix_id": "a", "event_index": 1},
        ]
    ).to_csv(run / "simulation_recovery_event_scores.csv", index=False)
    pd.DataFrame([{"matrix_id": "a", "true_model": "overall"}]).to_csv(
        run / "simulation_recovery_summary.csv",
        index=False,
    )
    pd.DataFrame([{"matrix_id": "a", "true_model": "overall"}]).to_csv(
        run / "simulation_recovery_confusion_matrix.csv",
        index=False,
    )

    table = audit_sweep_completeness(
        artifact_root=root,
        output=tmp_path / "completeness.csv",
        mode="simulation-recovery",
    )

    row = table.iloc[0]
    assert bool(row["artifact_complete"])
    assert not bool(row["included_in_final_ranking"])
    assert row["n_failure_rows"] == 1


def test_sweep_completeness_treats_blank_status_as_legacy_success(tmp_path):
    root = tmp_path / "artifacts"
    plan = root / "state-space-evidence-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame([{"id": "a"}]).to_csv(plan / "matrix.csv", index=False)

    run = root / "state-space-evidence-sweep-a"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "status": "",
                "matrix_id": "a",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            },
            {
                "status": None,
                "matrix_id": "a",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-momentum-exact-sparse",
                "log_evidence": -2.0,
            },
        ]
    ).to_csv(run / "event_model_evidence.csv", index=False)
    pd.DataFrame([{"matrix_id": "a", "model": "overall"}]).to_csv(
        run / "model_evidence_summary.csv",
        index=False,
    )

    table = audit_sweep_completeness(
        artifact_root=root,
        output=tmp_path / "completeness.csv",
        mode="state-space-evidence",
    )

    row = table.iloc[0]
    assert bool(row["artifact_complete"])
    assert bool(row["included_in_final_ranking"])
    assert row["n_success_rows"] == 2
    assert row["n_failure_rows"] == 0
    assert row["completeness_reason"] == "complete"


def test_sweep_summary_json_parses_string_false_flags(tmp_path):
    table = pd.DataFrame(
        [
            {
                "planned": "True",
                "artifact_complete": "True",
                "included_in_final_ranking": "True",
                "missing_matrix_cell": "False",
            },
            {
                "planned": "False",
                "artifact_complete": "False",
                "included_in_final_ranking": "False",
                "missing_matrix_cell": "True",
            },
        ]
    )
    output = tmp_path / "summary.json"

    _write_summary_json(
        table,
        output,
        mode="state-space-evidence",
        artifact_root=tmp_path / "artifacts",
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["planned_matrix_cells"] == 1
    assert summary["artifact_complete_cells"] == 1
    assert summary["included_in_final_ranking_cells"] == 1
    assert summary["missing_or_incomplete_cells"] == 1
