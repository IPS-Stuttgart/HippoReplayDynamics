from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.select_state_space_parameters import select_parameters


def _write(path: Path, name: str, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / name, index=False)


def _params(
    diffusion: float,
    momentum: float,
    initial: float,
    decay: float,
    top_k: int,
) -> dict[str, object]:
    return {
        "state_space_diffusion_sigma_cm_sqrt_s": diffusion,
        "state_space_momentum_sigma_cm_sqrt_s": momentum,
        "state_space_momentum_initial_sigma_cm_sqrt_s": initial,
        "state_space_momentum_velocity_decay": decay,
        "state_space_momentum_candidate_top_k": top_k,
    }


def test_select_parameters_prefers_recovered_momentum_config(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    strong_evidence_bad_recovery = _params(85.0, 85.0, 85.0, 0.95, 128)
    weaker_evidence_good_recovery = _params(60.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **strong_evidence_bad_recovery,
                "matrix_id": "evidence-a",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
                "median_momentum_minus_diffusion_log_evidence": 5.0,
                "mean_momentum_minus_diffusion_log_evidence": 5.5,
            },
            {
                **weaker_evidence_good_recovery,
                "matrix_id": "evidence-b",
                "events": 10,
                "momentum_beats_diffusion_events": 7,
                "median_momentum_minus_diffusion_log_evidence": 1.0,
                "mean_momentum_minus_diffusion_log_evidence": 1.5,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **strong_evidence_bad_recovery,
                "matrix_id": "recovery-a",
                "failures": 0,
                "overall_recovery_accuracy": 0.9,
                "momentum_recovery_accuracy": 0.0,
                "diffusion_recovery_accuracy": 1.0,
            },
            {
                **weaker_evidence_good_recovery,
                "matrix_id": "recovery-b",
                "failures": 0,
                "overall_recovery_accuracy": 0.8,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 0.6,
            },
        ],
    )

    tables = select_parameters(evidence, recovery, output=output, min_momentum_recovery_accuracy=0.5)

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-b"
    assert recommendation["recovery_matrix_id"] == "recovery-b"
    assert bool(recommendation["passes_recovery_gate"])
    assert recommendation["momentum_beats_diffusion_event_fraction"] == 0.7
    assert len(tables["candidates"]) == 1

    assert (output / "state_space_parameter_decision_table.csv").exists()
    assert (output / "state_space_parameter_candidates.csv").exists()
    assert (output / "state_space_parameter_recommendation.csv").exists()
    manifest = json.loads((output / "state_space_parameter_selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["recovery_gate"] == {
        "max_failures": 0,
        "min_momentum_recovery_accuracy": 0.5,
        "min_overall_recovery_accuracy": 0.5,
    }
    assert manifest["row_counts"]["candidate_rows"] == 1
    assert manifest["selected_parameters"]["state_space_diffusion_sigma_cm_sqrt_s"] == 60.0
    assert manifest["recommendation"]["evidence_matrix_id"] == "evidence-b"
    assert manifest["recommendation"]["recovery_matrix_id"] == "recovery-b"
    workflow_inputs = (output / "state_space_selected_workflow_inputs.yml").read_text(encoding="utf-8")
    assert "state_space_diffusion_sigma_cm_sqrt_s: 60.0" in workflow_inputs
    assert "state_space_momentum_velocity_decay: 0.95" in workflow_inputs
    assert "state_space_momentum_candidate_top_k: 128" in workflow_inputs
    cli_args = (output / "state_space_selected_cli_args.txt").read_text(encoding="utf-8")
    assert "--state-space-diffusion-sigma-cm-sqrt-s 60.0" in cli_args
    assert "--state-space-momentum-velocity-decay 0.95" in cli_args
    assert "--state-space-momentum-candidate-top-k 128" in cli_args


def test_select_parameters_falls_back_when_no_config_passes_gate(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    params = _params(85.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **params,
                "matrix_id": "evidence-a",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **params,
                "matrix_id": "recovery-a",
                "failures": 0,
                "overall_recovery_accuracy": 0.4,
                "momentum_recovery_accuracy": 0.4,
            },
        ],
    )

    tables = select_parameters(evidence, recovery, output=output, min_momentum_recovery_accuracy=0.5)

    assert tables["candidates"].empty
    recommendation = tables["recommendation"].iloc[0]
    assert not bool(recommendation["passes_recovery_gate"])
    assert "No configuration passed" in recommendation["recommendation_note"]
    assert (output / "state_space_selected_workflow_inputs.yml").exists()
    assert (output / "state_space_selected_cli_args.txt").exists()
    manifest = json.loads((output / "state_space_parameter_selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_counts"]["candidate_rows"] == 0
    assert not manifest["recommendation"]["passes_recovery_gate"]


def test_select_parameters_writes_leave_one_session_out_recommendations(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    rat1_favored = _params(40.0, 60.0, 85.0, 0.9, 128)
    rat2_favored = _params(110.0, 85.0, 85.0, 0.98, 256)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **rat1_favored,
                "requested_session": "Rat1/Open1",
                "matrix_id": "rat1-a",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
                "median_momentum_minus_diffusion_log_evidence": 4.0,
                "mean_momentum_minus_diffusion_log_evidence": 4.5,
            },
            {
                **rat2_favored,
                "requested_session": "Rat1/Open1",
                "matrix_id": "rat1-b",
                "events": 10,
                "momentum_beats_diffusion_events": 4,
                "median_momentum_minus_diffusion_log_evidence": -1.0,
                "mean_momentum_minus_diffusion_log_evidence": -1.5,
            },
            {
                **rat1_favored,
                "requested_session": "Rat2/Open1",
                "matrix_id": "rat2-a",
                "events": 10,
                "momentum_beats_diffusion_events": 3,
                "median_momentum_minus_diffusion_log_evidence": -2.0,
                "mean_momentum_minus_diffusion_log_evidence": -2.5,
            },
            {
                **rat2_favored,
                "requested_session": "Rat2/Open1",
                "matrix_id": "rat2-b",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
                "median_momentum_minus_diffusion_log_evidence": 5.0,
                "mean_momentum_minus_diffusion_log_evidence": 5.5,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **rat1_favored,
                "matrix_id": "recovery-a",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
            {
                **rat2_favored,
                "matrix_id": "recovery-b",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
        ],
    )

    tables = select_parameters(
        evidence,
        recovery,
        output=output,
        leave_one_session_out=True,
        session_column="requested_session",
    )

    loso = tables["leave_one_session_out"].set_index("held_out_session")
    assert loso.loc["Rat1/Open1", "state_space_diffusion_sigma_cm_sqrt_s"] == rat2_favored[
        "state_space_diffusion_sigma_cm_sqrt_s"
    ]
    assert loso.loc["Rat2/Open1", "state_space_diffusion_sigma_cm_sqrt_s"] == rat1_favored[
        "state_space_diffusion_sigma_cm_sqrt_s"
    ]
    assert loso.loc["Rat1/Open1", "train_sessions"] == "Rat2/Open1"
    assert loso.loc["Rat2/Open1", "train_sessions"] == "Rat1/Open1"
    assert (output / "state_space_loso_parameter_recommendations.csv").exists()
    assert (output / "state_space_loso_selected_workflow_inputs.yml").exists()
    assert (output / "state_space_loso_selected_cli_args.txt").exists()
    manifest = json.loads((output / "state_space_parameter_selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["leave_one_session_out"] == {
        "enabled": True,
        "folds": 2,
        "output_files": [
            "state_space_loso_parameter_recommendations.csv",
            "state_space_loso_selected_workflow_inputs.yml",
            "state_space_loso_selected_cli_args.txt",
        ],
        "requested_holdout_sessions": None,
        "session_column": "requested_session",
    }
