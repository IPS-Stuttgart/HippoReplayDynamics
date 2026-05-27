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
    predicted_top_k: int = 8,
    candidate_source: str = "emission",
) -> dict[str, object]:
    return {
        "state_space_diffusion_sigma_cm_sqrt_s": diffusion,
        "state_space_momentum_sigma_cm_sqrt_s": momentum,
        "state_space_momentum_initial_sigma_cm_sqrt_s": initial,
        "state_space_momentum_velocity_decay": decay,
        "state_space_momentum_candidate_top_k": top_k,
        "state_space_momentum_predicted_candidate_top_k": predicted_top_k,
        "state_space_momentum_candidate_source": candidate_source,
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

    tables = select_parameters(
        evidence,
        recovery,
        output=output,
        min_momentum_recovery_accuracy=0.5,
        force_strict_recovery_gate=True,
    )

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
    assert manifest["recovery_gate"]["max_failures"] == 0
    assert manifest["recovery_gate"]["min_momentum_recovery_accuracy"] == 0.5
    assert manifest["recovery_gate"]["min_overall_recovery_accuracy"] == 0.5
    assert manifest["recovery_gate"]["requested_metric"] == "auto"
    assert manifest["recovery_gate"]["resolved_metric"] == "strict"
    assert manifest["recovery_gate"]["momentum_column"] == "momentum_recovery_accuracy"
    assert manifest["recovery_gate"]["overall_column"] == "overall_recovery_accuracy"
    assert manifest["row_counts"]["candidate_rows"] == 1
    assert manifest["selected_parameters"]["state_space_diffusion_sigma_cm_sqrt_s"] == 60.0
    assert manifest["recommendation"]["evidence_matrix_id"] == "evidence-b"
    assert manifest["recommendation"]["recovery_matrix_id"] == "recovery-b"
    workflow_inputs = (output / "state_space_selected_workflow_inputs.yml").read_text(encoding="utf-8")
    assert "state_space_diffusion_sigma_cm_sqrt_s: 60.0" in workflow_inputs
    assert "state_space_momentum_velocity_decay: 0.95" in workflow_inputs
    assert "state_space_momentum_candidate_top_k: 128" in workflow_inputs
    assert "state_space_momentum_predicted_candidate_top_k: 8" in workflow_inputs
    cli_args = (output / "state_space_selected_cli_args.txt").read_text(encoding="utf-8")
    assert "--state-space-diffusion-sigma-cm-sqrt-s 60.0" in cli_args
    assert "--state-space-momentum-velocity-decay 0.95" in cli_args
    assert "--state-space-momentum-candidate-top-k 128" in cli_args
    assert "--state-space-momentum-predicted-candidate-top-k 8" in cli_args


def test_select_parameters_prefers_confident_momentum_wins(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    raw_win_only = _params(85.0, 85.0, 85.0, 0.95, 128)
    confident_win = _params(60.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **raw_win_only,
                "matrix_id": "evidence-raw",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
                "momentum_beats_diffusion_log5_events": 2,
                "median_momentum_minus_diffusion_log_evidence": 1.0,
                "mean_momentum_minus_diffusion_log_evidence": 1.5,
            },
            {
                **confident_win,
                "matrix_id": "evidence-confident",
                "events": 10,
                "momentum_beats_diffusion_events": 7,
                "momentum_beats_diffusion_log5_events": 6,
                "median_momentum_minus_diffusion_log_evidence": 5.0,
                "mean_momentum_minus_diffusion_log_evidence": 6.0,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **raw_win_only,
                "matrix_id": "recovery-raw",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
            {
                **confident_win,
                "matrix_id": "recovery-confident",
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
        force_strict_recovery_gate=True,
    )

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-confident"
    assert recommendation["momentum_beats_diffusion_log5_event_fraction"] == 0.6
    assert recommendation["momentum_beats_diffusion_event_fraction"] == 0.7


def test_select_parameters_prefers_calibrated_confidence_threshold(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    confidence = tmp_path / "confidence"
    output = tmp_path / "selection"
    stronger_fixed_log5 = _params(85.0, 85.0, 85.0, 0.95, 128)
    stronger_calibrated_threshold = _params(60.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **stronger_fixed_log5,
                "matrix_id": "evidence-log5",
                "events": 10,
                "momentum_beats_diffusion_events": 9,
                "momentum_beats_diffusion_log5_events": 7,
                "median_momentum_minus_diffusion_log_evidence": 8.0,
                "mean_momentum_minus_diffusion_log_evidence": 9.0,
            },
            {
                **stronger_calibrated_threshold,
                "matrix_id": "evidence-calibrated",
                "events": 10,
                "momentum_beats_diffusion_events": 8,
                "momentum_beats_diffusion_log5_events": 6,
                "median_momentum_minus_diffusion_log_evidence": 7.0,
                "mean_momentum_minus_diffusion_log_evidence": 8.0,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **stronger_fixed_log5,
                "matrix_id": "recovery-log5",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
            {
                **stronger_calibrated_threshold,
                "matrix_id": "recovery-calibrated",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
        ],
    )
    _write(
        confidence,
        "momentum_confidence_threshold_evidence_by_stratum.csv",
        [
            {
                "matrix_id": "evidence-log5",
                "margin_threshold": 6.0,
                "events": 10,
                "positive_model_claims": 3,
                "reference_model_claims": 0,
                "ambiguous_events": 7,
                "positive_claim_fraction": 0.3,
            },
            {
                "matrix_id": "evidence-calibrated",
                "margin_threshold": 6.0,
                "events": 10,
                "positive_model_claims": 5,
                "reference_model_claims": 0,
                "ambiguous_events": 5,
                "positive_claim_fraction": 0.5,
            },
        ],
    )

    tables = select_parameters(
        evidence,
        recovery,
        output=output,
        confidence_evidence=confidence,
        force_strict_recovery_gate=True,
    )

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-calibrated"
    assert recommendation["momentum_confidence_threshold"] == 6.0
    assert recommendation["momentum_confidence_claim_events"] == 5
    assert recommendation["momentum_confidence_claim_event_fraction"] == 0.5
    assert recommendation["momentum_beats_diffusion_log5_event_fraction"] == 0.6
    manifest = json.loads((output / "state_space_parameter_selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_paths"]["confidence_evidence"] == str(confidence)


def test_select_parameters_uses_confidence_recovery_safety_metrics(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    confidence = tmp_path / "confidence"
    output = tmp_path / "selection"
    unsafe = _params(85.0, 85.0, 85.0, 0.95, 128)
    safe = _params(60.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **unsafe,
                "matrix_id": "evidence-unsafe",
                "events": 10,
                "momentum_beats_diffusion_events": 9,
                "momentum_beats_diffusion_log5_events": 7,
                "median_momentum_minus_diffusion_log_evidence": 9.0,
                "mean_momentum_minus_diffusion_log_evidence": 20.0,
            },
            {
                **safe,
                "matrix_id": "evidence-safe",
                "events": 10,
                "momentum_beats_diffusion_events": 8,
                "momentum_beats_diffusion_log5_events": 6,
                "median_momentum_minus_diffusion_log_evidence": 7.0,
                "mean_momentum_minus_diffusion_log_evidence": 10.0,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **unsafe,
                "matrix_id": "recovery-unsafe",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
            {
                **safe,
                "matrix_id": "recovery-safe",
                "failures": 0,
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            },
        ],
    )
    _write(
        confidence,
        "momentum_confidence_threshold_evidence_by_stratum.csv",
        [
            {
                "matrix_id": "evidence-unsafe",
                "margin_threshold": 5.0,
                "events": 10,
                "positive_model_claims": 5,
                "reference_model_claims": 0,
                "ambiguous_events": 5,
                "positive_claim_fraction": 0.5,
            },
            {
                "matrix_id": "evidence-safe",
                "margin_threshold": 5.0,
                "events": 10,
                "positive_model_claims": 5,
                "reference_model_claims": 0,
                "ambiguous_events": 5,
                "positive_claim_fraction": 0.5,
            },
        ],
    )
    _write(
        confidence,
        "momentum_confidence_threshold_selection.csv",
        [
            {
                "matrix_id": "evidence-unsafe",
                "thresholded_binary_accuracy": 0.875,
                "positive_claim_recall": 1.0,
                "reference_specificity": 0.75,
                "false_positive_claims": 1,
                "false_negative_claims": 0,
                "passes_threshold_gate": False,
                "selection_status": "fallback_no_gate_pass",
                "threshold_scope": "stratum",
            },
            {
                "matrix_id": "evidence-safe",
                "thresholded_binary_accuracy": 1.0,
                "positive_claim_recall": 1.0,
                "reference_specificity": 1.0,
                "false_positive_claims": 0,
                "false_negative_claims": 0,
                "passes_threshold_gate": True,
                "selection_status": "passed_specificity_gate",
                "threshold_scope": "stratum",
            },
        ],
    )

    tables = select_parameters(
        evidence,
        recovery,
        output=output,
        confidence_evidence=confidence,
        force_strict_recovery_gate=True,
    )

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-safe"
    assert recommendation["momentum_confidence_recovery_false_positive_claims"] == 0
    assert recommendation["momentum_confidence_recovery_reference_specificity"] == 1.0
    assert bool(recommendation["momentum_confidence_recovery_passes_threshold_gate"])


def test_select_parameters_keeps_observation_calibration_dimensions_separate(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    params = _params(60.0, 60.0, 85.0, 0.95, 128, predicted_top_k=0)
    common = {
        **params,
        "state_space_max_step_sigma": 3.0,
        "state_space_valid_occupancy_threshold_s": 0.0,
        "spike_rate_scale": 2.0,
        "emission_negative_binomial_overdispersion": 0.0,
    }
    evidence_common = {**common, "time_bin_s": 0.004}
    recovery_common = {**common, "time_bin_ms": 4.0}
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **evidence_common,
                "matrix_id": "evidence-temp0375",
                "events": 10,
                "momentum_beats_diffusion_events": 9,
                "momentum_beats_diffusion_log5_events": 7,
                "median_momentum_minus_diffusion_log_evidence": 8.0,
                "mean_momentum_minus_diffusion_log_evidence": 20.0,
                "emission_likelihood_temperature": 0.375,
            },
            {
                **evidence_common,
                "matrix_id": "evidence-temp0425",
                "events": 10,
                "momentum_beats_diffusion_events": 8,
                "momentum_beats_diffusion_log5_events": 5,
                "median_momentum_minus_diffusion_log_evidence": 7.0,
                "mean_momentum_minus_diffusion_log_evidence": 18.0,
                "emission_likelihood_temperature": 0.425,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **recovery_common,
                "matrix_id": "recovery-temp0375",
                "failures": 0,
                "overall_recovery_accuracy": 0.875,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 0.75,
                "momentum_most_common_best_model": "sorted-spike-state-space-momentum-exact-sparse",
                "emission_likelihood_temperature": 0.375,
            },
            {
                **recovery_common,
                "matrix_id": "recovery-temp0425",
                "failures": 0,
                "overall_recovery_accuracy": 0.875,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 0.75,
                "momentum_most_common_best_model": "sorted-spike-state-space-momentum-exact-sparse",
                "emission_likelihood_temperature": 0.425,
            },
        ],
    )

    tables = select_parameters(evidence, recovery, output=output)

    assert len(tables["decision"]) == 2
    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-temp0375"
    assert recommendation["recovery_matrix_id"] == "recovery-temp0375"
    assert recommendation["emission_likelihood_temperature"] == 0.375
    assert recommendation["momentum_beats_diffusion_log5_event_fraction"] == 0.7
    assert bool(recommendation["passes_recovery_gate"])
    assert not bool(recommendation["uses_candidate_pruned_momentum"])
    workflow_inputs = (output / "state_space_selected_workflow_inputs.yml").read_text(encoding="utf-8")
    assert "emission_likelihood_temperature: 0.375" in workflow_inputs
    assert "time_bin_s: 0.004" in workflow_inputs


def test_select_parameters_loads_emission_calibration_recovery_artifact(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    params = _params(60.0, 60.0, 85.0, 0.95, 128, predicted_top_k=0)
    evidence_common = {
        **params,
        "state_space_max_step_sigma": 3.0,
        "state_space_valid_occupancy_threshold_s": 0.0,
        "time_bin_s": 0.004,
        "spike_rate_scale": 2.0,
        "emission_likelihood_temperature": 0.375,
        "emission_negative_binomial_overdispersion": 0.0,
    }
    recovery_common = {
        key: value
        for key, value in evidence_common.items()
        if key
        not in {
            "state_space_momentum_candidate_top_k",
            "state_space_momentum_predicted_candidate_top_k",
            "state_space_momentum_candidate_source",
        }
    }
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **evidence_common,
                "matrix_id": "evidence-temp0375",
                "events": 10,
                "momentum_beats_diffusion_events": 9,
                "momentum_beats_diffusion_log5_events": 7,
                "median_momentum_minus_diffusion_log_evidence": 8.0,
                "mean_momentum_minus_diffusion_log_evidence": 20.0,
            }
        ],
    )
    _write(
        recovery,
        "simulation_recovery_emission_calibration_config_ranked.csv",
        [
            {
                **recovery_common,
                "matrix_id": "recovery-temp0375",
                "failures": 0,
                "overall_recovery_accuracy": 0.875,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 0.75,
                "momentum_most_common_best_model": "sorted-spike-state-space-momentum-exact-sparse",
            }
        ],
    )

    tables = select_parameters(evidence, recovery, output=output)

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-temp0375"
    assert recommendation["recovery_matrix_id"] == "recovery-temp0375"
    assert recommendation["state_space_momentum_candidate_top_k"] == 128
    assert recommendation["state_space_momentum_predicted_candidate_top_k"] == 0
    assert recommendation["state_space_momentum_candidate_source"] == "emission"
    assert bool(recommendation["passes_recovery_gate"])


def test_select_parameters_matches_predicted_candidate_support_dimension(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    strong_unvalidated_support = _params(85.0, 85.0, 85.0, 0.95, 128, predicted_top_k=0)
    weaker_validated_support = _params(85.0, 85.0, 85.0, 0.95, 128, predicted_top_k=8)
    recovery_only_support = _params(110.0, 85.0, 85.0, 0.95, 128, predicted_top_k=8)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **strong_unvalidated_support,
                "matrix_id": "evidence-pk0",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
                "median_momentum_minus_diffusion_log_evidence": 5.0,
                "mean_momentum_minus_diffusion_log_evidence": 5.5,
            },
            {
                **weaker_validated_support,
                "matrix_id": "evidence-pk8",
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
                **weaker_validated_support,
                "matrix_id": "recovery-pk8",
                "failures": 0,
                "overall_recovery_accuracy": 0.8,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 0.6,
            },
            {
                **recovery_only_support,
                "matrix_id": "recovery-only",
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
        min_momentum_recovery_accuracy=0.5,
        force_strict_recovery_gate=True,
    )

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-pk8"
    assert recommendation["recovery_matrix_id"] == "recovery-pk8"
    assert recommendation["state_space_momentum_predicted_candidate_top_k"] == 8
    unvalidated = tables["decision"][
        tables["decision"]["evidence_matrix_id"].eq("evidence-pk0")
    ].iloc[0]
    assert not bool(unvalidated["has_recovery"])
    assert unvalidated["recovery_gate"] == "missing-recovery"
    recovery_only = tables["decision"][
        tables["decision"]["recovery_matrix_id"].eq("recovery-only")
    ].iloc[0]
    assert not bool(recovery_only["has_evidence"])
    assert recovery_only["recovery_gate"] == "missing-evidence"
    assert pd.isna(recovery_only["mean_momentum_minus_diffusion_log_evidence"])


def test_select_parameters_keeps_candidate_support_dimensions_separate(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    unsupported_high_evidence = _params(
        85.0,
        85.0,
        85.0,
        0.95,
        128,
        predicted_top_k=0,
        candidate_source="emission",
    )
    recovered_lower_evidence = _params(
        85.0,
        85.0,
        85.0,
        0.95,
        128,
        predicted_top_k=16,
        candidate_source="posterior",
    )
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **unsupported_high_evidence,
                "matrix_id": "evidence-pk0-emission",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
                "median_momentum_minus_diffusion_log_evidence": 5.0,
                "mean_momentum_minus_diffusion_log_evidence": 5.5,
            },
            {
                **recovered_lower_evidence,
                "matrix_id": "evidence-pk16-posterior",
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
                **recovered_lower_evidence,
                "matrix_id": "recovery-pk16-posterior",
                "failures": 0,
                "overall_certified_vs_exact_recovery_accuracy": 0.8,
                "momentum_certified_vs_exact_recovery_accuracy": 1.0,
                "overall_recovery_accuracy": 0.5,
                "momentum_recovery_accuracy": 0.0,
            },
        ],
    )

    tables = select_parameters(evidence, recovery, output=output, min_momentum_recovery_accuracy=0.5)

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-pk16-posterior"
    assert recommendation["recovery_matrix_id"] == "recovery-pk16-posterior"
    assert recommendation["state_space_momentum_predicted_candidate_top_k"] == 16
    assert recommendation["state_space_momentum_candidate_source"] == "posterior"
    assert bool(recommendation["passes_recovery_gate"])
    decision = tables["decision"].set_index("evidence_matrix_id")
    assert not bool(decision.loc["evidence-pk0-emission", "has_recovery"])
    workflow_inputs = (output / "state_space_selected_workflow_inputs.yml").read_text(encoding="utf-8")
    assert "state_space_momentum_predicted_candidate_top_k: 16" in workflow_inputs
    assert "state_space_momentum_candidate_source: posterior" in workflow_inputs
    cli_args = (output / "state_space_selected_cli_args.txt").read_text(encoding="utf-8")
    assert "--state-space-momentum-predicted-candidate-top-k 16" in cli_args
    assert "--state-space-momentum-candidate-source posterior" in cli_args


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


def test_select_parameters_auto_gate_uses_certified_recovery_columns_when_present(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    strong_evidence_bad_certified = _params(85.0, 85.0, 85.0, 0.95, 128)
    weaker_evidence_good_certified = _params(60.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **strong_evidence_bad_certified,
                "matrix_id": "evidence-a",
                "events": 10,
                "momentum_beats_diffusion_events": 10,
            },
            {
                **weaker_evidence_good_certified,
                "matrix_id": "evidence-b",
                "events": 10,
                "momentum_beats_diffusion_events": 7,
            },
        ],
    )
    _write(
        recovery,
        "simulation_recovery_sweep_config_ranked.csv",
        [
            {
                **strong_evidence_bad_certified,
                "matrix_id": "recovery-a",
                "failures": 0,
                "overall_recovery_accuracy": 0.5,
                "momentum_recovery_accuracy": 0.0,
                "overall_certified_vs_exact_recovery_accuracy": 0.4,
                "momentum_certified_vs_exact_recovery_accuracy": 0.2,
            },
            {
                **weaker_evidence_good_certified,
                "matrix_id": "recovery-b",
                "failures": 0,
                "overall_recovery_accuracy": 0.5,
                "momentum_recovery_accuracy": 0.0,
                "overall_certified_vs_exact_recovery_accuracy": 0.8,
                "momentum_certified_vs_exact_recovery_accuracy": 1.0,
            },
        ],
    )

    tables = select_parameters(evidence, recovery, output=output, min_momentum_recovery_accuracy=0.5)

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["evidence_matrix_id"] == "evidence-b"
    assert recommendation["recovery_gate_metric"] == "certified-vs-exact"
    assert recommendation["gate_momentum_recovery_accuracy"] == 1.0
    assert recommendation["momentum_recovery_gate_column"] == "momentum_certified_vs_exact_recovery_accuracy"
    manifest = json.loads((output / "state_space_parameter_selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recovery_gate"]["resolved_metric"] == "certified-vs-exact"


def test_candidate_pruned_strict_gate_is_blocked_without_override(tmp_path):
    evidence = tmp_path / "evidence"
    recovery = tmp_path / "recovery"
    output = tmp_path / "selection"
    params = _params(60.0, 85.0, 85.0, 0.95, 128)
    _write(
        evidence,
        "state_space_evidence_sweep_config_ranked.csv",
        [
            {
                **params,
                "matrix_id": "evidence-a",
                "events": 10,
                "momentum_beats_diffusion_events": 8,
                "mean_momentum_minus_diffusion_log_evidence": 2.0,
            }
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
                "overall_recovery_accuracy": 1.0,
                "momentum_recovery_accuracy": 1.0,
                "diffusion_recovery_accuracy": 1.0,
            }
        ],
    )

    tables = select_parameters(
        evidence, recovery, output=output, min_momentum_recovery_accuracy=0.5
    )

    decision = tables["decision"].iloc[0]
    assert decision["recovery_gate"] == "strict-gate-blocked"
    assert bool(decision["strict_candidate_recovery_gate_blocked"])
    assert not bool(decision["passes_recovery_gate"])
    assert tables["candidates"].empty

    forced = select_parameters(
        evidence,
        recovery,
        output=tmp_path / "selection-forced",
        min_momentum_recovery_accuracy=0.5,
        force_strict_recovery_gate=True,
    )
    assert bool(forced["recommendation"].iloc[0]["passes_recovery_gate"])


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
        force_strict_recovery_gate=True,
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
