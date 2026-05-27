from pathlib import Path


def test_state_space_evidence_sweep_workflow_defines_parameter_grid_and_summary_outputs():
    workflow = Path(".github/workflows/state-space-evidence-sweep.yml").read_text(encoding="utf-8")

    assert "name: State-space replay evidence parameter sweep" in workflow
    assert (
        'default: "sorted-spike-state-space-diffusion '
        "sorted-spike-state-space-momentum-exact-sparse "
        "sorted-spike-state-space-momentum "
        "sorted-spike-state-space-first-order-imm "
        'sorted-spike-state-space-imm"'
    ) in workflow
    assert "state_space_diffusion_sigma_cm_sqrt_s_values:" in workflow
    assert 'default: "40 60 85 110"' in workflow
    assert "state_space_momentum_sigma_cm_sqrt_s_values:" in workflow
    assert "state_space_momentum_velocity_decay_values:" in workflow
    assert 'default: "0.8 0.9 0.95 0.98"' in workflow
    assert "state_space_momentum_predicted_candidate_top_k_values:" in workflow
    assert "Matrix has {len(rows)} jobs; reduce inputs to 256 or fewer" in workflow
    assert "--state-space-diffusion-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-velocity-decay" in workflow
    assert "--state-space-momentum-predicted-candidate-top-k" in workflow
    assert "state_space_momentum_predicted_candidate_top_k" in workflow
    assert "MAX_STEP_SIGMA_VALUES" in workflow
    assert "state_space_max_step_sigma" in workflow
    assert 'f"step{slug(max_step_sigma)}-"' in workflow
    assert "state_space_valid_occupancy_threshold_s_values:" in workflow
    assert "VALID_OCCUPANCY_THRESHOLD_VALUES" in workflow
    assert 'f"occ{slug(valid_occupancy_threshold)}-"' in workflow
    assert "--state-space-valid-occupancy-threshold-s" in workflow
    assert "state_space_valid_occupancy_threshold_s" in workflow
    assert "TIME_BIN_S_VALUES" in workflow
    assert 'f"tb{slug(time_bin_s)}-"' in workflow
    assert "time_bin_s" in workflow
    assert "TIME_BIN_S: ${{ matrix.time_bin_s }}" in workflow
    assert "spike_rate_scale:" in workflow
    assert "SPIKE_RATE_SCALE_VALUES" in workflow
    assert 'f"rate{slug(spike_rate_scale)}-"' in workflow
    assert "--spike-rate-scale" in workflow
    assert "emission_likelihood_temperature_values:" in workflow
    assert "emission_negative_binomial_overdispersion_values:" in workflow
    assert "LIKELIHOOD_TEMPERATURE_VALUES" in workflow
    assert "NEGATIVE_BINOMIAL_OVERDISPERSION_VALUES" in workflow
    assert 'f"temp{slug(likelihood_temperature)}-"' in workflow
    assert 'f"nb{slug(negative_binomial_overdispersion)}-"' in workflow
    assert "--emission-likelihood-temperature" in workflow
    assert "--emission-negative-binomial-overdispersion" in workflow
    assert "state_space_evidence_sweep_config_ranked.csv" in workflow
    assert "state_space_evidence_sweep_momentum_ranked.csv" in workflow
    assert "scripts/marginalize_state_space_sweep.py" in workflow
    assert "state_space_marginalized_model_evidence_summary.csv" in workflow
    assert "state_space_marginalized_prior_weights.csv" in workflow
    assert "momentum_minus_diffusion_log_evidence" in workflow
    assert "MOMENTUM_CONFIDENT_LOG_EVIDENCE_THRESHOLD = 5.0" in workflow
    assert "momentum_beats_diffusion_log5_events" in workflow
    assert "momentum_ambiguous_vs_diffusion_log5_events" in workflow
    assert 'momentum_col = "sorted-spike-state-space-momentum-exact-sparse"' in workflow
    assert "pattern: state-space-evidence-sweep-*" in workflow
    assert _workflow_dispatch_input_count(workflow) <= 25


def _workflow_dispatch_input_count(workflow: str) -> int:
    in_inputs = False
    count = 0
    for line in workflow.splitlines():
        if line.strip() == "inputs:":
            in_inputs = True
            continue
        if in_inputs and line.startswith("permissions:"):
            break
        if in_inputs and line.startswith("      ") and not line.startswith("        "):
            count += 1
    return count
