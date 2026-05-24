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
    assert "spike_rate_scale:" in workflow
    assert "--spike-rate-scale" in workflow
    assert "state_space_evidence_sweep_config_ranked.csv" in workflow
    assert "state_space_evidence_sweep_momentum_ranked.csv" in workflow
    assert "scripts/marginalize_state_space_sweep.py" in workflow
    assert "state_space_marginalized_model_evidence_summary.csv" in workflow
    assert "state_space_marginalized_prior_weights.csv" in workflow
    assert "momentum_minus_diffusion_log_evidence" in workflow
    assert 'momentum_col = "sorted-spike-state-space-momentum-exact-sparse"' in workflow
    assert "pattern: state-space-evidence-sweep-*" in workflow
