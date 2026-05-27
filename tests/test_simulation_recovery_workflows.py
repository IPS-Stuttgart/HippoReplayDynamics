from pathlib import Path


def test_simulation_recovery_workflow_exposes_state_space_parameters():
    workflow = Path(".github/workflows/simulation-recovery.yml").read_text(encoding="utf-8")

    assert "sorted-spike-state-space-momentum-exact-sparse" in workflow
    assert "sorted-spike-state-space-first-order-imm" in workflow
    assert "state_space_diffusion_sigma_cm_sqrt_s:" in workflow
    assert "state_space_momentum_sigma_cm_sqrt_s:" in workflow
    assert "state_space_momentum_initial_sigma_cm_sqrt_s:" in workflow
    assert "--state-space-diffusion-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-initial-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-candidate-top-k" in workflow
    assert "--state-space-momentum-predicted-candidate-top-k" in workflow
    assert "max_synthetic_events:" in workflow
    assert "max_runtime_s:" in workflow
    assert "--checkpoint-output results/simulation-recovery" in workflow
    assert "--progress-log" in workflow


def test_simulation_recovery_sweep_workflow_defines_recovery_rankings():
    workflow = Path(".github/workflows/simulation-recovery-sweep.yml").read_text(encoding="utf-8")

    assert "name: Simulation recovery parameter sweep" in workflow
    assert 'default: "diffusion momentum"' in workflow
    assert (
        'default: "sorted-spike-state-space-diffusion '
        "sorted-spike-state-space-momentum-exact-sparse "
        "sorted-spike-state-space-momentum "
        "sorted-spike-state-space-first-order-imm "
        'sorted-spike-state-space-imm"'
    ) in workflow
    assert 'default: "6"' in workflow
    assert "state_space_diffusion_sigma_cm_sqrt_s_values:" in workflow
    assert 'default: "60 85 110"' in workflow
    assert "state_space_momentum_velocity_decay_values:" in workflow
    assert "state_space_momentum_predicted_candidate_top_k_values:" in workflow
    assert 'default: "0 4 8"' in workflow
    assert "state_space_momentum_predicted_candidate_top_k" in workflow
    assert "timeout-minutes: 360" in workflow
    assert "timeout-minutes: 350" in workflow
    assert "simulation_recovery_sweep_config_ranked.csv" in workflow
    assert "simulation_recovery_sweep_momentum_ranked.csv" in workflow
    assert "momentum_recovery_accuracy" in workflow
    assert "simulation_recovery_sweep_certified_vs_exact_summary.csv" in workflow
    assert "momentum_certified_vs_exact_recovery_accuracy" in workflow
    assert "simulation_recovery_diagnostic_event_table.csv" in workflow
    assert "simulation_recovery_sweep_diagnostic_summary.csv" in workflow
    assert "Matrix has {len(rows)} jobs; reduce inputs to 256 or fewer" in workflow
    assert "pattern: simulation-recovery-sweep-*" in workflow
    assert "max_synthetic_events:" in workflow
    assert "max_runtime_s:" in workflow
    assert "--checkpoint-output \"${out_dir}\"" in workflow
    assert "--progress-log" in workflow
    assert "simulation_recovery_partial_manifest.json" in workflow
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
