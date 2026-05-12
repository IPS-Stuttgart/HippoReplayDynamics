from pathlib import Path


def test_simulation_recovery_workflow_exposes_state_space_parameters():
    workflow = Path(".github/workflows/simulation-recovery.yml").read_text(encoding="utf-8")

    assert "state_space_diffusion_sigma_cm_sqrt_s:" in workflow
    assert "state_space_momentum_sigma_cm_sqrt_s:" in workflow
    assert "state_space_momentum_initial_sigma_cm_sqrt_s:" in workflow
    assert "--state-space-diffusion-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-initial-sigma-cm-sqrt-s" in workflow
    assert "--state-space-momentum-candidate-top-k" in workflow


def test_simulation_recovery_sweep_workflow_defines_recovery_rankings():
    workflow = Path(".github/workflows/simulation-recovery-sweep.yml").read_text(encoding="utf-8")

    assert "name: Simulation recovery parameter sweep" in workflow
    assert 'default: "diffusion momentum"' in workflow
    assert 'default: "6"' in workflow
    assert "state_space_diffusion_sigma_cm_sqrt_s_values:" in workflow
    assert 'default: "60 85 110"' in workflow
    assert "state_space_momentum_velocity_decay_values:" in workflow
    assert "simulation_recovery_sweep_config_ranked.csv" in workflow
    assert "simulation_recovery_sweep_momentum_ranked.csv" in workflow
    assert "momentum_recovery_accuracy" in workflow
    assert "Matrix has {len(rows)} jobs; reduce inputs to 256 or fewer" in workflow
    assert "pattern: simulation-recovery-sweep-*" in workflow
