from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_model_evidence_improved.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("benchmark_model_evidence_improved_registry", _SCRIPT)
assert _SPEC is not None
benchmark_model_evidence_improved = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(benchmark_model_evidence_improved)


def _args(models: str) -> argparse.Namespace:
    return argparse.Namespace(
        models=models,
        include_clusterless_defaults=False,
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_imm_switch_tau_s=0.0,
        time_bin_s=0.003,
        state_space_trajectory_imm_mode_stickiness=None,
        state_space_trajectory_imm_momentum_initial_probability=None,
        state_space_trajectory_imm_momentum_switch_probability=None,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_velocity_decay_tau_s=0.0,
        state_space_momentum_candidate_top_k=17,
        state_space_momentum_candidate_mass_threshold=None,
        state_space_momentum_candidate_min_k=1,
        state_space_momentum_candidate_max_k=0,
        state_space_momentum_predicted_candidate_top_k=5,
        state_space_momentum_candidate_source="emission",
        state_space_displacement_radius_bins=2,
        state_space_displacement_position_sigma_cm=0.0,
        state_space_displacement_transition_sigma_cm_sqrt_s=0.0,
        state_space_displacement_prior_sigma_cm=0.0,
        state_space_valid_occupancy_threshold_s=0.0,
        clusterless_mark_likelihood="local-kde",
        goal_state_space_transition_sigma_cm_sqrt_s=85.0,
        goal_state_space_drift_speed_cm_s=400.0,
        goal_state_space_max_step_sigma=4.0,
        valid_state_min_occupancy_s=0.02,
        valid_state_top_occupancy_fraction=None,
        valid_state_sigma_cm=5.0,
        valid_state_max_step_sigma=4.0,
        valid_state_grid_diagonal_neighbors=True,
        valid_state_grid_stay_probability=0.0,
    )


def test_improved_model_evidence_accepts_trajectory_imm_and_displacement_variants() -> None:
    args = _args(
        "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse "
        "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse "
        "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse "
        "sorted-spike-state-space-displacement-momentum "
        "clusterless-state-space-displacement-momentum"
    )

    models = benchmark_model_evidence_improved._models(
        args, SimpleNamespace(session_id="Rat1/Open1"), encoding=None
    )

    assert list(models) == [
        "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse",
        "sorted-spike-state-space-displacement-momentum",
        "clusterless-state-space-displacement-momentum",
    ]
    assert (
        models[
            "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse"
        ].config.trajectory_imm_momentum_initial_probability
        == 0.05
    )
    assert (
        models[
            "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse"
        ].config.trajectory_imm_momentum_switch_probability
        == 0.001
    )
    assert (
        models[
            "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse"
        ].config.trajectory_imm_mode_stickiness
        == 0.985
    )
    assert models["sorted-spike-state-space-displacement-momentum"].mode == "displacement-momentum"
    assert models["clusterless-state-space-displacement-momentum"].mode == "displacement-momentum"
