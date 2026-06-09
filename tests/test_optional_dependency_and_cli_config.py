from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tomllib

from hipporeplayimm.cli import _state_space_config_from_recovery_args
from hipporeplayimm.state_space import StateSpaceDecoderConfig


ROOT = Path(__file__).resolve().parents[1]


def _recovery_args(**overrides: object) -> Namespace:
    defaults = StateSpaceDecoderConfig()
    values: dict[str, object] = {
        "state_space_sigma_cm_sqrt_s": 85.0,
        "state_space_stationary_sigma_cm": defaults.stationary_sigma_cm,
        "state_space_diffusion_sigma_cm_sqrt_s": None,
        "state_space_max_step_sigma": defaults.max_step_sigma,
        "state_space_imm_mode_stickiness": defaults.imm_mode_stickiness,
        "state_space_trajectory_imm_mode_stickiness": defaults.trajectory_imm_mode_stickiness,
        "state_space_trajectory_imm_momentum_initial_probability": (
            defaults.trajectory_imm_momentum_initial_probability
        ),
        "state_space_trajectory_imm_momentum_switch_probability": (
            defaults.trajectory_imm_momentum_switch_probability
        ),
        "state_space_momentum_sigma_cm_sqrt_s": None,
        "state_space_momentum_initial_sigma_cm_sqrt_s": None,
        "state_space_momentum_velocity_decay": defaults.momentum_velocity_decay,
        "state_space_momentum_velocity_decay_tau_s": defaults.momentum_velocity_decay_tau_s,
        "state_space_momentum_candidate_top_k": defaults.momentum_candidate_top_k,
        "state_space_momentum_candidate_mass_threshold": defaults.momentum_candidate_mass_threshold,
        "state_space_momentum_candidate_min_k": defaults.momentum_candidate_min_k,
        "state_space_momentum_candidate_max_k": defaults.momentum_candidate_max_k,
        "state_space_momentum_predicted_candidate_top_k": (
            defaults.momentum_predicted_candidate_top_k
        ),
        "state_space_momentum_candidate_source": defaults.momentum_candidate_source,
        "state_space_displacement_radius_bins": defaults.displacement_radius_bins,
        "state_space_displacement_position_sigma_cm": (
            defaults.displacement_position_sigma_cm
        ),
        "state_space_displacement_transition_sigma_cm_sqrt_s": (
            defaults.displacement_transition_sigma_cm_sqrt_s
        ),
        "state_space_displacement_prior_sigma_cm": defaults.displacement_prior_sigma_cm,
        "state_space_valid_occupancy_threshold_s": (
            defaults.valid_occupancy_threshold_s
        ),
    }
    values.update(overrides)
    return Namespace(**values)


def test_pyrecest_is_not_a_core_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core_dependencies = tuple(pyproject["project"]["dependencies"])
    pyrecest_extra = tuple(pyproject["project"]["optional-dependencies"]["pyrecest"])

    assert not any(_dependency_name(dependency) == "pyrecest" for dependency in core_dependencies)
    assert any(_dependency_name(dependency) == "pyrecest" for dependency in pyrecest_extra)


def test_recovery_state_space_config_uses_shared_sigma_only_for_missing_specific_sigmas() -> None:
    config = _state_space_config_from_recovery_args(
        _recovery_args(state_space_sigma_cm_sqrt_s=123.0)
    )

    assert config.diffusion_sigma_cm_sqrt_s == 123.0
    assert config.momentum_sigma_cm_sqrt_s == 123.0
    assert config.momentum_initial_sigma_cm_sqrt_s == 123.0


def test_recovery_state_space_config_preserves_explicit_zero_sigma_overrides() -> None:
    config = _state_space_config_from_recovery_args(
        _recovery_args(
            state_space_sigma_cm_sqrt_s=123.0,
            state_space_diffusion_sigma_cm_sqrt_s=0.0,
            state_space_momentum_sigma_cm_sqrt_s=0.0,
            state_space_momentum_initial_sigma_cm_sqrt_s=0.0,
        )
    )

    assert config.diffusion_sigma_cm_sqrt_s == 0.0
    assert config.momentum_sigma_cm_sqrt_s == 0.0
    assert config.momentum_initial_sigma_cm_sqrt_s == 0.0


def _dependency_name(dependency: str) -> str:
    return dependency.split("@", 1)[0].strip().split(" ", 1)[0].lower()
