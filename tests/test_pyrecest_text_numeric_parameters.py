from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleIMMModel, PyRecEstGoalParticleModel


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_velocity_sigma_cm_s", "120.0", "initial_velocity_sigma_cm_s must be finite and positive"),
        ("process_noise_sigma_cm_s", np.str_("60.0"), "process_noise_sigma_cm_s must be finite and positive"),
        ("position_jump_sigma_cm", "25.0", "position_jump_sigma_cm must be finite and positive"),
        ("jump_probability", "0.03", "jump_probability must lie in \\[0, 1\\]"),
        ("goal_reset_probability", "0.02", "goal_reset_probability must lie in \\[0, 1\\]"),
        ("position_proposal_probability", "0.5", "position_proposal_probability must lie in \\[0, 1\\]"),
        ("position_proposal_ess_threshold", "0.5", "position_proposal_ess_threshold must lie in \\[0, 1\\]"),
    ],
)
def test_pyrecest_goal_particle_rejects_text_numeric_parameters(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PyRecEstGoalParticleModel(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode_stickiness", "0.95", "mode_stickiness must lie in \\[0, 1\\]"),
        ("stationary_velocity_decay", "0.0", "stationary_velocity_decay must lie in \\[0, 1\\]"),
        ("diffusion_velocity_decay", "0.0", "diffusion_velocity_decay must lie in \\[0, 1\\]"),
        ("momentum_velocity_decay", "0.95", "momentum_velocity_decay must lie in \\[0, 1\\]"),
        ("jump_fraction", "0.9", "jump_fraction must lie in \\[0, 1\\]"),
        ("jump_velocity_decay", np.array("0.25"), "jump_velocity_decay must lie in \\[0, 1\\]"),
    ],
)
def test_pyrecest_goal_particle_imm_rejects_text_numeric_parameters(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PyRecEstGoalParticleIMMModel(**{field: value})
