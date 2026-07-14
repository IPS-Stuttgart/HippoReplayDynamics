from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.pyrecest_models import (
    PyRecEstGoalParticleIMMModel,
    PyRecEstGoalParticleModel,
)


@pytest.mark.parametrize(
    ("model_type", "field", "value", "message"),
    [
        (
            PyRecEstGoalParticleModel,
            "initial_velocity_sigma_cm_s",
            b"120.0",
            "initial_velocity_sigma_cm_s must be finite and positive",
        ),
        (
            PyRecEstGoalParticleModel,
            "jump_probability",
            np.bytes_(b"0.03"),
            r"jump_probability must lie in \[0, 1\]",
        ),
        (
            PyRecEstGoalParticleModel,
            "goal_reset_probability",
            np.array(b"0.02"),
            r"goal_reset_probability must lie in \[0, 1\]",
        ),
        (
            PyRecEstGoalParticleIMMModel,
            "momentum_velocity_decay",
            np.array(b"0.95", dtype=object),
            r"momentum_velocity_decay must lie in \[0, 1\]",
        ),
    ],
)
def test_pyrecest_models_reject_byte_text_numeric_parameters(
    model_type: type[PyRecEstGoalParticleModel],
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        model_type(**{field: value})
