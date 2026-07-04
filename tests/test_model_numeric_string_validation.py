import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma_cm": "12.0"}, "sigma_cm"),
        ({"max_step_sigma": np.asarray("3.0")}, "max_step_sigma"),
    ],
)
def test_diffusion_model_rejects_string_numeric_parameters(kwargs, message):
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=message):
        DiffusionModel(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stationary_sigma_cm": "2.0"}, "stationary_sigma_cm"),
        ({"diffusion_sigma_cm": np.asarray("12.0")}, "diffusion_sigma_cm"),
        ({"momentum_sigma_cm": "12.0"}, "momentum_sigma_cm"),
        ({"velocity_decay": "0.95"}, "velocity_decay"),
        ({"mode_stickiness": np.asarray("0.94")}, "mode_stickiness"),
    ],
)
def test_candidate_kinematic_model_rejects_string_numeric_parameters(kwargs, message):
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=message):
        CandidateKinematicModel(**kwargs)


def test_model_numeric_parameters_still_accept_numeric_scalars():
    hipporeplayimm.apply_runtime_patches()

    diffusion = DiffusionModel(sigma_cm=np.float64(12.0), max_step_sigma=np.asarray(3.0))
    candidate = CandidateKinematicModel(
        stationary_sigma_cm=np.float64(2.0),
        diffusion_sigma_cm=np.asarray(12.0),
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=np.float64(0.94),
    )

    assert diffusion.sigma_cm == np.float64(12.0)
    assert candidate.velocity_decay == 0.95
