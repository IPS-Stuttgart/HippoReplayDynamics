from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_cm": True},
        {"max_step_sigma": np.bool_(True)},
    ],
)
def test_diffusion_model_rejects_boolean_numeric_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        DiffusionModel(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stationary_sigma_cm": True},
        {"diffusion_sigma_cm": np.bool_(True)},
        {"momentum_sigma_cm": True},
        {"velocity_decay": False},
        {"mode_stickiness": np.bool_(False)},
    ],
)
def test_candidate_model_rejects_boolean_numeric_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        CandidateKinematicModel(**kwargs)
