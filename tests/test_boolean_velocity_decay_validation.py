from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleIMMModel
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, _momentum_velocity_decays


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stationary_velocity_decay": True},
        {"diffusion_velocity_decay": np.bool_(False)},
        {"momentum_velocity_decay": True},
        {"jump_velocity_decay": np.bool_(False)},
    ],
)
def test_pyrecest_imm_rejects_boolean_velocity_decays(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        PyRecEstGoalParticleIMMModel(**kwargs)


@pytest.mark.parametrize("bad_decay", [True, np.bool_(False)])
def test_state_space_rejects_boolean_scalar_velocity_decay(bad_decay: object) -> None:
    hipporeplayimm.apply_runtime_patches()
    config = StateSpaceDecoderConfig(momentum_velocity_decay=bad_decay)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not boolean"):
        _momentum_velocity_decays(config, np.array([0.01, 0.02], dtype=float))
