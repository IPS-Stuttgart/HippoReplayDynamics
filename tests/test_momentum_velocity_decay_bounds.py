from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleIMMModel
from hipporeplayimm.state_space_displacement_momentum import (
    _duration_adjusted_decays as _displacement_duration_adjusted_decays,
)
from hipporeplayimm.state_space_model import (
    StateSpaceDecoderConfig,
    _momentum_prediction_multipliers,
    _momentum_velocity_decays,
)
from hipporeplayimm.state_space_sparse_momentum import (
    _duration_adjusted_decays as _sparse_momentum_duration_adjusted_decays,
)


def test_candidate_kinematic_model_rejects_amplifying_velocity_decay() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match=r"velocity_decay.*\[0, 1\]"):
        CandidateKinematicModel(velocity_decay=1.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stationary_velocity_decay": 1.01},
        {"diffusion_velocity_decay": 1.01},
        {"momentum_velocity_decay": 1.01},
        {"jump_velocity_decay": 1.01},
    ],
)
def test_pyrecest_imm_model_rejects_amplifying_velocity_decays(kwargs: dict[str, float]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match=r"velocity_decay.*\[0, 1\]"):
        PyRecEstGoalParticleIMMModel(**kwargs)


def test_state_space_decay_helpers_reject_amplifying_scalar_decay() -> None:
    hipporeplayimm.apply_runtime_patches()
    durations = np.array([0.01, 0.02], dtype=float)
    config = StateSpaceDecoderConfig(momentum_velocity_decay=1.01)

    with pytest.raises(ValueError, match=r"momentum_velocity_decay.*\[0, 1\]"):
        _momentum_velocity_decays(config, durations)
    with pytest.raises(ValueError, match=r"momentum_velocity_decay.*\[0, 1\]"):
        _momentum_prediction_multipliers(config, durations, fallback_dt=0.01)
    with pytest.raises(ValueError, match=r"momentum_velocity_decay.*\[0, 1\]"):
        _sparse_momentum_duration_adjusted_decays(config, durations, 0.01)
    with pytest.raises(ValueError, match=r"momentum_velocity_decay.*\[0, 1\]"):
        _displacement_duration_adjusted_decays(config, durations, 0.01)


def test_duration_tau_allows_unused_scalar_decay_above_one() -> None:
    hipporeplayimm.apply_runtime_patches()
    durations = np.array([0.01, 0.02], dtype=float)
    config = StateSpaceDecoderConfig(
        momentum_velocity_decay=1.01,
        momentum_velocity_decay_tau_s=1.0,
    )

    expected_decays = np.exp(-durations / 1.0)
    expected_multipliers = expected_decays * np.array([1.0, 2.0], dtype=float)

    np.testing.assert_allclose(_momentum_velocity_decays(config, durations), expected_decays)
    np.testing.assert_allclose(
        _momentum_prediction_multipliers(config, durations, fallback_dt=0.01),
        expected_multipliers,
    )
    np.testing.assert_allclose(
        _sparse_momentum_duration_adjusted_decays(config, durations, 0.01),
        expected_decays,
    )
    np.testing.assert_allclose(
        _displacement_duration_adjusted_decays(config, durations, 0.01),
        expected_decays,
    )
