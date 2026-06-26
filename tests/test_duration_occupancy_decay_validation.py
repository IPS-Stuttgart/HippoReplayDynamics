from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.duration_occupancy import _duration_adjusted_decays as _duration_occupancy_decays
from hipporeplayimm.state_space_displacement_momentum import (
    _duration_adjusted_decays as _displacement_decays,
)
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, _momentum_velocity_decays
from hipporeplayimm.state_space_sparse_momentum import (
    _duration_adjusted_decays as _sparse_decays,
)


def test_duration_occupancy_rejects_decay_above_unit_interval() -> None:
    hipporeplayimm.apply_runtime_patches()
    durations = np.array([0.01, 0.02], dtype=float)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _duration_occupancy_decays(1.01, durations, 0.01)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _duration_occupancy_decays(
            StateSpaceDecoderConfig(momentum_velocity_decay=1.01),
            durations,
            0.01,
        )


@pytest.mark.parametrize("bad_tau", [True, np.bool_(False)])
def test_momentum_decay_helpers_reject_boolean_tau(bad_tau: object) -> None:
    hipporeplayimm.apply_runtime_patches()
    durations = np.array([0.01, 0.02], dtype=float)
    config = StateSpaceDecoderConfig(momentum_velocity_decay_tau_s=bad_tau)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="momentum_velocity_decay_tau_s"):
        _momentum_velocity_decays(config, durations)

    with pytest.raises(TypeError, match="momentum_velocity_decay_tau_s"):
        _duration_occupancy_decays(config, durations, 0.01)

    with pytest.raises(TypeError, match="momentum_velocity_decay_tau_s"):
        _sparse_decays(config, durations, 0.01)

    with pytest.raises(TypeError, match="momentum_velocity_decay_tau_s"):
        _displacement_decays(config, durations, 0.01)


def test_duration_occupancy_rejects_boolean_scalar_decay() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="momentum_velocity_decay"):
        _duration_occupancy_decays(True, np.array([0.01], dtype=float), 0.01)
