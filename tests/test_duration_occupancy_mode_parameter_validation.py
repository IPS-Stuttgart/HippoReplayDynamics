from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.duration_occupancy as duration_occupancy
from hipporeplayimm import state_space as ss


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_duration_occupancy_rejects_boolean_imm_mode_stickiness(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()
    durations = np.array([0.002, 0.003], dtype=float)

    with pytest.raises(TypeError, match="imm_mode_stickiness"):
        duration_occupancy._mode_transition_matrices(ss, 3, value, 0.0, durations)

    with pytest.raises(TypeError, match="imm_mode_stickiness"):
        duration_occupancy._resolve_mode_transitions(ss, 3, value, None, len(durations))


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_duration_occupancy_rejects_boolean_imm_switch_tau(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()
    durations = np.array([0.002, 0.003], dtype=float)

    with pytest.raises(TypeError, match="imm_switch_tau_s"):
        duration_occupancy._mode_transition_matrices(ss, 3, 0.9, value, durations)
