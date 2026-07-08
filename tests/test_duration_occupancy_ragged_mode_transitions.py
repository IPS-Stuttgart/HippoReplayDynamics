from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import state_space as ss
from hipporeplayimm.duration_occupancy import _mode_transition_matrices
from hipporeplayimm.duration_occupancy_mode_transition_validation import (
    _validate_mode_transition_sequence,
)


def test_custom_duration_imm_mode_transition_rejects_ragged_matrix() -> None:
    ragged_transition = [
        [0.8, 0.2],
        [0.1],
    ]

    with pytest.raises(ValueError, match="rectangular numeric probability matrix"):
        _validate_mode_transition_sequence(
            [ragged_transition],
            n_modes=2,
            n_transitions=1,
        )


@pytest.mark.parametrize("bad_stickiness", [True, np.bool_(False), "0.9", b"0.9", np.str_("0.9")])
def test_generated_duration_imm_mode_transitions_reject_lossy_stickiness_scalars(
    bad_stickiness: object,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="mode_stickiness"):
        _mode_transition_matrices(
            ss,
            3,
            bad_stickiness,  # type: ignore[arg-type]
            0.0,
            np.array([0.002, 0.003]),
        )


@pytest.mark.parametrize("bad_tau", [True, np.bool_(False), "0.05", b"0.05", np.str_("0.05")])
def test_generated_duration_imm_mode_transitions_reject_lossy_tau_scalars(
    bad_tau: object,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="imm_switch_tau_s"):
        _mode_transition_matrices(
            ss,
            3,
            0.9,
            bad_tau,  # type: ignore[arg-type]
            np.array([0.002, 0.003]),
        )
