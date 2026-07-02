from __future__ import annotations

import numpy as np

from hipporeplayimm import state_space_trajectory_imm as trajectory_imm
from hipporeplayimm.model_parameter_validation import apply_model_parameter_validation_patch
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig


def test_trajectory_imm_tau_scales_explicit_momentum_switch_probability() -> None:
    apply_model_parameter_validation_patch()
    base_stickiness = 0.9
    momentum_switch = 0.04
    tau_s = 1.0
    config = StateSpaceDecoderConfig(
        mode="trajectory-imm-exact-sparse",
        trajectory_imm_mode_stickiness=base_stickiness,
        trajectory_imm_momentum_initial_probability=0.25,
        trajectory_imm_momentum_switch_probability=momentum_switch,
        imm_switch_tau_s=tau_s,
    )
    durations = np.asarray([0.01, 0.10, 0.50], dtype=float)

    matrices = trajectory_imm._trajectory_imm_mode_transition_matrices(config, base_stickiness, durations)

    assert len(matrices) == durations.size
    switch_fraction = momentum_switch / (1.0 - base_stickiness)
    for matrix, duration in zip(matrices, durations, strict=True):
        step_stickiness = float(np.exp(-float(duration) / tau_s))
        step_remaining = 1.0 - step_stickiness
        expected_momentum_switch = switch_fraction * step_remaining

        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]))
        np.testing.assert_allclose(
            np.diag(matrix),
            np.full(matrix.shape[0], step_stickiness),
        )
        np.testing.assert_allclose(
            matrix[: trajectory_imm._FIRST_ORDER_MODE_COUNT, trajectory_imm._MOMENTUM_MODE_INDEX],
            expected_momentum_switch,
        )
        np.testing.assert_allclose(
            matrix[trajectory_imm._MOMENTUM_MODE_INDEX, : trajectory_imm._FIRST_ORDER_MODE_COUNT],
            step_remaining / trajectory_imm._FIRST_ORDER_MODE_COUNT,
        )
