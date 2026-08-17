from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.duration_occupancy as duration_occupancy
import hipporeplayimm.state_space as state_space


def _momentum_config(*, decay: object = 0.95, tau_s: object = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        momentum_velocity_decay=decay,
        momentum_velocity_decay_tau_s=tau_s,
    )


def test_runtime_patches_restore_duration_occupancy_decay_input_validation_after_reload() -> None:
    module = importlib.reload(duration_occupancy)

    # importlib.reload() retains dynamically added module attributes, so this
    # stale sentinel survives even though the wrapped helper was redefined.
    assert getattr(module, "_duration_occupancy_velocity_decay_validation_patch_applied", False)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(
        TypeError,
        match=r"momentum_velocity_decay must be a numeric scalar, not boolean",
    ):
        module._duration_adjusted_decays(
            _momentum_config(decay=True),
            np.array([0.01], dtype=float),
            0.01,
        )

    with pytest.raises(
        TypeError,
        match=r"momentum_velocity_decay_tau_s must be a numeric scalar, not boolean",
    ):
        module._duration_adjusted_decays(
            _momentum_config(tau_s=True),
            np.array([0.01], dtype=float),
            0.01,
        )

    np.testing.assert_allclose(
        module._duration_adjusted_decays(
            _momentum_config(decay=0.9),
            np.array([0.01], dtype=float),
            0.01,
        ),
        np.array([0.9], dtype=float),
    )


def test_runtime_patches_restore_duration_occupancy_mode_parameter_validation_after_reload() -> None:
    module = importlib.reload(duration_occupancy)

    assert getattr(module, "_duration_occupancy_mode_parameter_validation_patch_applied", False)

    hipporeplayimm.apply_runtime_patches()

    durations = np.array([0.01], dtype=float)
    with pytest.raises(
        TypeError,
        match=r"imm_mode_stickiness must be a numeric scalar, not boolean",
    ):
        module._mode_transition_matrices(
            state_space,
            3,
            True,
            0.0,
            durations,
        )

    with pytest.raises(
        TypeError,
        match=r"imm_switch_tau_s must be a numeric scalar, not boolean",
    ):
        module._mode_transition_matrices(
            state_space,
            3,
            0.95,
            True,
            durations,
        )

    with pytest.raises(
        TypeError,
        match=r"imm_mode_stickiness must be a numeric scalar, not boolean",
    ):
        module._resolve_mode_transitions(
            state_space,
            3,
            True,
            None,
            1,
        )


def test_duration_occupancy_reload_refresh_is_idempotent() -> None:
    module = importlib.reload(duration_occupancy)
    hipporeplayimm.apply_runtime_patches()

    decay_helper = module._duration_adjusted_decays
    matrix_helper = module._mode_transition_matrices
    resolver = module._resolve_mode_transitions

    hipporeplayimm.apply_runtime_patches()

    assert module._duration_adjusted_decays is decay_helper
    assert module._mode_transition_matrices is matrix_helper
    assert module._resolve_mode_transitions is resolver
