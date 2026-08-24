from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm.state_space as state_space
from hipporeplayimm.continuous_time_imm_transition_patch import (
    _continuous_time_mode_transition_matrix,
)
from hipporeplayimm.duration_occupancy import (
    _mode_transition_matrices as duration_mode_transition_matrices,
)
from hipporeplayimm.state_space_displacement_imm import (
    _mode_transition_matrices as displacement_mode_transition_matrices,
)
from hipporeplayimm.state_space_trajectory_imm import (
    _trajectory_imm_mode_transition_matrices,
    _trajectory_imm_mode_transition_matrix,
)
from hipporeplayimm.state_space_utils import _mode_transition_matrix


def _assert_semigroup(factory, *, half_duration_s: float) -> None:
    half = factory(half_duration_s)
    full = factory(2.0 * half_duration_s)
    np.testing.assert_allclose(half @ half, full, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-13)
    assert np.all(full >= 0.0)


def test_continuous_time_mode_embedding_has_semigroup_and_stationary_limit() -> None:
    n_modes = 4
    tau_s = 0.1
    duration_s = 0.1
    base = _mode_transition_matrix(n_modes, 0.95)

    half = _continuous_time_mode_transition_matrix(base, duration_s / 2.0, tau_s)
    full = _continuous_time_mode_transition_matrix(base, duration_s, tau_s)
    np.testing.assert_allclose(half @ half, full, rtol=1.0e-12, atol=1.0e-12)

    expected_diagonal = 1.0 / n_modes + (1.0 - 1.0 / n_modes) * np.exp(
        -n_modes * duration_s / ((n_modes - 1) * tau_s)
    )
    np.testing.assert_allclose(
        np.diag(full),
        expected_diagonal,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    long_interval = _continuous_time_mode_transition_matrix(
        base,
        20.0 * tau_s,
        tau_s,
    )
    np.testing.assert_allclose(
        long_interval,
        np.full((n_modes, n_modes), 1.0 / n_modes),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_continuous_time_mode_embedding_rejects_lossy_transition_coercions() -> None:
    invalid_transitions = [
        np.array([[0.9 + 0.1j, 0.1 - 0.1j], [0.2, 0.8]], dtype=complex),
        np.array([[True, False], [False, True]], dtype=bool),
        np.array([["0.9", "0.1"], ["0.2", "0.8"]], dtype=str),
    ]

    for transition in invalid_transitions:
        with pytest.raises(ValueError, match="base_transition must contain real numeric values"):
            _continuous_time_mode_transition_matrix(transition, 0.05, 0.1)


def test_continuous_time_mode_embedding_rejects_nonreal_scalar_parameters() -> None:
    base = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)

    for duration in (True, "0.05", 0.05 + 0.01j):
        with pytest.raises(TypeError, match="duration_s must be a real finite scalar"):
            _continuous_time_mode_transition_matrix(base, duration, 0.1)

    for dwell in (True, "0.1", 0.1 + 0.01j):
        with pytest.raises(TypeError, match="mean_dwell_s must be a real finite scalar"):
            _continuous_time_mode_transition_matrix(base, 0.05, dwell)


def test_duration_aware_imm_wrappers_reject_complex_durations() -> None:
    durations = np.array([0.025 + 0.01j], dtype=complex)

    with pytest.raises(ValueError, match="durations must contain real numeric values"):
        duration_mode_transition_matrices(
            state_space,
            4,
            0.95,
            0.1,
            durations,
        )
    with pytest.raises(ValueError, match="durations must contain real numeric values"):
        displacement_mode_transition_matrices(
            4,
            0.95,
            0.1,
            durations,
        )

    trajectory_config = SimpleNamespace(
        imm_switch_tau_s=0.1,
        trajectory_imm_momentum_switch_probability=None,
    )
    with pytest.raises(ValueError, match="durations must contain real numeric values"):
        _trajectory_imm_mode_transition_matrices(
            trajectory_config,
            0.95,
            durations,
        )


def test_all_duration_aware_imm_families_are_semigroup_consistent() -> None:
    tau_s = 0.1
    half_duration_s = 0.025

    _assert_semigroup(
        lambda duration: duration_mode_transition_matrices(
            state_space,
            4,
            0.95,
            tau_s,
            np.asarray([duration]),
        )[0],
        half_duration_s=half_duration_s,
    )
    _assert_semigroup(
        lambda duration: displacement_mode_transition_matrices(
            4,
            0.95,
            tau_s,
            np.asarray([duration]),
        )[0],
        half_duration_s=half_duration_s,
    )

    trajectory_config = SimpleNamespace(
        imm_switch_tau_s=tau_s,
        trajectory_imm_momentum_switch_probability=None,
    )
    _assert_semigroup(
        lambda duration: _trajectory_imm_mode_transition_matrices(
            trajectory_config,
            0.95,
            np.asarray([duration]),
        )[0],
        half_duration_s=half_duration_s,
    )


def test_custom_trajectory_imm_switch_pattern_is_semigroup_consistent() -> None:
    config = SimpleNamespace(
        imm_switch_tau_s=0.1,
        trajectory_imm_momentum_switch_probability=0.04,
    )

    _assert_semigroup(
        lambda duration: _trajectory_imm_mode_transition_matrices(
            config,
            0.9,
            np.asarray([duration]),
        )[0],
        half_duration_s=0.025,
    )


def test_custom_trajectory_imm_uses_duration_invariant_reference_routing() -> None:
    config = SimpleNamespace(
        imm_switch_tau_s=0.1,
        trajectory_imm_momentum_switch_probability=0.04,
    )
    duration_s = 0.001

    observed = _trajectory_imm_mode_transition_matrices(
        config,
        0.9,
        np.asarray([duration_s]),
    )[0]
    reference_transition = _trajectory_imm_mode_transition_matrix(config, 0.9)
    expected = _continuous_time_mode_transition_matrix(
        reference_transition,
        duration_s,
        config.imm_switch_tau_s,
    )

    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)
