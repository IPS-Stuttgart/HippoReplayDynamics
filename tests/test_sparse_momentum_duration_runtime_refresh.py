from __future__ import annotations

import importlib

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import duration_occupancy, state_space_model, state_space_sparse_momentum


def test_runtime_patches_restore_sparse_duration_validation_after_reload() -> None:
    """A stale module sentinel must not leave reloaded duration helpers permissive."""

    sparse_momentum = importlib.reload(state_space_sparse_momentum)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="integer count"):
        sparse_momentum._coerce_transition_durations(
            [],
            n_time=2.5,
            fallback_dt=0.01,
        )
    with pytest.raises(TypeError, match="transition durations must be numeric, not boolean"):
        sparse_momentum._coerce_transition_durations(
            [True],
            n_time=2,
            fallback_dt=0.01,
        )


def test_runtime_patches_restore_duration_time_scale_guard_after_helper_replacement() -> None:
    """Replacing a wrapped helper must be repairable even when its module flag survives."""

    wrapped = duration_occupancy._time_scales
    original = getattr(wrapped, "__hipporeplayimm_original__")
    duration_occupancy._time_scales = original

    hipporeplayimm.apply_runtime_patches()

    durations = np.array([np.finfo(float).tiny, np.finfo(float).max], dtype=float)
    with pytest.raises(ValueError, match="momentum time scales must be finite and positive"):
        duration_occupancy._time_scales(durations)

    refreshed = duration_occupancy._time_scales
    hipporeplayimm.apply_runtime_patches()
    assert duration_occupancy._time_scales is refreshed


def test_runtime_patches_restore_prediction_multiplier_guard_after_helper_replacement() -> None:
    """Candidate augmentation must recover its finite-output guard after replacement."""

    # Other reload-focused tests may intentionally leave state_space_model with
    # its freshly defined source helper while module-level patch flags survive.
    # Start through the supported public refresh path so this regression is
    # independent of test order, then recreate only the stale-helper condition.
    hipporeplayimm.apply_runtime_patches()
    wrapped = state_space_model._momentum_prediction_multipliers
    original = getattr(wrapped, "__hipporeplayimm_original__")
    state_space_model._momentum_prediction_multipliers = original

    hipporeplayimm.apply_runtime_patches()

    durations = np.array([np.finfo(float).tiny, np.finfo(float).max], dtype=float)
    config = state_space_model.StateSpaceDecoderConfig(momentum_velocity_decay_tau_s=1.0)
    with pytest.raises(
        ValueError,
        match="momentum prediction multipliers must be finite and nonnegative",
    ):
        state_space_model._momentum_prediction_multipliers(
            config,
            durations,
            fallback_dt=0.01,
        )

    refreshed = state_space_model._momentum_prediction_multipliers
    hipporeplayimm.apply_runtime_patches()
    assert state_space_model._momentum_prediction_multipliers is refreshed
