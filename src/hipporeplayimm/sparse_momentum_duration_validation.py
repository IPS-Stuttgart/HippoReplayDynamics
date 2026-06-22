"""Runtime guards for duration-dependent momentum helpers.

The exact sparse and finite-displacement momentum decoders derive both velocity
decay and relative velocity scaling from transition durations.  Non-finite or
non-positive values should fail at the helper boundary instead of producing
NaN/inf transition parameters that later corrupt the dynamic program.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def apply_sparse_momentum_duration_validation_patch() -> None:
    """Install duration validation on momentum duration helper functions."""

    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum

    _patch_duration_helpers(sparse_momentum)
    _patch_duration_helpers(displacement_momentum)

    # These IMM modules import the helper functions by value.  Keep their module
    # aliases synchronized even if they were imported before this runtime patch.
    import hipporeplayimm.state_space_displacement_imm as displacement_imm
    import hipporeplayimm.state_space_trajectory_imm as trajectory_imm

    trajectory_imm._coerce_transition_durations = sparse_momentum._coerce_transition_durations
    trajectory_imm._duration_adjusted_decays = sparse_momentum._duration_adjusted_decays
    trajectory_imm._time_scales = sparse_momentum._time_scales
    displacement_imm._coerce_transition_durations = displacement_momentum._coerce_transition_durations
    displacement_imm._duration_adjusted_decays = displacement_momentum._duration_adjusted_decays
    displacement_imm._time_scales = displacement_momentum._time_scales


def _patch_duration_helpers(module: Any) -> None:
    if getattr(module, "_duration_validation_patch_applied", False):
        return

    original_coerce_transition_durations = module._coerce_transition_durations
    original_duration_adjusted_decays = module._duration_adjusted_decays
    original_time_scales = module._time_scales

    @wraps(original_coerce_transition_durations)
    def coerce_transition_durations(values: Any, *, n_time: int, fallback_dt: float) -> np.ndarray:
        fallback = float(fallback_dt)
        if not np.isfinite(fallback) or fallback <= 0.0:
            raise ValueError("fallback dt must be finite and positive")
        durations = np.asarray(list(values), dtype=float)
        expected = max(int(n_time) - 1, 0)
        if durations.shape == (expected,) and durations.size:
            _valid_transition_durations(durations)
        return original_coerce_transition_durations(
            durations,
            n_time=n_time,
            fallback_dt=fallback,
        )

    @wraps(original_duration_adjusted_decays)
    def duration_adjusted_decays(config: object, durations: Any, reference_dt: float) -> np.ndarray:
        return original_duration_adjusted_decays(
            config,
            _valid_transition_durations(durations),
            reference_dt,
        )

    @wraps(original_time_scales)
    def time_scales(durations: Any) -> np.ndarray:
        return original_time_scales(_valid_transition_durations(durations))

    module._coerce_transition_durations = coerce_transition_durations
    module._duration_adjusted_decays = duration_adjusted_decays
    module._time_scales = time_scales
    module._duration_validation_patch_applied = True


def _valid_transition_durations(durations: Any) -> np.ndarray:
    values = np.asarray(durations, dtype=float)
    if values.ndim != 1:
        raise ValueError("transition durations must be one-dimensional")
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    return values
