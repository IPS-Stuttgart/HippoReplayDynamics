"""Validate transition-duration metadata for exact finite-state decoders.

The exact sparse momentum, trajectory-IMM, and finite-displacement decoders use
transition durations to scale diffusion widths and velocity decay.  Legacy helper
functions fell back to uniform ``dt`` whenever duration metadata had the expected
length but contained non-finite or non-positive values, which can silently hide
corrupt event timing.  This patch keeps legacy length-mismatch fallback behavior
while rejecting corrupt supplied durations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np


def apply_transition_duration_validation_patch() -> None:
    """Reject non-finite or non-positive transition durations in exact decoders."""

    from . import state_space_displacement_imm as displacement_imm
    from . import state_space_displacement_momentum as displacement_momentum
    from . import state_space_sparse_momentum as sparse_momentum
    from . import state_space_trajectory_imm as trajectory_imm

    if getattr(sparse_momentum, "_transition_duration_validation_patch_applied", False):
        return

    sparse_decay = _validated_decay_helper(sparse_momentum._duration_adjusted_decays)
    displacement_decay = _validated_decay_helper(displacement_momentum._duration_adjusted_decays)

    sparse_momentum._coerce_transition_durations = _coerce_transition_durations
    sparse_momentum._duration_adjusted_decays = sparse_decay
    trajectory_imm._coerce_transition_durations = _coerce_transition_durations
    trajectory_imm._duration_adjusted_decays = sparse_decay

    displacement_momentum._coerce_transition_durations = _coerce_transition_durations
    displacement_momentum._duration_adjusted_decays = displacement_decay
    displacement_imm._coerce_transition_durations = _coerce_transition_durations
    displacement_imm._duration_adjusted_decays = displacement_decay

    sparse_momentum._transition_duration_validation_patch_applied = True
    displacement_momentum._transition_duration_validation_patch_applied = True
    trajectory_imm._transition_duration_validation_patch_applied = True
    displacement_imm._transition_duration_validation_patch_applied = True


def _coerce_transition_durations(
    values: Iterable[float],
    *,
    n_time: int,
    fallback_dt: float,
) -> np.ndarray:
    expected = max(int(n_time) - 1, 0)
    out = np.asarray(list(values), dtype=float)
    if out.shape != (expected,):
        dt = _positive_finite_scalar("fallback dt", fallback_dt)
        return np.full(expected, dt, dtype=float)
    _validate_transition_durations(out)
    return out


def _validated_decay_helper(helper: Callable[[Any, np.ndarray, float], np.ndarray]):
    def duration_adjusted_decays(config: Any, durations: np.ndarray, reference_dt: float) -> np.ndarray:
        durations = np.asarray(durations, dtype=float)
        _validate_transition_durations(durations)
        return helper(config, durations, reference_dt)

    duration_adjusted_decays._transition_duration_validation_wrapped = True  # type: ignore[attr-defined]
    return duration_adjusted_decays


def _validate_transition_durations(durations: np.ndarray) -> None:
    values = np.asarray(durations, dtype=float)
    if values.size == 0:
        return
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("transition durations must be finite and positive")


def _positive_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar
