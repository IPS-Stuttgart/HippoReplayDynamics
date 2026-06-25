"""Runtime guard for occupancy-masked emission metadata isolation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import operator
from typing import Any, Callable

import numpy as np


def apply_duration_occupancy_metadata_guard_patch() -> None:
    """Ensure derived duration/occupancy helper inputs stay isolated and valid."""

    from . import duration_occupancy as _duration_occupancy
    from . import state_space_utils as _state_space_utils

    _apply_transition_duration_validation()

    if getattr(_duration_occupancy, "_metadata_guard_patch_applied", False):
        return

    previous_candidate_selection = _duration_occupancy._candidate_selection_emissions
    if not hasattr(_duration_occupancy, "_uniform_probabilities"):
        _duration_occupancy._uniform_probabilities = _state_space_utils._uniform_probabilities
    previous_uniform_probabilities = _duration_occupancy._uniform_probabilities

    def _candidate_selection_emissions(emissions, valid_bin_mask):
        restricted = previous_candidate_selection(emissions, valid_bin_mask)
        if restricted is emissions:
            return restricted
        metadata = dict(getattr(restricted, "metadata", {}))
        return replace(restricted, metadata=metadata)

    def _uniform_probabilities(n_bins: int, valid_bin_mask=None):
        return previous_uniform_probabilities(_positive_integer_bin_count(n_bins), valid_bin_mask)

    _candidate_selection_emissions.__name__ = previous_candidate_selection.__name__
    _candidate_selection_emissions.__doc__ = previous_candidate_selection.__doc__
    _uniform_probabilities.__name__ = previous_uniform_probabilities.__name__
    _uniform_probabilities.__doc__ = previous_uniform_probabilities.__doc__

    _duration_occupancy._candidate_selection_emissions = _candidate_selection_emissions
    _duration_occupancy._uniform_probabilities = _uniform_probabilities
    _duration_occupancy._metadata_guard_patch_applied = True


def _positive_integer_bin_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_bins must be a positive integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError("n_bins must be a positive integer") from exc
    if count <= 0:
        raise ValueError("n_bins must be a positive integer")
    return int(count)


def _apply_transition_duration_validation() -> None:
    from . import state_space_displacement_imm as displacement_imm
    from . import state_space_displacement_momentum as displacement_momentum
    from . import state_space_sparse_momentum as sparse_momentum
    from . import state_space_trajectory_imm as trajectory_imm

    sparse_decay = _validated_decay_helper(sparse_momentum._duration_adjusted_decays)
    displacement_decay = _validated_decay_helper(displacement_momentum._duration_adjusted_decays)

    if not getattr(sparse_momentum, "_transition_duration_validation_patch_applied", False):
        sparse_momentum._coerce_transition_durations = _coerce_transition_durations
        sparse_momentum._duration_adjusted_decays = sparse_decay
        sparse_momentum._transition_duration_validation_patch_applied = True

    if not getattr(trajectory_imm, "_transition_duration_validation_patch_applied", False):
        trajectory_imm._coerce_transition_durations = _coerce_transition_durations
        trajectory_imm._duration_adjusted_decays = sparse_decay
        trajectory_imm._transition_duration_validation_patch_applied = True

    if not getattr(displacement_momentum, "_transition_duration_validation_patch_applied", False):
        displacement_momentum._coerce_transition_durations = _coerce_transition_durations
        displacement_momentum._duration_adjusted_decays = displacement_decay
        displacement_momentum._transition_duration_validation_patch_applied = True

    if not getattr(displacement_imm, "_transition_duration_validation_patch_applied", False):
        displacement_imm._coerce_transition_durations = _coerce_transition_durations
        displacement_imm._duration_adjusted_decays = displacement_decay
        displacement_imm._transition_duration_validation_patch_applied = True


def _coerce_transition_durations(
    values: Iterable[float],
    *,
    n_time: int,
    fallback_dt: float,
) -> np.ndarray:
    expected = max(int(n_time) - 1, 0)
    raw_values = list(values)
    if len(raw_values) == 0:
        dt = _positive_finite_scalar("fallback dt", fallback_dt)
        return np.full(expected, dt, dtype=float)

    out = np.asarray(raw_values, dtype=float)
    if out.ndim != 1:
        raise ValueError("transition durations must be one-dimensional")
    _validate_transition_durations(out)
    if out.shape != (expected,):
        raise ValueError(
            "transition durations must contain one finite positive value per transition; "
            f"expected shape {(expected,)}, got {out.shape}"
        )
    return out


def _validated_decay_helper(helper: Callable[[Any, np.ndarray, float], np.ndarray]):
    if getattr(helper, "_transition_duration_validation_wrapped", False):
        return helper

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
