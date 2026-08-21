"""Validate state-space per-bin sigma helper inputs.

State-space diffusion and momentum models convert noise specified in
``cm/sqrt(s)`` to a per-bin standard deviation. Python booleans are numeric
subclasses, and NumPy complex scalars can be coerced to floats by discarding the
imaginary component, so validate scalar types before float conversion. Keep the
guard at the shared helper boundary and at the duration-aware scorer's private
duplicate so direct and public import surfaces enforce the same scalar contract.
"""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any, Callable

import numpy as np

_STATE_SPACE_UTILS_PATCHED_FLAG = "_state_space_per_bin_sigma_validation_patch_applied"
_DURATION_OCCUPANCY_PATCHED_FLAG = "_duration_occupancy_per_bin_sigma_validation_patch_applied"
_MODE_TRANSITION_PATCHED_FLAG = "_state_space_mode_transition_string_validation_patch_applied"
_DURATION_MODE_TRANSITION_PATCHED_FLAG = (
    "_duration_occupancy_mode_transition_scalar_validation_patch_applied"
)
_PER_BIN_SIGMA_WRAPPER_VERSION = 2
_MAX_SCALAR_WRAPPER_DEPTH = 64


class _ComplexScalarError(TypeError):
    """Internal distinction for preserving public mode-parameter errors."""


def _coerce_real_numeric_scalar(name: str, value: Any) -> int | float:
    """Validate and normalize one possibly wrapped real numeric scalar.

    NumPy object scalars can recursively wrap other zero-dimensional arrays.
    Inspect every layer so booleans, numeric strings, complex values, and
    non-scalar arrays cannot hide inside object wrappers and later be accepted
    by ``float``. Return a plain Python scalar so downstream legacy helpers do
    not receive object arrays or ``Decimal`` values they cannot process.
    """

    current = value
    seen: set[int] = set()
    for _ in range(_MAX_SCALAR_WRAPPER_DEPTH):
        current_id = id(current)
        if current_id in seen:
            raise TypeError(f"{name} must be a numeric scalar")
        seen.add(current_id)

        try:
            raw = np.asarray(current)
        except (TypeError, ValueError, RecursionError) as exc:
            raise TypeError(f"{name} must be a numeric scalar") from exc
        if raw.ndim != 0:
            raise TypeError(f"{name} must be a numeric scalar")

        if raw.dtype == object:
            try:
                item = raw.item()
            except (TypeError, ValueError, RecursionError) as exc:
                raise TypeError(f"{name} must be a numeric scalar") from exc
            if item is current:
                # Decimal, Fraction, and similar scalar objects intentionally
                # have object dtype. Preserve them by normalizing once to the
                # float representation already required by the base helpers.
                # A self-referential object array is structural recursion.
                if isinstance(item, np.ndarray):
                    raise TypeError(f"{name} must be a numeric scalar")
                try:
                    return float(item)
                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                    RecursionError,
                ) as exc:
                    raise TypeError(f"{name} must be a numeric scalar") from exc
            current = item
            continue

        if np.issubdtype(raw.dtype, np.str_) or np.issubdtype(raw.dtype, np.bytes_):
            raise TypeError(f"{name} must be a numeric scalar, not string")
        if np.issubdtype(raw.dtype, np.complexfloating):
            raise _ComplexScalarError(f"{name} must be real-valued, not complex")
        if np.issubdtype(raw.dtype, np.bool_):
            raise TypeError(f"{name} must be numeric, not boolean")
        if np.issubdtype(raw.dtype, np.integer):
            return int(raw.item())
        if np.issubdtype(raw.dtype, np.floating):
            return float(raw.item())
        raise TypeError(f"{name} must be a numeric scalar")

    raise TypeError(f"{name} must be a numeric scalar")


def _reject_boolean_or_array_scalar(name: str, value: Any) -> int | float:
    """Reject lossy scalar inputs and return a normalized real scalar."""

    return _coerce_real_numeric_scalar(name, value)


def _coerce_unit_interval_mode_scalar(name: str, value: Any) -> int | float:
    """Normalize a mode probability while preserving its ValueError contract."""

    try:
        return _reject_boolean_or_array_scalar(name, value)
    except _ComplexScalarError as exc:
        if name == "mode_stickiness":
            message = f"{name} must be in [0, 1]"
        else:
            message = f"{name} must be finite and lie in [0, 1]"
        raise ValueError(message) from exc


def _coerce_nonnegative_mode_scalar(name: str, value: Any) -> int | float:
    """Normalize a nonnegative mode parameter with its existing error type."""

    try:
        return _reject_boolean_or_array_scalar(name, value)
    except _ComplexScalarError as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc


def _validate_per_bin_sigma_inputs(
    sigma_cm_sqrt_s: Any,
    dt_s: Any,
) -> tuple[int | float, int | float]:
    return (
        _reject_boolean_or_array_scalar("sigma_cm_sqrt_s", sigma_cm_sqrt_s),
        _reject_boolean_or_array_scalar("dt_s", dt_s),
    )


def _validated_per_bin_sigma(
    base: Callable[[Any, Any], float],
    sigma_cm_sqrt_s: Any,
    dt_s: Any,
) -> float:
    """Call a per-bin sigma helper and reject derived floating-point overflow."""

    numeric_sigma, numeric_dt = _validate_per_bin_sigma_inputs(
        sigma_cm_sqrt_s,
        dt_s,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        process_sigma = base(numeric_sigma, numeric_dt)
    try:
        finite = bool(np.isfinite(process_sigma))
    except TypeError as exc:
        raise TypeError("per-bin sigma helper must return a numeric scalar") from exc
    if not finite:
        raise ValueError(
            "sigma_cm_sqrt_s and dt_s must produce a finite per-bin sigma"
        )
    return float(process_sigma)


def _patch_state_space_utils_sigma() -> None:
    from . import state_space_utils

    observed = state_space_utils._per_bin_sigma
    if getattr(observed, _STATE_SPACE_UTILS_PATCHED_FLAG, None) == _PER_BIN_SIGMA_WRAPPER_VERSION:
        setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        return
    base = getattr(observed, "__hipporeplayimm_original__", observed)

    @wraps(base)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        return _validated_per_bin_sigma(base, sigma_cm_sqrt_s, dt_s)

    setattr(per_bin_sigma, _STATE_SPACE_UTILS_PATCHED_FLAG, _PER_BIN_SIGMA_WRAPPER_VERSION)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", base)
    state_space_utils._per_bin_sigma = per_bin_sigma
    setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        alias = getattr(module, "_per_bin_sigma", None)
        in_package = module_name == "hipporeplayimm" or module_name.startswith("hipporeplayimm.")
        if in_package and (alias is observed or alias is base):
            module._per_bin_sigma = per_bin_sigma


def _patch_duration_occupancy_sigma() -> None:
    from . import duration_occupancy

    observed = duration_occupancy._per_bin_sigma
    if getattr(observed, _DURATION_OCCUPANCY_PATCHED_FLAG, None) == _PER_BIN_SIGMA_WRAPPER_VERSION:
        setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)
        return
    base = getattr(observed, "__hipporeplayimm_original__", observed)

    @wraps(base)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        return _validated_per_bin_sigma(base, sigma_cm_sqrt_s, dt_s)

    setattr(per_bin_sigma, _DURATION_OCCUPANCY_PATCHED_FLAG, _PER_BIN_SIGMA_WRAPPER_VERSION)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", base)
    duration_occupancy._per_bin_sigma = per_bin_sigma
    setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)


def _patch_state_space_utils_mode_transition() -> None:
    from . import state_space_utils

    current = state_space_utils._mode_transition_matrix
    if getattr(current, _MODE_TRANSITION_PATCHED_FLAG, False):
        setattr(state_space_utils, _MODE_TRANSITION_PATCHED_FLAG, True)
        return

    @wraps(current)
    def mode_transition_matrix(n_modes, stickiness):
        numeric_stickiness = _coerce_unit_interval_mode_scalar(
            "mode_stickiness",
            stickiness,
        )
        return current(n_modes, numeric_stickiness)

    setattr(mode_transition_matrix, _MODE_TRANSITION_PATCHED_FLAG, True)
    setattr(mode_transition_matrix, "__hipporeplayimm_original__", current)
    state_space_utils._mode_transition_matrix = mode_transition_matrix
    setattr(state_space_utils, _MODE_TRANSITION_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        in_package = module_name == "hipporeplayimm" or module_name.startswith("hipporeplayimm.")
        if in_package and getattr(module, "_mode_transition_matrix", None) is current:
            module._mode_transition_matrix = mode_transition_matrix


def _patch_duration_occupancy_mode_transition() -> None:
    """Keep duration-aware IMM scalar validation aligned with shared helpers."""

    from . import duration_occupancy

    current = duration_occupancy._mode_transition_matrices
    if getattr(current, _DURATION_MODE_TRANSITION_PATCHED_FLAG, False):
        return

    @wraps(current)
    def mode_transition_matrices(
        ss,
        n_modes,
        mode_stickiness,
        imm_switch_tau_s,
        durations,
    ):
        numeric_stickiness = _coerce_unit_interval_mode_scalar(
            "mode_stickiness",
            mode_stickiness,
        )
        numeric_switch_tau = _coerce_nonnegative_mode_scalar(
            "imm_switch_tau_s",
            imm_switch_tau_s,
        )
        return current(
            ss,
            n_modes,
            numeric_stickiness,
            numeric_switch_tau,
            durations,
        )

    setattr(
        mode_transition_matrices,
        _DURATION_MODE_TRANSITION_PATCHED_FLAG,
        True,
    )
    setattr(mode_transition_matrices, "__hipporeplayimm_original__", current)
    duration_occupancy._mode_transition_matrices = mode_transition_matrices


def apply_state_space_sigma_validation_patch() -> None:
    """Install idempotent validation for state-space scalar conversion helpers."""

    _patch_state_space_utils_sigma()
    _patch_duration_occupancy_sigma()
    _patch_state_space_utils_mode_transition()
    _patch_duration_occupancy_mode_transition()


__all__ = ["apply_state_space_sigma_validation_patch"]
