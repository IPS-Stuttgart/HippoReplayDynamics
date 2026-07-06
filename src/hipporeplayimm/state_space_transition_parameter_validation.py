"""Validate state-space transition numeric parameters before float coercion.

Several state-space transition helpers historically called ``float(...)``
before validating Gaussian scale and duration parameters.  That silently
accepted malformed configuration values such as ``True`` or ``"85.0"``.
This patch keeps numeric NumPy scalars valid while rejecting booleans,
strings, and array-shaped scalar surrogates before scoring starts.
"""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

_STATE_SPACE_UTILS_PATCHED_FLAG = "_state_space_transition_parameter_validation_patch_applied"
_DURATION_OCCUPANCY_PATCHED_FLAG = "_duration_occupancy_transition_parameter_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _is_string_scalar(value: object) -> bool:
    if isinstance(value, _STRING_TYPES):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), _STRING_TYPES)
        except ValueError:
            return False
    return False


def _coerce_positive_float(name: str, value: object) -> float:
    """Return a finite positive scalar without accepting bool/string surrogates."""

    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not boolean")
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not string")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if array.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")
    try:
        numeric = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _replace_imported_module_aliases(attribute_name: str, original: object, replacement: object) -> None:
    """Replace by-value imports of patched state-space transition helpers."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, attribute_name, None) is original:
            setattr(module, attribute_name, replacement)


def _patch_state_space_utils_transition_helpers() -> None:
    from . import state_space_utils

    if getattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, False):
        return

    original_per_bin_sigma = state_space_utils._per_bin_sigma
    original_gaussian_transition_matrix = state_space_utils._gaussian_transition_matrix
    original_pairwise_gaussian_log_prob = state_space_utils._pairwise_gaussian_log_prob

    @wraps(original_per_bin_sigma)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        sigma = _coerce_positive_float("sigma_cm_sqrt_s", sigma_cm_sqrt_s)
        dt = _coerce_positive_float("dt_s", dt_s)
        return original_per_bin_sigma(sigma, dt)

    @wraps(original_gaussian_transition_matrix)
    def gaussian_transition_matrix(bin_centers, sigma_cm, max_step_sigma, valid_bin_mask=None):
        sigma = _coerce_positive_float("sigma_cm", sigma_cm)
        max_step = _coerce_positive_float("max_step_sigma", max_step_sigma)
        return original_gaussian_transition_matrix(
            bin_centers,
            sigma,
            max_step,
            valid_bin_mask=valid_bin_mask,
        )

    @wraps(original_pairwise_gaussian_log_prob)
    def pairwise_gaussian_log_prob(predicted, observed, sigma_cm):
        sigma = _coerce_positive_float("sigma_cm", sigma_cm)
        return original_pairwise_gaussian_log_prob(predicted, observed, sigma)

    for function in (
        per_bin_sigma,
        gaussian_transition_matrix,
        pairwise_gaussian_log_prob,
    ):
        setattr(function, _STATE_SPACE_UTILS_PATCHED_FLAG, True)

    setattr(per_bin_sigma, "__hipporeplayimm_original__", original_per_bin_sigma)
    setattr(gaussian_transition_matrix, "__hipporeplayimm_original__", original_gaussian_transition_matrix)
    setattr(pairwise_gaussian_log_prob, "__hipporeplayimm_original__", original_pairwise_gaussian_log_prob)

    state_space_utils._per_bin_sigma = per_bin_sigma
    state_space_utils._gaussian_transition_matrix = gaussian_transition_matrix
    state_space_utils._pairwise_gaussian_log_prob = pairwise_gaussian_log_prob
    _replace_imported_module_aliases("_per_bin_sigma", original_per_bin_sigma, per_bin_sigma)
    _replace_imported_module_aliases(
        "_gaussian_transition_matrix",
        original_gaussian_transition_matrix,
        gaussian_transition_matrix,
    )
    _replace_imported_module_aliases(
        "_pairwise_gaussian_log_prob",
        original_pairwise_gaussian_log_prob,
        pairwise_gaussian_log_prob,
    )
    setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)


def _patch_duration_occupancy_transition_helpers() -> None:
    from . import duration_occupancy

    if getattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, False):
        return

    original_per_bin_sigma = duration_occupancy._per_bin_sigma

    @wraps(original_per_bin_sigma)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        sigma = _coerce_positive_float("sigma_cm_sqrt_s", sigma_cm_sqrt_s)
        dt = _coerce_positive_float("dt", dt_s)
        return original_per_bin_sigma(sigma, dt)

    setattr(per_bin_sigma, _DURATION_OCCUPANCY_PATCHED_FLAG, True)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", original_per_bin_sigma)
    duration_occupancy._per_bin_sigma = per_bin_sigma
    _replace_imported_module_aliases("_per_bin_sigma", original_per_bin_sigma, per_bin_sigma)
    setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)


def apply_state_space_transition_parameter_validation_patch() -> None:
    """Install strict scalar validation for state-space transition parameters."""

    _patch_state_space_utils_transition_helpers()
    _patch_duration_occupancy_transition_helpers()


__all__ = ["apply_state_space_transition_parameter_validation_patch"]
