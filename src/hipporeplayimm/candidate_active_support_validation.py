"""Validate state-space candidate-support rows after active-support masking."""

from __future__ import annotations

from functools import wraps
import sys

import numpy as np

_PATCHED_FLAG = "_candidate_active_support_validation_patch_applied"
_COUNT_HELPERS_PATCHED_FLAG = "_candidate_count_validation_patch_applied"


def _validate_active_support_rows(values: np.ndarray) -> None:
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    finite_rows = np.any(np.isfinite(rows), axis=1)
    if not np.all(finite_rows):
        row = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(f"row {row} must contain at least one finite value on the active support")


def _coerce_nonnegative_integer_count(name: str, value: object) -> int:
    """Return a safe count value for candidate-support helpers."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative integer count, not boolean")
    try:
        arr = np.asarray(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a scalar integer count") from exc
    if arr.ndim != 0:
        raise TypeError(f"{name} must be a scalar integer count")
    if np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(f"{name} must be a nonnegative integer count, not boolean")
    if arr.dtype == object:
        value = arr.item()
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a nonnegative integer count, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative integer count") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite nonnegative integer count")
    return int(numeric)


def _refresh_helper_aliases(helper_name: str, original, patched) -> None:
    """Update already-imported hipporeplayimm module aliases for a patched helper."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, helper_name, None) is original:
            setattr(module, helper_name, patched)


def _apply_candidate_count_validation_patch() -> None:
    """Validate integer-valued candidate-count parameters before NumPy indexing."""

    from . import state_space_utils

    current_top = state_space_utils._top_candidate_indices
    if getattr(current_top, _COUNT_HELPERS_PATCHED_FLAG, False):
        top_candidate_indices = current_top
        original_top = getattr(current_top, "__hipporeplayimm_original__", current_top)
    else:
        original_top = current_top

        @wraps(original_top)
        def top_candidate_indices(log_emission, top_k):
            return original_top(
                log_emission,
                _coerce_nonnegative_integer_count("top_k", top_k),
            )

        setattr(top_candidate_indices, _COUNT_HELPERS_PATCHED_FLAG, True)
        setattr(top_candidate_indices, "__hipporeplayimm_original__", original_top)
        state_space_utils._top_candidate_indices = top_candidate_indices

    current_mass = state_space_utils._mass_retaining_candidate_indices
    if getattr(current_mass, _COUNT_HELPERS_PATCHED_FLAG, False):
        mass_retaining_candidate_indices = current_mass
        original_mass = getattr(current_mass, "__hipporeplayimm_original__", current_mass)
    else:
        original_mass = current_mass

        @wraps(original_mass)
        def mass_retaining_candidate_indices(
            log_emission,
            mass_threshold=None,
            *,
            top_k=None,
            min_k=1,
            max_k=0,
        ):
            return original_mass(
                log_emission,
                mass_threshold,
                top_k=(None if top_k is None else _coerce_nonnegative_integer_count("top_k", top_k)),
                min_k=_coerce_nonnegative_integer_count("min_k", min_k),
                max_k=_coerce_nonnegative_integer_count("max_k", max_k),
            )

        setattr(mass_retaining_candidate_indices, _COUNT_HELPERS_PATCHED_FLAG, True)
        setattr(mass_retaining_candidate_indices, "__hipporeplayimm_original__", original_mass)
        state_space_utils._mass_retaining_candidate_indices = mass_retaining_candidate_indices

    _refresh_helper_aliases("_top_candidate_indices", original_top, top_candidate_indices)
    _refresh_helper_aliases("_mass_retaining_candidate_indices", original_mass, mass_retaining_candidate_indices)


def apply_candidate_active_support_validation_patch() -> None:
    """Install validation for occupancy-masked candidate-support sources."""

    from . import state_space_model

    _apply_candidate_count_validation_patch()

    current = state_space_model._masked_candidate_support_log_values
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def masked_candidate_support_log_values(log_likelihood, valid_bin_mask):
        masked = current(log_likelihood, valid_bin_mask)
        _validate_active_support_rows(masked)
        return masked

    setattr(masked_candidate_support_log_values, _PATCHED_FLAG, True)
    setattr(masked_candidate_support_log_values, "__hipporeplayimm_original__", current)
    state_space_model._masked_candidate_support_log_values = masked_candidate_support_log_values


__all__ = ["apply_candidate_active_support_validation_patch"]
