"""Validate scalar parameters used by stabilized state-space Gaussian helpers."""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any, Callable

import numpy as np

from .candidate_active_support_validation import (
    _FULL_GRID_PAIRWISE_GAUSSIAN_WRAPPER_FLAG,
    _GAUSSIAN_TRANSITION_WRAPPER_FLAG,
    _PAIRWISE_GAUSSIAN_WRAPPER_FLAG,
    _SPARSE_GAUSSIAN_ROW_WRAPPER_FLAG,
)

_SCALAR_VALIDATION_FLAG = "_state_space_gaussian_scalar_validation_patch_applied"
_RUNTIME_REFRESH_WRAPPER_FLAG = "_state_space_gaussian_scalar_runtime_refresh_wrapper"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def _positive_finite_scalar(name: str, value: Any) -> float:
    """Return a positive finite real scalar without permissive Python coercions."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if raw.ndim != 0:
        raise TypeError(f"{name} must be a real numeric scalar")

    try:
        item = raw.item()
    except ValueError as exc:  # pragma: no cover - guarded by ndim above.
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean")
    if isinstance(item, _STRING_TYPES):
        raise TypeError(f"{name} must be numeric, not string")
    if isinstance(item, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be real-valued, not complex")

    try:
        numeric = float(item)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _mark_wrapper(
    wrapper: Callable[..., Any],
    original: Callable[..., Any],
    stability_marker: str,
) -> Callable[..., Any]:
    setattr(wrapper, _SCALAR_VALIDATION_FLAG, True)
    setattr(wrapper, "__hipporeplayimm_original__", original)
    if getattr(original, stability_marker, False):
        setattr(wrapper, stability_marker, True)
    return wrapper


def _has_scalar_validation(function: object) -> bool:
    """Return whether a wrapper chain already contains scalar validation."""

    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, _SCALAR_VALIDATION_FLAG, False):
            return True
        current = getattr(current, "__hipporeplayimm_original__", None)
    return False


def _synchronize_aliases(name: str, original: object, replacement: object) -> None:
    """Refresh package-local by-value imports of a wrapped Gaussian helper."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, name, None) is original:
            setattr(module, name, replacement)


def apply_state_space_gaussian_scalar_validation_patch() -> None:
    """Reject booleans, arrays, strings, complex values, and scalar overflow."""

    from . import state_space_sparse_momentum, state_space_utils

    current_dense = state_space_utils._gaussian_transition_matrix
    if not _has_scalar_validation(current_dense):

        @wraps(current_dense)
        def gaussian_transition_matrix(
            bin_centers,
            sigma_cm,
            max_step_sigma,
            valid_bin_mask=None,
        ):
            sigma = _positive_finite_scalar("sigma_cm", sigma_cm)
            max_step = _positive_finite_scalar("max_step_sigma", max_step_sigma)
            return current_dense(
                bin_centers,
                sigma,
                max_step,
                valid_bin_mask=valid_bin_mask,
            )

        gaussian_transition_matrix = _mark_wrapper(
            gaussian_transition_matrix,
            current_dense,
            _GAUSSIAN_TRANSITION_WRAPPER_FLAG,
        )
        state_space_utils._gaussian_transition_matrix = gaussian_transition_matrix
        _synchronize_aliases(
            "_gaussian_transition_matrix",
            current_dense,
            gaussian_transition_matrix,
        )

    current_sparse = state_space_sparse_momentum._finite_gaussian_row
    if not _has_scalar_validation(current_sparse):

        @wraps(current_sparse)
        def finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            predicted,
            *,
            sigma_cm,
            max_step_sigma,
        ):
            sigma = _positive_finite_scalar("sigma_cm", sigma_cm)
            max_step = _positive_finite_scalar("max_step_sigma", max_step_sigma)
            return current_sparse(
                centers,
                valid_indices,
                tree,
                predicted,
                sigma_cm=sigma,
                max_step_sigma=max_step,
            )

        finite_gaussian_row = _mark_wrapper(
            finite_gaussian_row,
            current_sparse,
            _SPARSE_GAUSSIAN_ROW_WRAPPER_FLAG,
        )
        state_space_sparse_momentum._finite_gaussian_row = finite_gaussian_row
        _synchronize_aliases(
            "_finite_gaussian_row",
            current_sparse,
            finite_gaussian_row,
        )

    current_pairwise = state_space_utils._pairwise_gaussian_log_prob
    if not _has_scalar_validation(current_pairwise):

        @wraps(current_pairwise)
        def pairwise_gaussian_log_prob(predicted, observed, sigma_cm):
            sigma = _positive_finite_scalar("sigma_cm", sigma_cm)
            return current_pairwise(predicted, observed, sigma)

        pairwise_gaussian_log_prob = _mark_wrapper(
            pairwise_gaussian_log_prob,
            current_pairwise,
            _PAIRWISE_GAUSSIAN_WRAPPER_FLAG,
        )
        state_space_utils._pairwise_gaussian_log_prob = pairwise_gaussian_log_prob
        _synchronize_aliases(
            "_pairwise_gaussian_log_prob",
            current_pairwise,
            pairwise_gaussian_log_prob,
        )

    current_normalized = state_space_utils._full_grid_normalized_pairwise_gaussian_log_prob
    if not _has_scalar_validation(current_normalized):

        @wraps(current_normalized)
        def full_grid_normalized_pairwise_gaussian_log_prob(
            predicted,
            observed,
            all_observed,
            sigma_cm,
            valid_bin_mask=None,
        ):
            sigma = _positive_finite_scalar("sigma_cm", sigma_cm)
            return current_normalized(
                predicted,
                observed,
                all_observed,
                sigma,
                valid_bin_mask=valid_bin_mask,
            )

        full_grid_normalized_pairwise_gaussian_log_prob = _mark_wrapper(
            full_grid_normalized_pairwise_gaussian_log_prob,
            current_normalized,
            _FULL_GRID_PAIRWISE_GAUSSIAN_WRAPPER_FLAG,
        )
        state_space_utils._full_grid_normalized_pairwise_gaussian_log_prob = (
            full_grid_normalized_pairwise_gaussian_log_prob
        )
        _synchronize_aliases(
            "_full_grid_normalized_pairwise_gaussian_log_prob",
            current_normalized,
            full_grid_normalized_pairwise_gaussian_log_prob,
        )


def _install_runtime_refresh_hook() -> None:
    """Reapply scalar validation when the public runtime patch hook refreshes support."""

    from . import candidate_active_support_validation as active_support

    current = active_support.apply_candidate_active_support_validation_patch
    if getattr(current, _RUNTIME_REFRESH_WRAPPER_FLAG, False):
        return

    @wraps(current)
    def apply_candidate_active_support_validation_patch() -> None:
        current()
        apply_state_space_gaussian_scalar_validation_patch()

    setattr(apply_candidate_active_support_validation_patch, _RUNTIME_REFRESH_WRAPPER_FLAG, True)
    setattr(apply_candidate_active_support_validation_patch, "__hipporeplayimm_original__", current)
    active_support.apply_candidate_active_support_validation_patch = (
        apply_candidate_active_support_validation_patch
    )


_install_runtime_refresh_hook()

__all__ = ["apply_state_space_gaussian_scalar_validation_patch"]
