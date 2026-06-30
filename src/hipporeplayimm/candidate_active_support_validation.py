"""Validate state-space candidate-support rows after active-support masking."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_candidate_active_support_validation_patch_applied"


def _validate_active_support_rows(values: np.ndarray) -> None:
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    finite_rows = np.any(np.isfinite(rows), axis=1)
    if not np.all(finite_rows):
        row = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(f"row {row} must contain at least one finite value on the active support")


def apply_candidate_active_support_validation_patch() -> None:
    """Install validation for occupancy-masked candidate-support sources."""

    from . import state_space_model

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
