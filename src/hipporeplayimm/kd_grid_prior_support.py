"""Preserve exact support when marginalizing KD hyperparameter grids."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_kd_grid_prior_support_patch_applied"


def apply_kd_grid_prior_support_patch() -> None:
    """Install validated KD grid-prior marginalization with exact zero mass."""

    from . import kd_reference as kd

    current = kd.marginalize_grid_log_evidence
    if getattr(current, _PATCHED_FLAG, False):
        return

    setattr(_marginalize_grid_log_evidence, _PATCHED_FLAG, True)
    kd.marginalize_grid_log_evidence = _marginalize_grid_log_evidence


def _contains_complex_numeric(value: object) -> bool:
    """Return whether a scalar or array-like value contains complex numerics."""

    try:
        values = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if np.issubdtype(values.dtype, np.complexfloating):
        return True
    if values.dtype == object:
        return any(isinstance(item, (complex, np.complexfloating)) for item in values.flat)
    return False


def _marginalize_grid_log_evidence(grid: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Marginalize a KD evidence grid without inventing excluded prior mass."""

    if _contains_complex_numeric(grid):
        raise ValueError("grid must contain real values")
    if _contains_complex_numeric(prior):
        raise ValueError("prior must contain real values")

    values = np.asarray(grid, dtype=float)
    weights = np.asarray(prior, dtype=float)
    if values.ndim < 2:
        raise ValueError("grid must have an event axis and at least one parameter axis")
    if values.shape[1:] != weights.shape:
        raise ValueError(f"grid shape {values.shape[1:]} does not match prior shape {weights.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("prior must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("prior cannot contain negative values")

    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError("prior must assign positive mass to at least one grid point")

    flat_values = values.reshape(values.shape[0], -1)
    flat_weights = weights.reshape(-1)
    support = positive.reshape(-1)
    supported_values = flat_values[:, support]
    if np.any(np.isnan(supported_values)) or np.any(np.isposinf(supported_values)):
        raise ValueError(
            "grid evidence with positive prior mass must contain only finite values or -inf"
        )
    supported_log_prior = np.log(flat_weights[support])
    supported_log_prior -= logsumexp(supported_log_prior)
    return logsumexp(supported_values + supported_log_prior[None, :], axis=1)


__all__ = ["apply_kd_grid_prior_support_patch"]
