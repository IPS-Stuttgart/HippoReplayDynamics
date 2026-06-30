"""Validate sparse-momentum bin-center arrays before KD-tree setup."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_ATTR = "_sparse_momentum_bin_center_validation_patch_applied"


def apply_sparse_momentum_bin_center_validation_patch() -> None:
    """Patch sparse-momentum center coercion with explicit shape/finite checks."""

    from . import state_space_sparse_momentum as sparse_momentum

    current = sparse_momentum._as_2d_centers
    if getattr(current, _PATCHED_ATTR, False):
        return

    @wraps(current)
    def as_2d_centers(bin_centers: Any) -> np.ndarray:
        centers = np.asarray(bin_centers, dtype=float)
        if centers.ndim == 1:
            centers = centers[:, None]
        if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] == 0:
            raise ValueError("bin_centers must have shape (n_bins, position_dim)")
        if not np.all(np.isfinite(centers)):
            raise ValueError("bin_centers must be finite")
        return centers

    setattr(as_2d_centers, _PATCHED_ATTR, True)
    sparse_momentum._as_2d_centers = as_2d_centers


__all__ = ["apply_sparse_momentum_bin_center_validation_patch"]
