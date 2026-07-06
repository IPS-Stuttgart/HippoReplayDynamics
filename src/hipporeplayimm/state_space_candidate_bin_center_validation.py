"""Validate bin centers for direct state-space candidate-support helpers."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_candidate_bin_center_validation_patch_applied"


def apply_state_space_candidate_bin_center_validation_patch() -> None:
    """Install bin-center validation on ``StateSpaceReplayModel.candidate_indices``."""

    from .state_space_model import StateSpaceReplayModel

    previous = StateSpaceReplayModel.candidate_indices
    if getattr(previous, _PATCHED_FLAG, False):
        return

    @wraps(previous)
    def candidate_indices(self, emissions, bin_centers=None, valid_bin_mask=None):
        centers = None
        if bin_centers is not None:
            centers = _coerce_candidate_bin_centers(bin_centers, int(emissions.n_bins))
        return previous(self, emissions, centers, valid_bin_mask)

    setattr(candidate_indices, _PATCHED_FLAG, True)
    setattr(candidate_indices, "__hipporeplayimm_original__", previous)
    StateSpaceReplayModel.candidate_indices = candidate_indices


def _coerce_candidate_bin_centers(bin_centers: Any, n_bins: int) -> np.ndarray:
    """Return finite 2D bin centers with one row per spatial bin."""

    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[1] == 0:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if centers.shape[0] != int(n_bins):
        raise ValueError("bin_centers must contain one row per emission spatial bin")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must be finite")
    return centers


__all__ = ["apply_state_space_candidate_bin_center_validation_patch"]
