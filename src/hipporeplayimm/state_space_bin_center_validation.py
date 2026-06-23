"""Validate state-space decoder bin-center inputs before scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_bin_center_validation_patch_applied"


def _coerce_state_space_bin_centers(bin_centers: Any, n_bins: int) -> np.ndarray:
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


def apply_state_space_bin_center_validation_patch() -> None:
    """Install bin-center validation for ``StateSpaceReplayModel.score``."""

    from . import state_space as ss

    if getattr(ss.StateSpaceReplayModel.score, _PATCHED_FLAG, False):
        return

    previous_score = ss.StateSpaceReplayModel.score

    def score(self, emissions, bin_centers, *args, **kwargs):
        centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
        return previous_score(self, emissions, centers, *args, **kwargs)

    score.__name__ = getattr(previous_score, "__name__", "score")
    score.__doc__ = getattr(previous_score, "__doc__", None)
    score.__module__ = getattr(previous_score, "__module__", __name__)
    setattr(score, _PATCHED_FLAG, True)
    setattr(score, "__hipporeplayimm_original__", previous_score)
    ss.StateSpaceReplayModel.score = score


__all__ = ["apply_state_space_bin_center_validation_patch"]
