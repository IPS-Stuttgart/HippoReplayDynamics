"""Validate state-space decoder score inputs before scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_bin_center_validation_patch_applied"


def _validate_state_space_log_likelihood(emissions: Any) -> None:
    """Reject malformed state-space emissions before candidate selection."""

    values = np.asarray(emissions.log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("emissions.log_likelihood must be two-dimensional")
    if values.shape[0] == 0:
        raise ValueError("emissions must contain at least one time bin")
    if values.shape[1] == 0:
        raise ValueError("emissions must contain at least one spatial bin")
    expected_shape = (int(emissions.n_time), int(emissions.n_bins))
    if values.shape != expected_shape:
        raise ValueError("emissions.log_likelihood shape must match emissions.n_time and emissions.n_bins")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("emissions.log_likelihood must not contain NaN or +inf")
    if not np.all(np.any(np.isfinite(values), axis=1)):
        raise ValueError("every emission row must contain at least one finite spatial-bin log likelihood")


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
    """Install state-space score input validation for ``StateSpaceReplayModel.score``."""

    from . import state_space as ss

    if getattr(ss.StateSpaceReplayModel.score, _PATCHED_FLAG, False):
        return

    previous_score = ss.StateSpaceReplayModel.score

    def score(self, emissions, bin_centers, *args, **kwargs):
        _validate_state_space_log_likelihood(emissions)
        centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
        return previous_score(self, emissions, centers, *args, **kwargs)

    score.__name__ = getattr(previous_score, "__name__", "score")
    score.__doc__ = getattr(previous_score, "__doc__", None)
    score.__module__ = getattr(previous_score, "__module__", __name__)
    setattr(score, _PATCHED_FLAG, True)
    if getattr(previous_score, "_native_duration_occupancy_aware", False):
        setattr(score, "_native_duration_occupancy_aware", True)
    setattr(score, "__hipporeplayimm_original__", previous_score)
    ss.StateSpaceReplayModel.score = score


__all__ = ["apply_state_space_bin_center_validation_patch"]
