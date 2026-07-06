"""Coerce flat bin-center vectors for momentum candidate augmentation."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_ATTR = "_momentum_candidate_center_coercion_patch_applied"


def _as_finite_2d_centers(bin_centers: Any) -> np.ndarray:
    """Return finite ``(n_bins, position_dim)`` centers, accepting flat 1D grids."""

    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] == 0:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must be finite")
    return centers


def apply_momentum_candidate_center_coercion_patch() -> None:
    """Install center coercion before predicted momentum candidates are built."""

    from . import state_space_model

    previous = state_space_model._augment_candidates_with_momentum_predictions
    if getattr(previous, _PATCHED_ATTR, False):
        return

    @wraps(previous)
    def augment_candidates_with_momentum_predictions(
        candidates,
        bin_centers,
        *,
        predicted_top_k: int,
        velocity_decay: float,
        velocity_decays: np.ndarray | None = None,
    ):
        centers = _as_finite_2d_centers(bin_centers)
        return previous(
            candidates,
            centers,
            predicted_top_k=predicted_top_k,
            velocity_decay=velocity_decay,
            velocity_decays=velocity_decays,
        )

    setattr(augment_candidates_with_momentum_predictions, _PATCHED_ATTR, True)
    setattr(augment_candidates_with_momentum_predictions, "__hipporeplayimm_original__", previous)
    state_space_model._augment_candidates_with_momentum_predictions = augment_candidates_with_momentum_predictions

    # Keep the public import surface in sync when this patch is applied while
    # hipporeplayimm.state_space is being imported.
    from . import state_space

    if getattr(state_space, "_augment_candidates_with_momentum_predictions", None) is previous:
        state_space._augment_candidates_with_momentum_predictions = augment_candidates_with_momentum_predictions


__all__ = ["apply_momentum_candidate_center_coercion_patch"]
