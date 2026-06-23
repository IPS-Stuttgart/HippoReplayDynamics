"""Runtime guards for PyRecEst-backed one-dimensional bin-center inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def apply_pyrecest_bin_center_validation_patch() -> None:
    """Normalize PyRecEst grid inputs to ``(n_bins, position_dim)`` arrays."""

    from . import pyrecest_models

    if getattr(pyrecest_models, "_pyrecest_bin_center_validation_patch_applied", False):
        return

    original_score = pyrecest_models.PyRecEstGoalParticleModel.score
    original_initial_replay_priors = pyrecest_models._initial_replay_priors

    @wraps(original_score)
    def score(self, emissions, bin_centers):
        centers = _as_2d_points(bin_centers, "bin_centers")
        if emissions.n_bins != centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        return original_score(self, emissions, centers)

    @wraps(original_initial_replay_priors)
    def _initial_replay_priors(bin_centers, initial_velocity_sigma_cm_s):
        return original_initial_replay_priors(
            _as_2d_points(bin_centers, "bin_centers"),
            initial_velocity_sigma_cm_s,
        )

    def _coerce_candidate_goals(candidate_goals, bin_centers):
        centers = _as_2d_points(bin_centers, "bin_centers")
        if candidate_goals is not None:
            goals = np.asarray(candidate_goals, dtype=float)
            if centers.shape[1] == 1 and goals.ndim == 1:
                goals = goals[:, None]
            invalid_shape = (
                goals.ndim != 2
                or goals.shape[1] != centers.shape[1]
                or goals.shape[0] == 0
                or not np.all(np.isfinite(goals))
            )
            if invalid_shape:
                raise ValueError(
                    "candidate_goals must have shape (n_goals, position_dim), "
                    "contain at least one row, and be finite"
                )
            return goals
        return pyrecest_models._farthest_point_subset(centers, max_points=32)

    def _farthest_point_subset(points: np.ndarray, max_points: int) -> np.ndarray:
        points = _as_2d_points(points, "bin_centers")
        max_points = int(max_points)
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        if points.shape[0] <= max_points:
            return points.copy()
        selected = [int(np.argmin(np.sum(points, axis=1)))]
        min_dist2 = np.full(points.shape[0], np.inf, dtype=float)
        for _ in range(1, max_points):
            last = points[selected[-1]]
            dist2 = np.sum((points - last[None, :]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
            selected.append(int(np.argmax(min_dist2)))
        return points[np.asarray(selected, dtype=int)]

    pyrecest_models.PyRecEstGoalParticleModel.score = score
    pyrecest_models._initial_replay_priors = _initial_replay_priors
    pyrecest_models._coerce_candidate_goals = _coerce_candidate_goals
    pyrecest_models._farthest_point_subset = _farthest_point_subset
    pyrecest_models._pyrecest_bin_center_validation_patch_applied = True


def _as_2d_points(values: Any, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_points, position_dim)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


__all__ = ["apply_pyrecest_bin_center_validation_patch"]
