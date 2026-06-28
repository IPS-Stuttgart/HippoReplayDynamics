"""Runtime guards for PyRecEst-backed one-dimensional bin-center inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_pyrecest_bin_center_validation_patch_applied"
_ORIGINALS_ATTR = "_pyrecest_bin_center_validation_originals"
_WRAPPER_MARKER = "_pyrecest_bin_center_validation_wrapper"


def _mark_bin_center_guard(wrapper):
    setattr(wrapper, _WRAPPER_MARKER, True)
    return wrapper


def _is_current_bin_center_guard(value: object) -> bool:
    return bool(getattr(value, _WRAPPER_MARKER, False))


def _wrappers_are_current(pyrecest_models) -> bool:
    try:
        return (
            _is_current_bin_center_guard(pyrecest_models.PyRecEstGoalParticleModel.score)
            and _is_current_bin_center_guard(pyrecest_models._initial_replay_priors)
            and _is_current_bin_center_guard(pyrecest_models._coerce_candidate_goals)
            and _is_current_bin_center_guard(pyrecest_models._farthest_point_subset)
            and _is_current_bin_center_guard(pyrecest_models._validate_probability)
            and _is_current_bin_center_guard(pyrecest_models._validate_positive_int)
            and _is_current_bin_center_guard(pyrecest_models._validate_positive_float)
            and _is_current_bin_center_guard(pyrecest_models._validate_nonnegative_float)
        )
    except AttributeError:
        return False


def _originals(pyrecest_models) -> dict[str, object]:
    originals = getattr(pyrecest_models, _ORIGINALS_ATTR, None)
    if originals is None:
        originals = {
            "score": pyrecest_models.PyRecEstGoalParticleModel.score,
            "initial_replay_priors": pyrecest_models._initial_replay_priors,
            "coerce_candidate_goals": pyrecest_models._coerce_candidate_goals,
            "farthest_point_subset": pyrecest_models._farthest_point_subset,
            "validate_probability": pyrecest_models._validate_probability,
            "validate_positive_int": pyrecest_models._validate_positive_int,
            "validate_positive_float": pyrecest_models._validate_positive_float,
            "validate_nonnegative_float": pyrecest_models._validate_nonnegative_float,
        }
        setattr(pyrecest_models, _ORIGINALS_ATTR, originals)
    return originals


def apply_pyrecest_bin_center_validation_patch() -> None:
    """Normalize PyRecEst grid inputs and reject bool-valued numeric parameters.

    The module-level applied flag is not enough by itself: tests or downstream
    code can replace a guarded helper while leaving the flag set. Re-checking
    wrapper markers lets ``apply_runtime_patches()`` refresh stale helpers.
    """

    from . import pyrecest_models

    originals = _originals(pyrecest_models)
    if getattr(pyrecest_models, _PATCHED_FLAG, False) and _wrappers_are_current(pyrecest_models):
        return

    original_score = originals["score"]
    original_initial_replay_priors = originals["initial_replay_priors"]
    original_coerce_candidate_goals = originals["coerce_candidate_goals"]
    original_farthest_point_subset = originals["farthest_point_subset"]
    original_validate_probability = originals["validate_probability"]
    original_validate_positive_int = originals["validate_positive_int"]
    original_validate_positive_float = originals["validate_positive_float"]
    original_validate_nonnegative_float = originals["validate_nonnegative_float"]

    @_mark_bin_center_guard
    @wraps(original_score)
    def score(self, emissions, bin_centers):
        centers = _as_2d_points(bin_centers, "bin_centers")
        if emissions.n_bins != centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        _validate_positive_int(self.n_particles, "n_particles", original_validate_positive_int)
        return original_score(self, emissions, centers)

    @_mark_bin_center_guard
    @wraps(original_initial_replay_priors)
    def _initial_replay_priors(bin_centers, initial_velocity_sigma_cm_s):
        return original_initial_replay_priors(
            _as_2d_points(bin_centers, "bin_centers"),
            initial_velocity_sigma_cm_s,
        )

    @_mark_bin_center_guard
    @wraps(original_coerce_candidate_goals)
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

    @_mark_bin_center_guard
    @wraps(original_farthest_point_subset)
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

    @_mark_bin_center_guard
    @wraps(original_validate_probability)
    def _patched_validate_probability(probability, name):
        _validate_probability(probability, name, original_validate_probability)

    @_mark_bin_center_guard
    @wraps(original_validate_positive_int)
    def _patched_validate_positive_int(value, name):
        _validate_positive_int(value, name, original_validate_positive_int)

    @_mark_bin_center_guard
    @wraps(original_validate_positive_float)
    def _patched_validate_positive_float(value, name):
        _validate_positive_float(value, name, original_validate_positive_float)

    @_mark_bin_center_guard
    @wraps(original_validate_nonnegative_float)
    def _patched_validate_nonnegative_float(value, name):
        _validate_nonnegative_float(value, name, original_validate_nonnegative_float)

    pyrecest_models.PyRecEstGoalParticleModel.score = score
    pyrecest_models._initial_replay_priors = _initial_replay_priors
    pyrecest_models._coerce_candidate_goals = _coerce_candidate_goals
    pyrecest_models._farthest_point_subset = _farthest_point_subset
    pyrecest_models._validate_probability = _patched_validate_probability
    pyrecest_models._validate_positive_int = _patched_validate_positive_int
    pyrecest_models._validate_positive_float = _patched_validate_positive_float
    pyrecest_models._validate_nonnegative_float = _patched_validate_nonnegative_float
    setattr(pyrecest_models, _PATCHED_FLAG, True)


def _is_bool_scalar(value: object) -> bool:
    array = np.asarray(value)
    if array.shape != ():
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _validate_probability(value: object, name: str, original_validate) -> None:
    if _is_bool_scalar(value):
        raise ValueError(f"{name} must lie in [0, 1]")
    try:
        original_validate(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must lie in [0, 1]") from exc


def _validate_positive_int(value: object, name: str, original_validate) -> None:
    if _is_bool_scalar(value):
        raise ValueError(f"{name} must be a positive integer")
    value_array = np.asarray(value)
    if value_array.shape != ():
        raise ValueError(f"{name} must be a positive integer")
    try:
        original_validate(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc


def _validate_positive_float(value: object, name: str, original_validate) -> None:
    if _is_bool_scalar(value):
        raise ValueError(f"{name} must be finite and positive")
    try:
        original_validate(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc


def _validate_nonnegative_float(value: object, name: str, original_validate) -> None:
    if _is_bool_scalar(value):
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        original_validate(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc


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
