"""Stabilize exact finite-displacement Gaussian transitions at extreme scales."""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from .candidate_active_support_validation import (
    _stable_standardized_gaussian_weights,
    _standardized_euclidean_distances,
)

_SHIFTED_WRAPPER_FLAG = "_stable_shifted_gaussian_transition_wrapper"
_DISPLACEMENT_WRAPPER_FLAG = "_stable_displacement_transition_wrapper"
_PRIOR_WRAPPER_FLAG = "_stable_displacement_prior_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_displacement_gaussian_stability_patch() -> None:
    """Patch Gaussian helpers used by the exact finite-displacement decoders."""

    from . import state_space_displacement_momentum as displacement_module

    _patch_shifted_gaussian_transition(displacement_module)
    _patch_displacement_transition(displacement_module)
    _patch_displacement_prior(displacement_module)


def _patch_shifted_gaussian_transition(displacement_module: Any) -> None:
    current = displacement_module._shifted_gaussian_transition_matrix
    if getattr(current, _SHIFTED_WRAPPER_FLAG, False):
        _synchronize_aliases("_shifted_gaussian_transition_matrix", current, current)
        return

    @wraps(current)
    def shifted_gaussian_transition_matrix(
        bin_centers,
        *,
        displacement: np.ndarray,
        sigma_cm: float,
        max_step_sigma: float,
        valid_bin_mask: np.ndarray | None = None,
    ) -> csr_matrix:
        sigma = float(sigma_cm)
        max_step = float(max_step_sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_cm must be finite and positive")
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step_sigma must be finite and positive")

        centers = displacement_module._as_2d_centers(bin_centers)
        n_bins = centers.shape[0]
        valid_mask = displacement_module._coerce_valid_bin_mask(valid_bin_mask, n_bins)
        allowed = np.arange(n_bins, dtype=int) if valid_mask is None else np.flatnonzero(valid_mask)
        try:
            shift = np.asarray(displacement, dtype=float).reshape(centers.shape[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "displacement must contain one value per position dimension"
            ) from exc
        if not np.all(np.isfinite(shift)):
            raise ValueError("displacement must be finite")

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for src, center in enumerate(centers):
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = center + shift
            if not np.all(np.isfinite(predicted)):
                raise ValueError("center plus displacement must remain finite")
            standardized_distance = _standardized_euclidean_distances(
                centers,
                predicted,
                sigma,
            )
            keep = standardized_distance <= max_step
            if valid_mask is not None:
                keep &= valid_mask
            if not np.any(keep):
                keep[int(allowed[int(np.argmin(standardized_distance[allowed]))])] = True
            dst = np.flatnonzero(keep)
            weights = _stable_standardized_gaussian_weights(
                standardized_distance[dst]
            )
            rows.extend(int(index) for index in dst)
            cols.extend([int(src)] * len(dst))
            data.extend(float(value) for value in weights)
        return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))

    setattr(shifted_gaussian_transition_matrix, _SHIFTED_WRAPPER_FLAG, True)
    setattr(shifted_gaussian_transition_matrix, _ORIGINAL_ATTR, current)
    displacement_module._shifted_gaussian_transition_matrix = shifted_gaussian_transition_matrix
    _synchronize_aliases(
        "_shifted_gaussian_transition_matrix",
        current,
        shifted_gaussian_transition_matrix,
    )


def _patch_displacement_transition(displacement_module: Any) -> None:
    current = displacement_module._displacement_transition_matrix
    if getattr(current, _DISPLACEMENT_WRAPPER_FLAG, False):
        _synchronize_aliases("_displacement_transition_matrix", current, current)
        return

    @wraps(current)
    def displacement_transition_matrix(
        vectors: np.ndarray,
        *,
        sigma_cm: float,
        decay: float,
    ) -> np.ndarray:
        sigma = float(sigma_cm)
        decay_value = float(decay)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("displacement transition sigma must be finite and positive")
        if not np.isfinite(decay_value) or decay_value < 0.0:
            raise ValueError("displacement velocity decay must be finite and nonnegative")
        points = np.asarray(vectors, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("displacement vectors must be a nonempty two-dimensional array")
        if not np.all(np.isfinite(points)):
            raise ValueError("displacement vectors must be finite")

        with np.errstate(over="ignore", invalid="ignore"):
            predicted = decay_value * points
        if not np.all(np.isfinite(predicted)):
            raise ValueError("decayed displacement vectors must remain finite")
        weights = np.empty((points.shape[0], points.shape[0]), dtype=float)
        for src, prediction in enumerate(predicted):
            distances = _standardized_euclidean_distances(points, prediction, sigma)
            weights[:, src] = _stable_standardized_gaussian_weights(distances)
        return weights

    setattr(displacement_transition_matrix, _DISPLACEMENT_WRAPPER_FLAG, True)
    setattr(displacement_transition_matrix, _ORIGINAL_ATTR, current)
    displacement_module._displacement_transition_matrix = displacement_transition_matrix
    _synchronize_aliases(
        "_displacement_transition_matrix",
        current,
        displacement_transition_matrix,
    )


def _patch_displacement_prior(displacement_module: Any) -> None:
    current = displacement_module._zero_centered_displacement_prior
    if getattr(current, _PRIOR_WRAPPER_FLAG, False):
        _synchronize_aliases("_zero_centered_displacement_prior", current, current)
        return

    @wraps(current)
    def zero_centered_displacement_prior(vectors: np.ndarray, sigma_cm: float) -> np.ndarray:
        sigma = float(sigma_cm)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("displacement prior sigma must be finite and positive")
        points = np.asarray(vectors, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("displacement vectors must be a nonempty two-dimensional array")
        if not np.all(np.isfinite(points)):
            raise ValueError("displacement vectors must be finite")
        distances = _standardized_euclidean_distances(
            points,
            np.zeros(points.shape[1], dtype=float),
            sigma,
        )
        return _stable_standardized_gaussian_weights(distances)

    setattr(zero_centered_displacement_prior, _PRIOR_WRAPPER_FLAG, True)
    setattr(zero_centered_displacement_prior, _ORIGINAL_ATTR, current)
    displacement_module._zero_centered_displacement_prior = zero_centered_displacement_prior
    _synchronize_aliases(
        "_zero_centered_displacement_prior",
        current,
        zero_centered_displacement_prior,
    )


def _synchronize_aliases(name: str, original: object, replacement: object) -> None:
    """Refresh package-local imports of a patched displacement helper."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, name, None) is original:
            setattr(module, name, replacement)


__all__ = ["apply_displacement_gaussian_stability_patch"]
