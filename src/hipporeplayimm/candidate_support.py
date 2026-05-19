"""Adaptive candidate-support helpers for pruned replay decoders."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

DEFAULT_CANDIDATE_MIN_MASS = 0.995
DEFAULT_CANDIDATE_MIN_LOG_MASS = float(np.log(DEFAULT_CANDIDATE_MIN_MASS))


def _candidate_min_log_mass_from_probability(value: float | None) -> float | None:
    """Convert a probability-mass threshold to log space.

    ``None`` and non-positive values disable mass-adaptive expansion. A value of
    one requests all finite likelihood support, subject to the optional
    ``max_candidates`` cap used by :func:`_adaptive_candidate_indices`.
    """

    if value is None:
        return None
    probability = float(value)
    if probability <= 0.0:
        return None
    if probability > 1.0:
        raise ValueError("candidate_min_mass must be in (0, 1] or non-positive to disable it")
    return float(np.log(probability))


def _candidate_min_mass_diagnostic(value: float | None) -> float:
    """Return a finite probability diagnostic for a log-mass threshold."""

    if value is None:
        return 0.0
    log_mass = float(value)
    if not np.isfinite(log_mass):
        return 0.0
    return float(np.exp(min(0.0, log_mass)))


def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    """Return finite candidates ordered by descending emission likelihood."""

    row = np.asarray(log_emission, dtype=float)
    if row.ndim != 1:
        raise ValueError("log_emission must be one-dimensional")
    if top_k <= 0 or top_k >= row.shape[0]:
        return np.arange(row.shape[0], dtype=int)
    selected = np.argpartition(row, -top_k)[-top_k:]
    return selected[np.argsort(row[selected])[::-1]]


def _adaptive_candidate_indices(
    log_emission: np.ndarray,
    top_k: int,
    *,
    min_log_mass: float | None = DEFAULT_CANDIDATE_MIN_LOG_MASS,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Select a top-k support enlarged until it covers enough emission mass.

    The returned support is still deterministic and sorted by descending local
    emission likelihood.  ``top_k`` acts as the minimum beam width, while
    ``min_log_mass`` requests a minimum fraction of the row's normalized
    emission mass.  ``max_candidates`` is an optional runtime guard; pass
    ``None`` or a non-positive value to allow the selector to grow to the full
    grid when the emission row is diffuse.
    """

    row = np.asarray(log_emission, dtype=float)
    if row.ndim != 1:
        raise ValueError("log_emission must be one-dimensional")
    n_bins = int(row.shape[0])
    if n_bins == 0:
        raise ValueError("log_emission must contain at least one bin")
    if top_k <= 0 or top_k >= n_bins:
        return np.arange(n_bins, dtype=int)

    cap = _candidate_cap(max_candidates, n_bins)
    finite = np.isfinite(row)
    if not np.any(finite):
        return np.arange(min(max(1, int(top_k)), cap), dtype=int)

    order = np.argsort(np.where(finite, row, -np.inf))[::-1]
    order = order[finite[order]]
    if order.size == 0:
        return np.arange(min(max(1, int(top_k)), cap), dtype=int)

    base_count = min(max(1, int(top_k)), int(order.size), cap)
    count = base_count
    if min_log_mass is not None and np.isfinite(float(min_log_mass)):
        target = min(0.0, float(min_log_mass))
        log_total = float(logsumexp(row[order]))
        cumulative_log_mass = np.logaddexp.accumulate(row[order]) - log_total
        count_from_mass = int(np.searchsorted(cumulative_log_mass + 1e-12, target, side="left") + 1)
        count = max(count, min(count_from_mass, int(order.size), cap))

    selected = order[:count]
    return selected[np.argsort(row[selected])[::-1]]


def _augment_candidates_with_spatial_halo(
    candidates: list[np.ndarray],
    bin_centers: np.ndarray,
    *,
    radius_cm: float,
) -> list[np.ndarray]:
    """Union every candidate set with grid bins inside a spatial halo."""

    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim != 2:
        raise ValueError("bin_centers must be a two-dimensional array")
    radius = float(radius_cm)
    if radius <= 0.0:
        return [np.asarray(curr, dtype=int).copy() for curr in candidates]
    if not np.isfinite(radius):
        full = np.arange(centers.shape[0], dtype=int)
        return [full.copy() for _ in candidates]

    radius2 = radius * radius
    output: list[np.ndarray] = []
    for curr in candidates:
        arr = np.asarray(curr, dtype=int)
        if arr.size == 0:
            output.append(arr.copy())
            continue
        selected_centers = centers[arr]
        delta = centers[:, None, :] - selected_centers[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        halo = np.flatnonzero(np.any(dist2 <= radius2, axis=1))
        output.append(np.union1d(arr, halo).astype(int, copy=False))
    return output


def _candidate_cap(max_candidates: int | None, n_bins: int) -> int:
    if max_candidates is None:
        return int(n_bins)
    value = int(max_candidates)
    if value <= 0:
        return int(n_bins)
    return min(value, int(n_bins))
