"""Shared helpers for state-space replay decoders."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .models import LOG_ZERO


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    return max(float(sigma_cm_sqrt_s) * np.sqrt(max(float(dt_s), np.finfo(float).tiny)), np.finfo(float).eps)


def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= log_emission.shape[0]:
        return np.arange(log_emission.shape[0], dtype=int)
    selected = np.argpartition(log_emission, -top_k)[-top_k:]
    return selected[np.argsort(log_emission[selected])[::-1]]


def _adaptive_candidate_indices(
    log_emission: np.ndarray,
    top_k: int,
    *,
    min_mass: float = 0.0,
    max_candidates: int = 0,
    halo_bin_centers: np.ndarray | None = None,
    halo_radius_cm: float = 0.0,
) -> np.ndarray:
    """Return emission-ranked candidates with optional mass/halo expansion.

    ``top_k`` is always honored as a minimum support size.  ``min_mass`` then
    adds enough emission-ranked bins to cover that fraction of the current
    row's normalized emission mass.  ``halo_radius_cm`` optionally adds a local
    spatial halo around selected bins.  A positive ``max_candidates`` caps the
    final support by dropping the lowest-emission bins, but never below
    ``top_k`` unless the grid itself is smaller.

    The function only changes the candidate support.  Downstream recursions
    still normalize priors and transitions over the full grid, so evidences
    remain conservative truncated full-grid evidences.
    """

    values = np.asarray(log_emission, dtype=float)
    if values.ndim != 1:
        raise ValueError("log_emission must be one-dimensional")
    n_bins = values.shape[0]
    if n_bins == 0:
        return np.asarray([], dtype=int)
    if not 0.0 <= float(min_mass) <= 1.0:
        raise ValueError("min_mass must be in [0, 1]")

    selected = _top_candidate_indices(values, int(top_k))
    if 0.0 < float(min_mass) < 1.0 and selected.size < n_bins:
        finite = np.isfinite(values)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            order = finite_indices[np.argsort(values[finite_indices])[::-1]]
            log_norm = logsumexp(values[order])
            if np.isfinite(log_norm):
                cumulative = np.cumsum(np.exp(values[order] - log_norm))
                n_required = int(np.searchsorted(cumulative, float(min_mass), side="left") + 1)
                selected = np.union1d(selected, order[:n_required])

    if halo_radius_cm > 0.0:
        if halo_bin_centers is None:
            raise ValueError("halo_bin_centers must be provided when halo_radius_cm is positive")
        centers = np.asarray(halo_bin_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[0] != n_bins:
            raise ValueError("halo_bin_centers must have shape (n_bins, n_dimensions)")
        radius2 = float(halo_radius_cm) * float(halo_radius_cm)
        anchor_centers = centers[np.asarray(selected, dtype=int)]
        delta = centers[:, None, :] - anchor_centers[None, :, :]
        in_halo = np.any(np.sum(delta * delta, axis=2) <= radius2, axis=1)
        selected = np.union1d(selected, np.flatnonzero(in_halo))

    selected = _rank_candidate_indices(values, selected)
    cap = int(max_candidates)
    if cap > 0 and selected.size > cap:
        minimum = n_bins if top_k <= 0 else min(max(int(top_k), 1), n_bins)
        cap = max(minimum, min(cap, n_bins))
        selected = selected[:cap]
    return selected.astype(int, copy=False)


def _rank_candidate_indices(log_emission: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return unique candidate indices sorted by descending emission score."""

    unique = np.unique(np.asarray(indices, dtype=int))
    if unique.size == 0:
        return unique
    values = np.asarray(log_emission, dtype=float)
    return unique[np.argsort(values[unique])[::-1]]


def _validate_candidate_indices(candidates: list[np.ndarray], n_time: int, n_bins: int) -> None:
    if len(candidates) != n_time:
        raise ValueError("candidate_indices must contain one array per emission time bin")
    for time_index, curr in enumerate(candidates):
        arr = np.asarray(curr)
        if arr.ndim != 1:
            raise ValueError(f"candidate_indices[{time_index}] must be one-dimensional")
        if arr.size == 0:
            raise ValueError(f"candidate_indices[{time_index}] must not be empty")
        if np.any((arr < 0) | (arr >= n_bins)):
            raise ValueError(f"candidate_indices[{time_index}] contains an out-of-range bin")


def _candidate_log_masses(log_likelihood: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
    return [
        float(logsumexp(log_likelihood[time_index, curr]) - logsumexp(log_likelihood[time_index]))
        for time_index, curr in enumerate(candidates)
    ]


def _gaussian_transition_matrix(bin_centers: np.ndarray, sigma_cm: float, max_step_sigma: float) -> csr_matrix:
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    n_bins = bin_centers.shape[0]
    radius2 = (sigma_cm * max_step_sigma) ** 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(bin_centers):
        delta = bin_centers - center[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius2
        if not np.any(keep):
            keep[int(np.argmin(dist2))] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma_cm * sigma_cm))
        weights /= float(weights.sum())
        rows.extend(int(idx) for idx in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))


def _full_grid_normalized_pairwise_gaussian_log_prob(
    predicted: np.ndarray,
    observed: np.ndarray,
    all_observed: np.ndarray,
    sigma_cm: float,
) -> np.ndarray:
    """Evaluate candidate log transitions with full-grid normalization."""

    log_kernel = _pairwise_gaussian_log_prob(predicted, observed, sigma_cm)
    log_normalizer = logsumexp(
        _pairwise_gaussian_log_prob(predicted, all_observed, sigma_cm),
        axis=1,
        keepdims=True,
    )
    return log_kernel - log_normalizer


def _pairwise_gaussian_log_prob(predicted: np.ndarray, observed: np.ndarray, sigma_cm: float) -> np.ndarray:
    delta = predicted[:, None, :] - observed[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    return -0.5 * dist2 / (sigma_cm * sigma_cm)


def _scaled_emissions(log_likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)
    if not np.all(np.any(finite, axis=1)):
        raise ValueError("every emission row must contain at least one finite value")
    offsets = np.max(np.where(finite, values, -np.inf), axis=1)
    shifted = np.where(finite, values - offsets[:, None], -np.inf)
    scaled = np.exp(np.clip(shifted, -745.0, 0.0))
    scaled[~finite] = 0.0
    return scaled, offsets


def _as_log_probs(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    out = np.full(probs.shape, LOG_ZERO, dtype=float)
    positive = probs > 0.0
    out[positive] = np.log(probs[positive])
    return out - logsumexp(out, axis=1)[:, None]


def _mean_entropy(trajectory_log_posterior: np.ndarray) -> float:
    posterior = np.exp(trajectory_log_posterior)
    return float(np.mean(-np.sum(posterior * trajectory_log_posterior, axis=1)))


def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    if n_modes < 2:
        return np.ones((n_modes, n_modes), dtype=float)
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("mode_stickiness must be in [0, 1]")
    off_diag = (1.0 - stickiness) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, stickiness)
    return matrix
