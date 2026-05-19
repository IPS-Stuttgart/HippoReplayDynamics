"""Adaptive candidate-support helpers for candidate-pruned replay models."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def adaptive_top_candidate_indices(
    log_emission: np.ndarray,
    top_k: int,
    *,
    min_log_mass: float | None = None,
) -> np.ndarray:
    """Return emission candidates with optional adaptive log-mass coverage.

    ``top_k`` remains the guaranteed minimum beam size.  If ``min_log_mass`` is
    finite, the returned support is enlarged in decreasing emission-likelihood
    order until the selected emission mass satisfies

        logsumexp(selected) - logsumexp(all bins) >= min_log_mass.

    Since this is a log mass fraction, useful values are non-positive; for
    example ``np.log(0.99)`` asks for at least 99% of the emission mass whenever
    that requires more than ``top_k`` bins.  Non-finite/``None`` keeps the legacy
    fixed top-k behavior.
    """

    values = np.asarray(log_emission, dtype=float)
    if values.ndim != 1:
        raise ValueError("log_emission must be one-dimensional")
    n_bins = int(values.shape[0])
    if n_bins == 0:
        return np.empty(0, dtype=int)
    if top_k <= 0 or top_k >= n_bins:
        return np.arange(n_bins, dtype=int)

    selected = np.argpartition(values, -top_k)[-top_k:]
    selected = selected[np.argsort(values[selected])[::-1]]
    if min_log_mass is None or not np.isfinite(float(min_log_mass)):
        return selected.astype(int, copy=False)

    threshold = float(min_log_mass)
    if threshold > 0.0:
        raise ValueError("min_log_mass must be a log mass fraction <= 0")

    total = float(logsumexp(values))
    if not np.isfinite(total):
        return selected.astype(int, copy=False)
    if float(logsumexp(values[selected]) - total) >= threshold:
        return selected.astype(int, copy=False)

    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return selected.astype(int, copy=False)
    ranked = finite[np.argsort(values[finite])[::-1]]
    keep = {int(index) for index in selected}
    running = -np.inf
    for index in ranked:
        keep.add(int(index))
        running = float(np.logaddexp(running, values[index]))
        if running - total >= threshold:
            break
    return np.fromiter((int(index) for index in ranked if int(index) in keep), dtype=int)


def augment_candidates_with_momentum_predictions(
    candidates: list[np.ndarray],
    bin_centers: np.ndarray,
    *,
    predicted_top_k: int,
    velocity_decay: float,
    prediction_halo_cm: float = 0.0,
) -> list[np.ndarray]:
    """Union emission candidates with states near momentum predictions.

    Forward predictions are built from adjacent high-emission candidate pairs;
    backward predictions add time-reversed support so the first bins are not
    forced to be emission-only.  ``prediction_halo_cm`` optionally adds all grid
    bins within a spatial radius around each predicted point instead of only the
    nearest bin.  This gives the second-order recursion a bounded local support
    around dynamically plausible states whose immediate emission rank is low.
    """

    if predicted_top_k <= 0:
        return [np.asarray(curr, dtype=int).copy() for curr in candidates]
    top_k = max(1, int(predicted_top_k))
    augmented = [set(int(index) for index in np.asarray(curr, dtype=int)) for curr in candidates]

    for time_index in range(2, len(candidates)):
        prev_prev = np.asarray(candidates[time_index - 2], dtype=int)[:top_k]
        prev = np.asarray(candidates[time_index - 1], dtype=int)[:top_k]
        if prev_prev.size == 0 or prev.size == 0:
            continue
        predictions = bin_centers[prev][None, :, :] + velocity_decay * (
            bin_centers[prev][None, :, :] - bin_centers[prev_prev][:, None, :]
        )
        _add_prediction_support(augmented[time_index], bin_centers, predictions, prediction_halo_cm)

    if abs(velocity_decay) > np.finfo(float).eps:
        for time_index in range(len(candidates) - 2):
            nxt = np.asarray(candidates[time_index + 1], dtype=int)[:top_k]
            nxt_nxt = np.asarray(candidates[time_index + 2], dtype=int)[:top_k]
            if nxt.size == 0 or nxt_nxt.size == 0:
                continue
            predictions = bin_centers[nxt][None, :, :] - (
                bin_centers[nxt_nxt][:, None, :] - bin_centers[nxt][None, :, :]
            ) / velocity_decay
            _add_prediction_support(augmented[time_index], bin_centers, predictions, prediction_halo_cm)

    return [np.fromiter(sorted(curr), dtype=int) for curr in augmented]


def _add_prediction_support(
    target: set[int],
    bin_centers: np.ndarray,
    predictions: np.ndarray,
    prediction_halo_cm: float,
) -> None:
    flat = predictions.reshape(-1, bin_centers.shape[1])
    if flat.size == 0:
        return
    radius = max(float(prediction_halo_cm), 0.0)
    radius2 = radius * radius
    for predicted in flat:
        dist2 = np.sum((bin_centers - predicted[None, :]) ** 2, axis=1)
        if radius > 0.0:
            target.update(int(index) for index in np.flatnonzero(dist2 <= radius2))
        else:
            target.add(int(np.argmin(dist2)))
