"""Calibration-only burst-time cell composition, without latent path inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from .assembly_predictive_control import validate_counts


@dataclass(frozen=True)
class BurstPhaseFit:
    probabilities: np.ndarray  # knots x cells
    global_probability: np.ndarray
    weighted_counts: np.ndarray
    n_calibration_bins: int


def event_phase(times, edges):
    times, edges = np.asarray(times, float), np.asarray(edges, float)
    if times.ndim != 1 or not len(times) or edges.shape != (len(times) + 1,):
        raise ValueError("aligned event times/edges required")
    if not np.isfinite(times).all() or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
        raise ValueError("finite strictly increasing event clock required")
    if not np.allclose(times, (edges[:-1] + edges[1:]) / 2, rtol=0, atol=1e-8):
        raise ValueError("count-bin centers must match edge midpoints")
    return (times - edges[0]) / (edges[-1] - edges[0])


def phase_weights(phase, n_knots):
    phase = np.asarray(phase, float)
    if phase.ndim != 1 or not len(phase) or not np.isfinite(phase).all() or (phase < 0).any() or (phase > 1).any():
        raise ValueError("phase must lie in [0,1]")
    if isinstance(n_knots, bool) or not isinstance(n_knots, int) or n_knots < 1:
        raise ValueError("positive integer knot count required")
    coordinate = np.clip(phase * n_knots - 0.5, 0, n_knots - 1)
    lo = np.floor(coordinate).astype(int)
    hi = np.minimum(lo + 1, n_knots - 1)
    fraction = coordinate - lo
    w = np.zeros((len(phase), n_knots))
    np.add.at(w, (np.arange(len(phase)), lo), 1 - fraction)
    np.add.at(w, (np.arange(len(phase)), hi), fraction)
    return w


def fit_burst_phase(sequences, phases, n_knots=5):
    if not sequences or len(sequences) != len(phases):
        raise ValueError("aligned nonempty calibration sequences required")
    arrays = [validate_counts(x) for x in sequences]
    if len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("cell identities must align across events")
    nc = arrays[0].shape[1]
    total = sum((x.sum(axis=0) for x in arrays), start=np.zeros(nc))
    global_p = (total + 100.0 / nc) / (total.sum() + 100.0)
    weighted = np.zeros((n_knots, nc))
    for x, phase in zip(arrays, phases, strict=True):
        w = phase_weights(phase, n_knots)
        if len(w) != len(x):
            raise ValueError("phase/count length mismatch")
        weighted += w.T @ x
    p = weighted + 100.0 * global_p[None, :]
    p /= p.sum(axis=1, keepdims=True)
    return BurstPhaseFit(p, global_p, weighted, sum(len(x) for x in arrays))


def predictive_score(counts, probabilities, held):
    x = validate_counts(counts)
    p = np.asarray(probabilities, float)
    held = np.asarray(held)
    if p.shape != x.shape or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("positive aligned time-by-cell probabilities required")
    if held.ndim != 1 or not len(held) or not np.issubdtype(held.dtype, np.integer) or len(np.unique(held)) != len(held) or (held < 0).any() or (held >= x.shape[1]).any():
        raise ValueError("valid distinct held-out indices required")
    y = x[:, held]
    q = p[:, held]
    q = q / q.sum(axis=1, keepdims=True)
    value = gammaln(y.sum(axis=1) + 1) - gammaln(y + 1).sum(axis=1) + (y * np.log(q)).sum(axis=1)
    return float(value.sum())


def predict_burst_phase(counts, phase, held, fit):
    p = phase_weights(phase, len(fit.probabilities)) @ fit.probabilities
    return predictive_score(counts, p, held)


def score_orders(counts, probabilities, held, orders):
    x = validate_counts(counts)
    p = np.asarray(probabilities, float)
    order = np.asarray(orders)
    if (
        order.ndim != 2
        or order.shape[1] != len(x)
        or not np.issubdtype(order.dtype, np.integer)
        or not np.array_equal(np.sort(order, axis=1), np.tile(np.arange(len(x)), (len(order), 1)))
    ):
        raise ValueError("whole-bin permutations required")
    predictive_score(x, p, held)
    y = x[:, held]
    q = p[:, held] / p[:, held].sum(axis=1, keepdims=True)
    coefficient = (gammaln(y.sum(axis=1) + 1) - gammaln(y + 1).sum(axis=1)).sum()
    return coefficient + np.einsum("ktc,tc->k", y[order], np.log(q))
