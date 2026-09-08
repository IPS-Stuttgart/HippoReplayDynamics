"""Forward-only neural forecasts with frozen and dwell-preserving controls."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .assembly_predictive_control import validate_counts
from .conditional_spatial_prediction import SpatialPredictionContext


def full_count_bins(counts, edges):
    x = validate_counts(counts)
    edges = np.asarray(edges, float)
    if edges.shape != (len(x) + 1,) or not np.isfinite(edges).all():
        raise ValueError("aligned finite count-bin edges required")
    widths = np.diff(edges)
    if (widths <= 0).any() or (widths > 0.02 + 1e-8).any() or not np.allclose(widths[:-1], 0.02, atol=1e-8, rtol=0):
        raise ValueError("only a final partial 20-ms bin is allowed")
    n = len(x) if np.isclose(widths[-1], 0.02, atol=1e-8, rtol=0) else len(x) - 1
    return x[:n], int(x[n:].sum())


class NeuralOperator:
    def __init__(self, initial, transition, occupancy):
        self.initial = np.asarray(initial, float)
        self.transition = np.asarray(transition, float)
        w = np.asarray(occupancy, float)
        n = len(self.initial)
        if n < 2 or w.shape != (n,) or self.transition.shape != (n, n):
            raise ValueError("aligned multiple-state parameters required")
        if not all(np.isfinite(a).all() and (a > 0).all() for a in (self.initial, self.transition, w)):
            raise ValueError("strictly positive finite neural parameters required")
        if not np.isclose(self.initial.sum(), 1) or not np.isclose(w.sum(), 1) or not np.allclose(self.transition.sum(axis=1), 1):
            raise ValueError("stochastic parameters required")
        d = np.diag(self.transition)
        self.dwell = (1 - d)[:, None] * w[None, :] / (1 - w)[:, None]
        np.fill_diagonal(self.dwell, d)
        np.testing.assert_allclose(self.dwell.sum(axis=1), 1, atol=1e-12)

    def step(self, q, dwell=False):
        return np.asarray(q) @ (self.dwell if dwell else self.transition)

    def expand(self, ll):
        return np.asarray(ll, float)

    def collapse(self, q):
        return np.asarray(q, float)


class SpatialOperator:
    def __init__(self, centers, imm=True):
        context = SpatialPredictionContext(centers)
        self.n_bins = len(centers)
        self.n_modes = 3 if imm else 1
        self.initial = np.full(self.n_modes * self.n_bins, 1 / (self.n_modes * self.n_bins))
        diffusion = context._gaussian_transition_matrix(context.centers, 60 * np.sqrt(0.02), 3.0)
        self.kernels = [context._gaussian_transition_matrix(context.centers, 2.0, 3.0), diffusion, None] if imm else [diffusion]
        self.mode = context._mode_transition_matrix(3, np.exp(-0.02 / 0.06)) if imm else np.ones((1, 1))
        for kernel in self.kernels:
            if kernel is not None:
                np.testing.assert_allclose(np.asarray(kernel.sum(axis=0)), 1, atol=1e-12)

    def step(self, q, dwell=False):
        q = np.asarray(q, float)
        shape = q.shape
        if q.shape[-1] != len(self.initial):
            raise ValueError("aligned latent state distribution required")
        x = q.reshape(-1, self.n_modes, self.n_bins)
        mixed = np.einsum("ij,bip->bjp", self.mode, x)
        out = np.empty_like(mixed)
        for j, kernel in enumerate(self.kernels):
            values = mixed[:, j]
            if kernel is None:
                out[:, j] = values.sum(axis=1, keepdims=True) / self.n_bins
            elif not dwell or self.n_bins == 1:
                out[:, j] = (kernel @ values.T).T
            else:
                diagonal = kernel.diagonal()
                remaining = (1 - diagonal) / (self.n_bins - 1)
                out[:, j] = values * (diagonal - remaining) + (values * remaining).sum(axis=1, keepdims=True)
        return out.reshape(shape)

    def expand(self, ll):
        ll = np.asarray(ll, float)
        if ll.ndim != 2 or ll.shape[1] != self.n_bins:
            raise ValueError("aligned position likelihood required")
        return np.tile(ll, (1, self.n_modes))

    def collapse(self, q):
        q = np.asarray(q, float)
        return q.reshape(-1, self.n_modes, self.n_bins).sum(axis=1)


def forward_filter(log_likelihood, operator):
    ll = operator.expand(log_likelihood)
    if ll.ndim != 2 or not len(ll) or ll.shape[1] != len(operator.initial) or not np.isfinite(ll).all():
        raise ValueError("finite aligned nonempty likelihood required")
    emission = np.exp(ll - ll.max(axis=1, keepdims=True))
    q = operator.initial.copy()
    filtered = np.empty_like(emission)
    for t, likelihood in enumerate(emission):
        if t:
            q = operator.step(q)
        q = q * likelihood
        total = q.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid filtered mass")
        q = q / total
        filtered[t] = q
    return filtered


def forecasts(log_likelihood, operator, horizons=(1, 2, 4)):
    if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("distinct positive integer horizons required")
    filtered = forward_filter(log_likelihood, operator)
    n = len(filtered)
    no_history = np.empty_like(filtered)
    prior = operator.initial.copy()
    for t in range(n):
        no_history[t] = prior
        prior = operator.step(prior)
    dynamic, dwell = filtered.copy(), filtered.copy()
    result = {}
    for h in range(1, max(horizons) + 1):
        dynamic = operator.step(dynamic)
        dwell = operator.step(dwell, dwell=True)
        if h not in horizons or h >= n:
            continue
        candidates = {"dynamic": dynamic[:-h], "dwell_only": dwell[:-h], "frozen": filtered[:-h], "no_history": no_history[h:], "same_time": filtered[h:]}
        result[h] = {name: operator.collapse(q).copy() for name, q in candidates.items()}
        for q in result[h].values():
            if not np.isfinite(q).all() or (q < -1e-12).any():
                raise ValueError("invalid forecast probabilities")
            np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-10)
    return result


def mixture_scores(probabilities, held_log_likelihood):
    q, ll = np.asarray(probabilities, float), np.asarray(held_log_likelihood, float)
    if q.shape != ll.shape or q.ndim != 2 or not len(q) or not np.isfinite(q).all() or not np.isfinite(ll).all() or (q < 0).any() or not np.allclose(q.sum(axis=1), 1):
        raise ValueError("normalized aligned forecast and held likelihood required")
    logq = np.full_like(q, -np.inf)
    np.log(q, out=logq, where=q > 0)
    return logsumexp(logq + ll, axis=1)
