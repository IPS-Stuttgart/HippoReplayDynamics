"""Matched reversible geometries and exact known-origin predictive checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import xlogy

STAY = float(np.exp(-0.02 / 0.06))
HORIZONS = (1, 2, 4)


def identities(rates):
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2 or min(rates.shape) < 2 or not np.isfinite(rates).all() or np.any(rates < 0) or np.any(rates.sum(axis=0) <= 0):
        raise ValueError("finite nonnegative cell-by-state rates with positive column totals required")
    return rates / rates.sum(axis=0, keepdims=True)


def physical_cost(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] < 4 or not np.isfinite(centers).all():
        raise ValueError("at least four finite spatial centers required")
    return cdist(centers, centers, metric="sqeuclidean")


def neural_cost(rates):
    p = np.sqrt(identities(rates))
    cost = np.clip(1 - p.T @ p, 0, 1)
    cost = (cost + cost.T) / 2
    np.fill_diagonal(cost, 0)
    return cost


def normalized_cost(cost):
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or len(cost) < 4 or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("finite nonnegative square cost required")
    if not np.allclose(cost, cost.T, atol=1e-12, rtol=0) or np.max(np.abs(np.diag(cost))) > 1e-12:
        raise ValueError("symmetric cost and zero diagonal required")
    positive = cost[np.triu_indices(len(cost), 1)]
    positive = positive[positive > 1e-12]
    if not len(positive):
        raise ValueError("uninformative metric: no separable states")
    scale = float(np.median(positive))
    return cost / scale, scale


def balanced_affinity(cost, beta, *, start=None, max_iter=10000, tolerance=1e-11):
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("finite nonnegative inverse scale required")
    w = np.exp(np.maximum(-beta * cost, -700))
    np.fill_diagonal(w, 0)
    logu = np.zeros(len(w)) if start is None else np.asarray(start, dtype=float).copy()
    for iteration in range(1, max_iter + 1):
        u = np.exp(logu)
        row = u * (w @ u)
        if not np.isfinite(row).all() or np.any(row <= 0):
            raise ValueError("matrix scaling became nonfinite")
        if np.max(np.abs(row - 1)) <= tolerance:
            b = u[:, None] * w * u[None, :]
            return b, logu, iteration
        logu -= 0.5 * np.log(row)
    raise ValueError("symmetric matrix scaling did not converge")


@dataclass
class MetricKernel:
    matrix: np.ndarray
    beta: float
    cost_scale: float
    log_scale: np.ndarray
    entropy: float
    target_entropy: float
    evaluations: int
    max_balance_iterations: int


def matched_kernel(cost, *, stay=STAY, entropy_fraction=0.5):
    if not 0 <= stay < 1 or not 0 < entropy_fraction < 1:
        raise ValueError("invalid fixed dwell or entropy fraction")
    cost, scale = normalized_cost(cost)
    target = entropy_fraction * np.log(len(cost) - 1)
    cache = {}

    def evaluate(beta):
        start = None if not cache else cache[min(cache, key=lambda key: abs(key - beta))][1]
        b, u, iterations = balanced_affinity(cost, beta, start=start)
        entropy = float(-xlogy(b, b).sum() / len(b))
        cache[beta] = (b, u, iterations, entropy)
        return entropy

    low, high = 0.0, 1.0
    evaluate(low)
    while evaluate(high) > target:
        high *= 2
        if high > 2**24:
            raise ValueError("metric cannot attain frozen transition entropy")
    beta = high
    for _ in range(50):
        beta = (low + high) / 2
        entropy = evaluate(beta)
        if abs(entropy - target) <= 1e-7:
            break
        if entropy > target:
            low = beta
        else:
            high = beta
    else:
        raise ValueError("entropy matching did not converge")
    b, u, _, entropy = cache[beta]
    matrix = (1 - stay) * b + stay * np.eye(len(b))
    validate_kernel(matrix, stay=stay, target_entropy=target)
    return MetricKernel(matrix, beta, scale, u, entropy, target, len(cache), max(v[2] for v in cache.values()))


def validate_kernel(matrix, *, stay=STAY, target_entropy=None):
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("invalid transition matrix")
    if np.max(np.abs(a.sum(axis=1) - 1)) > 1e-10 or np.max(np.abs(a - a.T)) > 1e-12 or np.max(np.abs(np.diag(a) - stay)) > 1e-12:
        raise ValueError("kernel violates stochasticity, uniform equilibrium or dwell")
    b = a.copy()
    np.fill_diagonal(b, 0)
    b /= 1 - stay
    entropy = float(-xlogy(b, b).sum() / len(b))
    if target_entropy is not None and abs(entropy - target_entropy) > 1.1e-7:
        raise ValueError("kernel violates entropy match")
    return entropy


def forecast_identities(kernel, rates, horizons=HORIZONS):
    p = identities(rates).T
    out = {}
    for step in range(1, max(horizons) + 1):
        p = kernel @ p
        if np.max(np.abs(p.sum(axis=1) - 1)) > 1e-9 or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("improper future identity distribution")
        if step in horizons:
            out[step] = p.copy()
    return out


def expected_difference(truth, first, second):
    for p in (truth, first, second):
        if not np.isfinite(p).all() or np.any(p <= 0) or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-9:
            raise ValueError("positive normalized identity predictions required")
    return float(np.mean(np.sum(truth * (np.log(first) - np.log(second)), axis=1)))


def oracle_screen(rates, held, physical, full_neural, train_neural):
    p = forecast_identities(physical, rates[held])
    n = forecast_identities(full_neural, rates[held])
    t = forecast_identities(train_neural, rates[held])
    rows = []
    for h in HORIZONS:
        row = {
            "horizon": h,
            "physical_true_physical_minus_train_neural": expected_difference(p[h], p[h], t[h]),
            "neural_true_train_neural_minus_physical": expected_difference(n[h], t[h], p[h]),
            "neural_true_oracle_neural_minus_physical": expected_difference(n[h], n[h], p[h]),
            "neural_true_oracle_minus_train_neural": expected_difference(n[h], n[h], t[h]),
        }
        for name, value in row.items():
            if name not in ("horizon", "neural_true_train_neural_minus_physical") and value < -1e-10:
                raise ValueError("proper-score KL inequality violated")
        rows.append(row)
    return rows
