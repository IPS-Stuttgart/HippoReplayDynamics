"""Grid-weight mixture inference; regional aggregation is downstream only."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.special import logsumexp


def grid_likelihood(counts, rates, observation, total_counts=None, full_rate_sum=None):
    counts, rates = np.asarray(counts), np.asarray(rates)
    if counts.ndim != 2 or rates.ndim != 2 or counts.shape[1] != rates.shape[0]:
        raise ValueError("invalid counts/rates shape")
    if not np.isfinite(counts).all() or not np.isfinite(rates).all() or (counts < 0).any() or (rates <= 0).any():
        raise ValueError("finite nonnegative counts and positive rates required")
    n, rate_sum = counts.sum(axis=1), rates.sum(axis=0)
    ll = counts @ np.log(rates)
    if observation == "conditional_multinomial":
        ll -= n[:, None] * np.log(rate_sum)
    elif observation == "poisson":
        ll -= 0.02 * rate_sum
    elif observation == "fixed_total_censored":
        total, full = np.asarray(total_counts), np.asarray(full_rate_sum)
        if total.shape != n.shape or full.shape != rate_sum.shape or not np.isfinite(total).all() or not np.isfinite(full).all() or (total < n).any() or (full <= 0).any():
            raise ValueError("valid whole-population counts and rates required")
        other = full - rate_sum
        if (other < -1e-9).any():
            raise ValueError("subset intensity exceeds full intensity")
        if np.all(np.abs(other) < 1e-9):
            if np.any(total != n):
                raise ValueError("unobserved count with no unobserved intensity")
        elif (other <= 0).any():
            raise ValueError("other category must have positive intensity at every bin")
        else:
            ll += (total - n)[:, None] * np.log(other)
        ll -= total[:, None] * np.log(full)
    else:
        raise ValueError("unknown observation model")
    return ll


def smoothing_matrix(grid):
    grid = np.asarray(grid, float)
    if len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("nonempty finite spatial grid required")
    tree = cKDTree(grid)
    spacing = float(np.min(tree.query(grid, k=2)[0][:, 1]))
    if spacing <= 0:
        raise ValueError("duplicate grid locations")
    pairs = np.asarray(sorted(tree.query_pairs(spacing * 1.01)), int)
    if len(pairs) == 0:
        raise ValueError("grid has no nearest-neighbor edges")
    i, j = pairs.T
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    values = np.concatenate([np.ones(2 * len(i)), -np.ones(2 * len(i))])
    return coo_matrix((values, (rows, cols)), shape=(len(grid), len(grid))).tocsr() * (len(grid) ** 2 / len(pairs))


def normalize_likelihood(log_likelihood):
    ll = np.asarray(log_likelihood, float)
    if ll.ndim != 2 or not len(ll) or ll.shape[1] < 2 or not np.isfinite(ll).all():
        raise ValueError("finite nonempty window-by-grid likelihood required")
    return np.exp(ll - ll.max(axis=1, keepdims=True))


def em_update(likelihood, pi, weights=None):
    w = np.ones(len(likelihood)) / len(likelihood) if weights is None else np.asarray(weights) / np.sum(weights)
    denominator = likelihood @ pi
    if (denominator <= 0).any():
        raise ValueError("zero mixture likelihood")
    updated = pi * (likelihood.T @ (w / denominator))
    return updated / updated.sum()


def fit_grid(log_likelihood, smooth, penalty=0.0, maxiter=2000, tolerance=1e-6):
    likelihood = normalize_likelihood(log_likelihood)
    n, k = likelihood.shape
    if smooth.shape != (k, k) or penalty < 0:
        raise ValueError("invalid penalty")
    flat = np.ones(k) / k
    first = em_update(likelihood, flat)
    pi = flat.copy()
    for _ in range(20):
        pi = em_update(likelihood, pi)
    first20 = pi.copy()

    def on_simplex(p):
        denominator = likelihood @ p
        if (denominator <= 0).any():
            raise ValueError("nonpositive mixture probability")
        lap = smooth @ p
        objective = -np.log(denominator).mean() + penalty * 0.5 * float(p @ lap)
        gradient = -(likelihood.T @ (1 / denominator)) / n + penalty * lap
        return float(objective), gradient

    # Normalize nonnegative coordinates; the radial term fixes their scale.
    # A final simplex dual gap certifies the convex objective, not regional identification.
    def objective(u):
        total = float(u.sum())
        if total <= 1e-20:
            return 1e100, np.full(k, -1e10)
        p = u / total
        value, gradient = on_simplex(p)
        return value + 0.5 * (total - 1) ** 2, (gradient - float(gradient @ p)) / total + (total - 1)

    iterations = 0
    message = ""
    for _ in range(3):
        result = minimize(
            objective, pi, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * k, options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50, "maxcor": 20}
        )
        pi = np.maximum(result.x, 0)
        pi /= pi.sum()
        value, gradient = on_simplex(pi)
        gap = max(0.0, float(gradient @ pi - gradient.min()))
        iterations += int(result.nit)
        message = str(result.message)
        if gap <= tolerance:
            break
    return {
        "pi": pi,
        "first": first,
        "em20": first20,
        "objective": value,
        "dual_gap": gap,
        "converged": bool(gap <= tolerance),
        "iterations": iterations,
        "em_fixed_point_l1": float(np.abs(em_update(likelihood, pi) - pi).sum()),
        "flat_likelihood": bool(np.max(np.ptp(likelihood, axis=1)) <= 1e-12),
        "optimizer_message": message,
    }


def predictive_score(log_likelihood, pi):
    with np.errstate(divide="ignore"):
        return float(logsumexp(log_likelihood + np.log(pi)[None, :], axis=1).mean())
