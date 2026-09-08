"""Position-free multinomial assembly controls with frozen cross-cell inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from .frozen_posterior_prediction import frozen_smoothed_marginal_log_score, posterior_sha256


@dataclass(frozen=True)
class AssemblyFit:
    probabilities: np.ndarray
    weights: np.ndarray
    global_probability: np.ndarray
    objective: float
    iterations: int
    converged: bool
    objective_trace: np.ndarray


def validate_counts(counts):
    k = np.asarray(counts)
    if k.ndim != 2 or not k.size or not np.isfinite(k).all() or (k < 0).any() or (k != np.floor(k)).any():
        raise ValueError("nonempty finite integer count matrix required")
    return k.astype(np.int64)


def multinomial_ll(counts, probabilities):
    k = validate_counts(counts)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] != k.shape[1] or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("positive finite aligned component probabilities required")
    p = p / p.sum(axis=0, keepdims=True)
    return k @ np.log(p) + (gammaln(k.sum(axis=1) + 1) - gammaln(k + 1).sum(axis=1))[:, None]


def fit_assembly(counts, n_components, seed, max_iter=1000):
    k = validate_counts(counts)
    informative = np.flatnonzero(k.sum(axis=1) > 0)
    if not isinstance(n_components, int) or n_components < 1 or len(informative) < n_components or max_iter < 1:
        raise ValueError("insufficient calibration for declared components")
    total = k.sum(axis=0).astype(float) + 100.0 / k.shape[1]
    reference = total / total.sum()
    alpha = 10.0
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(3):
        initial = k[rng.choice(informative, n_components, replace=False)].T + alpha * reference[:, None]
        p = initial / initial.sum(axis=0, keepdims=True)
        weights = np.ones(n_components) / n_components
        trace = []
        converged = False
        for iteration in range(max_iter + 1):
            joint = multinomial_ll(k, p) + np.log(weights)[None, :]
            likelihood = logsumexp(joint, axis=1)
            objective = float(likelihood.sum() + alpha * (reference[:, None] * np.log(p)).sum() + np.log(weights).sum() / n_components)
            if trace:
                change = objective - trace[-1]
                if change < -1e-8 * (1 + abs(trace[-1])):
                    raise ValueError("MAP-EM objective decreased")
                converged = abs(change) < 1e-8 * (1 + abs(trace[-1]))
            trace.append(objective)
            if converged or iteration == max_iter:
                break
            responsibility = np.exp(joint - likelihood[:, None])
            updated = k.T @ responsibility + alpha * reference[:, None]
            p = updated / updated.sum(axis=0, keepdims=True)
            weights = responsibility.sum(axis=0) + 1.0 / n_components
            weights /= weights.sum()
        fits.append(AssemblyFit(p, weights, reference, objective, iteration, converged, np.array(trace)))
    best = max(fits, key=lambda fit: fit.objective)
    if not best.converged:
        raise ValueError("highest calibration-objective fit did not converge")
    return best


def assignment_posterior(ll, weights, centers_s, mode):
    ll, weights = np.asarray(ll, float), np.asarray(weights, float)
    centers = np.asarray(centers_s, float)
    if (
        ll.ndim != 2
        or ll.shape[1] != len(weights)
        or centers.shape != (len(ll),)
        or (weights <= 0).any()
        or not np.isfinite(ll).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(centers).all()
        or (np.diff(centers) <= 0).any()
    ):
        raise ValueError("invalid likelihood, weights or times")
    logw = np.log(weights / weights.sum())
    if mode == "independent":
        q = ll + logw
        return q - logsumexp(q, axis=1, keepdims=True)
    if mode == "static":
        q = ll.sum(axis=0) + logw
        return np.tile(q - logsumexp(q), (len(ll), 1))
    if mode != "persistent":
        raise ValueError("unknown assembly dynamics")
    from scipy.sparse import csr_matrix

    from . import state_space
    from .duration_occupancy import _forward_backward_variable

    n = len(weights)
    w = weights / weights.sum()
    matrices = [csr_matrix(np.exp(-dt / 0.06) * np.eye(n) + (1 - np.exp(-dt / 0.06)) * w[:, None] * np.ones((1, n))) for dt in np.diff(centers)]
    adjusted = ll.copy()
    # The shared engine starts uniformly; this replaces only its initial prior.
    adjusted[0] += logw + np.log(n)
    _, posterior = _forward_backward_variable(state_space, adjusted, matrices)
    return posterior


def predict_heldout(counts, centers_s, train, held, fit):
    k = validate_counts(counts)
    train, held = np.asarray(train, int), np.asarray(held, int)
    if (
        not len(train)
        or not len(held)
        or len(set(train) | set(held)) != k.shape[1]
        or len(set(train) & set(held))
        or len(train) != len(set(train))
        or len(held) != len(set(held))
        or min(train.min(), held.min()) < 0
        or max(train.max(), held.max()) >= k.shape[1]
    ):
        raise ValueError("invalid full population partition")
    train_ll = multinomial_ll(k[:, train], fit.probabilities[train])
    post = {mode: assignment_posterior(train_ll, fit.weights, centers_s, mode) for mode in ("independent", "static", "persistent")}
    hashes = {m: posterior_sha256(q) for m, q in post.items()}
    held_ll = multinomial_ll(k[:, held], fit.probabilities[held])
    scores = {mode: frozen_smoothed_marginal_log_score(q, held_ll).total_log_score for mode, q in post.items()}
    scores["global"] = float(multinomial_ll(k[:, held], fit.global_probability[held, None]).sum())
    if any(hashes[m] != posterior_sha256(q) for m, q in post.items()):
        raise ValueError("held-out observations changed assignments")
    return scores, hashes
