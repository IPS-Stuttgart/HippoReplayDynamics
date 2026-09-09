"""Exact synthetic-generator classification and controlled information losses."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from .conditional_spatial_prediction import identity_likelihood
from .lagged_neural_prediction import NeuralOperator, mixture_scores
from .metric_finite_recovery import MODELS, decoded_origin, origin_target_indices, predict


def whole_evidence(log_likelihood, offsets, transitions):
    ll = np.asarray(log_likelihood, float)
    origin_target_indices(offsets)
    offsets = np.asarray(offsets, int)
    if ll.ndim != 2 or ll.shape[0] != offsets[-1] or not np.isfinite(ll).all() or ll.max() > 1e-8:
        raise ValueError("finite proper full-event log likelihood required")
    if not transitions or not set(transitions).issubset({"physical", "neural"}):
        raise ValueError("named metric kernels required")
    n = ll.shape[1]
    lengths = np.diff(offsets)
    result = {}
    for name, transition in transitions.items():
        operator = NeuralOperator(np.full(n, 1 / n), transition, np.full(n, 1 / n))
        q = np.tile(operator.initial, (len(lengths), 1))
        z = np.zeros(len(lengths))
        for t in range(lengths.max()):
            active = np.flatnonzero(lengths > t)
            prediction = operator.step(q[active]) if t else q[active]
            emission = ll[offsets[active] + t]
            maximum = emission.max(axis=1)
            mass = prediction * np.exp(emission - maximum[:, None])
            total = mass.sum(axis=1)
            if (total <= 0).any() or not np.isfinite(total).all():
                raise ValueError("invalid forward mass")
            q[active] = mass / total[:, None]
            z[active] += maximum + np.log(total)
        result[name] = z
    result["stationary"] = np.array([logsumexp(ll[a:b].sum(axis=0)) - np.log(n) for a, b in itertools.pairwise(offsets)])
    result["iid"] = np.array([(logsumexp(ll[a:b], axis=1) - np.log(n)).sum() for a, b in itertools.pairwise(offsets)])
    if any(not np.isfinite(z).all() or (z > 1e-8).any() for z in result.values()):
        raise ValueError("invalid whole-event probability")
    return result


def path_evidence(states, offsets, transitions):
    origin_target_indices(offsets)
    states, offsets = np.asarray(states), np.asarray(offsets, int)
    n = len(transitions["physical"])
    if states.shape != (offsets[-1],) or not np.array_equal(states, np.round(states)) or (states < 0).any() or (states >= n).any():
        raise ValueError("valid aligned latent path required")
    states = states.astype(int)
    result = {m: [] for m in MODELS}
    for a, b in itertools.pairwise(offsets):
        path = states[a:b]
        for name in ("physical", "neural"):
            p = transitions[name][path[:-1], path[1:]]
            if not np.isfinite(p).all() or (p <= 0).any():
                raise ValueError("positive matched metric probabilities required")
            result[name].append(-np.log(n) + np.log(p).sum())
        result["stationary"].append(-np.log(n) if (path == path[0]).all() else -np.inf)
        result["iid"].append(-(b - a) * np.log(n))
    return {k: np.asarray(v) for k, v in result.items()}


def lagged_evidence(x, y, train_rates, held_rates, offsets, two_step):
    origins, targets, groups = origin_target_indices(offsets)
    q = decoded_origin(x[origins], train_rates)
    forecasts = predict(q, two_step)
    held = identity_likelihood(y[targets], held_rates)
    counts = y[targets].sum(axis=1)
    scores = {}
    for model in MODELS:
        s = mixture_scores(forecasts[model], held)
        s[counts == 0] = 0
        scores[model] = np.add.reduceat(s, groups[:-1])
    support = np.add.reduceat(counts, groups[:-1])
    training = np.add.reduceat(x[origins].sum(axis=1), groups[:-1])
    return scores, (support > 0) & (training > 0)
