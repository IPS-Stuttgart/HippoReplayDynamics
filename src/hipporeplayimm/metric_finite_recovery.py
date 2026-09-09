"""Finite conditional-identity simulations and neutral-origin forecasts."""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise

import numpy as np
from scipy.special import logsumexp

from .conditional_spatial_prediction import identity_likelihood
from .lagged_neural_prediction import mixture_scores
from .physical_neural_metric import STAY, identities, physical_cost, validate_kernel

SEED = 20260909
GENERATORS = ("physical", "neural", "stationary", "iid")
CONDITIONS = ("matched", "gain_drift", "map_error")
SUPPORTS = (1, 4)
MODELS = ("physical", "neural", "stationary", "iid")
ORIGINS = ("decoded", "known")
TRIALS = 64


def rng_for(*parts):
    key = "|".join(map(str, (SEED, *parts))).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], "little"))


def reconstruct_kernel(cost, parameters, name):
    beta = float(parameters[name + "__beta"])
    scale = float(parameters[name + "__cost_scale"])
    u = np.exp(parameters[name + "__log_scale"])
    w = np.exp(np.maximum(-beta * cost / scale, -700))
    np.fill_diagonal(w, 0)
    a = (1 - STAY) * u[:, None] * w * u[None, :] + STAY * np.eye(len(w))
    validate_kernel(a, target_entropy=0.5 * np.log(len(w) - 1))
    return a


def observation_stress(rates, centers, tag):
    rng = rng_for(tag, "observation_stress")
    gains = np.exp(0.35 * rng.standard_normal(len(rates)) - 0.35**2 / 2)
    smooth = np.exp(-physical_cost(centers) / (2 * 16**2))
    smooth /= smooth.sum(axis=1, keepdims=True)
    noise = rng.standard_normal(rates.shape) @ smooth.T
    noise /= np.sqrt((smooth**2).sum(axis=1))[None, :]
    field = np.exp(0.35 * noise - 0.35**2 / 2)
    estimated = rates * field
    estimated *= rates.mean(axis=1, keepdims=True) / estimated.mean(axis=1, keepdims=True)
    return gains, estimated


def sample_path(kernel, length, rng):
    if length < 3:
        raise ValueError("at least three full bins required")
    path = np.empty(length, dtype=np.int32)
    path[0] = rng.integers(len(kernel))
    for t in range(1, length):
        p = kernel[path[t - 1]]
        path[t] = rng.choice(len(p), p=p / p.sum())
    return path


def sample_identities(rates, cells, states, totals, rng):
    p = identities(rates[cells])
    totals = np.asarray(totals)
    if totals.shape != states.shape or np.any(totals < 0) or not np.array_equal(totals, np.round(totals)):
        raise ValueError("aligned nonnegative integer count profile required")
    return np.array([rng.multinomial(int(n), p[:, int(s)]) for s, n in zip(states, totals, strict=True)], dtype=np.int32)


def origin_target_indices(offsets):
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or len(offsets) < 2 or not np.isfinite(offsets).all() or not np.array_equal(offsets, np.round(offsets)) or offsets[0] != 0 or np.any(np.diff(offsets) < 3):
        raise ValueError("nonempty full-bin trial offsets required")
    offsets = offsets.astype(int)
    origins = np.concatenate([np.arange(a, b - 2) for a, b in pairwise(offsets)])
    lengths = np.diff(offsets) - 2
    group = np.r_[0, np.cumsum(lengths)]
    return origins, origins + 2, group


def decoded_origin(counts, rates):
    ll = identity_likelihood(counts, rates)
    return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))


def predict(origin, two_step):
    out = {name: origin @ a for name, a in two_step.items()}
    out["stationary"] = origin.copy()
    out["iid"] = np.full_like(origin, 1 / origin.shape[1])
    for q in out.values():
        if not np.isfinite(q).all() or np.any(q < 0) or np.max(abs(q.sum(axis=1) - 1)) > 1e-8:
            raise ValueError("improper forecast")
    return out


def score_batch(train_counts, held_counts, train_rates, held_rates, states, offsets, two_step):
    origin, target, groups = origin_target_indices(offsets)
    states = np.asarray(states)
    if set(two_step) != {"physical", "neural"}:
        raise ValueError("both moving forecasts required")
    if (
        len(train_counts) != offsets[-1]
        or len(held_counts) != offsets[-1]
        or states.shape != (offsets[-1],)
        or not np.array_equal(states, np.round(states))
        or np.any(states < 0)
        or np.any(states >= train_rates.shape[1])
    ):
        raise ValueError("aligned complete trial data required")
    states = states.astype(int)
    q = decoded_origin(train_counts[origin], train_rates)
    predictions = {"decoded": predict(q, two_step)}
    known = {name: a[states[origin]].copy() for name, a in two_step.items()}
    known["stationary"] = np.eye(q.shape[1])[states[origin]]
    known["iid"] = np.full_like(q, 1 / q.shape[1])
    predictions["known"] = known
    # Held targets are evaluated only after both forecast arms have been formed.
    ll = identity_likelihood(held_counts[target], held_rates)
    held_spikes = held_counts[target].sum(axis=1)
    train_spikes = train_counts[origin].sum(axis=1)
    rows = []
    for arm in ORIGINS:
        scores = {}
        for model in MODELS:
            values = mixture_scores(predictions[arm][model], ll)
            values[held_spikes == 0] = 0
            if not np.isfinite(values).all() or np.max(values) > 1e-8:
                raise ValueError("invalid predictive log score")
            scores[model] = np.add.reduceat(values, groups[:-1])
        for i in range(len(offsets) - 1):
            lo, hi = groups[i : i + 2]
            rows.append(
                {
                    "simulation_index": i,
                    "origin": arm,
                    "n_target_bins": hi - lo,
                    "n_held_target_spikes": int(held_spikes[lo:hi].sum()),
                    "n_train_origin_spikes": int(train_spikes[lo:hi].sum()),
                    **{"score_" + model: float(scores[model][i]) for model in MODELS},
                }
            )
    return rows


def null_threshold(values, alpha=0.05):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or np.isnan(values).any() or not 0 < alpha < 1:
        raise ValueError("nonempty valid null calibration required")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else float("inf")
