"""Proper frozen-context held-out population prediction without dynamics."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import logsumexp

from .conditional_spatial_prediction import identity_likelihood
from .frozen_posterior_prediction import posterior_sha256


def context_priors(context_ids):
    contexts = np.asarray(context_ids)
    unique, counts = np.unique(contexts, return_counts=True)
    if not len(contexts):
        raise ValueError("nonempty contexts required")
    sizes = dict(zip(unique, counts, strict=True))
    return np.array([1 / len(unique) / sizes[c] for c in contexts])


def infer_contexts(counts, rate_maps, compositions, train, context_ids):
    counts = np.asarray(counts)
    train = np.asarray(train, int)
    prior = context_priors(context_ids)
    if len(rate_maps) != len(prior) or len(compositions) != len(prior):
        raise ValueError("template metadata mismatch")
    spatial_posts, marginal, global_ll = [], [], []
    for rates, composition in zip(rate_maps, compositions, strict=True):
        ll = identity_likelihood(counts[:, train], rates[train])
        normalizer = logsumexp(ll, axis=1)
        spatial_posts.append(ll - normalizer[:, None])
        marginal.append(float((normalizer - np.log(ll.shape[1])).sum()))
        global_ll.append(float(identity_likelihood(counts[:, train], composition[train, None]).sum()))
    weights = {}
    for name, values in (("iid", marginal), ("global", global_ll)):
        w = np.log(prior) + values
        weights[name] = w - logsumexp(w)
    return {"spatial_posts": spatial_posts, "weights": weights, "prior": prior, "context_ids": np.asarray(context_ids)}


def score_frozen_contexts(counts, rate_maps, compositions, held, inferred, current_context, event_global=None):
    counts = np.asarray(counts)
    held = np.asarray(held, int)
    context_ids = inferred["context_ids"]
    mask = context_ids == current_context
    if not mask.any() or not len(held):
        raise ValueError("known current context and heldout cells required")
    before = [posterior_sha256(x) for x in inferred["spatial_posts"]] + [posterior_sha256(x[None, :]) for x in inferred["weights"].values()]
    spatial, global_scores = [], []
    for rates, composition, post in zip(rate_maps, compositions, inferred["spatial_posts"], strict=True):
        ll = identity_likelihood(counts[:, held], rates[held])
        spatial.append(logsumexp(post + ll, axis=1))
        global_scores.append(identity_likelihood(counts[:, held], composition[held, None])[:, 0])
    result = {}
    for model, values in (("iid", spatial), ("global", global_scores)):
        values = np.asarray(values)
        w = inferred["weights"][model]
        result[f"mix_{model}"] = float(logsumexp(values + w[:, None], axis=0).sum())
        restricted = w[mask] - logsumexp(w[mask])
        result[f"current_{model}"] = float(logsumexp(values[mask] + restricted[:, None], axis=0).sum())
        result[f"blind_{model}"] = float(logsumexp(values + np.log(inferred["prior"])[:, None], axis=0).sum())
    if event_global is not None:
        result["event_global"] = float(identity_likelihood(counts[:, held], np.asarray(event_global)[held, None]).sum())
    after = [posterior_sha256(x) for x in inferred["spatial_posts"]] + [posterior_sha256(x[None, :]) for x in inferred["weights"].values()]
    if before != after:
        raise ValueError("heldout prediction changed training inference")
    if not np.isfinite(list(result.values())).all() or max(result.values()) > 1e-8:
        raise ValueError("invalid predictive log probabilities")
    for model, logw in inferred["weights"].items():
        for c in np.unique(context_ids):
            result[f"{model}_p_{c}"] = float(np.exp(logw[context_ids == c]).sum())
    result["training_hash"] = hashlib.sha256("|".join(before).encode()).hexdigest()
    result["heldout_used_for_inference"] = False
    return result


def predict_contexts(counts, rate_maps, compositions, train, held, context_ids, current_context, event_global=None):
    if sorted([*train, *held]) != list(range(np.asarray(counts).shape[1])) or not len(train) or not len(held):
        raise ValueError("disjoint nonempty complete neural partition required")
    inferred = infer_contexts(counts, rate_maps, compositions, train, context_ids)
    return score_frozen_contexts(counts, rate_maps, compositions, held, inferred, current_context, event_global)
