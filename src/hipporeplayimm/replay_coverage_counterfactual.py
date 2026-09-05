"""Paired spike thinning, pooled identities, and oracle restoration controls."""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def pooled_rates(full_rates, subset):
    """Retained cell maps plus the sum of removed maps: data-only coarsening."""
    rates = np.asarray(full_rates, dtype=float)
    removed = np.ones(len(rates), bool)
    removed[subset] = False
    if not removed.any():
        raise ValueError("pool requires removed cells")
    return np.vstack([rates[subset], rates[removed].sum(axis=0)])


def restored_rates(full_rates, subset):
    """Keep full population intensity; redistribute removed identities.

    Lambda'_i(x) = Lambda_full(x) * lambda_i(x) / Lambda_subset(x).
    This is a counterfactual encoding model, not a fitted biological correction.
    """
    rates = np.asarray(full_rates, dtype=float)
    if rates.ndim != 2 or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("finite positive cell-by-state rates required")
    retained = rates[subset]
    return retained * (rates.sum(axis=0) / retained.sum(axis=0))[None, :]


def paired_counts(path_rates, subsets, rate_scales, fine_s, seed):
    """Nested exposures with pooled identities and secondary oracle restoration.

    Draw the highest exposure once, then binomially thin to each lower exposure.
    Pooling sums only removed-cell counts, without using position or randomness.
    For the oracle control, reassign removed spikes at highest exposure; hypergeometric
    thinning pairs their retained labels at lower exposures. Native kept spikes
    are never changed. Every restoration has the SAME population total as its
    full reference in EVERY fine-time bin, not just the same expected total.
    """
    rates = np.asarray(path_rates, float)
    scales = sorted(rate_scales, reverse=True)
    if rates.ndim != 2 or min(rates.shape) < 1 or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("finite positive fine-time rates required")
    if not scales or len(set(scales)) != len(scales) or not np.isfinite(scales).all() or min(scales) <= 0 or not np.isfinite(fine_s) or fine_s <= 0:
        raise ValueError("invalid exposures or time step")
    for subset in subsets.values():
        subset = np.asarray(subset)
        if subset.ndim != 1 or not len(subset) or not np.issubdtype(subset.dtype, np.integer) or len(np.unique(subset)) != len(subset) or np.any((subset < 0) | (subset >= rates.shape[1])):
            raise ValueError("invalid cell subset")
    if 1. not in subsets or not np.array_equal(subsets[1.], np.arange(rates.shape[1])):
        raise ValueError("full reference subset required")
    seeds = np.random.SeedSequence(seed).spawn(1 + len(subsets))
    rng = np.random.default_rng(seeds[0])
    native = {scales[0]: rng.poisson(rates * fine_s * scales[0])}
    for high, low in pairwise(scales):
        native[low] = rng.binomial(native[high], low / high)
    result = {}
    for ordinal, (fraction, subset) in enumerate(sorted(subsets.items(), reverse=True)):
        for scale in scales:
            result[scale, "native", fraction] = native[scale][:, subset]
        if fraction == 1.:
            continue
        for scale in scales:
            kept = native[scale][:, subset]
            removed_total = native[scale].sum(axis=1) - kept.sum(axis=1)
            result[scale, "pooled_removed", fraction] = np.column_stack([kept, removed_total])
        rng = np.random.default_rng(seeds[ordinal + 1])
        kept_rates = rates[:, subset]
        probabilities = kept_rates / kept_rates.sum(axis=1)[:, None]
        high = scales[0]
        removed = native[high].sum(axis=1) - native[high][:, subset].sum(axis=1)
        extras = np.array([rng.multinomial(int(n), p) for n, p in zip(removed, probabilities, strict=True)])
        result[high, "count_restored", fraction] = native[high][:, subset] + extras
        for low in scales[1:]:
            removed = native[low].sum(axis=1) - native[low][:, subset].sum(axis=1)
            extras = np.array([rng.multivariate_hypergeometric(previous, int(n))
                               for previous, n in zip(extras, removed, strict=True)])
            result[low, "count_restored", fraction] = native[low][:, subset] + extras
    for (scale, family, fraction), counts in result.items():
        if family in {"count_restored", "pooled_removed"}:
            if not np.array_equal(counts.sum(axis=1), native[scale].sum(axis=1)):
                raise AssertionError("restoration changed population totals")
            if family == "count_restored" and np.any(counts < native[scale][:, subsets[fraction]]):
                raise AssertionError("restoration removed retained-cell spikes")
            if family == "pooled_removed" and not np.array_equal(counts[:, :-1], native[scale][:, subsets[fraction]]):
                raise AssertionError("pooling changed retained-cell spikes")
        for high in scales:
            if high > scale and np.any(counts > result[high, family, fraction]):
                raise AssertionError("exposure coupling is not nested")
    return result


def aggregate_fine_counts(counts, n_per_base):
    counts = np.asarray(counts)
    if counts.ndim != 2 or n_per_base < 1 or len(counts) % n_per_base:
        raise ValueError("fine counts must contain complete base bins")
    return counts.reshape(-1, n_per_base, counts.shape[1]).sum(axis=1)


def shuffle_base_path(path, permutation):
    """Whole 5 ms block null; retain within-block trajectory and observations."""
    n = path["n_per_base"]
    blocks = len(path["midpoints_cm"]) // n
    permutation = np.asarray(permutation)
    if not np.array_equal(np.sort(permutation), np.arange(blocks)):
        raise ValueError("invalid whole-bin permutation")
    shuffled = {}
    for key in ["midpoints_cm", "covariate"]:
        value = path[key]
        shuffled[key] = value.reshape((blocks, n) + value.shape[1:])[permutation].reshape(value.shape)
    edge_start = path["edges_cm"][:-1].reshape(blocks, n, 2)[permutation].reshape(-1, 2)
    last_edge = path["edges_cm"][(permutation[-1] + 1) * n]
    shuffled["edges_cm"] = np.vstack([edge_start, last_edge])
    shuffled["speed_cm_s"] = np.full(len(edge_start), np.nan)
    shuffled["fine_s"], shuffled["n_per_base"] = path["fine_s"], n
    return shuffled
