"""Split-population, sequence-based selection and independent context readout.

Evaluation counts never enter the sequence classifier. Content is sequenceless;
it cannot by itself establish ordered replay or future predictive information.
"""

import hashlib

import numpy as np
from scipy.special import logsumexp

from .tirole_two_track import decode_counts


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def split_populations(unit_indices, session, seed=20260918, n_splits=5):
    ids = np.asarray(unit_indices, int)
    if len(ids) < 20 or len(np.unique(ids)) != len(ids):
        raise ValueError("at least 20 unique common RUN units required")
    order = np.random.default_rng(stable_seed(seed, session, "detector")).permutation(ids)
    n_detector = max(3, int(np.ceil(0.2 * len(order))))
    detector = np.sort(order[:n_detector])
    remaining = order[n_detector:]
    splits = []
    for split in range(n_splits):
        perm = np.random.default_rng(stable_seed(seed, session, "split", split)).permutation(remaining)
        n_eval = max(4, int(np.ceil(0.3 * len(perm))))
        splits.append((np.sort(perm[n_eval:]), np.sort(perm[:n_eval])))
    return detector, splits


def nested_subsets(inference, session, split, repeat, seed=20260918):
    order = np.random.default_rng(stable_seed(seed, session, split, repeat, "coverage")).permutation(inference)
    return {fraction: np.sort(order[: max(1, int(np.ceil(fraction * len(order))))]) for fraction in (1.0, 0.75, 0.5, 0.25)}


def event_bin_counts(session, start, end, bin_s=0.02):
    if end <= start or bin_s <= 0:
        raise ValueError("positive interval and bin width required")
    n = int(np.floor((end - start + 1e-9) / bin_s))
    if n < 1:
        return np.zeros((0, len(session.unit_ids)), int)
    # No short final bin gets a full-duration likelihood.
    edges = start + np.arange(n + 1) * bin_s
    actual_end = edges[-1]
    a, b = np.searchsorted(session.spike_times, [start, actual_end], side="left")
    bins = np.searchsorted(edges, session.spike_times[a:b], side="right") - 1
    good = (bins >= 0) & (bins < n)
    return np.bincount(bins[good] * len(session.unit_ids) + session.spike_units[a:b][good], minlength=n * len(session.unit_ids)).reshape(n, -1)


def weighted_correlation(posterior, centers):
    """Absolute position/time correlation for each track, arbitrary batch axes."""
    p = np.asarray(posterior, float)
    if p.ndim < 3 or p.shape[-1] != len(centers) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("finite nonnegative ... x time x track x position weights required")
    mass = p.sum(axis=(-3, -1))
    t = np.arange(p.shape[-3], dtype=float)[:, None, None]
    x = np.asarray(centers)[None, None, :]
    den = np.maximum(mass, 1e-300)
    mt = (p * t).sum(axis=(-3, -1)) / den
    mx = (p * x).sum(axis=(-3, -1)) / den
    vt = (p * t * t).sum(axis=(-3, -1)) / den - mt * mt
    vx = (p * x * x).sum(axis=(-3, -1)) / den - mx * mx
    cov = (p * t * x).sum(axis=(-3, -1)) / den - mt * mx
    sd = np.sqrt(np.maximum(vt, 0) * np.maximum(vx, 0))
    value = np.divide(np.abs(cov), sd, out=np.zeros_like(sd), where=sd > 1e-12)
    return np.clip(value, 0, 1)


def field_shift_posteriors(counts, rates, valid, shifts, bin_s=0.02):
    """Null shifts are indexed by full session unit identity, then subset upstream."""
    shifts = np.asarray(shifts, int)
    if shifts.ndim != 3 or shifts.shape[1:] != rates.shape[:2]:
        raise ValueError("shifts must have shape null x track x unit")
    p = rates.shape[2]
    index = (np.arange(p)[None, None, None, :] - shifts[:, :, :, None]) % p
    lam = np.take_along_axis(np.broadcast_to(rates, (len(shifts), *rates.shape)), index, axis=-1)
    lam = np.maximum(lam, 1e-4)
    ll = np.einsum("tu,skup->stkp", counts, np.log(lam)) - bin_s * lam.sum(axis=2)[:, None, :, :]
    ll -= np.log(valid.sum(axis=1))[None, None, :, None]
    ll[:, :, ~valid] = -np.inf
    return np.exp(ll - logsumexp(ll, axis=(-2, -1), keepdims=True))


def classify_sequence(counts, rates, valid, centers, shifts, time_permutations, alpha=0.025):
    """Two established shuffle families; Bonferroni two-track eventwise alpha .05."""
    counts = np.asarray(counts)
    n_active = int((counts.sum(axis=0) > 0).sum())
    n_nonempty = int((counts.sum(axis=1) > 0).sum())
    result = {
        "n_active_inference": n_active,
        "n_nonempty_bins": n_nonempty,
        "sequence_eligible": n_active >= 5 and n_nonempty >= 5,
        "sequence_accepted": False,
        "inferred_track": -1,
        "best_weighted_correlation": np.nan,
        "track1_p_time": np.nan,
        "track2_p_time": np.nan,
        "track1_p_field": np.nan,
        "track2_p_field": np.nan,
    }
    if not result["sequence_eligible"]:
        return result
    if len(shifts) < 39 or len(time_permutations) != len(shifts) or time_permutations.shape[1] != len(counts):
        raise ValueError("adequate paired shuffle count and event-length permutations required")
    expected = np.arange(len(counts))
    if not np.all(np.sort(time_permutations, axis=1) == expected):
        raise ValueError("each temporal null must permute whole population time bins")
    posterior = decode_counts(counts, rates, 0.02, valid)
    posterior[counts.sum(axis=1) == 0] = 0
    real = weighted_correlation(posterior, centers)
    time_scores = weighted_correlation(posterior[time_permutations], centers)
    field_post = field_shift_posteriors(counts, rates, valid, shifts)
    field_post[:, counts.sum(axis=1) == 0] = 0
    field_scores = weighted_correlation(field_post, centers)
    p_time = (1 + (time_scores >= real[None, :] - 1e-12).sum(axis=0)) / (1 + len(shifts))
    p_field = (1 + (field_scores >= real[None, :] - 1e-12).sum(axis=0)) / (1 + len(shifts))
    accepted = (p_time < alpha) & (p_field < alpha)
    best = int(np.argmax(np.where(accepted, real, -np.inf))) if accepted.any() else int(np.argmax(real))
    result.update(inferred_track=best + 1, best_weighted_correlation=float(real[best]), sequence_accepted=bool(accepted.any()))
    for k in range(2):
        result[f"track{k + 1}_p_time"] = float(p_time[k])
        result[f"track{k + 1}_p_field"] = float(p_field[k])
    return result


def content_readout(counts, rates, valid, swaps, conditional_count=False):
    """Independent evaluation-cell sequenceless track posterior with track-ID null.

    Conditional-count sensitivity removes the per-bin total-rate Poisson term;
    it assesses population identity given counts, not the event's overall rate.
    Positive log odds indicate track 1. Zero-spike windows supply no content.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != rates.shape[1] or rates.shape[0] != 2 or np.any(counts < 0):
        raise ValueError("invalid two-track evaluation arrays")
    nonempty = counts.sum(axis=1) > 0
    if not nonempty.any():
        return {"log_odds": np.nan, "z_log_odds": np.nan, "track2_probability": np.nan, "n_eval_spikes": 0, "content_supported": False}
    if swaps.ndim != 2 or swaps.shape[1] != rates.shape[1] or len(swaps) < 39:
        raise ValueError("at least 39 cellwise track-ID nulls required")
    both = np.concatenate([np.zeros((1, rates.shape[1]), bool), np.asarray(swaps, bool)])
    lam = np.where(both[:, None, :, None], rates[::-1][None, :, :, :], rates[None, :, :, :])
    lam = np.maximum(lam, 1e-4)
    if conditional_count:
        lam = lam / lam.sum(axis=2, keepdims=True)
    ll = np.einsum("tu,skup->stkp", counts[nonempty], np.log(lam))
    if not conditional_count:
        ll -= 0.02 * lam.sum(axis=2)[:, None, :, :]
    ll -= np.log(valid.sum(axis=1))[None, None, :, None]
    ll[:, :, ~valid] = -np.inf
    posterior = np.exp(ll - logsumexp(ll, axis=(-2, -1), keepdims=True))
    mass = posterior.sum(axis=(1, 3))
    odds = np.log(np.maximum(mass[:, 0], 1e-300)) - np.log(np.maximum(mass[:, 1], 1e-300))
    sd = float(np.std(odds[1:], ddof=1))
    z = (odds[0] - np.mean(odds[1:])) / sd if sd > 1e-12 else np.nan
    return {
        "log_odds": float(odds[0]),
        "z_log_odds": float(z),
        "track2_probability": float(mass[0, 1] / mass[0].sum()),
        "n_eval_spikes": int(counts.sum()),
        "content_supported": bool(np.isfinite(z)),
    }
