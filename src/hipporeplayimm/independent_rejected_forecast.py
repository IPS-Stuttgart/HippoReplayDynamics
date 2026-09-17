"""Independent detection and genuine held-out forecasts of rejected events."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .conditional_spatial_prediction import identity_likelihood
from .lagged_neural_prediction import forward_filter, mixture_scores
from .training_continuity import classify_training, nested_half

SEED = 20260917
ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split", "level", "model", "horizon"]
BASELINES = ("matched_own", "matched_shared", "frozen", "no_history", "global")
GROUPS = ("rejected_with_opportunity", "lost_with_thinning", "geometric_pass", "all")


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).digest()[:4], "little")


def partitions(n, identity, n_splits=5):
    if n < 8 or n_splits < 1:
        raise ValueError("at least eight QC cells and one split required")
    perm = np.random.default_rng(seed(*identity, "detector")).permutation(n)
    detector = np.sort(perm[: max(2, round(0.2 * n))])
    population = np.setdiff1d(np.arange(n), detector)
    splits = []
    for split in range(n_splits):
        p = np.random.default_rng(seed(*identity, "inference", split)).permutation(len(population))
        held = np.sort(p[: max(2, round(0.3 * len(population)))])
        train = np.sort(p[len(held) :])
        half = nested_half(train, identity, split)
        if min(len(held), len(half)) < 2:
            raise ValueError("insufficient distinct cell support")
        splits.append((train, half, held))
    return detector, population, splits


def runs(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def detection_grid(position, supported_intervals):
    pos = np.asarray(position, float)
    intervals = np.asarray(supported_intervals, float)
    if pos.ndim != 2 or pos.shape[1] < 3 or len(pos) < 3 or not np.isfinite(pos[:, :3]).all() or (np.diff(pos[:, 0]) <= 0).any():
        raise ValueError("finite increasing position samples required")
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all() or (intervals[:, 1] <= intervals[:, 0]).any():
        raise ValueError("valid tracking-supported intervals required")
    dt = 0.001
    start = float(pos[0, 0])
    n = int(np.floor((pos[-1, 0] - start) / dt))
    centers = start + (np.arange(n) + 0.5) * dt
    speed = np.full(n, np.nan)
    raw_speed = np.linalg.norm(np.gradient(pos[:, 1:3], pos[:, 0], axis=0), axis=1)
    for lo, hi in intervals:
        p = pos[(pos[:, 0] >= lo) & (pos[:, 0] <= hi)]
        if len(p) < 3:
            continue
        breaks = np.r_[0, np.flatnonzero(np.diff(p[:, 0]) > 0.1) + 1, len(p)]
        for a, b in zip(breaks[:-1], breaks[1:], strict=True):
            x = p[a:b]
            if len(x) < 3:
                continue
            v = np.linalg.norm(np.gradient(x[:, 1:3], x[:, 0], axis=0), axis=1)
            v = gaussian_filter1d(v, 0.1 / np.median(np.diff(x[:, 0])), mode="nearest")
            # Leave edge frames out so every accepted bin has local support.
            left = np.searchsorted(centers, x[1, 0])
            right = np.searchsorted(centers, x[-2, 0], side="right")
            speed[left:right] = np.interp(centers[left:right], x[:, 0], v)
    # Bound raw encoding speed over each 1-ms bin, including tracking knots.
    for a, b in runs(np.isfinite(speed)):
        c = centers[a:b]
        raw = np.maximum(np.interp(c - 0.0005, pos[:, 0], raw_speed), np.interp(c + 0.0005, pos[:, 0], raw_speed))
        lo = np.searchsorted(pos[:, 0], c[0] - 0.0005)
        hi = np.searchsorted(pos[:, 0], c[-1] + 0.0005, side="right")
        indices = np.floor((pos[lo:hi, 0] - (c[0] - 0.0005)) / 0.001).astype(int)
        valid = (indices >= 0) & (indices < len(raw))
        np.maximum.at(raw, indices[valid], raw_speed[lo:hi][valid])
        speed[a:b] = np.maximum(speed[a:b], raw)
    return centers, speed


def detect_candidates(spikes, detector_ids, centers, speed):
    s = np.asarray(spikes, float)
    ids = np.asarray(detector_ids, int)
    centers, speed = np.asarray(centers, float), np.asarray(speed, float)
    if centers.ndim != 1 or len(centers) < 2 or centers.shape != speed.shape or not np.allclose(np.diff(centers), 0.001):
        raise ValueError("aligned 1-ms clock and speed required")
    if s.ndim != 2 or s.shape[1] != 2 or not np.isfinite(s).all() or len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("finite spike times/IDs and distinct detector cells required")
    selected = s[np.isin(s[:, 1], ids)]
    selected = selected[np.argsort(selected[:, 0], kind="stable")]
    start, dt = centers[0] - 0.0005, 0.001
    edges = start + np.arange(len(centers) + 1) * dt
    counts = np.histogram(selected[:, 0], edges)[0]
    smoothed = gaussian_filter1d(counts.astype(float), 10.0, mode="constant", truncate=4)
    immobile = np.isfinite(speed) & (speed < 5)
    if immobile.sum() < 10 or np.std(smoothed[immobile]) <= 0:
        return [], {"immobile_bins": int(immobile.sum()), "baseline_mean": None, "baseline_sd": None}
    mean, sd = float(smoothed[immobile].mean()), float(smoothed[immobile].std())
    z = (smoothed - mean) / sd
    output = []
    for a, b in runs(immobile & (z > 0)):
        duration = (b - a) * dt
        if duration < 0.05 - 1e-12 or duration > 2 + 1e-12 or z[a:b].max() <= 3:
            continue
        local = selected[np.searchsorted(selected[:, 0], edges[a]) : np.searchsorted(selected[:, 0], edges[b])]
        active = len(np.unique(local[:, 1]))
        if active < max(2, int(np.ceil(0.1 * len(ids)))):
            continue
        peak = a + int(np.argmax(z[a:b]))
        output.append(
            dict(
                event_id=len(output),
                start_s=float(edges[a]),
                end_s=float(edges[b]),
                peak_s=float(centers[peak]),
                duration_s=duration,
                detector_spikes=len(local),
                detector_active_cells=active,
                detector_peak_z=float(z[peak]),
                mean_speed_cm_s=float(speed[a:b].mean()),
            )
        )
    return output, dict(immobile_bins=int(immobile.sum()), baseline_mean=mean, baseline_sd=sd)


def event_counts(indexed_spikes, ids, start, end, bin_s=0.005):
    if end <= start or bin_s <= 0:
        raise ValueError("positive event and bin widths required")
    n = int(np.floor((end - start) / bin_s + 1e-8))
    edges = start + np.arange(n + 1) * bin_s
    if end - edges[-1] > 1e-9:
        edges = np.r_[edges, end]
    else:
        edges[-1] = end
    counts = np.column_stack([np.diff(np.searchsorted(indexed_spikes[int(i)], edges, side="left")) for i in ids])
    return counts, np.diff(edges)


def full_counts(base, durations):
    n = int(np.isclose(durations, 0.005, rtol=0, atol=1e-9).sum()) // 4 * 4
    counts = np.asarray(base)[:n].reshape(-1, 4, base.shape[1]).sum(axis=1)
    return counts, int(np.asarray(base)[n:].sum())


def classify(base, durations, rates, centers, train, half):
    mask = np.ones(len(centers), bool)
    full_path, _, full = classify_training(base, durations, rates, centers, train, mask)
    half_path, _, reduced = classify_training(base, durations, rates, centers, half, mask)
    a, b = full[0], reduced[0]
    return (
        {
            **{"full_" + k: v for k, v in a.items()},
            **{"half_" + k: v for k, v in b.items()},
            "rejected_with_opportunity": not a["geometric_pass"] and a["valid_frames"] >= 10,
            "lost_with_thinning": a["geometric_pass"] and not b["geometric_pass"],
            "geometric_pass": a["geometric_pass"],
        },
        full_path,
        half_path,
    )


class NullOperator:
    def __init__(self, original, matched):
        self.initial = original.initial.copy()
        self.original, self.matched = original, matched

    def expand(self, ll):
        return self.original.expand(ll)

    def collapse(self, q):
        return self.original.collapse(q)

    def step(self, q):
        return self.matched.step(q)


def forecast_distributions(train_counts, emissions, operator, matched, horizons=(1, 2, 4)):
    ll = identity_likelihood(train_counts, emissions)
    filtered = forward_filter(ll, operator)
    null_filtered = forward_filter(ll, NullOperator(operator, matched))
    prior, no_history = operator.initial.copy(), []
    for _ in ll:
        no_history.append(prior.copy())
        prior = operator.step(prior)
    no_history = np.asarray(no_history)
    dynamic, shared, own = filtered.copy(), filtered.copy(), null_filtered.copy()
    result = {}
    for h in range(1, max(horizons) + 1):
        dynamic = operator.step(dynamic)
        shared, own = matched.step(shared), matched.step(own)
        if h in horizons and h < len(ll):
            result[h] = {
                k: operator.collapse(q).copy()
                for k, q in {
                    "dynamic": dynamic[:-h],
                    "matched_shared": shared[:-h],
                    "matched_own": own[:-h],
                    "frozen": filtered[:-h],
                    "no_history": no_history[h:],
                }.items()
            }
    return result


def score_predictions(predictions, held_counts, held_emissions, global_probability):
    result = []
    for h, candidates in predictions.items():
        targets = held_counts[h:]
        hl = identity_likelihood(targets, held_emissions)
        empty = targets.sum(axis=1) == 0
        row = dict(horizon=h, n_target_bins=len(targets), n_heldout_target_spikes=int(targets.sum()))
        for name, q in candidates.items():
            value = mixture_scores(q, hl)
            value[empty] = 0
            row["score_" + name] = float(value.sum())
        row["score_global"] = float(identity_likelihood(targets, np.asarray(global_probability)[:, None]).sum())
        result.append(row)
    return result


def aggregate(rows):
    if rows.empty or rows.duplicated(KEY).any():
        raise ValueError("unique nonempty event/split/model rows required")
    ok = rows.status.eq("scored")
    columns = ["score_dynamic"] + ["score_" + k for k in BASELINES]
    if not np.isfinite(rows.loc[ok, columns]).all().all():
        raise ValueError("nonfinite scored values")
    pieces = []
    for group in GROUPS:
        mask = np.ones(len(rows), bool) if group == "all" else rows[group].astype(bool)
        local = rows[ok & mask].copy()
        for b in BASELINES:
            local_delta = local[KEY + ["n_heldout_target_spikes", "n_target_bins"]].copy()
            local_delta["group"], local_delta["contrast"] = group, "dynamic_minus_" + b
            local_delta["delta"] = (local.score_dynamic - local["score_" + b]).to_numpy()
            local_delta["delta_per_spike"] = local_delta.delta / local_delta.n_heldout_target_spikes.replace(0, np.nan)
            local_delta["delta_per_bin"] = local_delta.delta / local_delta.n_target_bins
            pieces.append(local_delta)
    splits = pd.concat(pieces, ignore_index=True)
    factors = ["level", "model", "horizon", "group", "contrast"]
    ek = ID + ["event_id"] + factors
    metrics = ["delta", "delta_per_spike", "delta_per_bin"]
    events = splits.groupby(ek, as_index=False).agg(
        **{m: (m, "median") for m in metrics},
        qualifying_splits=("split", "nunique"),
        informative_splits=("delta_per_spike", "count"),
    )
    sessions = events.groupby(ID + factors, as_index=False)[metrics].mean()
    animals = sessions.groupby(["dataset", "animal"] + factors, as_index=False)[metrics].mean()
    summary, loo = [], []
    for key, g in animals.groupby(["dataset"] + factors):
        ids = dict(zip(["dataset"] + factors, key, strict=True))
        for metric in metrics:
            finite = g[np.isfinite(g[metric])]
            v = finite.sort_values("animal")[metric].to_numpy()
            n = len(v)
            if not n:
                continue
            draws = np.array(list(itertools.product(range(n), repeat=n)))
            lo, hi = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
            nonzero = v[v != 0]
            from scipy.stats import binomtest

            p = binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else 1.0
            summary.append(
                ids | dict(metric=metric, mean=float(v.mean()), ci_low=float(lo), ci_high=float(hi), animals=n, positive_animals=int((v > 0).sum()), sign_p_value=float(p))
            )
            for rat in finite.animal:
                x = finite[finite.animal.ne(rat)][metric]
                loo.append(ids | dict(metric=metric, omitted_animal=rat, mean=float(x.mean()), remaining_animals=len(x)))
    return splits, events, sessions, animals, pd.DataFrame(summary), pd.DataFrame(loo)
