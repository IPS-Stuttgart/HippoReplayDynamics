"""Nested observation coverage with paired independent predictive validation."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .independent_rejected_forecast import BASELINES, ID

FRACTIONS = (1.0, 0.75, 0.5, 0.25)
REPEATS = 10
KEY = ID + ["event_id", "split", "repeat", "fraction", "arm"]
CONTRASTS = ["dynamic_minus_" + b for b in BASELINES]


def subsets(train, old_half, identity, split, repeat):
    train, half = np.sort(np.asarray(train, int)), np.sort(np.asarray(old_half, int))
    if len(train) < 8 or len(np.unique(train)) != len(train) or (train < 0).any():
        raise ValueError("at least eight unique inference cells required")
    if len(half) != len(train) // 2 or not set(half) < set(train) or len(np.unique(half)) != len(half):
        raise ValueError("nested old-half population required")
    if repeat < 0 or split < 0:
        raise ValueError("nonnegative repeat and split required")
    key = "|".join(map(str, (20260918, *identity, split, repeat)))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    order = np.r_[rng.permutation(half), rng.permutation(np.setdiff1d(train, half))] if repeat == 0 else rng.permutation(train)
    return {f: np.sort(order[: max(2, int(np.floor(len(train) * f)))]) for f in FRACTIONS}


def attach_contrasts(rows):
    x = rows.copy()
    if x.empty or x.duplicated(KEY).any():
        raise ValueError("nonempty unique score rows required")
    ok = x.status.eq("scored")
    if not ok.eq(x.n_full_bins.gt(2)).all():
        raise ValueError("incorrect temporal availability")
    scores = ["score_dynamic"] + ["score_" + b for b in BASELINES]
    if not np.isfinite(x.loc[ok, scores]).all().all() or not x.loc[~ok, scores].isna().all().all():
        raise ValueError("invalid score status or values")
    anchor = x[x.fraction.eq(1)].copy()
    if anchor.duplicated(ID + ["event_id", "split"]).any():
        raise ValueError("duplicated full-population reference")
    for b in BASELINES:
        name = "dynamic_minus_" + b
        x[name] = (x.score_dynamic - x["score_" + b]) / x.n_heldout_target_spikes.replace(0, np.nan)
    anchor = x[x.fraction.eq(1)][ID + ["event_id", "split", *CONTRASTS, "geometric_pass", "valid_frames", "n_heldout_target_spikes"]]
    anchor = anchor.rename(columns={c: "full_" + c for c in [*CONTRASTS, "geometric_pass", "valid_frames", "n_heldout_target_spikes"]})
    x = x.merge(anchor, on=ID + ["event_id", "split"], validate="many_to_one", how="left")
    if x.full_geometric_pass.isna().any() or not x.n_heldout_target_spikes.eq(x.full_n_heldout_target_spikes).all():
        raise ValueError("missing anchor or changed evaluation targets")
    for name in CONTRASTS:
        x["paired_change_" + name] = x[name] - x["full_" + name]
    x["pass_change"] = x.geometric_pass.astype(int) - x.full_geometric_pass.astype(int)
    return x


def group_masks(x):
    full, reduced = x.full_geometric_pass.astype(bool), x.geometric_pass.astype(bool)
    return {
        "all": np.ones(len(x), bool),
        "full_pass": full,
        "full_supported": x.full_valid_frames.ge(10),
        "lost": full & ~reduced,
        "lost_supported": full & ~reduced & x.valid_frames.ge(10),
        "gained": ~full & reduced,
        "retained": full & reduced,
    }


def event_tables(rows):
    x = attach_contrasts(rows)
    factors = ["arm", "fraction", "group"]
    metrics = CONTRASTS + ["paired_change_" + c for c in CONTRASTS] + ["full_" + c for c in CONTRASTS]
    results = []
    for group, mask in group_masks(x).items():
        g = x.loc[mask].assign(group=group)
        if g.empty:
            continue
        per_split = g.groupby(ID + ["event_id", "split"] + factors, as_index=False).agg(
            **{c: (c, "median") for c in metrics},
            acceptance=("geometric_pass", "mean"),
            pass_change=("pass_change", "mean"),
            mean_inference_cells=("n_train_cells", "mean"),
            mean_inference_spikes=("n_train_spikes", "mean"),
            n_heldout_target_spikes=("n_heldout_target_spikes", "first"),
            qualifying_repeats=("repeat", "size"),
        )
        event = per_split.groupby(ID + ["event_id"] + factors, as_index=False).agg(
            **{c: (c, "median") for c in metrics},
            **{c: (c, "mean") for c in ["acceptance", "pass_change", "mean_inference_cells", "mean_inference_spikes"]},
            n_heldout_target_spikes=("n_heldout_target_spikes", "median"),
            qualifying_repeats=("qualifying_repeats", "sum"),
            qualifying_splits=("split", "size"),
            informative_splits=(CONTRASTS[0], "count"),
        )
        results.append(event)
    return pd.concat(results, ignore_index=True)


def summaries(events):
    if events.empty:
        raise ValueError("empty event table")
    factors = ["arm", "fraction", "group"]
    metrics = CONTRASTS + ["paired_change_" + c for c in CONTRASTS] + ["full_" + c for c in CONTRASTS]
    metrics += ["acceptance", "pass_change", "mean_inference_cells", "mean_inference_spikes"]
    sessions = events.groupby(ID + factors, as_index=False)[metrics].mean()
    animals = sessions.groupby(["dataset", "animal"] + factors, as_index=False)[metrics].mean()
    records, loo = [], []
    for key, g in animals.groupby(["dataset"] + factors):
        ids = dict(zip(["dataset"] + factors, key, strict=True))
        for metric in metrics:
            finite = g[np.isfinite(g[metric])].sort_values("animal")
            v = finite[metric].to_numpy()
            n = len(v)
            if n == 0:
                continue
            draws = np.array(list(itertools.product(range(n), repeat=n)))
            lo, hi = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
            nonzero = v[v != 0]
            records.append(
                ids
                | {
                    "metric": metric,
                    "mean": v.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "animals": n,
                    "positive_animals": int((v > 0).sum()),
                    "sign_p_value": binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else 1.0,
                }
            )
            for animal in finite.animal:
                loo.append(ids | {"metric": metric, "omitted_animal": animal, "mean": finite.loc[finite.animal.ne(animal), metric].mean()})
    return sessions, animals, pd.DataFrame(records), pd.DataFrame(loo)


def retention_gate(summary, dataset, arm, group, expected_animals):
    x = summary[summary.dataset.eq(dataset) & summary.arm.eq(arm) & summary.fraction.eq(0.5) & summary.group.eq(group)]
    main = x[x.metric.eq("dynamic_minus_matched_own")]
    other = x[x.metric.isin(["dynamic_minus_no_history", "dynamic_minus_global"])]
    return bool(
        len(main) == 1
        and len(other) == 2
        and main.animals.eq(expected_animals).all()
        and main.positive_animals.eq(expected_animals).all()
        and main.ci_low.gt(0).all()
        and other.animals.eq(expected_animals).all()
        and other["mean"].gt(0).all()
    )
