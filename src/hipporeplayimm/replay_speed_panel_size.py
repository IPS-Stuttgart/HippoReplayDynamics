"""Per-event sufficient statistics for nested candidate-budget recovery."""

from __future__ import annotations

from itertools import pairwise, product

import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage import continuity_metrics, decode_independent
from hipporeplayimm.replay_coverage_geometry import window_counts
from hipporeplayimm.replay_coverage_recovery import spatial_covariate
from hipporeplayimm.replay_speed_identifiability import bootstrap_slope, event_slope

READOUTS = list(product(["map", "posterior_mean"], ["unfiltered", "at_least_2cells_3spikes"], ["all", "selected"]))
READOUT = ["estimator", "bin_filter", "selection"]
DIAGNOSTICS = ["contributing_steps", "continuous", "valid_windows", "total_windows"]


def profile_cycle(profiles, count, seed):
    data = profiles[["source_event_index", "n_base_bins"]].sort_values("source_event_index").reset_index(drop=True)
    if count < 1 or data.empty or data.source_event_index.duplicated().any():
        raise ValueError("positive count and unique nonempty source profiles required")
    if not np.isfinite(data.to_numpy()).all() or (data.to_numpy() != data.to_numpy().astype(int)).any() or (data.n_base_bins < 4).any():
        raise ValueError("finite integral profiles with at least four base bins required")
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.permutation(len(data)) for _ in range((count + len(data) - 1) // len(data))])[:count]
    plan = data.iloc[indices].reset_index(drop=True)
    plan.insert(0, "synthetic_ordinal", np.arange(count))
    plan["profile_cycle"] = np.arange(count) // len(data)
    return plan


def decode_event_moments(fine_chunks, model):
    if not fine_chunks:
        raise ValueError("nonempty candidate panel required")
    chunks = [window_counts(fine, 20, 5) for fine in fine_chunks]
    if any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("every candidate must support a decoding window")
    offsets = np.r_[0, np.cumsum([len(c) for c in chunks])]
    decoded = decode_independent(np.concatenate(chunks), model["rates_hz"] * 3., model["grid_cm"], .02)
    del decoded["posterior"]
    moments = np.full((len(chunks), len(READOUTS), 4), np.nan)
    diagnostics = np.zeros((len(chunks), len(READOUTS), len(DIAGNOSTICS)), dtype=np.int32)
    for estimator, support in product(["map", "posterior_mean"], [False, True]):
        for event, (lo, hi, counts) in enumerate(zip(offsets[:-1], offsets[1:], chunks, strict=True)):
            points = decoded[estimator][lo:hi]
            valid = ((counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)) if support else np.ones(len(counts), bool)
            core = continuity_metrics(points, valid_bins=valid)
            idx = np.arange(0, len(points), 4)
            keep = np.array([valid[a:b + 1].all() for a, b in pairwise(idx)], bool)
            selected = keep & (idx[:-1] >= core["continuous_start"]) & (idx[1:] < core["continuous_end_exclusive"]) & core["continuity_pass"]
            speed = np.linalg.norm(np.diff(points[idx], axis=0), axis=1) / .02 / 1000.
            q = spatial_covariate((points[idx[:-1]] + points[idx[1:]]) / 2, model["domain_cm"])
            for selection, mask in [("all", keep), ("selected", selected)]:
                key = (estimator, "at_least_2cells_3spikes" if support else "unfiltered", selection)
                column = READOUTS.index(key)
                if mask.any():
                    x, y = q[mask], speed[mask]
                    moments[event, column] = [v.mean() for v in [x, y, x * x, x * y]]
                diagnostics[event, column] = [int(mask.sum()), int(core["continuity_pass"]), int(valid.sum()), len(valid)]
    return moments, diagnostics


def validate_event_statistics(moments, diagnostics):
    if moments.ndim != 3 or moments.shape[1:] != (8, 4) or diagnostics.shape != moments.shape or len(moments) == 0:
        raise ValueError("nonempty event/readout/moment arrays required")
    if not np.isfinite(diagnostics).all() or (diagnostics < 0).any() or (diagnostics != diagnostics.astype(int)).any():
        raise ValueError("invalid event diagnostics")
    steps, continuous, valid, total = np.moveaxis(diagnostics, 2, 0)
    if (continuous > 1).any() or (valid > total).any():
        raise ValueError("invalid continuity/support diagnostics")
    finite = np.isfinite(moments).all(axis=2)
    missing = np.isnan(moments).all(axis=2)
    if not (finite | missing).all() or not np.array_equal(finite, steps > 0):
        raise ValueError("moment availability must match contributing steps")
    if ((steps[:, 1::2] > 0) & (continuous[:, 1::2] == 0)).any() or (steps[:, 1::2] > steps[:, ::2]).any():
        raise ValueError("selected steps require an eligible subset")


def prefix_panels(moments, diagnostics, budgets, seed, bootstrap=True):
    validate_event_statistics(moments, diagnostics)
    if not budgets or list(budgets) != sorted(set(budgets)) or min(budgets) < 1 or max(budgets) > len(moments):
        raise ValueError("positive unique increasing budgets inside the panel required")
    rows = []
    for budget in budgets:
        for column, readout in enumerate(READOUTS):
            values, diag = moments[:budget, column], diagnostics[:budget, column]
            statistic = event_slope(values)
            lower, upper = np.nan, np.nan
            if bootstrap:
                statistic, lower, upper = bootstrap_slope(values, [seed, budget, column])
            finite = np.isfinite(values).all(axis=1)
            variance = float(values[finite, 2].mean() - values[finite, 0].mean() ** 2) if finite.any() else np.nan
            rows.append({**dict(zip(READOUT, readout, strict=True)), "candidate_budget": budget,
                "statistic": statistic, "naive_lower": lower, "naive_upper": upper,
                "contributing_events": int(finite.sum()), "source_events": budget,
                "continuous_events": int(diag[:, 1].sum()), "valid_windows": int(diag[:, 2].sum()),
                "total_windows": int(diag[:, 3].sum()), "contributing_steps": int(diag[:, 0].sum()),
                "spatial_variance": variance, "statistic_status": "available" if np.isfinite(statistic)
                    else "fewer_than_five_events" if finite.sum() < 5 else "insufficient_spatial_variance"})
    return pd.DataFrame(rows)
