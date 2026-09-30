"""Segment readouts and finite-bank unknown-mixture stress tests."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from hipporeplayimm.regional_content_frontier import calls_from_bf, regional_log_bf
from hipporeplayimm.regional_readout_endpoint import ContinuousDensity, fit_prevalence


def count_subwindows(identities, templates, n_cells, duration_ms, frozen_last_starts):
    if duration_ms not in (20, 40, 60, 100):
        raise ValueError("unsupported segment duration")
    k = duration_ms // 20
    if len(identities) != len(templates["times"]):
        raise ValueError("identity/template mismatch")
    result = np.zeros((len(templates["endpoints"]), k, n_cells), np.uint16)
    for i, end in enumerate(templates["endpoints"]):
        steps = round((end-templates["starts"][i])/.005)
        edges = templates["starts"][i] + .005*(steps-4*np.arange(k, -1, -1))
        if abs(edges[-1]-end) > 1e-8:
            raise ValueError("endpoint is not on the native 5-ms grid")
        edges[-1] = end
        edges[-2] = frozen_last_starts[i]
        if edges[0] < templates["starts"][i] - 1e-9:
            raise ValueError("segment would extend outside native event")
        a, b = templates["offsets"][i:i+2]
        times, cells = templates["times"][a:b], identities[a:b]
        take = (times >= edges[0]) & (times < edges[-1])
        bins = np.searchsorted(edges, times[take], side="right") - 1
        flat = np.bincount(bins*n_cells+cells[take], minlength=k*n_cells)
        if flat.max(initial=0) > np.iinfo(np.uint16).max:
            raise ValueError("spike-count overflow")
        result[i] = flat.reshape(k, n_cells)
    return result


def independent_any_bf(bin_bf, bin_totals, area):
    score, totals = np.asarray(bin_bf, float), np.asarray(bin_totals)
    if score.ndim != 2 or score.shape != totals.shape or not 0 < area < 1 or not np.isfinite(score).all():
        raise ValueError("invalid independent-bin inputs")
    neutral = np.where(totals == 0, 0., score)
    if score.shape[1] == 1:
        return neutral[:, 0].copy()
    log_odds = neutral + np.log(area / (1-area))
    log_not_bin = -np.logaddexp(0., log_odds)
    log_not = log_not_bin.sum(axis=1)
    preceding = np.column_stack([np.zeros(len(score)), np.cumsum(log_not_bin, axis=1)[:, :-1]])
    log_any = logsumexp(-np.logaddexp(0., -log_odds)+preceding, axis=1)
    log_not_prior = score.shape[1] * np.log1p(-area)
    value = log_any-log_not-(np.log(-np.expm1(log_not_prior))-log_not_prior)
    value[totals.sum(axis=1) == 0] = 0.
    return value


def segment_features(counts, rates, region):
    n = np.asarray(counts)
    if n.ndim != 3:
        raise ValueError("expected event x window x cell counts")
    e, k, c = n.shape
    totals = n.sum(axis=2)
    bf = regional_log_bf(n.reshape(-1, c), rates, region, exposure=.020).reshape(e, k)
    independent = independent_any_bf(bf, totals, float(np.mean(region)))
    pooled = regional_log_bf(n.sum(axis=1), rates, region, exposure=k*.020)
    pooled[totals.sum(axis=1) == 0] = 0.
    return {"independent_bin_any_home": independent, "pooled_counts": pooled,
            "totals": totals.sum(axis=1), "silent_bin_fraction": (totals == 0).mean(axis=1)}


class Calibration:
    """Calibration-only emissions, including the exact neutral-score atom."""

    def __init__(self, scores, labels, representation):
        x, z = np.asarray(scores), np.asarray(labels)
        if x.shape != z.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isin(z, [0, 1]).all():
            raise ValueError("invalid calibration")
        if representation not in ("ternary", "continuous_zero_mass"):
            raise ValueError("unknown representation")
        self.representation = representation
        self.models = []
        for label in (0, 1):
            s = x[z == label]
            if not len(s):
                raise ValueError("missing calibration class")
            if representation == "ternary":
                if not np.isin(s, [0, 1, 2]).all():
                    raise ValueError("invalid ternary calibration")
                h = np.bincount(s.astype(int), minlength=3) + .5
                self.models.append(h/h.sum())
            else:
                zero = s == 0
                self.models.append(((zero.sum()+.5)/(len(s)+1),
                                    ContinuousDensity(s[~zero]) if (~zero).any() else None))

    def likelihoods(self, query):
        q = np.asarray(query)
        shape = q.shape
        q = q.ravel()
        if not np.isfinite(q).all():
            raise ValueError("nonfinite query")
        columns = []
        for model in self.models:
            if self.representation == "ternary":
                if not np.isin(q, [0, 1, 2]).all():
                    raise ValueError("invalid ternary query")
                columns.append(model[q.astype(int)])
            else:
                p0, density = model
                values = np.full(q.shape, p0)
                if (q != 0).any():
                    if density is None:
                        raise ValueError("nonzero query without nonzero calibration support")
                    values[q != 0] = (1-p0)*density(q[q != 0])
                columns.append(values)
        return np.column_stack(columns).reshape(*shape, 2)


def represented_scores(score, totals, representation):
    return calls_from_bf(score, totals) if representation == "ternary" else np.asarray(score)


def empirical_envelope(likelihoods, truth):
    f = np.asarray(likelihoods)
    if f.ndim != 3 or f.shape[2] != 2 or not 0 <= truth <= 1:
        raise ValueError("expected generator x event x class likelihoods")
    estimates = np.asarray([fit_prevalence(x) for x in f])
    valid = bool(np.isfinite(estimates).all())
    lower, upper = (float(estimates.min()), float(estimates.max())) if valid else (0., 1.)
    worst = max(abs(lower-truth), abs(upper-truth))
    return {"lower_estimate": lower, "upper_estimate": upper, "worst_absolute_error": worst,
            "within_5pp": bool(valid and worst <= .05), "identified": valid,
            "pure_estimates": estimates}


def simplex_grid(n=7, denominator=4):
    def compositions(total, slots):
        if slots == 1:
            yield [total]
        else:
            for first in range(total+1):
                for rest in compositions(total-first, slots-1):
                    yield [first, *rest]
    return np.asarray(list(compositions(denominator, n)), float)/denominator


def weighted_mixture_estimates(likelihoods, weights):
    """Vectorized concave MLE; weights apply to empirical panels, not labels."""
    f, w = np.asarray(likelihoods, float), np.asarray(weights, float)
    if f.ndim != 3 or f.shape[2] != 2 or not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError("invalid likelihoods")
    if w.ndim != 2 or w.shape[1] != len(f) or not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(axis=1), 1):
        raise ValueError("convex weights required")
    f = np.maximum(f/f.max(axis=2, keepdims=True), 1e-250)
    a, b = f[..., 0], f[..., 1]
    d = b-a
    def gradient(p):
        divisor = (1-p[:, None, None])*a + p[:, None, None]*b
        return (w*(d/divisor).sum(axis=2)).sum(axis=1)
    left, right = np.zeros(len(w)), np.ones(len(w))
    g0, g1 = gradient(left), gradient(right)
    for _ in range(45):
        middle = (left+right)/2
        rising = gradient(middle) > 0
        left = np.where(rising, middle, left)
        right = np.where(rising, right, middle)
    estimates = np.where(g0 <= 0, 0., np.where(g1 >= 0, 1., (left+right)/2))
    flat = (w*np.max(np.abs(d), axis=1)[None, :]).sum(axis=1) < 1e-10
    estimates[flat] = np.nan
    return estimates


def regional_mass(score, area, k, readout):
    prior = 1-(1-area)**k if readout == "independent_bin_any_home" else area
    return expit(score+np.log(prior/(1-prior)))
