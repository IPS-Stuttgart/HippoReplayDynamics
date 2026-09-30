"""Frozen readout and temporal-aggregation diagnostics on known-truth banks."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm


class ContinuousDensity:
    """Calibration-only transformed KDE with a direct, non-clamped tail."""

    def __init__(self, scores):
        x = np.arcsinh(np.asarray(scores, float))
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("finite nonempty calibration required")
        self.center = float(x.mean())
        self.kde = None
        sd = float(x.std(ddof=1)) if len(x) > 1 else 0.
        self.bandwidth = max(.05, sd * len(x)**(-.2))
        if sd > 1e-12:
            self.kde = gaussian_kde(x, bw_method=self.bandwidth/sd)
        self.grid = np.linspace(x.min()-8*self.bandwidth, x.max()+8*self.bandwidth, 2049)
        self.density = self._direct(self.grid)

    def _direct(self, x):
        return (self.kde(x) if self.kde is not None
                else norm.pdf(x, loc=self.center, scale=self.bandwidth))

    def __call__(self, scores):
        x = np.arcsinh(np.asarray(scores, float))
        if not np.isfinite(x).all():
            raise ValueError("finite evaluation scores required")
        result = np.interp(x, self.grid, self.density)
        outside = (x < self.grid[0]) | (x > self.grid[-1])
        if outside.any():
            result[outside] = self._direct(x[outside])
        return np.maximum(result, np.finfo(float).tiny)


def class_likelihoods(cal_scores, cal_truth, query, readout):
    y, z, q = np.asarray(cal_scores), np.asarray(cal_truth), np.asarray(query)
    if y.shape != z.shape or not np.isin(z, [0, 1]).all() or not all(np.any(z == c) for c in (0, 1)):
        raise ValueError("two calibration classes required")
    if readout == "ternary":
        if not np.isin(y, [0, 1, 2]).all() or not np.isin(q, [0, 1, 2]).all():
            raise ValueError("invalid ternary values")
        emission = np.array([np.bincount(y[z == c].astype(int), minlength=3)+.5 for c in (0, 1)])
        emission /= emission.sum(axis=1, keepdims=True)
        return emission[:, q.astype(int)].T
    if readout not in ("continuous_neutral", "continuous_native"):
        raise ValueError("unknown readout")
    return np.column_stack([ContinuousDensity(y[z == c])(q) for c in (0, 1)])


def fit_prevalence(likelihoods):
    f = np.asarray(likelihoods, float)
    if f.ndim != 2 or f.shape[1] != 2 or not len(f) or not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError("positive finite (events, 2) likelihoods required")
    f = np.maximum(f / f.max(axis=1, keepdims=True), 1e-250)
    difference = f[:, 1]-f[:, 0]
    if np.max(np.abs(difference)) < 1e-10:
        return np.nan
    def gradient(p):
        return np.sum(difference / ((1-p)*f[:, 0]+p*f[:, 1]))
    if gradient(0) <= 0:
        return 0.
    if gradient(1) >= 0:
        return 1.
    return float(brentq(gradient, 0., 1., xtol=1e-10))


def oracle_prevalence(likelihoods, generators):
    f, g = np.asarray(likelihoods), np.asarray(generators)
    if len(f) != len(g):
        raise ValueError("generator shape mismatch")
    fits = [(np.sum(g == name), fit_prevalence(f[g == name])) for name in np.unique(g)]
    if not all(np.isfinite(p) for _, p in fits):
        return np.nan
    return float(sum(n*p for n, p in fits)/len(f))


def count_terminal(identities, templates, n_cells, duration_s):
    """Count unchanged spike identities in half-open terminal subwindows."""
    from hipporeplayimm.selection_matched_regional import endpoint_interval
    if duration_s <= 0 or duration_s > .020:
        raise ValueError("duration must be in (0, .020]")
    result, cursor = [], 0
    for event in templates:
        t = event["times"]
        ids = identities[cursor:cursor+len(t)]
        cursor += len(t)
        start, end = endpoint_interval(event["start"], event["end"])
        left = start if duration_s == .020 else end-duration_s
        take = (t >= left) & (t < end)
        result.append(np.bincount(ids[take], minlength=n_cells))
    if cursor != len(identities):
        raise ValueError("identity/template size mismatch")
    return np.asarray(result, np.uint16)
