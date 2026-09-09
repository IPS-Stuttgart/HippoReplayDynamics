"""Streaming path quadrature for the frozen unknown-path clock experiment."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from hipporeplayimm.unknown_path_clocks import summarize as summarize_bank

SUPPORTS = (1024, 4096, 8192)
BANKS = (0, 1)
TEACHERS = ("original_bank_0", "original_bank_1")
CHUNK = 128


def rng(*parts):
    digest = hashlib.sha256(("dense_path_clocks_v1|20260909|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


class RateIntegral:
    """Reuse the exact piecewise-linear antiderivative across native bin counts."""

    def __init__(self, clock, values):
        self.t = np.asarray(clock, float)
        self.y = np.asarray(values, float)
        if self.t.ndim != 1 or self.y.ndim != 2 or len(self.t) != len(self.y) or len(self.t) < 2:
            raise ValueError("aligned rate values and clock knots required")
        dt = np.diff(self.t)
        if not np.isfinite(self.t).all() or not np.isfinite(self.y).all() or (dt <= 0).any() or self.t[0] != 0 or self.t[-1] != 1 or (self.y <= 0).any():
            raise ValueError("positive rates and increasing normalized clock required")
        self.slope = np.diff(self.y, axis=0) / dt[:, None]
        self.prefix = np.vstack([np.zeros(self.y.shape[1]), np.cumsum(dt[:, None] * (self.y[:-1] + self.y[1:]) / 2, axis=0)])

    def average(self, n):
        if n < 1 or int(n) != n:
            raise ValueError("positive integer bin count required")
        edge = np.linspace(0, 1, int(n) + 1)
        i = np.minimum(np.searchsorted(self.t, edge, side="right") - 1, len(self.t) - 2)
        d = (edge - self.t[i])[:, None]
        cumulative = self.prefix[i] + self.y[i] * d + 0.5 * self.slope[i] * d**2
        average = np.diff(cumulative, axis=0) * n
        if (average <= 0).any() or not np.isfinite(average).all():
            raise ValueError("invalid integrated rate")
        return average


class PathAccumulator:
    def __init__(self, counts):
        x = np.asarray(counts)
        if x.ndim != 3 or min(x.shape) < 1 or (x < 0).any() or not np.equal(x, np.floor(x)).all():
            raise ValueError("nonempty event-time-cell integer counts required")
        self.shape = x.shape
        self.counts = [csr_matrix(x[:, t]) for t in range(x.shape[1])]
        self.coherent = np.full(x.shape[0], -np.inf)
        self.reset = np.full(x.shape[:2], -np.inf)
        self.n_paths = 0

    def add(self, log_probabilities):
        p = np.asarray(log_probabilities)
        if p.ndim != 3 or p.shape[1:] != self.shape[1:] or len(p) < 1 or not np.isfinite(p).all() or not np.allclose(logsumexp(p, axis=2), 0, atol=1e-10):
            raise ValueError("normalized aligned path-time-cell log probabilities required")
        event_path = np.zeros((self.shape[0], len(p)))
        for t, x in enumerate(self.counts):
            contribution = x @ np.ascontiguousarray(p[:, t].T)
            event_path += contribution
            self.reset[:, t] = np.logaddexp(self.reset[:, t], logsumexp(contribution, axis=1))
        self.coherent = np.logaddexp(self.coherent, logsumexp(event_path, axis=1))
        self.n_paths += len(p)

    def scores(self):
        if self.n_paths < 1:
            raise ValueError("empty path integral")
        return self.coherent - np.log(self.n_paths), (self.reset - np.log(self.n_paths)).sum(axis=1)


def summarize(fits, repeats=50, supports=SUPPORTS, banks=BANKS):
    if set(fits.candidate_bank) != set(banks):
        raise ValueError("complete independent scorer libraries required")
    collected = [[], [], []]
    for bank in banks:
        source = fits[fits.candidate_bank.eq(bank)].rename(columns={"source_teacher": "teacher"})
        frames = summarize_bank(source, repeats=repeats, supports=supports, teachers=TEACHERS)
        for output, frame in zip(collected, frames, strict=True):
            output.append(frame.rename(columns={"teacher": "source_teacher"}).assign(candidate_bank=bank))
    detail, summary, gates = [pd.concat(tables, ignore_index=True) for tables in collected]
    convergence = []
    if len(banks) != 2 or len(supports) < 2:
        raise ValueError("two banks and at least two support levels required for convergence")
    largest, previous = sorted(supports)[-2:][::-1]
    for (dataset, teacher), group in detail.groupby(["dataset", "source_teacher"]):
        high = group[group.support.eq(largest)]
        a, b = [high[high.candidate_bank.eq(bank)].set_index(["scenario", "repeat"]).sort_index() for bank in banks]
        bank_error = np.abs(a.phi_hat - b.phi_hat)
        for bank in banks:
            lo = group[group.support.eq(previous) & group.candidate_bank.eq(bank)].set_index(["scenario", "repeat"]).sort_index()
            hi = high[high.candidate_bank.eq(bank)].set_index(["scenario", "repeat"]).sort_index()
            shift = np.abs(hi.phi_hat - lo.phi_hat)
            endpoint_shift = np.maximum(np.abs(hi.phi_low - lo.phi_low), np.abs(hi.phi_high - lo.phi_high))
            convergence.append(
                {
                    "dataset": dataset,
                    "source_teacher": teacher,
                    "candidate_bank": bank,
                    "support": largest,
                    "previous_support": previous,
                    "median_cross_bank_phi_difference": bank_error.median(),
                    "p95_cross_bank_phi_difference": bank_error.quantile(0.95),
                    "median_support_phi_shift": shift.median(),
                    "p95_support_phi_shift": shift.quantile(0.95),
                    "median_support_interval_endpoint_shift": endpoint_shift.median(),
                    "integration_stable": bool(bank_error.median() <= 0.025 and shift.median() <= 0.025 and endpoint_shift.median() <= 0.05),
                }
            )
    return detail, summary, gates, pd.DataFrame(convergence)
