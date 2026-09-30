"""Dependence-robust regional prevalence and conditional identification sets."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp
from scipy.stats import beta

DEFAULT_LOG_BF = float(np.log(3))


def patterns(k=4):
    if not isinstance(k, int) or not 1 <= k <= 6:
        raise ValueError("one to six views required")
    return np.asarray(list(product(range(3), repeat=k)), dtype=int)


def pattern_counts(readouts):
    y = np.asarray(readouts)
    if y.ndim != 2 or not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("nonempty ternary readouts required")
    k = y.shape[1]
    patterns(k)
    ids = y.astype(int) @ (3 ** np.arange(k - 1, -1, -1))
    return np.bincount(ids, minlength=3**k)


def binomial_interval(successes, total, alpha):
    x, n = np.broadcast_arrays(np.asarray(successes, float), np.asarray(total, float))
    if not 0 < alpha < 1 or not np.isfinite(x).all() or not np.isfinite(n).all():
        raise ValueError("invalid binomial input")
    if np.any((x < 0) | (x > n) | (n < 0) | (x != np.floor(x)) | (n != np.floor(n))):
        raise ValueError("invalid binomial counts")
    lo = np.where(x > 0, beta.ppf(alpha / 2, x, n - x + 1), 0.)
    hi = np.where(x < n, beta.ppf(1 - alpha / 2, x + 1, n - x), 1.)
    return np.stack([lo, hi], axis=-1)


def calibration_intervals(readouts, truth, alpha=.025):
    y, z = np.asarray(readouts), np.asarray(truth)
    pattern_counts(y)
    if z.shape != (len(y),) or not np.isin(z, [0, 1]).all():
        raise ValueError("binary calibration truth required")
    counts = np.zeros((2, y.shape[1], 3), int)
    totals = np.array([(z == c).sum() for c in (0, 1)])
    for c, k, r in product(range(2), range(y.shape[1]), range(3)):
        counts[c, k, r] = ((z == c) & (y[:, k] == r)).sum()
    intervals = binomial_interval(counts, totals[:, None, None], alpha / counts.size)
    return intervals, counts, totals


def decode_region(counts, rates, region, groups, dt=.02, delta=DEFAULT_LOG_BF):
    n, r, mask = np.asarray(counts), np.asarray(rates, float), np.asarray(region, bool)
    if n.ndim != 2 or r.ndim != 2 or n.shape[1] != r.shape[0]:
        raise ValueError("count/rate dimension mismatch")
    if not np.isfinite(n).all() or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("invalid counts")
    if not np.isfinite(r).all() or np.any(r <= 0) or mask.shape != (r.shape[1],):
        raise ValueError("invalid map or mask")
    if not 0 < mask.mean() < 1 or not np.isfinite(dt) or dt <= 0 or delta <= 0:
        raise ValueError("empty region/complement or invalid readout settings")
    ids = np.concatenate(groups)
    if len(ids) != len(np.unique(ids)) or not np.isin(ids, np.arange(len(r))).all():
        raise ValueError("groups must be disjoint valid cell indices")
    masses, bfs, calls, spikes = [], [], [], []
    prior_odds = np.log(mask.mean()) - np.log1p(-mask.mean())
    for g in groups:
        if not len(g):
            raise ValueError("empty population")
        ll = n[:, g] @ np.log(r[g]) - dt * r[g].sum(axis=0)
        inside, outside = logsumexp(ll[:, mask], axis=1), logsumexp(ll[:, ~mask], axis=1)
        bf = inside - outside - prior_odds
        q = np.exp(inside - np.logaddexp(inside, outside))
        total = n[:, g].sum(axis=1)
        # Category 1 is neutral. Preserve raw silence evidence, but abstain in calls.
        y = np.where(bf > delta, 2, np.where(bf < -delta, 0, 1))
        y[total == 0] = 1
        masses.append(q)
        bfs.append(bf)
        calls.append(y)
        spikes.append(total)
    return {"mass": np.asarray(masses).T, "log_bf": np.asarray(bfs).T,
                "calls": np.asarray(calls).T, "spikes": np.asarray(spikes).T}


def coverage_groups(rates, region, cell_ids, k=4):
    r = np.asarray(rates, float)
    ids = np.asarray(cell_ids)
    if len(r) < k or ids.shape != (len(r),) or len(np.unique(ids)) != len(ids):
        raise ValueError("insufficient or duplicate cells")
    score = r[:, region].mean(axis=1) / np.maximum(r.mean(axis=1), 1e-15)
    order = np.lexsort((ids, score))
    return [g.astype(int) for g in np.array_split(order, k)], score


@dataclass
class IdentificationSet:
    a: np.ndarray
    b: np.ndarray
    table: np.ndarray

    def optimize(self, objective, conditional_pattern=None):
        c = np.asarray(objective, float)
        size = 2 * len(self.table)
        if c.shape != (size,):
            raise ValueError("bad objective")
        a, b = self.a, self.b
        eq, rhs = np.ones((1, size)), np.ones(1)
        if conditional_pattern is not None:
            j = int(conditional_pattern)
            if not 0 <= j < len(self.table):
                raise ValueError("invalid conditional pattern")
            # v = w/P(pattern), t = 1/P(pattern); sum(v)=t, denominator(v)=1.
            a, b = np.column_stack([a, -b]), np.zeros(len(b))
            eq = np.zeros((2, size + 1))
            eq[0, :size], eq[0, -1] = 1, -1
            eq[1, [j, len(self.table) + j]] = 1
            rhs, c = np.array([0., 1.]), np.r_[c, 0.]
        solutions = [linprog(sign * c, A_ub=a, b_ub=b, A_eq=eq, b_eq=rhs,
                             bounds=(0, None), method="highs") for sign in (1, -1)]
        if any(s.status == 2 for s in solutions):
            return {"lower": np.nan, "upper": np.nan, "status": "incompatible"}
        if not all(s.success for s in solutions):
            raise RuntimeError([s.message for s in solutions])
        values = [float(c @ s.x) for s in solutions]
        if values[0] > values[1] + 1e-7 or min(values) < -1e-7 or max(values) > 1 + 1e-7:
            raise RuntimeError("invalid probability bounds")
        return {"lower": float(np.clip(values[0], 0, 1)), "upper": float(np.clip(values[1], 0, 1)), "status": "feasible"}

    def prevalence(self):
        return self.optimize(np.r_[np.zeros(len(self.table)), np.ones(len(self.table))])

    def conditional(self, pattern):
        c = np.zeros(2 * len(self.table))
        c[len(self.table) + int(pattern)] = 1
        return self.optimize(c, conditional_pattern=pattern)


def identification_set(observed_intervals, calibration, slack=0.):
    ci, cal = np.asarray(observed_intervals, float), np.asarray(calibration, float)
    if cal.ndim != 4 or cal.shape[0] != 2 or cal.shape[2:] != (3, 2):
        raise ValueError("calibration must have shape (2,views,3,2)")
    tab = patterns(cal.shape[1])
    if ci.shape != (len(tab), 2) or not 0 <= slack <= 1:
        raise ValueError("bad intervals or transfer slack")
    for intervals in (ci, cal):
        if not np.isfinite(intervals).all() or np.any(intervals < 0) or np.any(intervals > 1) or np.any(intervals[..., 0] > intervals[..., 1]):
            raise ValueError("invalid probability interval")
    cal = np.stack([np.maximum(0, cal[..., 0] - slack), np.minimum(1, cal[..., 1] + slack)], axis=-1)
    m, rows, bounds = len(tab), [], []
    for j in range(m):
        v = np.zeros(2 * m)
        v[[j, m + j]] = 1
        rows.extend([v, -v])
        bounds.extend([ci[j, 1], -ci[j, 0]])
    for z, k, r in product(range(2), range(cal.shape[1]), range(3)):
        indicator = (tab[:, k] == r).astype(float)
        hi, lo = np.zeros(2 * m), np.zeros(2 * m)
        hi[z*m:(z+1)*m] = indicator - cal[z, k, r, 1]
        lo[z*m:(z+1)*m] = cal[z, k, r, 0] - indicator
        rows.extend([hi, lo])
        bounds.extend([0., 0.])
    return IdentificationSet(np.asarray(rows), np.asarray(bounds), tab)


def observed_intervals(readouts, alpha=.025):
    count = pattern_counts(readouts)
    return binomial_interval(count, count.sum(), alpha / len(count))


def fit_latent_class(readouts, seed=0, starts=8, max_iter=1000, tolerance=1e-8):
    y = np.asarray(readouts)
    hist = pattern_counts(y)
    tab = patterns(y.shape[1])
    rng, fits = np.random.default_rng(seed), []
    for attempt in range(starts):
        theta = rng.dirichlet(np.ones(3), size=(2, y.shape[1]))
        mixing, before, converged = .5, -np.inf, False
        for iteration in range(max_iter):
            logp = np.array([np.log(max(mixing if z else 1-mixing, 1e-12)) +
                            np.log(theta[z, np.arange(y.shape[1])[None, :], tab]).sum(axis=1) for z in (0, 1)]).T
            normalizer = logsumexp(logp, axis=1)
            likelihood = float(hist @ normalizer)
            resp = np.exp(logp - normalizer[:, None])
            if iteration and abs(likelihood-before) < tolerance * (1 + abs(before)):
                converged = True
                break
            before = likelihood
            weights = hist[:, None] * resp
            mixing = np.clip(weights[:, 1].sum()/hist.sum(), 1e-9, 1-1e-9)
            for z, k, r in product(range(2), range(y.shape[1]), range(3)):
                theta[z, k, r] = max(weights[tab[:, k] == r, z].sum(), 1e-10)
            theta /= theta.sum(axis=2, keepdims=True)
        logp = np.array([np.log(max(mixing if z else 1-mixing, 1e-12)) +
                        np.log(theta[z, np.arange(y.shape[1])[None, :], tab]).sum(axis=1) for z in (0, 1)]).T
        normalizer = logsumexp(logp, axis=1)
        likelihood = float(hist @ normalizer)
        resp = np.exp(logp-normalizer[:, None])
        positive = int(np.argmax(theta[:, :, 2].mean(axis=1)))
        q = resp[:, positive]
        pi = mixing if positive else 1-mixing
        fits.append({"pi": float(pi), "theta": theta[[1-positive, positive]], "q": q,
                         "likelihood": likelihood, "converged": converged, "iterations": iteration+1})
    best = max(fits, key=lambda f: f["likelihood"])
    codes = y.astype(int) @ (3 ** np.arange(y.shape[1]-1, -1, -1))
    return dict(**best, event_probability=best["q"][codes],
                multistart_prevalence_range=float(np.ptp([f["pi"] for f in fits])))

