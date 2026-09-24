"""Conditional two-period spatial-alignment score with Poisson nuisances removed."""

import numpy as np
from scipy.special import expit, gammaln


def fixed_total_moments(counts, log_odds, feature, target_total):
    """Moments of the feature sum under weighted fixed-size Bernoulli selection."""
    raw = np.asarray(counts)
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
        raise ValueError("noninteger_counts")
    c = raw.astype(int)
    odds = np.asarray(log_odds, float)
    a = np.asarray(feature, float)
    if not (c.shape == odds.shape == a.shape) or np.any(c < 0):
        raise ValueError("invalid_shapes_or_counts")
    if not np.isfinite(odds).all() or not np.isfinite(a).all():
        raise ValueError("nonfinite_feature_or_odds")
    n, m = int(c.sum()), int(target_total)
    if m != target_total or m < 0 or m > n:
        raise ValueError("invalid_target_total")
    if m == 0:
        return 0.0, 0.0
    if m == n:
        return float(c @ a), 0.0
    if m > n // 2:
        mu, var = fixed_total_moments(c, -odds, a, n - m)
        return float(c @ a) - mu, var
    # A common log-odds tilt cancels after conditioning on M and avoids underflow.
    lo, hi = -float(odds.max()) - 50, -float(odds.min()) + 50
    for _ in range(60):
        mid = (lo + hi) / 2
        if c @ expit(odds + mid) < m:
            lo = mid
        else:
            hi = mid
    tilted = odds + (lo + hi) / 2
    center = float(c @ a) / n
    a = a - center
    z = np.zeros(m + 1)
    first, second = z.copy(), z.copy()
    z[0] = 1
    for count, log_w, value in zip(c, tilted, a, strict=True):
        if count == 0:
            continue
        j = np.arange(min(int(count), m) + 1)
        log_p = gammaln(count + 1) - gammaln(j + 1) - gammaln(count - j + 1) + j * log_w
        p = np.exp(log_p - log_p.max())
        term = j * value
        zz = np.convolve(z, p)[: m + 1]
        ff = (np.convolve(first, p) + np.convolve(z, p * term))[: m + 1]
        ss = (np.convolve(second, p) + 2 * np.convolve(first, p * term) + np.convolve(z, p * term**2))[: m + 1]
        scale = zz.max()
        if scale <= 0:
            raise ValueError("conditional_underflow")
        z, first, second = zz / scale, ff / scale, ss / scale
    if z[m] <= 0:
        raise ValueError("impossible_target_total")
    mean = first[m] / z[m]
    variance = second[m] / z[m] - mean**2
    if variance < -1e-8:
        raise ValueError("negative_conditional_variance")
    return float(mean + center * m), float(max(0, variance))


def reference_feature(rate, common):
    rate = np.asarray(rate, float)
    if np.any(rate <= 0) or not np.isfinite(rate).all():
        raise ValueError("invalid_reference")
    a = np.log(rate)
    sd = float(a[common].std()) if np.any(common) else 0.0
    if sd < 1e-10:
        return np.zeros_like(a)
    return (a - a[common].mean()) / sd


def spatial_score(before, after, before_exposure, after_exposure, feature):
    b, y = np.asarray(before), np.asarray(after)
    t0, t1, a = [np.asarray(v, float) for v in (before_exposure, after_exposure, feature)]
    if not (b.shape == y.shape == t0.shape == t1.shape == a.shape):
        raise ValueError("shape_mismatch")
    if not all(np.isfinite(v).all() for v in (b, y, t0, t1, a)):
        raise ValueError("nonfinite_input")
    if np.any(b < 0) or np.any(y < 0) or np.any(b != np.floor(b)) or np.any(y != np.floor(y)):
        raise ValueError("noninteger_or_negative_counts")
    if np.any(t0 < 0) or np.any(t1 < 0) or np.any((t0 == 0) & (b > 0)) or np.any((t1 == 0) & (y > 0)):
        raise ValueError("counts_without_exposure")
    common = (t0 > 0) & (t1 > 0)
    c = (b + y)[common].astype(int)
    m = int(y[common].sum())
    mu, var = fixed_total_moments(c, np.log(t1[common] / t0[common]), a[common], m)
    return {
        "score": float(y[common] @ a[common] - mu),
        "information": var,
        "before_common_spikes": int(b[common].sum()),
        "after_common_spikes": m,
        "excluded_spikes": int((b + y)[~common].sum()),
        "informative": bool(var > 1e-10),
    }


def anchor_score(predictors, scores, information):
    x, u, v = [np.asarray(a, float) for a in (predictors, scores, information)]
    if not (x.shape == u.shape == v.shape) or not all(np.isfinite(a).all() for a in (x, u, v)) or np.any(v < 0):
        raise ValueError("invalid_association_inputs")
    if v.sum() <= 1e-10:
        return 0.0, 0.0
    centered = x - np.dot(x, v) / v.sum()
    return float(centered @ u), float((centered**2) @ v)
