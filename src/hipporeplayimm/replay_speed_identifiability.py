"""Scalar inverse calibration with explicit abstention and transfer checks."""

from __future__ import annotations

import numpy as np
from scipy.stats import t


def event_slope(moments, minimum_events=5, minimum_variance=.01):
    """Equal-event moments [E(x), E(y), E(x*x), E(x*y)], y already normalized."""
    values = np.asarray(moments, float).reshape(-1, 4)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < minimum_events:
        return np.nan
    x, y, xx, xy = values.mean(axis=0)
    variance = xx - x*x
    return float((xy-x*y)/variance) if variance >= minimum_variance else np.nan


def bootstrap_slope(moments, seed, draws=200):
    values = np.asarray(moments, float).reshape(-1, 4)
    values = values[np.isfinite(values).all(axis=1)]
    point = event_slope(values)
    if not np.isfinite(point):
        return point, -np.inf, np.inf
    samples = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    means = values[samples].mean(axis=1)
    variance = means[:, 2]-means[:, 0]**2
    good = variance >= .01
    slopes = np.full(draws, np.nan)
    slopes[good] = (means[good, 3]-means[good, 0]*means[good, 1])/variance[good]
    # Do not silently discard failed resamples and report an over-tight interval.
    if not good.all():
        return point, -np.inf, np.inf
    lo, hi = np.quantile(slopes, [.025, .975])
    return point, float(lo), float(hi)


def fit_inverse(statistic, gradient):
    x, y = np.asarray(statistic, float), np.asarray(gradient, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("invalid training arrays")
    good = np.isfinite(x)
    x, y = x[good], y[good]
    model = {"n_fit": len(x), "status": "insufficient_fit", "intercept": np.nan,
             "slope": np.nan, "x_mean": np.nan, "sxx": np.nan, "residual_sd": np.nan}
    if len(x) < 20:
        return model
    sxx = float(np.sum((x-x.mean())**2))
    if sxx < 1e-12:
        return {**model, "status": "uninformative_fit"}
    slope = float(np.sum((x-x.mean())*(y-y.mean())) / sxx)
    intercept = float(y.mean()-slope*x.mean())
    residual = y-(intercept+slope*x)
    return {"n_fit": len(x), "status": "fitted", "intercept": intercept, "slope": slope,
            "x_mean": float(x.mean()), "sxx": sxx,
            "residual_sd": float(np.sqrt(np.sum(residual**2)/(len(x)-2)))}


def conformal_radius(model, statistic, gradient, alpha=.05):
    x, y = np.asarray(statistic, float), np.asarray(gradient, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(y).all() or not 0 < alpha < 1:
        raise ValueError("invalid calibration arrays or alpha")
    rank = int(np.ceil((len(x)+1)*(1-alpha)))
    if model["status"] != "fitted" or rank > len(x):
        return np.inf
    residuals = np.full(len(x), np.inf)
    good = np.isfinite(x)
    residuals[good] = np.abs(y[good]-(model["intercept"]+model["slope"]*x[good]))
    return float(np.sort(residuals)[rank-1])


def inverse_intervals(model, radius, statistic):
    if model["status"] != "fitted" or not np.isfinite(statistic):
        return {name: (np.nan, -np.inf, np.inf) for name in ["inverse_gaussian", "inverse_conformal"]}
    point = model["intercept"]+model["slope"]*statistic
    gaussian = t.ppf(.975, model["n_fit"]-2) * model["residual_sd"] * np.sqrt(
        1+1/model["n_fit"]+(statistic-model["x_mean"])**2/model["sxx"])
    return {"inverse_gaussian": (point, point-gaussian, point+gaussian),
            "inverse_conformal": (point, point-radius, point+radius)}


def interval_decision(lower, upper, truth, equivalence_bound):
    if not np.isfinite(truth) or not np.isfinite(equivalence_bound) or equivalence_bound <= 0 or np.isnan([lower, upper]).any() or lower > upper:
        raise ValueError("invalid interval, truth or equivalence bound")
    finite = bool(np.isfinite([lower, upper]).all())
    inside = abs(truth) < equivalence_bound
    claim = finite and lower > -equivalence_bound and upper < equivalence_bound
    return {"finite_interval": finite, "covered": lower <= truth <= upper,
            "interval_width": upper-lower,
            "nonzero_claim": finite and (lower > 0 or upper < 0),
            "equivalence_claim": claim, "truth_inside_equivalence": inside,
            "false_equivalence": claim and not inside,
            "decision": "equivalent_in_surrogate" if claim else "nonzero_in_surrogate" if finite and (lower > 0 or upper < 0) else "inconclusive" if finite else "abstain"}
