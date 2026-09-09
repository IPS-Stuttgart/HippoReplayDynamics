"""Likelihood mixtures for simulated populations, not hard event labels."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

MODELS = ("physical", "neural", "stationary", "iid")
CONDITIONS = ("exact", "train_geometry", "gain_drift")
SCENARIOS = (0.25, 0.5, 0.75)
SIZES = (32, 128)
REPEATS = 50
EPS = 1e-9
LR95 = 3.841458820694124


def simplex_fit(log_scores, initial=None):
    ll = np.asarray(log_scores, dtype=float)
    if ll.ndim != 2 or min(ll.shape) < 2 or not np.isfinite(ll).all():
        raise ValueError("finite nonempty observation-by-model log scores required")
    shift = ll.max(axis=1)
    x = np.exp(ll - shift[:, None])
    _n, k = x.shape
    start = np.full(k, 1 / k) if initial is None else np.asarray(initial, float)
    if start.shape != (k,) or not np.isfinite(start).all() or (start < 0).any() or not np.isclose(start.sum(), 1):
        raise ValueError("simplex initial weights required")
    start = EPS + (1 - k * EPS) * start / start.sum()

    def objective(w):
        mass = x @ w
        return -np.log(mass).mean(), -(x / mass[:, None]).mean(axis=0)

    result = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=[(EPS, 1)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)},
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w = result.x
    gradient = (x / (x @ w)[:, None]).mean(axis=0)
    free = w > EPS * 10
    level = gradient[free].mean()
    certificate = max(float(np.max(abs(gradient[free] - level), initial=0)), float(np.max(gradient[~free] - level, initial=0)), abs(w.sum() - 1))
    if not result.success or certificate > 2e-5 or np.any(w < EPS / 2):
        raise RuntimeError(f"mixture optimizer failed: {result.message}; KKT={certificate}")
    return w, float(np.log(x @ w).sum() + shift.sum()), certificate


def population_fit(log_scores):
    ll = np.asarray(log_scores, float)
    if ll.ndim != 2 or ll.shape[1] != 4 or len(ll) < 2 or not np.isfinite(ll).all():
        raise ValueError("four finite log likelihoods per event required")
    ll = ll - ll.max(axis=1, keepdims=True)
    weights, maximum, certificate = simplex_fit(ll)
    moving = weights[0] + weights[1]
    estimate = weights[1] / moving
    profiles = {}

    def profile(phi):
        phi = float(phi)
        if phi not in profiles:
            with np.errstate(divide="ignore"):
                dynamic = np.logaddexp(ll[:, 0] + np.log1p(-phi), ll[:, 1] + np.log(phi))
            w, value, error = simplex_fit(np.column_stack([dynamic, ll[:, 2:]]), [moving, weights[2], weights[3]])
            if value > maximum + 1e-5:
                raise RuntimeError("profile exceeds unconstrained optimum")
            profiles[phi] = w, value, error
        return profiles[phi]

    def boundary(phi):
        return 2 * (maximum - profile(phi)[1]) - LR95

    low = 0.0 if boundary(0) <= 0 else brentq(boundary, 0, estimate, xtol=1e-7)
    high = 1.0 if boundary(1) <= 0 else brentq(boundary, estimate, 1, xtol=1e-7)
    half = profile(0.5)
    out = {
        "phi_hat": estimate,
        "phi_low": low,
        "phi_high": high,
        "relative_log_likelihood": maximum,
        "null_lr": max(0.0, 2 * (maximum - half[1])),
        "moving_weight": moving,
        "kkt_error": certificate,
    }
    out.update({"weight_" + m: float(w) for m, w in zip(MODELS, weights, strict=True)})
    for label, point in (("low", low), ("high", high), ("null", 0.5)):
        w, value, error = profile(point)
        out[label + "_profile_log_likelihood"] = value
        out[label + "_kkt_error"] = error
        out.update({label + "_weight_" + m: float(v) for m, v in zip(("moving", "stationary", "iid"), w, strict=True)})
    out["direction"] = "neural" if low > 0.5 else "physical" if high < 0.5 else "undetermined"
    return out


def wilson(successes, total):
    if total < 1:
        raise ValueError("nonempty Monte Carlo denominator required")
    p, z = successes / total, 1.959963984540054
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return center - radius, center + radius


def summarize(fits, repeats=REPEATS, sizes=SIZES):
    keys = ["dataset", "scenario", "repeat", "events_per_recording", "condition"]
    expected = set(product(("pfeiffer_foster", "tanni2022"), SCENARIOS, range(repeats), sizes, CONDITIONS))
    if fits.empty or fits.duplicated(keys).any() or set(fits[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("complete unique population ensembles required")
    x = fits.copy()
    values = x[["phi_hat", "phi_low", "phi_high"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or (x.phi_low > x.phi_high).any():
        raise ValueError("finite ordered mixture estimates required")
    x["error"] = x.phi_hat - x.scenario
    x["covered"] = (x.phi_low <= x.scenario + 1e-8) & (x.phi_high >= x.scenario - 1e-8)
    x["directional_claim"] = (x.phi_low > 0.5) | (x.phi_high < 0.5)
    x["correct_direction"] = ((x.scenario > 0.5) & (x.phi_low > 0.5)) | ((x.scenario < 0.5) & (x.phi_high < 0.5))
    rows = []
    for key, group in x.groupby(["dataset", "condition", "events_per_recording", "scenario"]):
        n = len(group)
        row = dict(zip(["dataset", "condition", "events_per_recording", "scenario"], key, strict=True))
        row.update(
            n_replicates=n,
            mean_estimate=group.phi_hat.mean(),
            bias=group.error.mean(),
            rmse=np.sqrt(np.mean(group.error**2)),
            median_interval_width=(group.phi_high - group.phi_low).median(),
        )
        for col in ("covered", "directional_claim", "correct_direction"):
            count = int(group[col].sum())
            low, high = wilson(count, n)
            row.update({col + "_fraction": count / n, col + "_mc_low": low, col + "_mc_high": high})
        rows.append(row)
    summary = pd.DataFrame(rows)
    gates = []
    for key, group in summary.groupby(["dataset", "condition", "events_per_recording"]):
        null = group[group.scenario.eq(0.5)].iloc[0]
        power = group[group.scenario.ne(0.5)].correct_direction_fraction.min()
        coverage, bias = group.covered_fraction.min(), group.bias.abs().max()
        row = dict(zip(["dataset", "condition", "events_per_recording"], key, strict=True))
        row.update(
            min_direction_power=power,
            min_coverage=coverage,
            max_absolute_bias=bias,
            null_false_direction_fraction=null.directional_claim_fraction,
            practical_pass=bool(power >= 0.8 and coverage >= 0.9 and bias <= 0.1 and null.directional_claim_fraction <= 0.05),
        )
        gates.append(row)
    return x, summary, pd.DataFrame(gates)
