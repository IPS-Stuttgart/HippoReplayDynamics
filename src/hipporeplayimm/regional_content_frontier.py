"""Known-truth regional discrimination and full-population calibration checks."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import rankdata

from hipporeplayimm.regional_content_bounds import identification_set


def regional_log_bf(counts, rates, region, exposure=.02, conditional=False):
    n, r, mask = np.asarray(counts), np.asarray(rates), np.asarray(region, bool)
    if n.ndim != 2 or r.ndim != 2 or n.shape[1] != r.shape[0]:
        raise ValueError("count/rate shape mismatch")
    if not np.isfinite(n).all() or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("invalid spike counts")
    if not np.isfinite(r).all() or np.any(r <= 0) or mask.shape != (r.shape[1],):
        raise ValueError("invalid rates or mask")
    if not mask.any() or mask.all():
        raise ValueError("empty region or complement")
    e = np.broadcast_to(np.asarray(exposure, float), (len(n),))
    if not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError("invalid exposure")
    output = []
    for start in range(0, len(n), 1024):
        part = n[start:start+1024]
        ll = part @ np.log(r)
        if conditional:
            ll -= part.sum(axis=1)[:, None]*np.log(r.sum(axis=0))
        else:
            ll -= e[start:start+1024, None]*r.sum(axis=0)
        output.append(logsumexp(ll[:, mask], axis=1)-np.log(mask.sum())
                      -logsumexp(ll[:, ~mask], axis=1)+np.log((~mask).sum()))
    return np.concatenate(output) if output else np.empty(0)


def discrimination(log_bf, truth):
    score, z = np.asarray(log_bf, float), np.asarray(truth)
    if score.shape != z.shape or not np.isfinite(score).all() or not np.isin(z, [0, 1]).all():
        raise ValueError("invalid scores/truth")
    positive, negative = int(z.sum()), int((z == 0).sum())
    if not positive or not negative:
        return {"auc": np.nan, "balanced_error": np.nan, "balanced_brier": np.nan, "balanced_log_loss": np.nan}
    auc = (rankdata(score)[z == 1].sum()-positive*(positive+1)/2)/(positive*negative)
    q = expit(score)
    # Ties have half loss; neither class gets a special zero-evidence advantage.
    mistakes = np.where(score == 0, .5, (score > 0) != z)
    per_class = lambda loss: float(.5*(np.mean(loss[z == 0])+np.mean(loss[z == 1])))
    return {"auc": float(auc), "balanced_error": per_class(mistakes),
                "balanced_brier": per_class((q-z)**2),
                "balanced_log_loss": per_class(np.logaddexp(0, score)-z*score)}


def calls_from_bf(log_bf, totals):
    score = np.asarray(log_bf)
    calls = np.where(score > np.log(3), 2, np.where(score < -np.log(3), 0, 1))
    calls[np.asarray(totals) == 0] = 1
    return calls


def thin_counts(counts, desired, rng):
    n, desired = np.asarray(counts), np.asarray(desired)
    if n.ndim != 2 or desired.shape != (len(n),) or np.any(n < 0) or np.any(desired < 0):
        raise ValueError("invalid thinning inputs")
    total = n.sum(axis=1)
    p = np.divide(desired, total, out=np.zeros(len(n), float), where=total > 0).clip(0, 1)
    return rng.binomial(n, p[:, None]), p, desired > total


def mixture_fit(calibration, observed):
    cal, obs = np.asarray(calibration, float), np.asarray(observed, float)
    if cal.shape != (2, 3) or obs.shape != (3,) or np.any(cal < 0) or np.any(obs < 0):
        raise ValueError("invalid category counts")
    if np.any(cal.sum(axis=1) == 0) or obs.sum() == 0:
        return {"prevalence": np.nan, "fit_tv": np.nan, "separation": np.nan, "status": "missing_class"}
    emission = (cal+.5)/(cal.sum(axis=1)[:, None]+1.5)
    distance = float(np.abs(emission[1]-emission[0]).sum()/2)
    if distance < 1e-10:
        return {"prevalence": np.nan, "fit_tv": float(np.abs(obs/obs.sum()-emission[0]).sum()/2),
                    "separation": distance, "status": "unidentified"}
    def loss(pi):
        return float(-obs @ np.log((1-pi)*emission[0]+pi*emission[1]))
    fit = minimize_scalar(loss, bounds=(0, 1), method="bounded")
    if not fit.success:
        raise RuntimeError(fit.message)
    pi = min((0., float(fit.x), 1.), key=loss)
    fitted = (1-pi)*emission[0]+pi*emission[1]
    return {"prevalence": pi, "fit_tv": float(np.abs(fitted-obs/obs.sum()).sum()/2),
                "separation": distance, "status": "fit"}


def block_histograms(calls, blocks, truth=None):
    y, block = np.asarray(calls), np.asarray(blocks)
    if y.shape != block.shape or not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("invalid blocked calls")
    _, codes = np.unique(block, return_inverse=True)
    if truth is None:
        result = np.zeros((codes.max()+1, 3), int)
        np.add.at(result, (codes, y), 1)
    else:
        z = np.asarray(truth)
        if z.shape != y.shape or not np.isin(z, [0, 1]).all():
            raise ValueError("invalid calibration labels")
        result = np.zeros((codes.max()+1, 2, 3), int)
        np.add.at(result, (codes, z, y), 1)
    return result


def bootstrap_compatibility(cal_blocks, target_blocks, rng, repeats=200):
    cal, obs = cal_blocks.sum(axis=0), target_blocks.sum(axis=0)
    estimate = mixture_fit(cal, obs)
    if np.any(cal.sum(axis=1) == 0):
        return estimate, [{"slack": s, "lower": 0., "upper": 1., "status": "missing_class"} for s in (0., .05, .10, 1.)], {}
    cp, op, estimates = [], [], []
    for _ in range(repeats):
        c = cal_blocks[rng.integers(len(cal_blocks), size=len(cal_blocks))].sum(axis=0)
        o = target_blocks[rng.integers(len(target_blocks), size=len(target_blocks))].sum(axis=0)
        cp.append(np.divide(c, c.sum(axis=1)[:, None], out=np.full((2, 3), np.nan), where=c.sum(axis=1)[:, None] > 0))
        op.append(o/o.sum())
        estimates.append(mixture_fit(c, o)["prevalence"])
    cp, op, estimates = np.asarray(cp), np.asarray(op), np.asarray(estimates)
    # Include the empirical point; bootstrap ranges are compatibility diagnostics.
    ci = np.moveaxis(np.nanquantile(cp, [.025, .975], axis=0), 0, -1)
    for z in (0, 1):
        if np.isnan(cp[:, z]).any():
            ci[z, :, 0], ci[z, :, 1] = 0, 1
    oi = np.quantile(op, [.025, .975], axis=0).T
    for intervals, point in ((ci, cal/cal.sum(axis=1)[:, None]), (oi, obs/obs.sum())):
        intervals[..., 0] = np.minimum(intervals[..., 0], point)
        intervals[..., 1] = np.maximum(intervals[..., 1], point)
    rows = [dict(slack=s, **identification_set(oi, ci[:, None], s).prevalence()) for s in (0., .05, .10, 1.)]
    return estimate, rows, {"calibration_bootstrap": cp, "target_bootstrap": op,
                               "prevalence_bootstrap": estimates, "calibration_intervals": ci, "observed_intervals": oi}


def generate_counts(rates, region, pool, per_class, rng, generator, multiplier=1, gain=1., dt=.02):
    z = np.repeat([0, 1], per_class)
    states = np.r_[rng.choice(np.flatnonzero(~region), per_class), rng.choice(np.flatnonzero(region), per_class)]
    if generator == "poisson":
        n = rng.poisson(rates[:, states].T*dt*gain*multiplier)
        requested = np.full(len(z), -1)
    elif generator == "conditional_multinomial":
        requested = rng.choice(pool, len(z))*multiplier
        probs = rates[:, states].T.copy()
        probs /= probs.sum(axis=1)[:, None]
        n = np.asarray([rng.multinomial(int(k), p) for k, p in zip(requested, probs, strict=True)])
    else:
        raise ValueError("unknown generator")
    return n, z, states, requested
