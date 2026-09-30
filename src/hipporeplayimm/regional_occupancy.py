"""Assumption-explicit time-occupancy calibration, without dynamics inference."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from hipporeplayimm.regional_blind_bank import circle_interval
from hipporeplayimm.regional_content_frontier import regional_log_bf


def window_occupancy(grid, ages, nodes, kind, center, duration_ms, radius=20.):
    """Exact dwell, returned in chronological (oldest-bin-first) order."""
    if duration_ms not in (20, 40, 60, 100) or kind not in ("stationary", "moving", "jumping"):
        raise ValueError("unsupported window or dynamics")
    grid, ages, nodes = np.asarray(grid), np.asarray(ages), np.asarray(nodes)
    if len(ages) != len(nodes) or ages[0] != 0 or (np.diff(ages) <= 0).any() or ages[-1] < duration_ms/1000-1e-9:
        raise ValueError("invalid or incomplete path")
    k = duration_ms//20
    result = np.zeros(k)
    for b in range(k):
        left, right = b*.02, (b+1)*.02
        for j in range(len(ages)-1):
            lo, hi = max(left, ages[j]), min(right, ages[j+1])
            if hi <= lo:
                continue
            start = grid[nodes[j]]
            if kind == "moving":
                velocity = (grid[nodes[j+1]]-start)/(ages[j+1]-ages[j])
                p, q = start+velocity*(lo-ages[j]), start+velocity*(hi-ages[j])
            else:
                p = q = start
            interval = circle_interval(p, q, center, radius)
            if interval is not None:
                result[b] += (hi-lo)*(interval[1]-interval[0])/.02
    if (result < -1e-8).any() or (result > 1+1e-8).any():
        raise ValueError("invalid geometric occupancy")
    return np.clip(result[::-1], 0, 1)


def bin_masses(counts, rates, region):
    shape = counts.shape[:-1]
    flat = counts.reshape(-1, counts.shape[-1])
    area = float(np.mean(region))
    if not 0 < area < 1:
        raise ValueError("Home and non-Home support required")
    q = expit(regional_log_bf(flat, rates, region, exposure=.02)+logit(area))
    neutral = np.where(flat.sum(axis=1) == 0, area, q)
    return {"neutral_silence": neutral.reshape(shape), "poisson_silence": q.reshape(shape)}


def emission_means(q, occupancy, definition="time_conditional"):
    q, o = np.asarray(q, float).ravel(), np.asarray(occupancy, float).ravel()
    if q.shape != o.shape or not len(q) or not np.isfinite(q).all() or not np.isfinite(o).all():
        raise ValueError("finite paired nonempty readouts required")
    if ((q < 0) | (q > 1) | (o < 0) | (o > 1)).any():
        raise ValueError("probabilities must lie in [0,1]")
    if definition == "time_conditional":
        wi, wo = o, 1-o
    elif definition == "pure_window":
        wi, wo = (o >= 1-1e-8).astype(float), (o <= 1e-8).astype(float)
    else:
        raise ValueError("unknown emission definition")
    inside, outside = float(wi.sum()), float(wo.sum())
    s = float(q@wi/inside) if inside > 0 else np.nan
    f = float(q@wo/outside) if outside > 0 else np.nan
    return {"s": s, "f": f, "s_minus_f": s-f, "home_window_equivalents": inside,
            "nonhome_window_equivalents": outside, "windows": len(q),
            "mean_mass": float(q.mean()), "true_occupancy": float(o.mean()),
            "mixed_window_fraction": float(((o > 1e-8) & (o < 1-1e-8)).mean()),
            "support_complete": bool(np.isfinite(s) and np.isfinite(f))}


def invariance(s, f, expected=7, tolerance=.02):
    s, f = np.asarray(s, float), np.asarray(f, float)
    complete = len(s) == expected and len(f) == expected and np.isfinite(s).all() and np.isfinite(f).all()
    ds, df = (float(np.ptp(s)), float(np.ptp(f))) if complete else (np.nan, np.nan)
    return {"s_range": ds, "f_range": df, "support_complete": bool(complete),
            "invariance_pass": bool(complete and ds <= tolerance and df <= tolerance)}


def correct_occupancy(qbar, s, f):
    if not np.isfinite([qbar, s, f]).all() or s-f <= 1e-12:
        return np.nan
    return float((qbar-f)/(s-f))
