"""Known-path experiments using empirical rate maps, with no motion prior."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .encoding import _positions_to_flat_bins
from .replay_coverage import continuity_metrics, decode_independent
from .replay_coverage_validation import posterior_coverage

BASE_S = .005
WINDOW_S = .020


def fold_box(points, bounds):
    """Specular rectangular reflection, including arbitrarily long steps."""
    lo, hi = np.asarray(bounds, dtype=float)
    width = hi - lo
    if not np.isfinite(bounds).all() or np.any(width <= 0):
        raise ValueError("invalid simulation domain")
    phase = (np.asarray(points) - lo) % (2 * width)
    return lo + width - np.abs(phase - width), np.where(phase <= width, 1., -1.)


def spatial_covariate(points, bounds):
    """Horizontal physical coordinate, NOT inferred distance to PF arena walls."""
    return 2 * (np.asarray(points)[..., 0] - bounds[0, 0]) / (bounds[1, 0] - bounds[0, 0]) - 1


def simulate_path(n_base, bounds, kind, gradient, speed_cm_s, seed, fine_s=.001):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("invalid simulation domain")
    if not np.isfinite(fine_s) or fine_s <= 0 or int(n_base) != n_base:
        raise ValueError("invalid simulation time grid")
    n_base = int(n_base)
    n_per_base = round(BASE_S / fine_s)
    if n_base < 4 or n_per_base < 1 or not np.isclose(n_per_base * fine_s, BASE_S):
        raise ValueError("need at least 20 ms and an integer fine/base ratio")
    if not np.isfinite([gradient, speed_cm_s]).all() or abs(gradient) >= 1 or speed_cm_s <= 0:
        raise ValueError("invalid speed field")
    rng = np.random.default_rng(seed)
    origin = rng.uniform(bounds[0], bounds[1])
    angle = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    n = n_base * n_per_base
    edges, midpoint = np.empty((n + 1, 2)), np.empty((n, 2))
    speed = np.empty(n)
    edges[0] = origin
    if kind == "stationary":
        edges[:] = origin
        midpoint[:] = origin
        speed[:] = 0
    elif kind == "discontinuous":
        snapshots = rng.uniform(bounds[0], bounds[1], size=(n_base, 2))
        midpoint[:] = np.repeat(snapshots, n_per_base, axis=0)
        edges[:-1] = midpoint
        edges[-1] = midpoint[-1]
        speed[:] = np.nan
    elif kind == "continuous":
        for k in range(n):
            local_speed = speed_cm_s * (1 + gradient * spatial_covariate(edges[k], bounds))
            predictor, _ = fold_box(edges[k] + .5 * fine_s * local_speed * direction, bounds)
            speed[k] = speed_cm_s * (1 + gradient * spatial_covariate(predictor, bounds))
            delta = fine_s * speed[k] * direction
            midpoint[k], _ = fold_box(edges[k] + .5 * delta, bounds)
            edges[k + 1], signs = fold_box(edges[k] + delta, bounds)
            direction *= signs
    else:
        raise ValueError("unknown truth trajectory class")
    return {"edges_cm": edges, "midpoints_cm": midpoint, "speed_cm_s": speed,
            "covariate": spatial_covariate(midpoint, bounds), "fine_s": fine_s,
            "n_per_base": n_per_base}


def map_interpolator(rates, x_edges, y_edges):
    """Bilinear empirical-map surrogate; rate values clamp at grid-center edges."""
    x, y = (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2
    values = rates.reshape(rates.shape[0], len(x), len(y)).transpose(1, 2, 0)
    interpolator = RegularGridInterpolator((x, y), values, method="linear", bounds_error=True)

    def evaluate(points):
        return interpolator(np.clip(points, [x[0], y[0]], [x[-1], y[-1]]))

    return evaluate


def fine_population_totals(base_totals, n_per_base, seed):
    totals = np.asarray(base_totals)
    if totals.ndim != 1 or not np.isfinite(totals).all() or np.any(totals < 0) or np.any(totals != np.floor(totals)):
        raise ValueError("population totals must be nonnegative integers")
    if n_per_base < 1 or int(n_per_base) != n_per_base:
        raise ValueError("invalid fine-bin count")
    if not len(totals):
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.multinomial(int(n), np.full(n_per_base, 1 / n_per_base)) for n in totals])


def sample_observations(path_rates, path, source_totals, subsets, rate_scale, seed):
    """Return native removals and count-restored subsets; never redraw outcomes."""
    rates = np.asarray(path_rates)
    if rates.ndim != 2 or not np.isfinite(rates).all() or np.any(rates <= 0) or not np.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError("path rates and scale must be finite and positive")
    sub_seeds = np.random.SeedSequence(seed).spawn(4)
    n_per_base = path["n_per_base"]
    n_base = len(source_totals)
    if rates.shape[0] != n_base * n_per_base:
        raise ValueError("source-count and path duration mismatch")
    native = np.random.default_rng(sub_seeds[0]).poisson(rates * path["fine_s"] * rate_scale)
    gain_rng = np.random.default_rng(sub_seeds[1])
    gain_sd = np.sqrt(np.log(2.0))
    gain_width = round(WINDOW_S / path["fine_s"])
    block_gain = gain_rng.lognormal(-gain_sd**2 / 2, gain_sd, int(np.ceil(len(rates) / gain_width)))
    gain = np.repeat(block_gain, gain_width)[:len(rates)]
    fluctuating = gain_rng.poisson(rates * (path["fine_s"] * rate_scale * gain[:, None]))
    totals = fine_population_totals(source_totals, n_per_base, sub_seeds[2])
    uniforms = np.random.default_rng(sub_seeds[3]).random(int(totals.sum()))
    results = {}
    for fraction, subset in subsets.items():
        subset = np.asarray(subset)
        if subset.ndim != 1 or not len(subset) or not np.issubdtype(subset.dtype, np.integer) or len(np.unique(subset)) != len(subset) or np.any((subset < 0) | (subset >= rates.shape[1])):
            raise ValueError("invalid cell subset")
        def aggregate(counts):
            return counts.reshape(n_base, n_per_base, counts.shape[1]).sum(axis=1)
        results["poisson_oracle", fraction] = aggregate(native[:, subset])
        results["shared_gain_mismatch", fraction] = aggregate(fluctuating[:, subset])
        restored = np.zeros((len(rates), len(subset)), dtype=np.int64)
        offset = 0
        for k, total in enumerate(totals):
            p = rates[k, subset]
            cdf = np.cumsum(p / p.sum())
            cdf[-1] = 1.
            cells = np.searchsorted(cdf, uniforms[offset:offset + total], side="right")
            restored[k] = np.bincount(cells, minlength=len(subset))
            offset += total
        results["fixed_count", fraction] = aggregate(restored)
    return results, {"mean_shared_gain": float(gain.mean()), "min_shared_gain": float(gain.min()),
                     "max_shared_gain": float(gain.max())}


def truth_windows(path):
    n_per_base = path["n_per_base"]
    n_fine_window = 4 * n_per_base
    starts = np.arange(0, len(path["midpoints_cm"]) - n_fine_window + 1, n_per_base)
    prefix = np.vstack([np.zeros((1, 2)), np.cumsum(path["midpoints_cm"], axis=0)])
    means = (prefix[starts + n_fine_window] - prefix[starts]) / n_fine_window
    center_indices = starts + n_fine_window // 2
    indices = np.arange(0, len(means), 4)
    center_xy = path["edges_cm"][center_indices]
    true_speed, q = [], []
    for a, b in zip(center_indices[indices[:-1]], center_indices[indices[1:]], strict=True):
        true_speed.append(path["speed_cm_s"][a:b].mean())
        q.append(path["covariate"][a:b].mean())
    return {"window_mean_cm": means, "center_cm": center_xy,
            "speed_cm_s": np.asarray(true_speed), "true_covariate": np.asarray(q),
            "window_mean_chord_speed_cm_s": np.linalg.norm(np.diff(means[indices], axis=0), axis=1) / WINDOW_S}


def true_state_indices(points, x_edges, y_edges, valid):
    flat = _positions_to_flat_bins(points, x_edges, y_edges)
    lookup = np.full(len(valid), -1, dtype=int)
    lookup[valid] = np.arange(np.count_nonzero(valid))
    result = np.full(len(flat), -1, dtype=int)
    ok = (flat >= 0) & (flat < len(lookup))
    result[ok] = lookup[flat[ok]]
    return result


def decode_batches(counts, rates, grid, truth_indices, likelihood):
    if len(truth_indices) != len(counts):
        raise ValueError("truth/count length mismatch")
    chunks = []
    for lo in range(0, len(counts), 512):
        decoded = decode_independent(counts[lo:lo + 512], rates, grid, WINDOW_S, likelihood=likelihood)
        decoded["hpd95"] = posterior_coverage(decoded.pop("posterior"), truth_indices[lo:lo + 512])["hpd95"]
        chunks.append(decoded)
    if not chunks:
        return {key: np.empty((0, grid.shape[1])) if key in {"map", "posterior_mean"} else np.empty(0) for key in ["map", "posterior_mean", "posterior_rms_cm", "posterior_entropy_nats", "hpd95"]}
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def paired_speed_moments(path, counts, truth, bounds, apply_support):
    """Equal-event regression sufficient statistics; preserve all failed windows."""
    good = (counts.sum(axis=1) >= 3) & (np.count_nonzero(counts, axis=1) >= 2)
    valid = good if apply_support else np.ones(len(path), dtype=bool)
    selected = continuity_metrics(path, valid_bins=valid)
    idx = np.arange(0, len(path), 4)
    speed = np.linalg.norm(np.diff(path[idx], axis=0), axis=1) / WINDOW_S
    q_decoded = spatial_covariate((path[idx[:-1]] + path[idx[1:]]) / 2, bounds)
    pairs = valid[idx[:-1]] & valid[idx[1:]]
    core = (idx >= selected["continuous_start"]) & (idx < selected["continuous_end_exclusive"])
    core_pairs = pairs & core[:-1] & core[1:] & selected["continuity_pass"]
    rows = {}
    for name, keep in [("all", pairs), ("selected", core_pairs)]:
        rows[f"{name}_steps"] = int(keep.sum())
        rows[f"{name}_median_true_arclength_speed_cm_s"] = float(np.median(truth["speed_cm_s"][keep])) if keep.any() else np.nan
        rows[f"{name}_median_true_chord_speed_cm_s"] = float(np.median(truth["window_mean_chord_speed_cm_s"][keep])) if keep.any() else np.nan
        rows[f"{name}_median_speed_error_vs_arclength_cm_s"] = float(np.median(speed[keep] - truth["speed_cm_s"][keep])) if keep.any() else np.nan
        rows[f"{name}_median_speed_error_vs_chord_cm_s"] = float(np.median(speed[keep] - truth["window_mean_chord_speed_cm_s"][keep])) if keep.any() else np.nan
        for axis, x in [("true_coordinate", truth["true_covariate"]), ("decoded_coordinate", q_decoded)]:
            for readout, y in [("decoded", speed), ("true_arclength", truth["speed_cm_s"]), ("true_chord", truth["window_mean_chord_speed_cm_s"])]:
                finite = keep & np.isfinite(x) & np.isfinite(y)
                prefix = f"{name}__{axis}__{readout}"
                rows[f"{prefix}__steps"] = int(finite.sum())
                for label, values in [("x", x), ("y", y), ("xx", x * x), ("xy", x * y)]:
                    rows[f"{prefix}__mean_{label}"] = float(np.mean(values[finite])) if finite.any() else np.nan
    return rows


def gradient_from_moments(frame, prefix, speed_scale, min_events=5, min_variance=.01):
    """Equal-event slope; repeated windows do not give long events extra weight."""
    cols = [f"{prefix}__mean_{suffix}" for suffix in ["x", "y", "xx", "xy"]]
    local = frame.loc[frame[f"{prefix}__steps"].gt(0), cols].dropna()
    if local.empty:
        return {"events": 0, "spatial_variance": np.nan, "normalized_slope": np.nan, "status": "no_steps"}
    x, y, xx, xy = local.mean().to_numpy()
    variance = xx - x * x
    status = "available" if len(local) >= min_events and variance >= min_variance else "insufficient_events_or_spatial_spread"
    return {"events": len(local), "spatial_variance": float(variance),
            "normalized_slope": float((xy - x * y) / variance / speed_scale) if status == "available" else np.nan,
            "status": status}
