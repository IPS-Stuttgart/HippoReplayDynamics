"""Train-only behavioral validation for replay recording-coverage calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import ReplaySession
from .encoding import _positions_to_flat_bins, _speed_cm_s, _times_in_intervals
from .replay_coverage_data import (
    CoverageInputConfig,
    array_sha256,
    fit_coverage_population,
    prepare_position_support,
)


def subtract_block(intervals, start, end):
    """Remove one guarded test block without joining separated RUN intervals."""
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("invalid excluded time block")
    result = []
    for lo, hi in np.asarray(intervals, dtype=float).reshape(-1, 2):
        if hi <= start or lo >= end:
            result.append((lo, hi))
        else:
            if start > lo:
                result.append((lo, start))
            if end < hi:
                result.append((end, hi))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def training_session(full: ReplaySession, test_start_s, test_end_s, guard_s, maximum_gap_s):
    """Remove test/guard positions AND spikes before grid, unit, or map fitting."""
    if not np.isfinite(guard_s) or guard_s < 0:
        raise ValueError("guard must be finite and nonnegative")
    intervals = subtract_block(full.run_times, test_start_s - guard_s, test_end_s + guard_s)
    keep_position = _times_in_intervals(full.position[:, 0], intervals)
    # Canonical RUN intervals include both endpoints; test/guard endpoints must
    # nevertheless be removed before grid fitting and boundary derivatives.
    keep_position &= (full.position[:, 0] < test_start_s - guard_s) | (full.position[:, 0] > test_end_s + guard_s)
    if keep_position.sum() < 4:
        raise ValueError("insufficient training position samples")
    train = replace(full, position=full.position[keep_position], run_times=intervals)
    bounds = full.metadata.get("arena_bounds_cm")
    train, _ = prepare_position_support(train, maximum_gap_s, bounds)
    keep_spikes = _times_in_intervals(full.spikes[:, 0], train.run_times)
    spikes = full.spikes[keep_spikes]
    if not len(spikes):
        raise ValueError("no training spikes")
    return replace(train, spikes=spikes)


def fit_training_fold(full, test_start_s, test_end_s, guard_s, config: CoverageInputConfig):
    train = training_session(full, test_start_s, test_end_s, guard_s, config.maximum_position_gap_s)
    arrays, units, meta = fit_coverage_population(train, config)
    cells = arrays["unit_qc_mask"].astype(bool)
    valid = arrays["valid_spatial_bins"].astype(bool)
    bounds = train.metadata.get("arena_bounds_cm")
    if bounds is not None:
        bounds = np.asarray(bounds)
        valid &= ((arrays["bin_centers_cm"] >= bounds[0]) & (arrays["bin_centers_cm"] <= bounds[1])).all(axis=1)
    if cells.sum() < 5 or valid.sum() < 2:
        raise ValueError(f"insufficient training-only encoding support: {cells.sum()} units, {valid.sum()} states")
    model = {
        "rates_hz": arrays["rates_hz"][cells][:, valid],
        "grid_cm": arrays["bin_centers_cm"][valid],
        "cell_ids": arrays["cell_ids"][cells],
        "x_edges_cm": arrays["x_edges_cm"], "y_edges_cm": arrays["y_edges_cm"],
        "valid_spatial_bins": valid, "training_intervals": train.run_times,
    }
    meta.update({
        "test_start_s": float(test_start_s), "test_end_s": float(test_end_s),
        "guard_s": float(guard_s), "training_positions_sha256": array_sha256(train.position),
        "training_spikes_sha256": array_sha256(train.spikes),
        "training_unit_ids_sha256": array_sha256(model["cell_ids"]),
        "training_rates_sha256": array_sha256(model["rates_hz"]),
        "training_spatial_states": int(valid.sum()),
        "full_RUN_unit_QC_used": False, "heldout_positions_used_for_grid": False,
    })
    return model, units, meta


def heldout_centers(full, start_s, end_s, max_window_s, step_s, max_samples, seed, min_speed_cm_s=10.0):
    """Choose test windows from behavior only, before reading test spike counts."""
    if not np.isfinite([start_s, end_s, max_window_s, step_s, min_speed_cm_s]).all() or end_s <= start_s or min_speed_cm_s < 0:
        raise ValueError("invalid held-out interval or speed threshold")
    if max_samples < 1 or step_s < max_window_s or max_window_s <= 0:
        raise ValueError("invalid held-out sampling configuration")
    values = []
    half = max_window_s / 2
    for lo, hi in full.run_times:
        first, last = max(lo, start_s) + half, min(hi, end_s) - half
        if last < first:
            continue
        indices = np.arange(np.ceil(first / step_s), np.floor(last / step_s) + 1)
        values.extend(indices * step_s)
    centers = np.unique(values)
    if not len(centers):
        return centers, 0
    speed = _speed_cm_s(full.position[:, 0], full.position[:, 1:3])
    center_speed = np.interp(centers, full.position[:, 0], speed)
    centers = centers[center_speed >= min_speed_cm_s]
    eligible = len(centers)
    if eligible > max_samples:
        centers = np.sort(np.random.default_rng(seed).choice(centers, max_samples, replace=False))
    return centers, eligible


def count_windows(indexed_spikes, cell_ids, centers_s, width_s):
    if width_s <= 0 or not np.isfinite(width_s):
        raise ValueError("invalid window duration")
    counts = np.empty((len(centers_s), len(cell_ids)), dtype=np.int64)
    for index, cell in enumerate(cell_ids):
        times = indexed_spikes[int(cell)]
        counts[:, index] = np.searchsorted(times, centers_s + width_s / 2, side="left") - np.searchsorted(times, centers_s - width_s / 2, side="left")
    return counts


def window_truth(position, centers_s, width_s):
    """Exact integral of piecewise-linear tracking, plus the center position."""
    t = position[:, 0]
    xy = position[:, 1:3]
    if len(t) < 2 or not np.all(np.diff(t) > 0) or not np.isfinite(position).all():
        raise ValueError("tracking must be finite and monotonic")
    centers_s = np.asarray(centers_s)
    lo, hi = centers_s - width_s / 2, centers_s + width_s / 2
    if not np.isfinite(width_s) or not np.isfinite(centers_s).all() or width_s <= 0 or (lo < t[0]).any() or (hi > t[-1]).any():
        raise ValueError("test windows outside tracking")
    dt = np.diff(t)
    slopes = np.diff(xy, axis=0) / dt[:, None]
    prefix = np.vstack([np.zeros((1, 2)), np.cumsum((xy[:-1] + xy[1:]) * dt[:, None] / 2, axis=0)])
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

    def integrals(query):
        index = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 2)
        delta = query - t[index]
        integral = prefix[index] + xy[index] * delta[:, None] + .5 * slopes[index] * delta[:, None] ** 2
        distance = lengths[index] + np.linalg.norm(slopes[index], axis=1) * delta
        return integral, distance

    lower, lower_distance = integrals(lo)
    upper, upper_distance = integrals(hi)
    return {
        "center_xy": np.column_stack([np.interp(centers_s, t, xy[:, j]) for j in range(2)]),
        "window_mean_xy": (upper - lower) / width_s,
        "mean_behavior_speed_cm_s": (upper_distance - lower_distance) / width_s,
    }


def truth_state_indices(model, xy):
    flat = _positions_to_flat_bins(xy, model["x_edges_cm"], model["y_edges_cm"])
    lookup = np.full(len(model["valid_spatial_bins"]), -1, dtype=int)
    lookup[model["valid_spatial_bins"]] = np.arange(model["valid_spatial_bins"].sum())
    indices = np.full(len(xy), -1, dtype=int)
    within = (flat >= 0) & (flat < len(lookup))
    indices[within] = lookup[flat[within]]
    return indices


def posterior_coverage(posterior, truth_indices, masses=(.5, .8, .95)):
    """Discrete HPD coverage; unsupported truth is a miss, ties are included."""
    posterior = np.asarray(posterior)
    raw_indices = np.asarray(truth_indices)
    if not np.isfinite(raw_indices).all() or np.any(raw_indices != np.floor(raw_indices)):
        raise ValueError("truth indices must be finite integers")
    indices = raw_indices.astype(int)
    if posterior.ndim != 2 or indices.shape != (len(posterior),) or np.any(indices < -1) or np.any(indices >= posterior.shape[1]):
        raise ValueError("posterior/truth shape mismatch")
    if not np.isfinite(posterior).all() or (posterior < 0).any() or not np.allclose(posterior.sum(axis=1), 1):
        raise ValueError("invalid posterior probability")
    if any(not 0 < mass <= 1 for mass in masses):
        raise ValueError("invalid credible mass")
    supported = indices >= 0
    threshold = np.full(len(indices), np.inf)
    threshold[supported] = posterior[np.flatnonzero(supported), indices[supported]]
    greater_mass = np.sum(posterior * (posterior > threshold[:, None]), axis=1)
    return {f"hpd{int(mass * 100)}": supported & (greater_mass < mass) for mass in masses}
