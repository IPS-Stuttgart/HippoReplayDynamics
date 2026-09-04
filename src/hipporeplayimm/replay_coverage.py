"""Observation-model and path metrics for replay coverage calibration.

There is no transition prior or temporal smoothing in this decoder. Conditioning
on a population spike total is explicit, not approximated by rescaling all rate
maps by a single constant.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, xlogy


def decode_independent(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    grid_cm: np.ndarray,
    duration_s: float,
    *,
    likelihood: str = "poisson",
) -> dict[str, np.ndarray]:
    """Decode bins independently under a uniform prior over supplied states.

    ``conditional_multinomial`` conditions on N=sum(counts) in EACH window:
    log L(x) = sum_i n_i log(lambda_i(x)/sum_j lambda_j(x)).
    ``poisson`` additionally uses the information in the total count:
    log L(x) = sum_i n_i log(lambda_i(x)) - dt * sum_i lambda_i(x).
    Terms constant over x are omitted in both cases. Rate maps must be strictly
    positive; callers must explicitly declare any floor used in their model.
    """
    counts = np.asarray(counts, dtype=float)
    rates = np.asarray(rates_hz, dtype=float)
    grid = np.asarray(grid_cm, dtype=float)
    if counts.ndim != 2 or rates.ndim != 2 or grid.ndim != 2:
        raise ValueError("counts, rates, and grid must be matrices")
    if counts.shape[1] != rates.shape[0] or rates.shape[1] != grid.shape[0]:
        raise ValueError("counts (time,cells), rates (cells,states), grid (states,dim) mismatch")
    if min(rates.shape) < 1 or grid.shape[1] < 1:
        raise ValueError("at least one cell, state, and spatial dimension are required")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("counts must be finite nonnegative integers")
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("rates must be finite and strictly positive")
    if not np.isfinite(grid).all() or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("grid must be finite and duration positive")
    if likelihood == "conditional_multinomial":
        log_rates = np.log(rates) - np.log(rates.sum(axis=0))[None, :]
        log_likelihood = counts @ log_rates
    elif likelihood == "poisson":
        expected = rates * duration_s
        log_likelihood = counts @ np.log(expected) - expected.sum(axis=0)[None, :]
    else:
        raise ValueError(f"unknown likelihood: {likelihood}")
    if not np.isfinite(log_likelihood).all():
        raise ValueError("nonfinite likelihood; check count/rate scales")
    posterior = np.exp(log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True))
    mean = posterior @ grid
    second_moment = posterior @ np.sum(grid**2, axis=1)
    rms = np.sqrt(np.maximum(second_moment - np.sum(mean**2, axis=1), 0.0))
    return {
        "posterior": posterior,
        "map": grid[np.argmax(posterior, axis=1)],
        "posterior_mean": mean,
        "posterior_rms_cm": rms,
        "posterior_entropy_nats": -np.sum(xlogy(posterior, posterior), axis=1),
    }


def continuity_metrics(
    positions_cm: np.ndarray,
    *,
    valid_bins: np.ndarray | None = None,
    max_jump_cm: float = 20.0,
    min_frames: int = 10,
    min_displacement_cm: float = 40.0,
) -> dict[str, object]:
    """Longest valid run with strict adjacent jump < threshold; never bridge gaps.

    This is only a geometric screening heuristic, NOT a shuffle-validated replay
    test. In a tie the earliest longest run wins, regardless of displacement.
    """
    positions = np.asarray(positions_cm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 1:
        raise ValueError("positions must have shape (frames, spatial_dimensions)")
    if not np.isfinite(max_jump_cm) or max_jump_cm <= 0 or min_frames < 1:
        raise ValueError("jump threshold and frame count must be positive")
    if not np.isfinite(min_displacement_cm) or min_displacement_cm < 0:
        raise ValueError("displacement threshold must be finite and nonnegative")
    valid = np.isfinite(positions).all(axis=1)
    if valid_bins is not None:
        support = np.asarray(valid_bins)
        if support.shape != valid.shape or support.dtype != bool:
            raise ValueError("valid_bins must be a Boolean vector matching frames")
        valid &= support
    best_start, best_end = 0, 0
    start: int | None = None
    for index in range(len(positions)):
        if not valid[index]:
            start = None
            continue
        if start is None or np.linalg.norm(positions[index] - positions[index - 1]) >= max_jump_cm:
            start = index
        if index + 1 - start > best_end - best_start:
            best_start, best_end = start, index + 1
    run = positions[best_start:best_end]
    displacement = float(np.linalg.norm(run[-1] - run[0])) if len(run) > 1 else 0.0
    passed = len(run) >= min_frames and displacement >= min_displacement_cm
    return {
        "continuous_start": best_start,
        "continuous_end_exclusive": best_end,
        "continuous_frames": len(run),
        "continuous_displacement_cm": displacement,
        "continuity_pass": bool(passed),
    }


def path_metrics(
    positions_cm: np.ndarray,
    true_positions_cm: np.ndarray,
    *,
    step_s: float,
    valid_bins: np.ndarray | None = None,
) -> dict[str, object]:
    positions = np.asarray(positions_cm, dtype=float)
    truth = np.asarray(true_positions_cm, dtype=float)
    if positions.shape != truth.shape or not np.isfinite(truth).all():
        raise ValueError("finite truth must have the same shape as decoded positions")
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("step_s must be positive")
    continuity = continuity_metrics(positions, valid_bins=valid_bins)
    valid = np.isfinite(positions).all(axis=1)
    if valid_bins is not None:
        valid &= valid_bins
    adjacent = valid[:-1] & valid[1:]
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    true_jumps = np.linalg.norm(np.diff(truth, axis=0), axis=1)
    errors = np.linalg.norm(positions - truth, axis=1)
    start, end = continuity["continuous_start"], continuity["continuous_end_exclusive"]
    selected_jumps = jumps[start:max(start, end - 1)]
    return {
        **continuity,
        "valid_bins": int(valid.sum()),
        "valid_adjacent_steps": int(adjacent.sum()),
        "large_jump_fraction": float(np.mean(jumps[adjacent] >= 20.0)) if adjacent.any() else np.nan,
        "median_position_error_cm": float(np.median(errors[valid])) if valid.any() else np.nan,
        "median_speed_cm_s": float(np.median(jumps[adjacent] / step_s)) if adjacent.any() else np.nan,
        "mean_speed_cm_s": float(np.mean(jumps[adjacent] / step_s)) if adjacent.any() else np.nan,
        "median_true_speed_cm_s": float(np.median(true_jumps[adjacent] / step_s)) if adjacent.any() else np.nan,
        "selected_median_speed_cm_s": float(np.median(selected_jumps / step_s)) if continuity["continuity_pass"] and selected_jumps.size else np.nan,
    }
