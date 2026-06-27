"""Trajectory-posterior quality metrics for replay decoding outputs."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def trajectory_quality_metrics(
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    times: np.ndarray | None = None,
    *,
    prefix: str = "trajectory",
) -> dict[str, float | int]:
    """Summarize posterior trajectory geometry and certainty.

    Metrics are based on both posterior-mean and MAP paths.  They are not a
    substitute for evidence; they flag whether a high-evidence event corresponds
    to an interpretable path.
    """

    logp = np.asarray(trajectory_log_posterior, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    if logp.ndim != 2:
        raise ValueError("trajectory_log_posterior must have shape (time, bins)")
    if logp.shape[0] == 0 or logp.shape[1] == 0:
        raise ValueError("trajectory_log_posterior must contain at least one time bin and one position bin")
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[0] != logp.shape[1]:
        raise ValueError("bin_centers must have shape (bins,) or (bins, position_dim)")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must contain finite values")
    if np.any(np.isnan(logp)) or np.any(np.isposinf(logp)):
        raise ValueError("trajectory_log_posterior cannot contain NaN or +inf")
    row_log_norm = logsumexp(logp, axis=1)
    if not np.all(np.isfinite(row_log_norm)):
        raise ValueError("trajectory_log_posterior rows must contain finite posterior mass")
    normalized = logp - row_log_norm[:, None]
    posterior = np.exp(normalized)
    mean_path = posterior @ centers
    map_bins = np.argmax(normalized, axis=1)
    map_path = centers[map_bins]
    durations = _transition_durations(times, logp.shape[0])
    mean_steps = np.linalg.norm(np.diff(mean_path, axis=0), axis=1) if logp.shape[0] > 1 else np.empty(0)
    map_steps = np.linalg.norm(np.diff(map_path, axis=0), axis=1) if logp.shape[0] > 1 else np.empty(0)
    displacement = _distance(mean_path[0], mean_path[-1])
    path_length = float(np.sum(mean_steps))
    total_time = float(np.sum(durations)) if durations.size else float(max(logp.shape[0] - 1, 1))
    entropy = _posterior_entropy(posterior, normalized)
    spread = _posterior_spread(posterior, centers, mean_path)
    return {
        f"{prefix}_time_bins": int(logp.shape[0]),
        f"{prefix}_posterior_mean_path_length_cm": path_length,
        f"{prefix}_posterior_mean_displacement_cm": float(displacement),
        f"{prefix}_posterior_mean_linearity": _safe_ratio(displacement, path_length),
        f"{prefix}_posterior_mean_speed_cm_s": _safe_ratio(path_length, total_time),
        f"{prefix}_direction_consistency": _direction_consistency(mean_path),
        f"{prefix}_map_path_length_cm": float(np.sum(map_steps)),
        f"{prefix}_map_step_median_cm": float(np.median(map_steps)) if map_steps.size else 0.0,
        f"{prefix}_map_step_p95_cm": float(np.quantile(map_steps, 0.95)) if map_steps.size else 0.0,
        f"{prefix}_mean_entropy": float(np.mean(entropy)) if entropy.size else np.nan,
        f"{prefix}_terminal_entropy": float(entropy[-1]) if entropy.size else np.nan,
        f"{prefix}_mean_spread_cm": float(np.mean(spread)) if spread.size else np.nan,
        f"{prefix}_terminal_spread_cm": float(spread[-1]) if spread.size else np.nan,
    }


def _transition_durations(times: np.ndarray | None, n_time: int) -> np.ndarray:
    if n_time <= 1:
        return np.empty(0, dtype=float)
    if times is None:
        return np.ones(n_time - 1, dtype=float)
    arr = np.asarray(times, dtype=float)
    if arr.shape != (n_time,):
        raise ValueError("times must contain one timestamp per trajectory row")
    if not np.all(np.isfinite(arr)):
        raise ValueError("times must contain finite values")
    diffs = np.diff(arr)
    if np.any(diffs <= 0.0):
        raise ValueError("times must be strictly increasing")
    return diffs


def _posterior_entropy(posterior: np.ndarray, log_posterior: np.ndarray) -> np.ndarray:
    """Return entropy while treating impossible bins as zero-contribution mass."""

    terms = np.zeros_like(posterior, dtype=float)
    np.multiply(
        posterior,
        log_posterior,
        out=terms,
        where=np.isfinite(log_posterior),
    )
    return -np.sum(terms, axis=1)


def _posterior_spread(posterior: np.ndarray, centers: np.ndarray, mean_path: np.ndarray) -> np.ndarray:
    delta = centers[None, :, :] - mean_path[:, None, :]
    dist2 = np.sum(delta * delta, axis=2)
    return np.sqrt(np.sum(posterior * dist2, axis=1))


def _direction_consistency(path: np.ndarray) -> float:
    steps = np.diff(path, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    keep = lengths > np.finfo(float).eps
    if not np.any(keep):
        return 0.0
    unit = steps[keep] / lengths[keep, None]
    return float(np.linalg.norm(np.sum(unit, axis=0)) / unit.shape[0])


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom > np.finfo(float).eps else 0.0
