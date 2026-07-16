"""Trajectory-posterior quality metrics for replay decoding outputs."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .models import LOG_ZERO

_LOG_ZERO_ROW_THRESHOLD = LOG_ZERO / 2.0
_BOOL_OR_TEXT_DTYPE_KINDS = {"b", "S", "U"}
_PATH_RANGE_ERROR = "trajectory path geometry exceeds floating-point range"


def trajectory_quality_metrics(
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    times: np.ndarray | None = None,
    *,
    prefix: str = "trajectory",
) -> dict[str, float | int]:
    """Summarize posterior trajectory geometry and certainty.

    Metrics are based on both posterior-mean and MAP paths.  They are not a
    substitute for evidence; they flag whether a scored event has an interpretable
    path.
    """

    logp = _as_numeric_real_array(trajectory_log_posterior, "trajectory_log_posterior")
    centers = _as_numeric_real_array(bin_centers, "bin_centers")
    if logp.ndim != 2:
        raise ValueError("trajectory_log_posterior must have shape (time, bins)")
    if logp.shape[0] == 0 or logp.shape[1] == 0:
        raise ValueError("trajectory_log_posterior must contain at least one time bin and one position bin")
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[0] != logp.shape[1] or centers.shape[1] == 0:
        raise ValueError("bin_centers must have shape (bins,) or (bins, position_dim) with position_dim >= 1")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must contain finite values")
    if np.any(np.isnan(logp)) or np.any(np.isposinf(logp)):
        raise ValueError("trajectory_log_posterior cannot contain NaN or +inf")
    row_log_norm = logsumexp(logp, axis=1)
    if not np.all(np.isfinite(row_log_norm)) or np.any(row_log_norm <= _LOG_ZERO_ROW_THRESHOLD):
        raise ValueError("trajectory_log_posterior rows must contain positive finite posterior mass")
    normalized = logp - row_log_norm[:, None]
    posterior = np.exp(normalized)
    with np.errstate(over="ignore", invalid="ignore"):
        mean_path = posterior @ centers
    if not np.all(np.isfinite(mean_path)):
        raise ValueError(_PATH_RANGE_ERROR)
    map_bins = np.argmax(normalized, axis=1)
    map_path = centers[map_bins]
    durations = _transition_durations(times, logp.shape[0])
    _, mean_steps = _path_steps(mean_path)
    _, map_steps = _path_steps(map_path)
    displacement = _distance(mean_path[0], mean_path[-1])
    with np.errstate(over="ignore", invalid="ignore"):
        path_length = float(np.sum(mean_steps))
        map_path_length = float(np.sum(map_steps))
        total_time = float(np.sum(durations)) if durations.size else float(max(logp.shape[0] - 1, 1))
    if not np.isfinite(path_length) or not np.isfinite(map_path_length):
        raise ValueError(_PATH_RANGE_ERROR)
    if not np.isfinite(total_time):
        raise ValueError("total trajectory duration exceeds floating-point range")
    entropy = _posterior_entropy(posterior, normalized)
    spread = _posterior_spread(posterior, centers, mean_path)
    return {
        f"{prefix}_time_bins": int(logp.shape[0]),
        f"{prefix}_posterior_mean_path_length_cm": path_length,
        f"{prefix}_posterior_mean_displacement_cm": float(displacement),
        f"{prefix}_posterior_mean_linearity": _safe_ratio(displacement, path_length),
        f"{prefix}_posterior_mean_speed_cm_s": _safe_ratio(path_length, total_time),
        f"{prefix}_direction_consistency": _direction_consistency(mean_path),
        f"{prefix}_map_path_length_cm": map_path_length,
        f"{prefix}_map_step_median_cm": float(np.median(map_steps)) if map_steps.size else 0.0,
        f"{prefix}_map_step_p95_cm": float(np.quantile(map_steps, 0.95)) if map_steps.size else 0.0,
        f"{prefix}_mean_entropy": float(np.mean(entropy)) if entropy.size else np.nan,
        f"{prefix}_terminal_entropy": float(entropy[-1]) if entropy.size else np.nan,
        f"{prefix}_mean_spread_cm": float(np.mean(spread)) if spread.size else np.nan,
        f"{prefix}_terminal_spread_cm": float(spread[-1]) if spread.size else np.nan,
    }


def _as_numeric_real_array(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind in _BOOL_OR_TEXT_DTYPE_KINDS:
        raise ValueError(f"{name} must contain numeric real values, not boolean or text")
    if raw.dtype.kind == "c":
        raise ValueError(f"{name} must contain numeric real values, not complex values")
    if raw.dtype.kind == "O":
        for item in raw.ravel():
            if isinstance(item, (bool, np.bool_, str, bytes, np.bytes_)):
                raise ValueError(f"{name} must contain numeric real values, not boolean or text")
            if isinstance(item, (complex, np.complexfloating)):
                raise ValueError(f"{name} must contain numeric real values, not complex values")
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric real values") from exc


def _transition_durations(times: np.ndarray | None, n_time: int) -> np.ndarray:
    if times is None:
        return np.ones(n_time - 1, dtype=float) if n_time > 1 else np.empty(0, dtype=float)
    arr = _as_numeric_real_array(times, "times")
    if arr.shape != (n_time,):
        raise ValueError("times must contain one timestamp per trajectory row")
    if not np.all(np.isfinite(arr)):
        raise ValueError("times must contain finite values")
    if n_time <= 1:
        return np.empty(0, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        diffs = np.diff(arr)
    if not np.all(np.isfinite(diffs)):
        raise ValueError("timestamp differences exceed floating-point range")
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
    """Return spread without zero-mass or intermediate-squaring overflow."""

    positive_mass = posterior > 0.0
    delta = np.zeros((posterior.shape[0], centers.shape[0], centers.shape[1]), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(
            centers[None, :, :],
            mean_path[:, None, :],
            out=delta,
            where=positive_mass[:, :, None],
        )
    if not np.all(np.isfinite(delta)):
        raise ValueError("posterior spread exceeds floating-point range")
    scale = np.max(np.abs(delta), axis=(1, 2))
    expanded_scale = scale[:, None, None]
    scaled_delta = np.divide(
        delta,
        expanded_scale,
        out=np.zeros_like(delta, dtype=float),
        where=expanded_scale > 0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_spread2 = np.sum(
            posterior[:, :, None] * scaled_delta * scaled_delta,
            axis=(1, 2),
        )
        spread = scale * np.sqrt(scaled_spread2)
    if not np.all(np.isfinite(spread)):
        raise ValueError("posterior spread exceeds floating-point range")
    return spread


def _path_steps(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.diff(path, axis=0)
    if not np.all(np.isfinite(steps)):
        raise ValueError(_PATH_RANGE_ERROR)
    lengths = _stable_euclidean_norm(steps, axis=1)
    if not np.all(np.isfinite(lengths)):
        raise ValueError(_PATH_RANGE_ERROR)
    return steps, lengths


def _stable_euclidean_norm(values: np.ndarray, *, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scale = np.max(np.abs(arr), axis=axis)
    expanded_scale = np.expand_dims(scale, axis=axis)
    scaled = np.divide(
        arr,
        expanded_scale,
        out=np.zeros_like(arr, dtype=float),
        where=expanded_scale > 0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * np.sqrt(np.sum(scaled * scaled, axis=axis))


def _direction_consistency(path: np.ndarray) -> float:
    steps, lengths = _path_steps(path)
    keep = lengths > np.finfo(float).eps
    if not np.any(keep):
        return 0.0
    unit = steps[keep] / lengths[keep, None]
    return float(np.linalg.norm(np.sum(unit, axis=0)) / unit.shape[0])


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if not np.all(np.isfinite(delta)):
        raise ValueError(_PATH_RANGE_ERROR)
    distance = float(_stable_euclidean_norm(delta, axis=0))
    if not np.isfinite(distance):
        raise ValueError(_PATH_RANGE_ERROR)
    return distance


def _safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom > np.finfo(float).eps else 0.0
