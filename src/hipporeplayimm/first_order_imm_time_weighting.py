"""Duration-weight first-order IMM summaries for partial replay bins."""

from __future__ import annotations

from functools import wraps
from threading import RLock

import numpy as np


_PATCH_ATTR = "_first_order_imm_time_weighting_aware"
_ORIGINAL_ATTR = "_first_order_imm_time_weighting_original"
_PATCH_LOCK = RLock()
_MODE_NAMES = ("stationary", "diffusion", "fragmented")


def _validated_bin_durations(
    emissions: object,
    n_time: int,
    fallback_dt_s: float,
) -> np.ndarray:
    """Return one finite positive exposure duration per posterior row."""

    values = getattr(emissions, "bin_durations", None)
    if values is None:
        values = np.full(int(n_time), float(fallback_dt_s), dtype=float)
    durations = np.asarray(values, dtype=float)
    if durations.shape != (int(n_time),):
        raise ValueError("bin_durations must contain one duration per IMM posterior row")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("bin_durations must contain finite positive durations")
    return durations


def _longest_true_duration(active: np.ndarray, durations: np.ndarray) -> float:
    """Return the longest contiguous active-bin exposure duration."""

    best = 0.0
    current = 0.0
    for is_active, duration in zip(active, durations, strict=True):
        if bool(is_active):
            current += float(duration)
            best = max(best, current)
        else:
            current = 0.0
    return best


def _duration_weighted_mode_summary(
    mode_posterior: np.ndarray,
    bin_durations: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Summarize mode occupancy using physical bin exposure rather than row count."""

    mode = np.asarray(mode_posterior, dtype=float)
    durations = np.asarray(bin_durations, dtype=float)
    if mode.ndim != 2 or mode.shape[1] != len(_MODE_NAMES):
        raise ValueError("first-order IMM mode posterior must have shape (time, 3)")
    if durations.shape != (mode.shape[0],):
        raise ValueError("bin_durations must contain one duration per IMM posterior row")
    if not np.all(np.isfinite(mode)) or np.any(mode < 0.0):
        raise ValueError("first-order IMM mode posterior must contain finite nonnegative values")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("bin_durations must contain finite positive durations")

    row_mass = mode.sum(axis=1)
    if np.any(row_mass <= 0.0) or not np.all(np.isfinite(row_mass)):
        raise ValueError("first-order IMM mode posterior rows must contain positive finite mass")
    normalized = mode / row_mass[:, None]

    duration_scale = float(np.max(durations))
    scaled_durations = durations / duration_scale
    scaled_total_duration = float(np.sum(scaled_durations))
    if not np.isfinite(scaled_total_duration) or scaled_total_duration <= 0.0:
        raise ValueError("total bin duration must be finite and positive")
    weights = scaled_durations / scaled_total_duration
    event_probability = weights @ normalized

    map_mode = np.argmax(normalized, axis=1)
    nonstationary = map_mode != 0
    stationary_fraction = float(np.sum(weights[~nonstationary]))
    nonstationary_fraction = float(np.sum(weights[nonstationary]))

    with np.errstate(divide="ignore", invalid="ignore"):
        log_mode = np.where(normalized > 0.0, np.log(normalized), 0.0)
    entropy_by_bin = -np.sum(
        np.where(normalized > 0.0, normalized * log_mode, 0.0),
        axis=1,
    )
    mean_entropy = float(weights @ entropy_by_bin)

    return {
        "event_probability": np.asarray(event_probability, dtype=float),
        "fraction_time_map_stationary": stationary_fraction,
        "fraction_time_map_nonstationary": nonstationary_fraction,
        "longest_nonstationary_bout_s": _longest_true_duration(
            nonstationary,
            durations,
        ),
        "mean_mode_entropy": mean_entropy,
    }


def _apply_duration_weighted_diagnostics(
    result: object,
    mode_posterior: np.ndarray,
    bin_durations: np.ndarray,
) -> None:
    """Replace row-count summaries in an EventScore with exposure-weighted values."""

    summary = _duration_weighted_mode_summary(mode_posterior, bin_durations)
    diagnostics = getattr(result, "diagnostics")
    event_probability = np.asarray(summary["event_probability"], dtype=float)
    for index, name in enumerate(_MODE_NAMES):
        diagnostics[f"state_space_mode_{name}_event_probability"] = float(
            event_probability[index]
        )
    diagnostics["state_space_imm_nonstationary_event_probability"] = float(
        event_probability[1:].sum()
    )
    diagnostics["state_space_imm_fraction_time_map_stationary"] = float(
        summary["fraction_time_map_stationary"]
    )
    diagnostics["state_space_imm_fraction_time_map_nonstationary"] = float(
        summary["fraction_time_map_nonstationary"]
    )
    diagnostics["state_space_imm_longest_nonstationary_bout_s"] = float(
        summary["longest_nonstationary_bout_s"]
    )
    diagnostics["state_space_imm_mean_mode_entropy"] = float(
        summary["mean_mode_entropy"]
    )
    diagnostics["state_space_imm_time_weighting"] = "bin_duration"


def apply_first_order_imm_time_weighting_patch() -> None:
    """Install duration-weighted first-order IMM event diagnostics."""

    import hipporeplayimm.duration_occupancy as duration_occupancy

    current = duration_occupancy._score_state_space_duration_with_occupancy
    if getattr(current, _PATCH_ATTR, False):
        return

    original_score = current

    @wraps(original_score)
    def score_with_time_weighted_diagnostics(
        self,
        emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory: bool = True,
    ):
        if getattr(self, "mode", None) != "first-order-imm":
            return original_score(
                self,
                emissions,
                bin_centers,
                candidate_indices=candidate_indices,
                occupancy_s=occupancy_s,
                return_trajectory=return_trajectory,
            )

        captured_mode_posterior: np.ndarray | None = None
        with _PATCH_LOCK:
            original_diagnostics = (
                duration_occupancy._first_order_imm_content_diagnostics
            )

            def capture_mode_posterior(
                mode_posterior,
                trajectory_log_posterior,
                centers,
                dt_s,
            ):
                nonlocal captured_mode_posterior
                captured_mode_posterior = np.asarray(
                    mode_posterior,
                    dtype=float,
                ).copy()
                return original_diagnostics(
                    mode_posterior,
                    trajectory_log_posterior,
                    centers,
                    dt_s,
                )

            duration_occupancy._first_order_imm_content_diagnostics = (
                capture_mode_posterior
            )
            try:
                result = original_score(
                    self,
                    emissions,
                    bin_centers,
                    candidate_indices=candidate_indices,
                    occupancy_s=occupancy_s,
                    return_trajectory=return_trajectory,
                )
            finally:
                duration_occupancy._first_order_imm_content_diagnostics = (
                    original_diagnostics
                )

        if captured_mode_posterior is None:
            return result

        durations = _validated_bin_durations(
            emissions,
            captured_mode_posterior.shape[0],
            float(emissions.dt),
        )
        _apply_duration_weighted_diagnostics(
            result,
            captured_mode_posterior,
            durations,
        )
        return result

    setattr(score_with_time_weighted_diagnostics, _PATCH_ATTR, True)
    setattr(score_with_time_weighted_diagnostics, _ORIGINAL_ATTR, original_score)
    duration_occupancy._score_state_space_duration_with_occupancy = (
        score_with_time_weighted_diagnostics
    )
