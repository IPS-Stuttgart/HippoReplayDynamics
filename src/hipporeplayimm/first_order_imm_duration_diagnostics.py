"""Preserve partial-bin duration metadata for first-order IMM diagnostics."""

from __future__ import annotations

from functools import wraps
from threading import RLock

import numpy as np
from scipy.special import logsumexp

from .duration_dynamics import DurationFloat, transition_durations_s

_DIAGNOSTIC_PATCH_LOCK = RLock()
_DISTANCE_PATCH_ATTR = "_first_order_imm_distance_overflow_safe"
_DISTANCE_ORIGINAL_ATTR = "_first_order_imm_distance_overflow_original"
_LOG_OFFSET_VALIDATION_PATCH_ATTR = "_first_order_imm_log_offset_validation_safe"
_LOG_OFFSET_VALIDATION_ORIGINAL_ATTR = (
    "_first_order_imm_log_offset_validation_original"
)
_PATH_RANGE_ERROR = "first-order IMM path geometry exceeds floating-point range"
_DURATION_RANGE_ERROR = "first-order IMM total duration exceeds floating-point range"


def _install_log_offset_safe_validation() -> None:
    """Validate finite log-posterior mass without raw exponential underflow."""

    from . import first_order_imm_diagnostics_validation as validation

    current = validation._validate_first_order_imm_content_inputs
    if getattr(current, _LOG_OFFSET_VALIDATION_PATCH_ATTR, False):
        return

    @wraps(current)
    def validate_log_offset_safe_inputs(
        mode_posterior: np.ndarray,
        trajectory_log_posterior: np.ndarray,
        bin_centers: np.ndarray,
        dt_s: float,
    ) -> None:
        trajectory = np.asarray(trajectory_log_posterior, dtype=float)
        if trajectory.ndim == 2:
            with np.errstate(invalid="ignore"):
                row_log_mass = logsumexp(trajectory, axis=1)
            finite_rows = np.isfinite(row_log_mass)
            if np.any(finite_rows):
                trajectory = trajectory.copy()
                trajectory[finite_rows] -= row_log_mass[finite_rows, None]
                trajectory_log_posterior = trajectory
        current(
            mode_posterior,
            trajectory_log_posterior,
            bin_centers,
            dt_s,
        )

    setattr(
        validate_log_offset_safe_inputs,
        _LOG_OFFSET_VALIDATION_PATCH_ATTR,
        True,
    )
    setattr(
        validate_log_offset_safe_inputs,
        _LOG_OFFSET_VALIDATION_ORIGINAL_ATTR,
        current,
    )
    validation._validate_first_order_imm_content_inputs = (
        validate_log_offset_safe_inputs
    )


def _install_overflow_safe_content_diagnostics() -> None:
    """Keep representable path diagnostics finite for large coordinates."""

    from . import first_order_imm_diagnostics_validation as validation

    current = validation._compute_first_order_imm_content_diagnostics
    if getattr(current, _DISTANCE_PATCH_ATTR, False):
        return

    @wraps(current)
    def overflow_safe_content_diagnostics(
        mode_posterior: np.ndarray,
        trajectory_log_posterior: np.ndarray,
        bin_centers: np.ndarray,
        dt_s: float,
    ) -> dict[str, float | int]:
        # The historical implementation uses np.linalg.norm, whose intermediate
        # squaring can overflow even when the Euclidean distance is representable.
        with np.errstate(over="ignore", invalid="ignore"):
            diagnostics = current(
                mode_posterior,
                trajectory_log_posterior,
                bin_centers,
                dt_s,
            )

        trajectory = np.asarray(trajectory_log_posterior, dtype=float)
        centers = np.asarray(bin_centers, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            row_log_mass = logsumexp(trajectory, axis=1)
            posterior = np.exp(trajectory - row_log_mass[:, None])
            expected_position = posterior @ centers
        if not np.all(np.isfinite(expected_position)):
            raise ValueError(_PATH_RANGE_ERROR)

        if expected_position.shape[0] > 1:
            with np.errstate(over="ignore", invalid="ignore"):
                step_deltas = np.diff(expected_position, axis=0)
            if not np.all(np.isfinite(step_deltas)):
                raise ValueError(_PATH_RANGE_ERROR)
            with np.errstate(over="ignore", invalid="ignore"):
                steps = np.hypot.reduce(step_deltas, axis=1)
                path_length = float(np.sum(steps))
                final_delta = expected_position[-1] - expected_position[0]
            if not np.all(np.isfinite(steps)) or not np.all(np.isfinite(final_delta)):
                raise ValueError(_PATH_RANGE_ERROR)
            with np.errstate(over="ignore", invalid="ignore"):
                net = float(np.hypot.reduce(final_delta))
            transition_durations = validation._diagnostic_transition_durations(
                dt_s,
                expected_position.shape[0],
                float(dt_s),
            )
            with np.errstate(over="ignore", invalid="ignore"):
                duration = float(np.sum(transition_durations))
            if not np.isfinite(duration):
                raise ValueError(_DURATION_RANGE_ERROR)
            duration = max(duration, np.finfo(float).tiny)
        else:
            path_length = 0.0
            net = 0.0
            duration = float(dt_s)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            speed = path_length / duration
        if (
            not np.isfinite(path_length)
            or not np.isfinite(net)
            or not np.isfinite(speed)
        ):
            raise ValueError(_PATH_RANGE_ERROR)

        diagnostics[
            "state_space_imm_posterior_expected_path_length_cm"
        ] = path_length
        diagnostics["state_space_imm_posterior_net_displacement_cm"] = net
        diagnostics["state_space_imm_posterior_path_speed_cm_s"] = speed
        return diagnostics

    setattr(overflow_safe_content_diagnostics, _DISTANCE_PATCH_ATTR, True)
    setattr(overflow_safe_content_diagnostics, _DISTANCE_ORIGINAL_ATTR, current)
    validation._compute_first_order_imm_content_diagnostics = (
        overflow_safe_content_diagnostics
    )


def _validated_bin_durations(emissions, n_time: int) -> np.ndarray | None:
    """Return explicit emission-bin durations aligned with diagnostic rows."""

    values = getattr(emissions, "bin_durations", None)
    if values is None:
        return None
    try:
        durations = np.asarray(values, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "bin_durations must contain finite positive durations"
        ) from exc
    if durations.shape != (int(n_time),):
        raise ValueError("bin_durations must contain one duration per emission row")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("bin_durations must contain finite positive durations")
    return durations


def _longest_active_bin_duration(
    mode_posterior: np.ndarray,
    bin_durations: np.ndarray,
) -> float:
    """Measure the longest nonstationary bout from exact emission-bin widths."""

    mode = np.asarray(mode_posterior, dtype=float)
    active = np.argmax(mode, axis=1) != 0
    best = 0.0
    current = 0.0
    for is_active, duration in zip(active, bin_durations, strict=True):
        if not is_active:
            current = 0.0
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            current += float(duration)
        if not np.isfinite(current):
            raise ValueError(_DURATION_RANGE_ERROR)
        best = max(best, current)
    return best


def apply_first_order_imm_duration_diagnostics_patch() -> None:
    """Install a compatibility wrapper around the duration-aware scorer.

    The native scorer computes variable transition durations before it calls the
    first-order IMM content diagnostics, but older call sites pass only scalar
    ``dt`` into the diagnostic helper. Wrap that helper during first-order IMM
    scoring so duration-dependent content diagnostics still see the explicit
    per-transition durations. Bout occupancy is measured from the emission-bin
    widths rather than approximated from center-to-center transition durations.
    """

    _install_log_offset_safe_validation()
    _install_overflow_safe_content_diagnostics()

    import hipporeplayimm.duration_occupancy as duration_occupancy

    current = duration_occupancy._score_state_space_duration_with_occupancy
    if getattr(current, "_first_order_imm_duration_diagnostics_aware", False):
        return

    original_score = current

    @wraps(original_score)
    def score_with_duration_diagnostics(
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

        with _DIAGNOSTIC_PATCH_LOCK:
            original_diagnostics = (
                duration_occupancy._first_order_imm_content_diagnostics
            )

            def diagnostics_with_transition_durations(
                mode_post,
                trajectory,
                centers,
                dt_s,
            ):
                diagnostic_dt = dt_s
                if getattr(diagnostic_dt, "transition_durations", None) is None:
                    durations = transition_durations_s(emissions)
                    if durations.size:
                        diagnostic_dt = DurationFloat(float(diagnostic_dt), durations)
                diagnostics = original_diagnostics(
                    mode_post,
                    trajectory,
                    centers,
                    diagnostic_dt,
                )
                bin_durations = _validated_bin_durations(
                    emissions,
                    np.asarray(mode_post).shape[0],
                )
                if bin_durations is not None:
                    diagnostics = dict(diagnostics)
                    diagnostics[
                        "state_space_imm_longest_nonstationary_bout_s"
                    ] = _longest_active_bin_duration(
                        mode_post,
                        bin_durations,
                    )
                return diagnostics

            duration_occupancy._first_order_imm_content_diagnostics = (
                diagnostics_with_transition_durations
            )
            try:
                return original_score(
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

    score_with_duration_diagnostics._first_order_imm_duration_diagnostics_aware = (
        True
    )
    score_with_duration_diagnostics.__duration_occupancy_previous_score__ = (
        original_score
    )
    duration_occupancy._score_state_space_duration_with_occupancy = (
        score_with_duration_diagnostics
    )
