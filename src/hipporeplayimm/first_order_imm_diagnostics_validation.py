"""Runtime validation for first-order IMM content diagnostics."""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np

_PATCH_ATTR = "_first_order_imm_diagnostics_validation_patch"
_ORIGINAL_ATTR = "_first_order_imm_diagnostics_validation_original"
_DURATION_ALIAS_ATTR = "_first_order_imm_duration_diagnostics_alias_patch"
_DURATION_SOURCE_ATTR = "_first_order_imm_duration_diagnostics_source_patch"
_DURATION_SCORE_ATTR = "_first_order_imm_duration_diagnostics_score_patch"
_RECORDING_ENABLED_ATTR = "_first_order_imm_duration_diagnostics_recording_enabled"
_LAST_DURATIONS_ATTR = "_first_order_imm_diagnostic_transition_durations"


def _validate_first_order_imm_content_inputs(
    mode_posterior: np.ndarray,
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    dt_s: float,
) -> None:
    """Reject non-finite diagnostic inputs before MAP/path summaries are computed."""

    mode = np.asarray(mode_posterior, dtype=float)
    trajectory = np.asarray(trajectory_log_posterior, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)

    if mode.ndim != 2 or mode.shape[1] != 3:
        raise ValueError("first-order IMM mode posterior must have shape (time, 3)")
    if trajectory.ndim != 2 or trajectory.shape[0] != mode.shape[0]:
        raise ValueError("trajectory posterior must have one row per mode-posterior time bin")
    if centers.ndim != 2 or centers.shape[0] != trajectory.shape[1] or centers.shape[1] < 1:
        raise ValueError("bin_centers must contain one coordinate row per spatial bin")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must be finite")

    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    if not np.all(np.isfinite(mode)):
        raise ValueError("first-order IMM mode posterior must be finite")
    if np.any(mode < 0.0):
        raise ValueError("first-order IMM mode posterior must be nonnegative")
    mode_mass = mode.sum(axis=1)
    if not np.all(np.isfinite(mode_mass)) or np.any(mode_mass <= 0.0):
        raise ValueError("first-order IMM mode posterior rows must contain positive finite mass")

    if np.any(np.isnan(trajectory)) or np.any(trajectory == np.inf):
        raise ValueError("trajectory posterior must not contain NaN or +inf")
    trajectory_mass = np.exp(trajectory).sum(axis=1)
    if not np.all(np.isfinite(trajectory_mass)) or np.any(trajectory_mass <= 0.0):
        raise ValueError("trajectory posterior rows must contain positive finite mass")


def _diagnostic_transition_durations(
    dt_s: float,
    n_time: int,
    fallback_dt_s: float,
) -> np.ndarray:
    """Return center-to-center diagnostic durations for adjacent posterior rows."""

    n_transitions = max(int(n_time) - 1, 0)
    transition_durations = getattr(dt_s, "transition_durations", None)
    if transition_durations is None:
        return np.full(n_transitions, float(fallback_dt_s), dtype=float)

    durations = np.asarray(transition_durations, dtype=float)
    if durations.shape != (n_transitions,):
        raise ValueError("transition_durations must contain one value per adjacent time-bin pair")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("transition_durations must contain finite positive durations")
    return durations


def _longest_active_run_duration(
    active: np.ndarray,
    transition_durations: np.ndarray,
    fallback_dt_s: float,
) -> float:
    """Measure the longest contiguous active run using explicit transition durations."""

    active = np.asarray(active, dtype=bool)
    if active.size == 0:
        return 0.0
    bin_duration = float(np.median(transition_durations)) if transition_durations.size else float(fallback_dt_s)
    best = 0.0
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is None:
            continue
        if value and index != active.size - 1:
            continue
        stop = index if value else index - 1
        duration = bin_duration
        if stop > start:
            duration += float(np.sum(transition_durations[start:stop]))
        best = max(best, duration)
        start = None
    return best


def _compute_first_order_imm_content_diagnostics(
    mode_posterior: np.ndarray,
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    dt_s: float,
) -> dict[str, float | int]:
    """Compute content diagnostics while respecting explicit transition durations."""

    mode = np.asarray(mode_posterior, dtype=float)
    trajectory = np.asarray(trajectory_log_posterior, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    dt = float(dt_s)
    transition_durations = _diagnostic_transition_durations(dt_s, mode.shape[0], dt)

    map_mode = np.argmax(mode, axis=1)
    nonstationary = map_mode != 0
    starts = nonstationary & np.concatenate(([True], ~nonstationary[:-1]))
    bout_count = int(starts.sum())
    longest_duration = _longest_active_run_duration(nonstationary, transition_durations, dt)

    posterior = np.exp(trajectory)
    row_mass = posterior.sum(axis=1)
    valid = row_mass > 0.0
    posterior[valid] = posterior[valid] / row_mass[valid, None]
    expected_position = posterior @ centers
    if len(expected_position) > 1:
        steps = np.linalg.norm(np.diff(expected_position, axis=0), axis=1)
        path_length = float(np.nansum(steps))
        net = float(np.linalg.norm(expected_position[-1] - expected_position[0]))
        duration = max(float(np.sum(transition_durations)), np.finfo(float).tiny)
    else:
        path_length = 0.0
        net = 0.0
        duration = dt

    return {
        "state_space_imm_fraction_time_map_stationary": float(np.mean(~nonstationary)),
        "state_space_imm_fraction_time_map_nonstationary": float(np.mean(nonstationary)),
        "state_space_imm_nonstationary_bout_count": bout_count,
        "state_space_imm_longest_nonstationary_bout_s": longest_duration,
        "state_space_imm_posterior_expected_path_length_cm": path_length,
        "state_space_imm_posterior_net_displacement_cm": net,
        "state_space_imm_posterior_path_speed_cm_s": path_length / duration,
    }


def _matching_transition_durations(values: object, n_time: int) -> np.ndarray | None:
    """Return stored durations only when they match this diagnostic call."""

    if values is None:
        return None
    try:
        durations = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None

    expected_shape = (max(int(n_time) - 1, 0),)
    if durations.shape != expected_shape:
        return None
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        return None
    return durations


def _wrap_helper(
    helper: Callable[..., dict[str, float | int]],
) -> Callable[..., dict[str, float | int]]:
    if getattr(helper, _PATCH_ATTR, False):
        return helper

    def validated_first_order_imm_content_diagnostics(
        mode_posterior: np.ndarray,
        trajectory_log_posterior: np.ndarray,
        bin_centers: np.ndarray,
        dt_s: float,
    ) -> dict[str, float | int]:
        _validate_first_order_imm_content_inputs(
            mode_posterior,
            trajectory_log_posterior,
            bin_centers,
            dt_s,
        )
        return _compute_first_order_imm_content_diagnostics(
            mode_posterior,
            trajectory_log_posterior,
            bin_centers,
            dt_s,
        )

    setattr(validated_first_order_imm_content_diagnostics, _PATCH_ATTR, True)
    setattr(validated_first_order_imm_content_diagnostics, _ORIGINAL_ATTR, helper)
    return validated_first_order_imm_content_diagnostics


def _duration_occupancy_module():
    return sys.modules.get("hipporeplayimm.duration_occupancy")


def _clear_duration_occupancy_transition_durations() -> None:
    module = _duration_occupancy_module()
    if module is not None:
        setattr(module, _LAST_DURATIONS_ATTR, None)


def _stored_duration_occupancy_durations(n_time: int) -> np.ndarray | None:
    module = _duration_occupancy_module()
    if module is None:
        return None
    values = getattr(module, _LAST_DURATIONS_ATTR, None)
    _clear_duration_occupancy_transition_durations()
    return _matching_transition_durations(values, n_time)


def _record_duration_occupancy_transition_durations() -> None:
    module = _duration_occupancy_module()
    if module is None or not hasattr(module, "transition_durations_s"):
        return
    current = module.transition_durations_s
    if getattr(current, _DURATION_SOURCE_ATTR, False):
        return

    def recording_transition_durations(emissions):
        durations = current(emissions)
        if getattr(module, _RECORDING_ENABLED_ATTR, False):
            setattr(module, _LAST_DURATIONS_ATTR, np.asarray(durations, dtype=float).copy())
        else:
            setattr(module, _LAST_DURATIONS_ATTR, None)
        return durations

    setattr(recording_transition_durations, _DURATION_SOURCE_ATTR, True)
    setattr(recording_transition_durations, _ORIGINAL_ATTR, current)
    module.transition_durations_s = recording_transition_durations


def _record_duration_occupancy_score_context() -> None:
    module = _duration_occupancy_module()
    if module is None or not hasattr(module, "_score_state_space_duration_with_occupancy"):
        return
    current = module._score_state_space_duration_with_occupancy
    if getattr(current, _DURATION_SCORE_ATTR, False):
        return

    def score_with_first_order_duration_recording(*args, **kwargs):
        model = args[0] if args else kwargs.get("self")
        previous = bool(getattr(module, _RECORDING_ENABLED_ATTR, False))
        setattr(module, _RECORDING_ENABLED_ATTR, getattr(model, "mode", None) == "first-order-imm")
        try:
            return current(*args, **kwargs)
        finally:
            setattr(module, _RECORDING_ENABLED_ATTR, previous)
            if not previous:
                _clear_duration_occupancy_transition_durations()

    setattr(score_with_first_order_duration_recording, _DURATION_SCORE_ATTR, True)
    setattr(score_with_first_order_duration_recording, _ORIGINAL_ATTR, current)
    module._score_state_space_duration_with_occupancy = score_with_first_order_duration_recording


def _wrap_duration_occupancy_alias(
    helper: Callable[..., dict[str, float | int]],
) -> Callable[..., dict[str, float | int]]:
    """Preserve transition durations for the duration-aware scorer's helper alias."""

    if getattr(helper, _DURATION_ALIAS_ATTR, False):
        return helper

    def duration_aware_first_order_imm_content_diagnostics(
        mode_posterior: np.ndarray,
        trajectory_log_posterior: np.ndarray,
        bin_centers: np.ndarray,
        dt_s: float,
    ) -> dict[str, float | int]:
        try:
            durations = getattr(dt_s, "transition_durations", None)
            if durations is None:
                mode = np.asarray(mode_posterior)
                n_time = int(mode.shape[0]) if mode.ndim else 0
                durations = _stored_duration_occupancy_durations(n_time)
            if durations is not None:
                from .duration_dynamics import DurationFloat

                dt_s = DurationFloat(float(dt_s), durations)
            return helper(
                mode_posterior,
                trajectory_log_posterior,
                bin_centers,
                dt_s,
            )
        finally:
            _clear_duration_occupancy_transition_durations()

    setattr(duration_aware_first_order_imm_content_diagnostics, _DURATION_ALIAS_ATTR, True)
    setattr(duration_aware_first_order_imm_content_diagnostics, _ORIGINAL_ATTR, helper)
    return duration_aware_first_order_imm_content_diagnostics


def _patch_loaded_alias(module_name: str, helper: Callable[..., dict[str, float | int]]) -> None:
    module = sys.modules.get(module_name)
    if module is None or not hasattr(module, "_first_order_imm_content_diagnostics"):
        return
    if module_name == "hipporeplayimm.duration_occupancy":
        current = getattr(module, "_first_order_imm_content_diagnostics")
        if getattr(current, _DURATION_ALIAS_ATTR, False):
            return
        alias = _wrap_duration_occupancy_alias(helper)
    else:
        alias = helper
    setattr(module, "_first_order_imm_content_diagnostics", alias)


def apply_first_order_imm_diagnostics_validation_patch() -> None:
    """Install validation on the shared helper and already-imported aliases."""

    import hipporeplayimm.state_space_utils as state_space_utils

    helper = _wrap_helper(state_space_utils._first_order_imm_content_diagnostics)
    state_space_utils._first_order_imm_content_diagnostics = helper
    _record_duration_occupancy_transition_durations()
    _record_duration_occupancy_score_context()
    _patch_loaded_alias("hipporeplayimm.state_space", helper)
    _patch_loaded_alias("hipporeplayimm.duration_occupancy", helper)
