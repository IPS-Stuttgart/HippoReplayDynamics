"""Runtime validation for first-order IMM content diagnostics."""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np

_PATCH_ATTR = "_first_order_imm_diagnostics_validation_patch"
_ORIGINAL_ATTR = "_first_order_imm_diagnostics_validation_original"
_DURATION_ALIAS_ATTR = "_first_order_imm_duration_diagnostics_alias_patch"


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
        return helper(
            mode_posterior,
            trajectory_log_posterior,
            bin_centers,
            dt_s,
        )

    setattr(validated_first_order_imm_content_diagnostics, _PATCH_ATTR, True)
    setattr(validated_first_order_imm_content_diagnostics, _ORIGINAL_ATTR, helper)
    return validated_first_order_imm_content_diagnostics


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
        durations = getattr(dt_s, "transition_durations", None)
        if durations is None:
            try:
                durations = sys._getframe(1).f_locals.get("durations")
            except ValueError:
                durations = None
        if durations is not None:
            from .duration_dynamics import DurationFloat

            dt_s = DurationFloat(float(dt_s), durations)
        return helper(
            mode_posterior,
            trajectory_log_posterior,
            bin_centers,
            dt_s,
        )

    setattr(duration_aware_first_order_imm_content_diagnostics, _DURATION_ALIAS_ATTR, True)
    setattr(duration_aware_first_order_imm_content_diagnostics, _ORIGINAL_ATTR, helper)
    return duration_aware_first_order_imm_content_diagnostics


def _patch_loaded_alias(module_name: str, helper: Callable[..., dict[str, float | int]]) -> None:
    module = sys.modules.get(module_name)
    if module is not None and hasattr(module, "_first_order_imm_content_diagnostics"):
        alias = (
            _wrap_duration_occupancy_alias(helper)
            if module_name == "hipporeplayimm.duration_occupancy"
            else helper
        )
        setattr(module, "_first_order_imm_content_diagnostics", alias)


def apply_first_order_imm_diagnostics_validation_patch() -> None:
    """Install validation on the shared helper and already-imported aliases."""

    import hipporeplayimm.state_space_utils as state_space_utils

    helper = _wrap_helper(state_space_utils._first_order_imm_content_diagnostics)
    state_space_utils._first_order_imm_content_diagnostics = helper
    _patch_loaded_alias("hipporeplayimm.state_space", helper)
    _patch_loaded_alias("hipporeplayimm.duration_occupancy", helper)
