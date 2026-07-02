"""Runtime guards for duration-dependent momentum helpers.

The exact sparse and finite-displacement momentum decoders derive both velocity
decay and relative velocity scaling from transition durations.  Non-finite or
non-positive values should fail at the helper boundary instead of producing
NaN/inf transition parameters that later corrupt the dynamic program.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from .sparse_momentum_bin_center_validation import apply_sparse_momentum_bin_center_validation_patch

_SPARSE_SCORE_CONFIG_PATCHED_FLAG = "_sparse_momentum_config_validation_patch_applied"
_TRAJECTORY_IMM_CONFIG_PATCHED_FLAG = "_trajectory_imm_sparse_config_validation_patch_applied"


def apply_sparse_momentum_duration_validation_patch() -> None:
    """Install duration and exact-sparse momentum config validation patches."""

    apply_sparse_momentum_bin_center_validation_patch()

    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum

    _patch_duration_helpers(sparse_momentum)
    _patch_duration_helpers(displacement_momentum)

    # These IMM modules import the helper functions by value.  Keep their module
    # aliases synchronized even if they were imported before this runtime patch.
    import hipporeplayimm.state_space_displacement_imm as displacement_imm
    import hipporeplayimm.state_space_trajectory_imm as trajectory_imm

    trajectory_imm._coerce_transition_durations = sparse_momentum._coerce_transition_durations
    trajectory_imm._duration_adjusted_decays = sparse_momentum._duration_adjusted_decays
    trajectory_imm._time_scales = sparse_momentum._time_scales
    displacement_imm._coerce_transition_durations = displacement_momentum._coerce_transition_durations
    displacement_imm._duration_adjusted_decays = displacement_momentum._duration_adjusted_decays
    displacement_imm._time_scales = displacement_momentum._time_scales
    displacement_imm._duration_scale_at = displacement_momentum._duration_scale_at

    _patch_exact_sparse_momentum_config(
        sparse_momentum,
        "_score_sparse_momentum_exact",
        _SPARSE_SCORE_CONFIG_PATCHED_FLAG,
        include_diffusion=False,
    )
    _patch_exact_sparse_momentum_config(
        trajectory_imm,
        "_score_trajectory_imm_exact_sparse",
        _TRAJECTORY_IMM_CONFIG_PATCHED_FLAG,
        include_diffusion=True,
    )


def _patch_duration_helpers(module: Any) -> None:
    if getattr(module, "_duration_validation_patch_applied", False):
        return

    original_duration_adjusted_decays = module._duration_adjusted_decays
    original_time_scales = module._time_scales
    original_duration_scale_at = getattr(module, "_duration_scale_at", None)

    @wraps(module._coerce_transition_durations)
    def coerce_transition_durations(values: Any, *, n_time: int, fallback_dt: float) -> np.ndarray:
        expected = max(_coerce_count_scalar("n_time", n_time) - 1, 0)
        fallback = _coerce_positive_float_scalar("fallback dt", fallback_dt, "fallback dt must be finite and positive")

        raw_values = list(values)
        if len(raw_values) == 0:
            return np.full(expected, fallback, dtype=float)

        durations = np.asarray(raw_values, dtype=float)
        if durations.ndim != 1:
            raise ValueError("transition durations must be one-dimensional")
        if durations.shape != (expected,):
            raise ValueError(f"transition durations must have shape {(expected,)}, got {durations.shape}")
        return _valid_transition_durations(durations)

    @wraps(original_duration_adjusted_decays)
    def duration_adjusted_decays(config: object, durations: Any, reference_dt: float) -> np.ndarray:
        reference = _coerce_positive_float_scalar("reference dt", reference_dt, "reference dt must be finite and positive")
        return original_duration_adjusted_decays(
            config,
            _valid_transition_durations(durations),
            reference,
        )

    @wraps(original_time_scales)
    def time_scales(durations: Any) -> np.ndarray:
        return original_time_scales(_valid_transition_durations(durations))

    module._coerce_transition_durations = coerce_transition_durations
    module._duration_adjusted_decays = duration_adjusted_decays
    module._time_scales = time_scales
    if original_duration_scale_at is not None:

        @wraps(original_duration_scale_at)
        def duration_scale_at(durations: Any, transition_index: int, reference_dt: float) -> float:
            reference = _coerce_positive_float_scalar("reference dt", reference_dt, "reference dt must be finite and positive")
            return original_duration_scale_at(
                _valid_transition_durations(durations),
                transition_index,
                reference,
            )

        module._duration_scale_at = duration_scale_at
    module._duration_validation_patch_applied = True


def _patch_exact_sparse_momentum_config(module: Any, score_name: str, patched_flag: str, *, include_diffusion: bool) -> None:
    original_score = getattr(module, score_name)
    if getattr(original_score, patched_flag, False):
        return

    @wraps(original_score)
    def score(
        emissions: Any,
        bin_centers: Any,
        config: object,
        transition_durations_s: Any,
        *,
        valid_bin_mask: Any = None,
        return_trajectory: bool = True,
    ):
        _validate_exact_sparse_momentum_config(config, include_diffusion=include_diffusion)
        return original_score(
            emissions,
            bin_centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )

    setattr(score, patched_flag, True)
    setattr(score, "__hipporeplayimm_original__", original_score)
    setattr(module, score_name, score)


def _validate_exact_sparse_momentum_config(config: object, *, include_diffusion: bool) -> None:
    _validate_config_positive_scalar(config, "max_step_sigma", 4.0)
    if include_diffusion:
        _validate_config_positive_scalar(config, "diffusion_sigma_cm_sqrt_s", 85.0)
    _validate_config_positive_scalar(config, "momentum_sigma_cm_sqrt_s", 85.0)
    _validate_config_positive_scalar(config, "momentum_initial_sigma_cm_sqrt_s", 85.0)


def _validate_config_positive_scalar(config: object, name: str, default: float) -> None:
    _coerce_positive_float_scalar(name, getattr(config, name, default), f"{name} must be finite and positive")


def _is_boolean_scalar(value: object) -> bool:
    """Return True for Python, NumPy, and object-wrapped boolean scalars."""

    if isinstance(value, (bool, np.bool_)):
        return True
    arr = np.asarray(value)
    if arr.ndim != 0:
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        try:
            return isinstance(arr.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _reject_array_shaped_scalar(name: str, value: object) -> None:
    """Reject values that NumPy/Python might coerce from an array to a scalar."""

    try:
        arr = np.asarray(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if arr.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")


def _coerce_count_scalar(name: str, value: object) -> int:
    _reject_array_shaped_scalar(name, value)
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be an integer count, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer count") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.floor(numeric):
        raise TypeError(f"{name} must be a non-negative integer count")
    return int(numeric)


def _coerce_positive_float_scalar(name: str, value: object, message: str) -> float:
    _reject_array_shaped_scalar(name, value)
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(message)
    return numeric


def _valid_transition_durations(durations: Any) -> np.ndarray:
    values = np.asarray(durations, dtype=float)
    if values.ndim != 1:
        raise ValueError("transition durations must be one-dimensional")
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    return values
