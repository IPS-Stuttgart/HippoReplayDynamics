"""Runtime validation for custom duration-aware IMM mode transitions."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_ATTR = "_duration_occupancy_mode_transition_validation_patch"
_DECAY_PATCH_ATTR = "_duration_occupancy_decay_output_validation_patch"
_DISPLACEMENT_MOMENTUM_SINGLE_BIN_PATCH_ATTR = (
    "_displacement_momentum_single_bin_evidence_support_patch"
)
_DISPLACEMENT_IMM_SINGLE_BIN_PATCH_ATTR = "_displacement_imm_single_bin_evidence_support_patch"
_ORIGINAL_ATTR = "_duration_occupancy_mode_transition_validation_original"
_DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT = "degenerate_single_bin"
_TEXT_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _contains_boolean_values(values: np.ndarray) -> bool:
    """Return True only when matrix entries are actual boolean scalars."""

    if np.issubdtype(values.dtype, np.bool_):
        return True
    if values.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in values.flat)
    return False


def _contains_text_values(values: np.ndarray) -> bool:
    """Return True when matrix entries are text scalars NumPy could parse."""

    if values.dtype.kind in {"S", "U"}:
        return True
    if values.dtype == object:
        return any(isinstance(item, _TEXT_SCALAR_TYPES) for item in values.flat)
    return False


def _coerce_integer_count(value: Any, name: str, *, minimum: int) -> int:
    """Return an integer count without bool or array-scalar coercion."""

    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if arr.ndim != 0:
        raise TypeError(f"{name} must be an integer scalar")
    item = arr.item()
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer scalar")
    try:
        count = operator.index(item)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if count < int(minimum):
        raise ValueError(f"{name} must be at least {int(minimum)}")
    return int(count)


def _validate_mode_transition_sequence(
    mode_transitions: Sequence[Any],
    *,
    n_modes: int,
    n_transitions: int,
) -> list[np.ndarray]:
    """Validate custom source-row-stochastic mode-transition matrices."""

    mode_count = _coerce_integer_count(n_modes, "n_modes", minimum=1)
    transition_count = _coerce_integer_count(n_transitions, "n_transitions", minimum=0)
    if len(mode_transitions) != transition_count:
        raise ValueError("mode_transitions must contain one matrix per transition")

    expected_shape = (mode_count, mode_count)
    resolved: list[np.ndarray] = []
    for transition_index, matrix in enumerate(mode_transitions):
        try:
            raw_values = np.asarray(matrix)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"mode transition matrix {transition_index} must be a rectangular numeric probability matrix"
            ) from exc
        if _contains_boolean_values(raw_values):
            raise ValueError(
                f"mode transition matrix {transition_index} must contain numeric probabilities, not booleans"
            )
        if _contains_text_values(raw_values):
            raise ValueError(
                f"mode transition matrix {transition_index} must contain numeric probabilities, not strings"
            )
        try:
            values = raw_values.astype(float, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"mode transition matrix {transition_index} must contain numeric probabilities"
            ) from exc
        if values.shape != expected_shape:
            raise ValueError("mode transition matrices must be square with one row and column per mode")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"mode transition matrix {transition_index} must contain finite probabilities")
        if np.any(values < 0.0):
            raise ValueError(f"mode transition matrix {transition_index} must contain nonnegative probabilities")
        row_sums = values.sum(axis=1)
        if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
            raise ValueError(f"mode transition matrix {transition_index} rows must contain positive finite probability mass")
        if not np.allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError(f"mode transition matrix {transition_index} rows must sum to 1")
        resolved.append(values)
    return resolved


def _validate_decay_probabilities(decays: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Return validated per-transition velocity-decay probabilities."""

    values = np.asarray(decays, dtype=float)
    expected_shape = np.asarray(durations, dtype=float).shape
    if values.shape != expected_shape:
        raise ValueError("momentum velocity decays must contain one value per transition duration")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("momentum velocity decays must be finite probabilities in [0, 1]")
    return values


def _wrap_duration_adjusted_decays(helper: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    if getattr(helper, _DECAY_PATCH_ATTR, False):
        return helper

    @wraps(helper)
    def duration_adjusted_decays(config_or_decay, durations, reference_dt):
        out = helper(config_or_decay, durations, reference_dt)
        return _validate_decay_probabilities(out, np.asarray(durations, dtype=float))

    setattr(duration_adjusted_decays, _DECAY_PATCH_ATTR, True)
    setattr(duration_adjusted_decays, _ORIGINAL_ATTR, helper)
    return duration_adjusted_decays


def _is_single_bin_event(emissions: Any) -> bool:
    try:
        return int(getattr(emissions, "n_time")) <= 1
    except (AttributeError, TypeError, ValueError):
        return False


def _mark_single_bin_displacement_support(
    diagnostics: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    updated = dict(diagnostics)
    updated[f"{prefix}_evidence_support"] = _DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    updated[f"{prefix}_degenerate_reason"] = "single_time_bin_random_marginal"
    updated[f"{prefix}_required_min_time_bins"] = 2
    return updated


def _wrap_displacement_momentum_single_bin_support(helper: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(helper, _DISPLACEMENT_MOMENTUM_SINGLE_BIN_PATCH_ATTR, False):
        return helper

    @wraps(helper)
    def score_displacement_momentum_exact(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory: bool = True,
    ):
        logp, trajectory, terminal, displacement_post, diagnostics = helper(
            emissions,
            bin_centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        if _is_single_bin_event(emissions):
            diagnostics = _mark_single_bin_displacement_support(
                diagnostics,
                "state_space_displacement_momentum",
            )
        return logp, trajectory, terminal, displacement_post, diagnostics

    setattr(
        score_displacement_momentum_exact,
        _DISPLACEMENT_MOMENTUM_SINGLE_BIN_PATCH_ATTR,
        True,
    )
    setattr(score_displacement_momentum_exact, _ORIGINAL_ATTR, helper)
    return score_displacement_momentum_exact


def _wrap_displacement_imm_single_bin_support(helper: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(helper, _DISPLACEMENT_IMM_SINGLE_BIN_PATCH_ATTR, False):
        return helper

    @wraps(helper)
    def score_displacement_imm_exact(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory: bool = True,
    ):
        logp, trajectory, terminal, mode_post, displacement_post, diagnostics = helper(
            emissions,
            bin_centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        if _is_single_bin_event(emissions):
            diagnostics = _mark_single_bin_displacement_support(
                diagnostics,
                "state_space_displacement_imm",
            )
        return logp, trajectory, terminal, mode_post, displacement_post, diagnostics

    setattr(score_displacement_imm_exact, _DISPLACEMENT_IMM_SINGLE_BIN_PATCH_ATTR, True)
    setattr(score_displacement_imm_exact, _ORIGINAL_ATTR, helper)
    return score_displacement_imm_exact


def _wrap_resolver(resolver: Callable[..., list[np.ndarray]]) -> Callable[..., list[np.ndarray]]:
    if getattr(resolver, _PATCH_ATTR, False):
        return resolver

    def _resolve_mode_transitions(
        ss,
        n_modes: int,
        mode_stickiness: float,
        mode_transitions,
        n_transitions: int,
    ) -> list[np.ndarray]:
        if mode_transitions is None:
            return resolver(ss, n_modes, mode_stickiness, mode_transitions, n_transitions)
        return _validate_mode_transition_sequence(
            mode_transitions,
            n_modes=n_modes,
            n_transitions=n_transitions,
        )

    _resolve_mode_transitions.__name__ = getattr(resolver, "__name__", "_resolve_mode_transitions")
    _resolve_mode_transitions.__doc__ = getattr(resolver, "__doc__", None)
    setattr(_resolve_mode_transitions, _PATCH_ATTR, True)
    setattr(_resolve_mode_transitions, _ORIGINAL_ATTR, resolver)
    return _resolve_mode_transitions


def _patch_displacement_single_bin_evidence_support() -> None:
    from . import state_space_displacement_imm, state_space_displacement_momentum

    state_space_displacement_momentum._score_displacement_momentum_exact = (
        _wrap_displacement_momentum_single_bin_support(
            state_space_displacement_momentum._score_displacement_momentum_exact
        )
    )
    state_space_displacement_imm._score_displacement_imm_exact = (
        _wrap_displacement_imm_single_bin_support(
            state_space_displacement_imm._score_displacement_imm_exact
        )
    )


def apply_duration_occupancy_mode_transition_validation_patch() -> None:
    """Install validation for duration-aware IMM transition helpers."""

    from . import duration_occupancy

    duration_occupancy._resolve_mode_transitions = _wrap_resolver(duration_occupancy._resolve_mode_transitions)
    duration_occupancy._duration_adjusted_decays = _wrap_duration_adjusted_decays(duration_occupancy._duration_adjusted_decays)
    _patch_displacement_single_bin_evidence_support()


__all__ = [
    "apply_duration_occupancy_mode_transition_validation_patch",
]
