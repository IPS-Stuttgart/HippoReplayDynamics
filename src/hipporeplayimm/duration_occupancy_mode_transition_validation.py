"""Runtime validation for custom duration-aware IMM mode transitions."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_PATCH_ATTR = "_duration_occupancy_mode_transition_validation_patch"
_ORIGINAL_ATTR = "_duration_occupancy_mode_transition_validation_original"


def _contains_boolean_values(values: np.ndarray) -> bool:
    """Return True only when matrix entries are actual boolean scalars."""

    if np.issubdtype(values.dtype, np.bool_):
        return True
    if values.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in values.flat)
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
        raw_values = np.asarray(matrix)
        if _contains_boolean_values(raw_values):
            raise ValueError(
                f"mode transition matrix {transition_index} must contain numeric probabilities, not booleans"
            )
        try:
            values = raw_values.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
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


def apply_duration_occupancy_mode_transition_validation_patch() -> None:
    """Install validation for externally supplied duration-aware IMM transitions."""

    from . import duration_occupancy

    duration_occupancy._resolve_mode_transitions = _wrap_resolver(duration_occupancy._resolve_mode_transitions)


__all__ = ["apply_duration_occupancy_mode_transition_validation_patch"]
