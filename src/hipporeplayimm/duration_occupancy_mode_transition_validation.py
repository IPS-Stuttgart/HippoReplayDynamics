"""Runtime validation for custom duration-aware IMM mode transitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_PATCH_ATTR = "_duration_occupancy_mode_transition_validation_patch"
_ORIGINAL_ATTR = "_duration_occupancy_mode_transition_validation_original"


def _validate_mode_transition_sequence(
    mode_transitions: Sequence[Any],
    *,
    n_modes: int,
    n_transitions: int,
) -> list[np.ndarray]:
    """Validate custom source-row-stochastic mode-transition matrices."""

    if len(mode_transitions) != int(n_transitions):
        raise ValueError("mode_transitions must contain one matrix per transition")

    expected_shape = (int(n_modes), int(n_modes))
    resolved: list[np.ndarray] = []
    for transition_index, matrix in enumerate(mode_transitions):
        values = np.asarray(matrix, dtype=float)
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
            n_modes=int(n_modes),
            n_transitions=int(n_transitions),
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
