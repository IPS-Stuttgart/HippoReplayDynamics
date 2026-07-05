"""Validation helper for mass-retaining candidate thresholds."""

from __future__ import annotations

import numpy as np

from . import state_space_utils as _state_space_utils


def _coerce_mass_threshold(value: object) -> float:
    """Return a finite numeric scalar threshold without boolean/array coercion."""

    if _state_space_utils._is_boolean_scalar(value):
        raise TypeError("mass_threshold must be numeric, not boolean")
    try:
        arr = np.asarray(value)
    except ValueError as exc:
        raise TypeError("mass_threshold must be a numeric scalar") from exc
    if arr.ndim != 0:
        raise TypeError("mass_threshold must be a numeric scalar")
    try:
        threshold = float(arr.item())
    except (TypeError, ValueError) as exc:
        raise ValueError("mass_threshold must be in (0, 1]") from exc
    if not np.isfinite(threshold):
        raise ValueError("mass_threshold must be in (0, 1]")
    return threshold


def validated_mass_retaining_candidate_indices(
    log_emission: np.ndarray,
    mass_threshold: float | None = None,
    *,
    top_k: int | None = None,
    min_k: int = 1,
    max_k: int = 0,
) -> np.ndarray:
    """Select candidates after strict scalar validation of mass_threshold."""

    if mass_threshold is None:
        validated_threshold = None
    else:
        validated_threshold = _coerce_mass_threshold(mass_threshold)
    return _state_space_utils._mass_retaining_candidate_indices(
        log_emission,
        validated_threshold,
        top_k=top_k,
        min_k=min_k,
        max_k=max_k,
    )
