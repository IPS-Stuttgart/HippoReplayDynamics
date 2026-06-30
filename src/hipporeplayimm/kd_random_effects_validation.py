"""Validate KD random-effects Gibbs sampler options before sampling."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_kd_random_effects_validation_patch_applied"


def apply_kd_random_effects_validation_patch() -> None:
    """Install strict option validation on KD random-effects summaries."""

    from . import kd_reference

    current = kd_reference.random_effects_model_probabilities
    if getattr(current, _PATCHED_FLAG, False):
        setattr(kd_reference, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def random_effects_model_probabilities(
        log_evidence,
        models,
        prior: float = 10.0,
        n_iterations: int = 500,
        burnin: int = 50,
    ):
        prior_value, n_iterations_value, burnin_value = _validate_sampler_options(
            prior=prior,
            n_iterations=n_iterations,
            burnin=burnin,
        )
        return current(
            log_evidence,
            models,
            prior=prior_value,
            n_iterations=n_iterations_value,
            burnin=burnin_value,
        )

    setattr(random_effects_model_probabilities, _PATCHED_FLAG, True)
    setattr(random_effects_model_probabilities, "__hipporeplayimm_original__", current)
    kd_reference.random_effects_model_probabilities = random_effects_model_probabilities
    setattr(kd_reference, _PATCHED_FLAG, True)


def _validate_sampler_options(*, prior: Any, n_iterations: Any, burnin: Any) -> tuple[float, int, int]:
    """Return normalized sampler options or raise before producing NaN summaries."""

    prior_array = np.asarray(prior)
    if prior_array.shape != ():
        raise ValueError("prior must be a scalar")
    try:
        prior_value = float(prior_array)
    except (TypeError, ValueError) as exc:
        raise TypeError("prior must be numeric") from exc
    if not np.isfinite(prior_value) or prior_value <= 0.0:
        raise ValueError("prior must be finite and positive")

    n_iterations_value = _coerce_non_boolean_integer(n_iterations, "n_iterations")
    if n_iterations_value <= 0:
        raise ValueError("n_iterations must be positive")

    burnin_value = _coerce_non_boolean_integer(burnin, "burnin")
    if burnin_value < 0 or burnin_value >= n_iterations_value:
        raise ValueError("burnin must be non-negative and less than n_iterations")

    return prior_value, n_iterations_value, burnin_value


def _coerce_non_boolean_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not boolean")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


__all__ = ["apply_kd_random_effects_validation_patch"]
