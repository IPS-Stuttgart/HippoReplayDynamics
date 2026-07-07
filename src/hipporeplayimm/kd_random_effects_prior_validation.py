"""Reject boolean KD random-effects prior values before float coercion."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_kd_random_effects_prior_validation_patch_applied"


def _is_boolean_scalar(value: Any) -> bool:
    """Return whether value is a scalar boolean."""

    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.shape != ():
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        try:
            return isinstance(raw.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def apply_kd_random_effects_prior_validation_patch() -> None:
    """Install a boolean-prior guard on KD random-effects summaries."""

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
        if _is_boolean_scalar(prior):
            raise TypeError("prior must be numeric, not boolean")
        return current(
            log_evidence,
            models,
            prior=prior,
            n_iterations=n_iterations,
            burnin=burnin,
        )

    setattr(random_effects_model_probabilities, _PATCHED_FLAG, True)
    setattr(random_effects_model_probabilities, "__hipporeplayimm_original__", current)
    kd_reference.random_effects_model_probabilities = random_effects_model_probabilities
    setattr(kd_reference, _PATCHED_FLAG, True)


__all__ = ["apply_kd_random_effects_prior_validation_patch"]
