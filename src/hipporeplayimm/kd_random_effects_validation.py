"""Validate KD random-effects Gibbs sampler options before sampling."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_kd_random_effects_validation_patch_applied"


def _is_boolean_scalar(value: Any) -> bool:
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


def _contains_complex_values(value: Any) -> bool:
    """Return whether an array-like input contains complex numeric values."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(raw.dtype, np.complexfloating):
        return True
    if raw.dtype == object:
        return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)
    return False


def apply_kd_random_effects_validation_patch() -> None:
    """Install strict option validation on KD random-effects summaries."""

    from . import kd_reference

    current = kd_reference.random_effects_model_probabilities
    if getattr(current, _PATCHED_FLAG, False):
        setattr(kd_reference, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def random_effects_model_probabilities(log_evidence, models, prior: float = 10.0, n_iterations: int = 500, burnin: int = 50):
        evidence_values, model_values = _validate_model_evidence_inputs(log_evidence, models)
        prior_value, n_iterations_value, burnin_value = _validate_sampler_options(prior=prior, n_iterations=n_iterations, burnin=burnin)
        if np.any(np.isneginf(evidence_values)):
            return _random_effects_with_exact_impossible_support(
                evidence_values,
                model_values,
                prior=prior_value,
                n_iterations=n_iterations_value,
                burnin=burnin_value,
            )
        return current(evidence_values, model_values, prior=prior_value, n_iterations=n_iterations_value, burnin=burnin_value)

    setattr(random_effects_model_probabilities, _PATCHED_FLAG, True)
    setattr(random_effects_model_probabilities, "__hipporeplayimm_original__", current)
    kd_reference.random_effects_model_probabilities = random_effects_model_probabilities
    setattr(kd_reference, _PATCHED_FLAG, True)


def _random_effects_with_exact_impossible_support(
    evidence_values: np.ndarray,
    model_values: list[Any],
    *,
    prior: float,
    n_iterations: int,
    burnin: int,
) -> list[dict[str, Any]]:
    """Run the Gibbs sampler while retaining mixed finite/negative-infinite rows."""

    invalid_rows = np.any(np.isnan(evidence_values) | np.isposinf(evidence_values), axis=1)
    has_finite_support = np.any(np.isfinite(evidence_values), axis=1)
    usable = evidence_values[~invalid_rows & has_finite_support]
    if usable.size == 0:
        return [{"model": model, "p_model": np.nan, "p_exceedance": np.nan} for model in model_values]

    rng = np.random.default_rng(0)
    centered = usable - usable.max(axis=1, keepdims=True)
    n_events, n_models = centered.shape
    gibbs = np.zeros((n_iterations, n_models), dtype=float)
    alpha = np.ones(n_models, dtype=float) * prior
    for iteration in range(n_iterations):
        r_m = rng.dirichlet(alpha)
        gibbs[iteration] = r_m
        log_assignment = centered + np.log(r_m)[None, :]
        assignment_probs = np.exp(log_assignment - logsumexp(log_assignment, axis=1, keepdims=True))
        counts = np.zeros(n_models, dtype=int)
        for event_index in range(n_events):
            counts += rng.multinomial(1, assignment_probs[event_index])
        alpha = prior + counts

    posterior = gibbs[burnin:]
    p_models = posterior.mean(axis=0)
    p_exceedance = (posterior == posterior.max(axis=1, keepdims=True)).mean(axis=0)
    return [
        {
            "model": model,
            "p_model": float(p_models[index]),
            "p_exceedance": float(p_exceedance[index]),
        }
        for index, model in enumerate(model_values)
    ]


def _validate_model_evidence_inputs(log_evidence: Any, models: Any) -> tuple[np.ndarray, list[Any]]:
    if _contains_complex_values(log_evidence):
        raise ValueError("log_evidence must contain real values, not complex values")
    try:
        evidence_values = np.asarray(log_evidence, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("log_evidence must be a numeric two-dimensional array") from exc
    if evidence_values.ndim != 2:
        raise ValueError("log_evidence must be a two-dimensional array")
    if isinstance(models, (str, bytes)):
        raise TypeError("models must be a sequence of model names")
    try:
        model_values = list(models)
    except TypeError as exc:
        raise TypeError("models must be a sequence of model names") from exc
    if evidence_values.shape[1] == 0:
        raise ValueError("models must contain at least one model")
    if len(model_values) != evidence_values.shape[1]:
        raise ValueError("models length must match log_evidence columns")
    return evidence_values, model_values


def _validate_sampler_options(*, prior: Any, n_iterations: Any, burnin: Any) -> tuple[float, int, int]:
    if _is_boolean_scalar(prior):
        raise TypeError("prior must be numeric, not boolean")
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
