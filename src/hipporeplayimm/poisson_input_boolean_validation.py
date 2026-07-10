"""Validate Poisson inputs and preserve exact zero-rate likelihood support."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_poisson_input_boolean_validation_patch_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_WRAPPER_VERSION = 2


def _contains_boolean_values(value: object) -> bool:
    """Return True for boolean arrays, including object arrays containing bools."""

    try:
        array = np.asarray(value)
    except ValueError:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in array.flat)
    return False


def _reject_boolean_array(name: str, value: object) -> None:
    if _contains_boolean_values(value):
        raise ValueError(f"{name} must contain numeric values, not boolean values")


def _reusable_cell_weights(cell_weights):
    """Materialize one-shot iterables for validation and support repair."""

    if (
        cell_weights is None
        or np.isscalar(cell_weights)
        or isinstance(cell_weights, np.ndarray)
    ):
        return cell_weights
    try:
        return tuple(cell_weights)
    except TypeError:
        return cell_weights


def _floored_zero_count_log_probability(overdispersion: float) -> float:
    """Return the implementation's log PMF for zero count at its rate floor."""

    tiny = np.finfo(float).tiny
    if overdispersion == 0.0:
        return -tiny
    size = 1.0 / overdispersion
    if not np.isfinite(size):
        return 0.0
    return float(size * (np.log(size) - np.log(size + tiny)))


def _restore_exact_zero_rate_support(
    log_likelihood: np.ndarray,
    *,
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    cell_weights: np.ndarray,
    likelihood_temperature: float,
    negative_binomial_overdispersion: float,
) -> np.ndarray:
    """Undo the numeric rate floor wherever the configured rate is exactly zero."""

    zero_rate = rates_hz == 0.0
    if not np.any(zero_rate):
        return log_likelihood

    active_zero_rate = zero_rate & (cell_weights[:, None] > 0.0)
    impossible = np.any(
        (spike_counts[:, :, None] > 0.0) & active_zero_rate[None, :, :],
        axis=1,
    )

    corrected = np.asarray(log_likelihood, dtype=float).copy()
    zero_count_weight = np.einsum(
        "tc,cb,c->tb",
        spike_counts == 0.0,
        zero_rate,
        cell_weights,
        optimize=True,
    )
    floored_log_probability = _floored_zero_count_log_probability(
        negative_binomial_overdispersion
    )
    corrected -= floored_log_probability * zero_count_weight / likelihood_temperature
    corrected[impossible] = -np.inf
    return corrected


def apply_poisson_input_boolean_validation_patch() -> None:
    """Install input guards and exact support handling for count emissions."""

    from . import encoding

    current = encoding._poisson_log_emissions
    if getattr(current, _PATCHED_FLAG, None) == _WRAPPER_VERSION:
        setattr(encoding, _PATCHED_FLAG, True)
        return
    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def poisson_log_emissions(
        spike_counts,
        rates_hz,
        dt,
        *,
        spike_rate_scale=1.0,
        likelihood_temperature=1.0,
        cell_weights=None,
        negative_binomial_overdispersion=0.0,
    ):
        _reject_boolean_array("spike_counts", spike_counts)
        _reject_boolean_array("rates_hz", rates_hz)
        reusable_weights = _reusable_cell_weights(cell_weights)
        log_likelihood = original(
            spike_counts,
            rates_hz,
            dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            cell_weights=reusable_weights,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )

        counts = np.asarray(spike_counts, dtype=float)
        rates = np.asarray(rates_hz, dtype=float)
        weights = encoding._emission_cell_weights(reusable_weights, counts.shape[1])
        return _restore_exact_zero_rate_support(
            log_likelihood,
            spike_counts=counts,
            rates_hz=rates,
            cell_weights=weights,
            likelihood_temperature=float(likelihood_temperature),
            negative_binomial_overdispersion=float(negative_binomial_overdispersion),
        )

    setattr(poisson_log_emissions, _PATCHED_FLAG, _WRAPPER_VERSION)
    setattr(poisson_log_emissions, _ORIGINAL_ATTR, original)
    encoding._poisson_log_emissions = poisson_log_emissions
    setattr(encoding, _PATCHED_FLAG, True)


__all__ = ["apply_poisson_input_boolean_validation_patch"]
