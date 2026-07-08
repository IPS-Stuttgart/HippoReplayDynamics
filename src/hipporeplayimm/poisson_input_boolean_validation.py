"""Reject boolean Poisson emission inputs."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_poisson_input_boolean_validation_patch_applied"


def _contains_boolean_values(value: object) -> bool:
    """Return True for boolean scalars/arrays, including object arrays containing bools."""

    if value is None:
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in array.flat)
    return False


def _reject_boolean_array(name: str, value: object) -> None:
    if _contains_boolean_values(value):
        raise ValueError(f"{name} must contain numeric values, not boolean values")


def apply_poisson_input_boolean_validation_patch() -> None:
    """Install a guard before Poisson emissions coerce inputs to floats."""

    from . import encoding

    current = encoding._poisson_log_emissions
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
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
        _reject_boolean_array("dt", dt)
        _reject_boolean_array("spike_rate_scale", spike_rate_scale)
        _reject_boolean_array("likelihood_temperature", likelihood_temperature)
        _reject_boolean_array("cell_weights", cell_weights)
        _reject_boolean_array(
            "negative_binomial_overdispersion",
            negative_binomial_overdispersion,
        )
        return current(
            spike_counts,
            rates_hz,
            dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            cell_weights=cell_weights,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )

    setattr(poisson_log_emissions, _PATCHED_FLAG, True)
    setattr(poisson_log_emissions, "__hipporeplayimm_original__", current)
    encoding._poisson_log_emissions = poisson_log_emissions
    setattr(encoding, _PATCHED_FLAG, True)


__all__ = ["apply_poisson_input_boolean_validation_patch"]
