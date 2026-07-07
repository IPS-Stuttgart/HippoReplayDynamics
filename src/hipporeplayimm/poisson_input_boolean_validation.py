"""Reject invalid Poisson emission and emission-tensor inputs."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_poisson_input_boolean_validation_patch_applied"
_LOG_EMISSION_TENSOR_FLAG = "_log_emission_tensor_validation_patch_applied"


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


def _validate_log_emission_tensor_spike_counts(spike_counts: object) -> None:
    """Reject non-count spike-count arrays on direct LogEmissionTensor construction."""

    _reject_boolean_array("spike_counts", spike_counts)
    try:
        counts = np.asarray(spike_counts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("spike_counts must contain numeric count values") from exc
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("spike_counts must contain finite nonnegative values")
    if not np.all(np.isclose(counts, np.rint(counts), rtol=0.0, atol=0.0)):
        raise ValueError("spike_counts must contain integer-valued counts")


def apply_poisson_input_boolean_validation_patch() -> None:
    """Install guards before emission inputs can poison likelihood scoring."""

    from . import encoding

    current = encoding._poisson_log_emissions
    if not getattr(current, _PATCHED_FLAG, False):

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

    _patch_log_emission_tensor_validation(encoding)
    setattr(encoding, _PATCHED_FLAG, True)


def _patch_log_emission_tensor_validation(encoding) -> None:
    current = encoding.LogEmissionTensor.__post_init__
    if getattr(current, _LOG_EMISSION_TENSOR_FLAG, False):
        return

    @wraps(current)
    def __post_init__(self) -> None:
        current(self)
        if np.any(np.isnan(self.log_likelihood)):
            raise ValueError("log_likelihood must not contain NaN")
        _validate_log_emission_tensor_spike_counts(self.spike_counts)

    setattr(__post_init__, _LOG_EMISSION_TENSOR_FLAG, True)
    setattr(__post_init__, "__hipporeplayimm_original__", current)
    encoding.LogEmissionTensor.__post_init__ = __post_init__


__all__ = ["apply_poisson_input_boolean_validation_patch"]
