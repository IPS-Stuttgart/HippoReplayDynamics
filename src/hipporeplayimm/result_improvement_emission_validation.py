"""Validate replay-calibrated result-improvement emission parameters."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
from scipy.special import betaln

_PATCHED_FLAG = "_result_improvement_emission_validation_patch_applied"
_BUILD_SORTED_EMISSIONS_WRAPPER_FLAG = "_replay_calibrated_emission_parameter_validation_wrapper"
_BUILD_SORTED_EMISSIONS_WRAPPER_VERSION = 2
_GAMMA_POISSON_WRAPPER_FLAG = "_replay_calibrated_gamma_poisson_stability_wrapper"
_GAMMA_POISSON_WRAPPER_VERSION = 1
_ORIGINAL_ATTR = "__hipporeplayimm_emission_validation_original__"
_GAMMA_POISSON_ORIGINAL_ATTR = "__hipporeplayimm_gamma_poisson_original__"


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _is_boolean_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in array.flat)
    return False


def _is_text_scalar(value: object) -> bool:
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (str, bytes, np.str_, np.bytes_))
        except ValueError:
            return False
    return False


def _is_text_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in array.flat)
    return False


def _reject_array_shaped_scalar(name: str, value: object) -> None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if array.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")
    if _is_text_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not text")


def _reject_boolean_scalar(name: str, value: object) -> None:
    _reject_array_shaped_scalar(name, value)
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not boolean")


def _reject_boolean_numeric(name: str, value: object) -> None:
    if _is_boolean_scalar(value) or _is_boolean_array(value):
        raise TypeError(f"{name} must be numeric, not boolean")
    if _is_text_scalar(value) or _is_text_array(value):
        raise TypeError(f"{name} must be numeric, not text")


def _finite_positive_scalar(name: str, value: object) -> float:
    _reject_boolean_scalar(name, value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _finite_nonnegative_scalar(name: str, value: object) -> float:
    _reject_boolean_scalar(name, value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _max_gain_scalar(value: object) -> float:
    _reject_boolean_scalar("max_gain", value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_gain must be finite and at least 1.0") from exc
    if not np.isfinite(numeric) or numeric < 1.0:
        raise ValueError("max_gain must be finite and at least 1.0")
    return numeric


def _required_attr(source: object, name: str) -> Any:
    try:
        return getattr(source, name)
    except AttributeError as exc:
        raise ValueError(f"{name} must be provided") from exc


def _validate_replay_calibrated_emission_parameters(config: object | None, calibration: object | None) -> None:
    """Reject malformed scalars before the builder's historical ``float(...)`` coercions."""

    if config is not None:
        _finite_positive_scalar("time_bin_s", _required_attr(config, "time_bin_s"))
        _finite_positive_scalar("spike_rate_scale", _required_attr(config, "spike_rate_scale"))
        _finite_positive_scalar("likelihood_temperature", _required_attr(config, "likelihood_temperature"))
        _finite_nonnegative_scalar(
            "negative_binomial_overdispersion",
            _required_attr(config, "negative_binomial_overdispersion"),
        )
        cell_weights = getattr(config, "cell_weights", None)
        if cell_weights is not None:
            _reject_boolean_numeric("cell_weights", cell_weights)

    if calibration is not None:
        _finite_nonnegative_scalar("gain_prior_count", _required_attr(calibration, "gain_prior_count"))
        _max_gain_scalar(_required_attr(calibration, "max_gain"))
        _finite_positive_scalar(
            "negative_binomial_dispersion",
            _required_attr(calibration, "negative_binomial_dispersion"),
        )


def _stable_gamma_poisson_log_emissions(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_durations: np.ndarray,
    *,
    dispersion: float,
) -> np.ndarray:
    """Evaluate Gamma-Poisson emissions continuously at the Poisson limit."""

    r = float(dispersion)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("dispersion must be finite and positive")
    dt = np.asarray(bin_durations, dtype=float)
    if dt.ndim != 1 or dt.shape[0] != spike_counts.shape[0]:
        raise ValueError("bin_durations must contain one duration per time bin")
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("all bin durations must be finite and positive")

    mean = np.maximum(
        dt[:, None, None] * np.asarray(rates_hz, dtype=float)[None, :, :],
        np.finfo(float).tiny,
    )
    counts = np.asarray(spike_counts, dtype=float)[:, :, None]
    counts, mean = np.broadcast_arrays(counts, mean)

    combination = np.zeros_like(mean, dtype=float)
    positive_counts = counts > 0.0
    combination[positive_counts] = (
        -np.log(counts[positive_counts])
        - betaln(r, counts[positive_counts])
    )
    log_scaled_mean = np.log(mean) - np.log(r)
    log_success_probability = -np.logaddexp(0.0, log_scaled_mean)
    log_failure_probability = -np.logaddexp(0.0, -log_scaled_mean)
    return np.sum(
        combination
        + r * log_success_probability
        + counts * log_failure_probability,
        axis=1,
    )


def _patch_gamma_poisson_stability(result_improvement_extensions: Any) -> None:
    """Replace cancellation-prone Gamma-Poisson evaluation idempotently."""

    current = result_improvement_extensions._negative_binomial_log_emissions
    if getattr(current, _GAMMA_POISSON_WRAPPER_FLAG, None) == _GAMMA_POISSON_WRAPPER_VERSION:
        return

    @wraps(current)
    def stable_gamma_poisson_log_emissions(
        spike_counts,
        rates_hz,
        bin_durations,
        *,
        dispersion,
    ):
        return _stable_gamma_poisson_log_emissions(
            spike_counts,
            rates_hz,
            bin_durations,
            dispersion=dispersion,
        )

    setattr(
        stable_gamma_poisson_log_emissions,
        _GAMMA_POISSON_WRAPPER_FLAG,
        _GAMMA_POISSON_WRAPPER_VERSION,
    )
    setattr(stable_gamma_poisson_log_emissions, _GAMMA_POISSON_ORIGINAL_ATTR, current)
    result_improvement_extensions._negative_binomial_log_emissions = stable_gamma_poisson_log_emissions


def _restore_exact_zero_rate_support(emissions: object, encoding: object, config: object | None) -> object:
    """Make spikes at exactly unsupported place-field bins impossible again.

    The replay-calibrated builder historically floors every rate to the smallest
    positive float before applying gains.  That turns an exact zero-rate support
    constraint into a merely tiny likelihood.  Positive replay gains cannot make
    an exact zero rate positive, so the original encoding still defines the
    correct impossible-bin mask.
    """

    rates = np.asarray(getattr(encoding, "rates_hz"), dtype=float)
    counts = np.asarray(getattr(emissions, "spike_counts"), dtype=float)
    log_likelihood = np.asarray(getattr(emissions, "log_likelihood"), dtype=float)
    if rates.ndim != 2 or counts.ndim != 2 or log_likelihood.ndim != 2:
        raise ValueError("replay-calibrated emissions and rates must be two-dimensional")
    if rates.shape[0] != counts.shape[1] or rates.shape[1] != log_likelihood.shape[1]:
        raise ValueError("encoding rates must match replay emission cell and spatial dimensions")
    if counts.shape[0] != log_likelihood.shape[0]:
        raise ValueError("spike counts must contain one row per replay emission row")

    zero_rate = rates == 0.0
    if not np.any(zero_rate) or counts.size == 0:
        return emissions

    metadata = dict(getattr(emissions, "metadata", {}) or {})
    emission_model = str(metadata.get("sorted_spike_emission_model", "poisson")).strip().lower()
    if emission_model == "poisson":
        from . import encoding as encoding_module

        effective_config = encoding_module.EmissionConfig() if config is None else config
        cell_weights = encoding_module._emission_cell_weights(
            getattr(effective_config, "cell_weights", None),
            counts.shape[1],
        )
    else:
        cell_weights = np.ones(counts.shape[1], dtype=float)

    active_zero_rate = zero_rate & (np.asarray(cell_weights, dtype=float)[:, None] > 0.0)
    impossible = np.any(
        (counts[:, :, None] > 0.0) & active_zero_rate[None, :, :],
        axis=1,
    )
    if not np.any(impossible):
        return emissions

    corrected = log_likelihood.copy()
    corrected[impossible] = -np.inf
    emissions.log_likelihood = corrected
    return emissions


def apply_result_improvement_emission_validation_patch() -> None:
    """Install scalar guards, stable Gamma-Poisson numerics, and exact support."""

    from . import result_improvement_extensions

    _patch_gamma_poisson_stability(result_improvement_extensions)

    current = result_improvement_extensions.build_sorted_emissions_with_replay_calibration
    if getattr(current, _BUILD_SORTED_EMISSIONS_WRAPPER_FLAG, None) == _BUILD_SORTED_EMISSIONS_WRAPPER_VERSION:
        setattr(result_improvement_extensions, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def build_sorted_emissions_with_replay_calibration(session, encoding, ripple, config=None, calibration=None):
        _validate_replay_calibrated_emission_parameters(config, calibration)
        emissions = current(session, encoding, ripple, config, calibration)
        return _restore_exact_zero_rate_support(emissions, encoding, config)

    setattr(
        build_sorted_emissions_with_replay_calibration,
        _BUILD_SORTED_EMISSIONS_WRAPPER_FLAG,
        _BUILD_SORTED_EMISSIONS_WRAPPER_VERSION,
    )
    setattr(build_sorted_emissions_with_replay_calibration, _ORIGINAL_ATTR, current)
    result_improvement_extensions.build_sorted_emissions_with_replay_calibration = build_sorted_emissions_with_replay_calibration
    setattr(result_improvement_extensions, _PATCHED_FLAG, True)


__all__ = ["apply_result_improvement_emission_validation_patch"]
