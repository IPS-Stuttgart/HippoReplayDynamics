"""Validate Poisson inputs and preserve exact zero-rate likelihood support."""

from __future__ import annotations

from functools import wraps
import sys

import numpy as np
from scipy.special import betaln, gammaln

_PATCHED_FLAG = "_poisson_input_boolean_validation_patch_applied"
_NEGATIVE_BINOMIAL_PATCHED_FLAG = (
    "_negative_binomial_poisson_limit_patch_applied"
)
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_WRAPPER_VERSION = 4


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


def _reject_nonfinite_expected_count_scaling(
    rates_hz: object,
    dt: object,
    spike_rate_scale: float,
) -> None:
    """Reject finite inputs whose expected-count multiplication overflows."""

    rates = np.asarray(rates_hz, dtype=float)
    durations = np.asarray(dt, dtype=float)
    if rates.size == 0 or durations.size == 0:
        return
    max_rate = float(np.max(rates))
    max_duration = (
        float(durations) if durations.ndim == 0 else float(np.max(durations))
    )
    with np.errstate(over="ignore", invalid="ignore"):
        max_expected = max_rate * max_duration * spike_rate_scale
    if not np.isfinite(max_expected):
        raise ValueError("scaled expected spike counts must be finite")


def _expected_counts(
    rates_hz: np.ndarray,
    dt: float | np.ndarray,
    spike_rate_scale: float,
    n_time: int,
) -> np.ndarray:
    """Return one expected-count tensor while retaining exact zero rates."""

    durations = np.asarray(dt, dtype=float)
    if durations.ndim == 0:
        raw = rates_hz[None, :, :] * float(durations) * spike_rate_scale
    else:
        raw = (
            durations[:, None, None]
            * rates_hz[None, :, :]
            * spike_rate_scale
        )
    if raw.shape[0] == 1 and n_time != 1:
        raw = np.broadcast_to(
            raw,
            (n_time, rates_hz.shape[0], rates_hz.shape[1]),
        )
    return np.where(
        rates_hz[None, :, :] == 0.0,
        0.0,
        np.maximum(raw, np.finfo(float).tiny),
    )


def _poisson_log_terms(counts: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Return elementwise Poisson log-PMF terms."""

    return counts * np.log(mean) - mean - gammaln(counts + 1.0)


def _stable_negative_binomial_log_terms(
    counts: np.ndarray,
    expected: np.ndarray,
    negative_binomial_overdispersion: float,
) -> np.ndarray:
    """Evaluate the mean/overdispersion negative-binomial log PMF stably.

    Writing the combinatorial coefficient as a difference of two ``gammaln``
    values loses all significant digits when ``1 / overdispersion`` is much
    larger than the observed count.  The beta-function identity

    ``Gamma(r + k) / (Gamma(r) Gamma(k + 1)) = 1 / (k B(r, k))``

    avoids that subtraction.  Log-add-exp forms keep the success and failure
    probabilities stable at both very small and very large overdispersion.
    When the reciprocal shape itself overflows, the distribution is
    numerically indistinguishable from its Poisson limit.
    """

    count_values = np.asarray(counts, dtype=float)
    mean_values = np.maximum(
        np.asarray(expected, dtype=float),
        np.finfo(float).tiny,
    )
    count_values, mean_values = np.broadcast_arrays(
        count_values,
        mean_values,
    )
    overdispersion = float(negative_binomial_overdispersion)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        size = 1.0 / overdispersion
    if not np.isfinite(size):
        return _poisson_log_terms(count_values, mean_values)

    combination = np.zeros_like(mean_values, dtype=float)
    positive_counts = count_values > 0.0
    combination[positive_counts] = (
        -np.log(count_values[positive_counts])
        - betaln(size, count_values[positive_counts])
    )

    log_scaled_mean = np.log(overdispersion) + np.log(mean_values)
    log_success_probability = -np.logaddexp(0.0, log_scaled_mean)
    log_failure_probability = -np.logaddexp(0.0, -log_scaled_mean)
    return (
        combination
        + size * log_success_probability
        + count_values * log_failure_probability
    )


def _patch_negative_binomial_poisson_limit(encoding) -> None:
    """Install stable negative-binomial evaluation including its Poisson limit."""

    current = encoding._negative_binomial_log_emissions
    if getattr(current, _NEGATIVE_BINOMIAL_PATCHED_FLAG, False):
        return
    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def stable_negative_binomial_log_emissions(
        counts,
        expected,
        negative_binomial_overdispersion,
    ):
        overdispersion = float(negative_binomial_overdispersion)
        if not np.isfinite(overdispersion) or overdispersion <= 0.0:
            return original(
                counts,
                expected,
                negative_binomial_overdispersion,
            )
        return _stable_negative_binomial_log_terms(
            counts,
            expected,
            overdispersion,
        )

    setattr(
        stable_negative_binomial_log_emissions,
        _NEGATIVE_BINOMIAL_PATCHED_FLAG,
        True,
    )
    setattr(stable_negative_binomial_log_emissions, _ORIGINAL_ATTR, original)
    encoding._negative_binomial_log_emissions = (
        stable_negative_binomial_log_emissions
    )


def _restore_exact_zero_rate_support(
    log_likelihood: np.ndarray,
    *,
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    dt: float | np.ndarray,
    spike_rate_scale: float,
    cell_weights: np.ndarray,
    likelihood_temperature: float,
    negative_binomial_overdispersion: float,
) -> np.ndarray:
    """Recompute bins whose active cells include an exactly zero rate."""

    zero_rate = rates_hz == 0.0
    active_zero_rate = zero_rate & (cell_weights[:, None] > 0.0)
    affected_bins = np.any(active_zero_rate, axis=0)
    if not np.any(affected_bins):
        return log_likelihood

    counts = spike_counts[:, :, None]
    expected = _expected_counts(
        rates_hz,
        dt,
        spike_rate_scale,
        spike_counts.shape[0],
    )
    safe_expected = np.where(zero_rate[None, :, :], 1.0, expected)

    if negative_binomial_overdispersion == 0.0:
        terms = _poisson_log_terms(counts, safe_expected)
    else:
        terms = _stable_negative_binomial_log_terms(
            counts,
            safe_expected,
            negative_binomial_overdispersion,
        )
    terms = np.where(zero_rate[None, :, :], 0.0, terms)
    exact = np.einsum("tcb,c->tb", terms, cell_weights, optimize=True)
    exact /= likelihood_temperature

    impossible = np.any(
        (counts > 0.0) & active_zero_rate[None, :, :],
        axis=1,
    )
    exact[impossible] = -np.inf

    corrected = np.asarray(log_likelihood, dtype=float).copy()
    corrected[:, affected_bins] = exact[:, affected_bins]
    return corrected


def _wrapped_function_aliases(*functions: object) -> tuple[object, ...]:
    """Return wrapper-chain members whose imported aliases may be stale."""

    pending = list(functions)
    aliases: list[object] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        aliases.append(current)
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
        stored_original = getattr(current, _ORIGINAL_ATTR, None)
        if stored_original is not None and stored_original is not current:
            pending.append(stored_original)
    return tuple(aliases)


def _synchronize_poisson_log_emission_aliases(
    original: object,
    active: object,
) -> None:
    """Refresh package modules that imported any earlier Poisson wrapper."""

    stale_aliases = _wrapped_function_aliases(original, active)
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("hipporeplayimm"):
            continue
        current_alias = getattr(module, "_poisson_log_emissions", None)
        if current_alias is active:
            continue
        if any(current_alias is stale for stale in stale_aliases):
            setattr(module, "_poisson_log_emissions", active)


def apply_poisson_input_boolean_validation_patch() -> None:
    """Install input guards and exact support handling for count emissions."""

    from . import encoding

    _patch_negative_binomial_poisson_limit(encoding)

    current = encoding._poisson_log_emissions
    if getattr(current, _PATCHED_FLAG, None) == _WRAPPER_VERSION:
        _synchronize_poisson_log_emission_aliases(
            getattr(current, _ORIGINAL_ATTR, current),
            current,
        )
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
        with np.errstate(over="ignore", invalid="ignore"):
            log_likelihood = original(
                spike_counts,
                rates_hz,
                dt,
                spike_rate_scale=spike_rate_scale,
                likelihood_temperature=likelihood_temperature,
                cell_weights=reusable_weights,
                negative_binomial_overdispersion=(
                    negative_binomial_overdispersion
                ),
            )
        _reject_nonfinite_expected_count_scaling(
            rates_hz,
            dt,
            float(spike_rate_scale),
        )

        counts = np.asarray(spike_counts, dtype=float)
        rates = np.asarray(rates_hz, dtype=float)
        weights = encoding._emission_cell_weights(
            reusable_weights,
            counts.shape[1],
        )
        return _restore_exact_zero_rate_support(
            log_likelihood,
            spike_counts=counts,
            rates_hz=rates,
            dt=dt,
            spike_rate_scale=float(spike_rate_scale),
            cell_weights=weights,
            likelihood_temperature=float(likelihood_temperature),
            negative_binomial_overdispersion=float(
                negative_binomial_overdispersion
            ),
        )

    setattr(poisson_log_emissions, _PATCHED_FLAG, _WRAPPER_VERSION)
    setattr(poisson_log_emissions, _ORIGINAL_ATTR, original)
    encoding._poisson_log_emissions = poisson_log_emissions
    _synchronize_poisson_log_emission_aliases(
        original,
        poisson_log_emissions,
    )
    setattr(encoding, _PATCHED_FLAG, True)


__all__ = ["apply_poisson_input_boolean_validation_patch"]
