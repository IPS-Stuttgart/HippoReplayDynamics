"""Validate Poisson inputs and preserve exact zero-rate likelihood support."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
import operator
import sys

import numpy as np
from scipy.special import betaln, gammaln


_PATCHED_FLAG = "_poisson_input_boolean_validation_patch_applied"
_NEGATIVE_BINOMIAL_PATCHED_FLAG = (
    "_negative_binomial_poisson_limit_patch_applied"
)
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_WRAPPER_VERSION = 6


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


def _contains_complex_values(value: object) -> bool:
    """Return True for complex arrays, including nested object scalar wrappers."""

    pending = [value]
    seen_objects: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, (complex, np.complexfloating)):
            return True
        marker = id(current)
        if marker in seen_objects:
            continue
        seen_objects.add(marker)
        try:
            array = np.asarray(current)
        except (TypeError, ValueError):
            continue
        if np.issubdtype(array.dtype, np.complexfloating):
            return True
        if array.dtype != object:
            continue
        if array.ndim == 0:
            try:
                item = array.item()
            except (TypeError, ValueError):
                continue
            if item is not current:
                pending.append(item)
            continue
        pending.extend(array.reshape(-1))
    return False


def _reject_complex_array(name: str, value: object) -> None:
    if _contains_complex_values(value):
        raise ValueError(f"{name} must contain real numeric values, not complex values")


def _coerce_text_spike_count(value: str | bytes) -> int:
    """Parse one integral textual spike count without binary-float rounding."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("spike_counts must contain numeric counts") from exc
    else:
        text = value
    text = text.strip()
    if not text:
        raise ValueError("spike_counts must contain numeric counts")
    try:
        return int(text, 10)
    except ValueError:
        try:
            numeric = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("spike_counts must contain numeric counts") from exc
        if not numeric.is_finite() or numeric < 0:
            raise ValueError("spike_counts must be finite and nonnegative")
        integral = numeric.to_integral_value()
        if numeric != integral:
            raise ValueError("spike_counts must contain integer counts")
        return int(integral)


def _coerce_exact_spike_count(value: object, integer_info: np.iinfo) -> int:
    """Return one exact nonnegative platform-integer spike count."""

    try:
        item = np.asarray(value).item()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spike_counts must contain numeric counts") from exc

    if isinstance(item, (bool, np.bool_)):
        raise ValueError(
            "spike_counts must contain numeric counts, not boolean values"
        )
    if isinstance(item, (int, np.integer)):
        count = int(item)
    elif isinstance(item, Decimal):
        if not item.is_finite() or item < 0:
            raise ValueError("spike_counts must be finite and nonnegative")
        integral = item.to_integral_value()
        if item != integral:
            raise ValueError("spike_counts must contain integer counts")
        count = int(integral)
    elif isinstance(item, (str, bytes, np.str_, np.bytes_)):
        text_value = (
            bytes(item) if isinstance(item, (bytes, np.bytes_)) else str(item)
        )
        count = _coerce_text_spike_count(text_value)
    elif isinstance(item, (complex, np.complexfloating)):
        raise ValueError("spike_counts must contain real integer counts")
    elif isinstance(item, (float, np.floating)):
        if not bool(np.isfinite(item)) or item < 0:
            raise ValueError("spike_counts must be finite and nonnegative")
        if not bool(item.is_integer()):
            raise ValueError("spike_counts must contain integer counts")
        count = int(item)
    else:
        try:
            count = int(operator.index(item))
        except TypeError:
            try:
                count = int(item)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("spike_counts must contain numeric counts") from exc
            try:
                exact = item == count
            except (TypeError, ValueError, OverflowError):
                exact = False
            if not isinstance(exact, (bool, np.bool_)) or not bool(exact):
                raise ValueError("spike_counts must contain integer counts")

    if count < 0:
        raise ValueError("spike_counts must be finite and nonnegative")
    if count > int(integer_info.max):
        raise ValueError("spike_counts must fit into integer count range")
    return count


def _coerce_spike_counts_exact(spike_counts: object) -> np.ndarray:
    """Validate spike-count matrices exactly before any binary-float conversion."""

    try:
        raw_counts = np.asarray(spike_counts)
    except (TypeError, ValueError) as exc:
        raise ValueError("spike_counts must contain numeric counts") from exc
    if raw_counts.ndim != 2:
        raise ValueError("spike_counts must be two-dimensional")
    if np.issubdtype(raw_counts.dtype, np.bool_) or (
        raw_counts.dtype == object and _contains_boolean_values(raw_counts)
    ):
        raise ValueError(
            "spike_counts must contain numeric counts, not boolean values"
        )
    if np.issubdtype(raw_counts.dtype, np.complexfloating):
        raise ValueError("spike_counts must contain real integer counts")

    integer_info = np.iinfo(np.dtype(int))
    if np.issubdtype(raw_counts.dtype, np.integer):
        if np.issubdtype(raw_counts.dtype, np.signedinteger) and np.any(
            raw_counts < 0
        ):
            raise ValueError("spike_counts must be finite and nonnegative")
        if raw_counts.size and int(np.max(raw_counts)) > int(integer_info.max):
            raise ValueError("spike_counts must fit into integer count range")
        return raw_counts.astype(int, copy=False)

    if np.issubdtype(raw_counts.dtype, np.floating):
        if not np.all(np.isfinite(raw_counts)) or np.any(raw_counts < 0):
            raise ValueError("spike_counts must be finite and nonnegative")
        rounded = np.rint(raw_counts)
        if not np.array_equal(raw_counts, rounded):
            raise ValueError("spike_counts must contain integer counts")
        if rounded.size and int(np.max(rounded)) > int(integer_info.max):
            raise ValueError("spike_counts must fit into integer count range")
        return rounded.astype(int)

    counts = np.empty(raw_counts.shape, dtype=int)
    for index, value in np.ndenumerate(raw_counts):
        counts[index] = _coerce_exact_spike_count(value, integer_info)
    return counts


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
        module_name = getattr(module, "__name__", "")
        if module_name != "hipporeplayimm" and not module_name.startswith(
            "hipporeplayimm."
        ):
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
        exact_counts = _coerce_spike_counts_exact(spike_counts)
        _reject_boolean_array("rates_hz", rates_hz)
        _reject_complex_array("rates_hz", rates_hz)
        reusable_weights = _reusable_cell_weights(cell_weights)
        with np.errstate(over="ignore", invalid="ignore"):
            log_likelihood = original(
                exact_counts,
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

        counts = np.asarray(exact_counts, dtype=float)
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
