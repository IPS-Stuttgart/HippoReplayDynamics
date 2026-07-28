from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_COUNT_WRAPPER_ATTR = "_simulation_recovery_count_validation_wrapper"
_FINITE_SCALAR_WRAPPER_ATTR = "_simulation_recovery_positive_finite_scalar_validation_wrapper"
_LATENT_PATH_WRAPPER_ATTR = "_simulation_recovery_latent_path_n_time_validation_wrapper"
_REPLAY_EVENT_WRAPPER_ATTR = "_simulation_recovery_replay_event_n_time_validation_wrapper"
_PRIOR_WRAPPER_ATTR = "_simulation_recovery_valid_bins_prior_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_simulation_recovery_count_validation_patch() -> None:
    from . import simulation_recovery

    _patch_positive_finite_scalar(simulation_recovery)
    _patch_simulate_latent_path(simulation_recovery)
    _patch_simulate_replay_event(simulation_recovery)
    _patch_emissions_from_counts(simulation_recovery)
    _patch_valid_bins_and_prior(simulation_recovery)
    simulation_recovery._count_validation_patch_applied = True


def _patch_positive_finite_scalar(simulation_recovery: Any) -> None:
    current = simulation_recovery._positive_finite_scalar
    if getattr(current, _FINITE_SCALAR_WRAPPER_ATTR, False):
        return

    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def positive_finite_scalar_with_boolean_guard(name: str, value: Any) -> float:
        if _is_boolean_scalar(value):
            raise TypeError(f"{name} must be numeric, not boolean")
        if _contains_text_values(value):
            raise TypeError(f"{name} must be numeric, not text")
        return original(name, value)

    setattr(positive_finite_scalar_with_boolean_guard, _FINITE_SCALAR_WRAPPER_ATTR, True)
    setattr(positive_finite_scalar_with_boolean_guard, _ORIGINAL_ATTR, original)
    simulation_recovery._positive_finite_scalar = positive_finite_scalar_with_boolean_guard


def _patch_simulate_latent_path(simulation_recovery: Any) -> None:
    current = simulation_recovery.simulate_latent_path
    if getattr(current, _LATENT_PATH_WRAPPER_ATTR, False):
        return

    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def simulate_latent_path_with_validated_n_time(
        encoding: Any,
        *,
        true_model: str,
        n_time: Any,
        dt: Any,
        rng: Any,
        state_space: Any = None,
    ):
        return original(
            encoding,
            true_model=true_model,
            n_time=_positive_integer_scalar("n_time", n_time),
            dt=dt,
            rng=rng,
            state_space=state_space,
        )

    setattr(simulate_latent_path_with_validated_n_time, _LATENT_PATH_WRAPPER_ATTR, True)
    setattr(simulate_latent_path_with_validated_n_time, _ORIGINAL_ATTR, original)
    simulation_recovery.simulate_latent_path = simulate_latent_path_with_validated_n_time


def _patch_simulate_replay_event(simulation_recovery: Any) -> None:
    current = simulation_recovery.simulate_replay_event
    if getattr(current, _REPLAY_EVENT_WRAPPER_ATTR, False):
        return

    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def simulate_replay_event_with_validated_n_time(
        encoding: Any,
        *,
        true_model: str,
        n_time: Any,
        dt: Any,
        rng: Any,
        spike_rate_scale: float = 1.0,
        likelihood_temperature: float = 1.0,
        negative_binomial_overdispersion: float = 0.0,
        state_space: Any = None,
    ):
        return original(
            encoding,
            true_model=true_model,
            n_time=_positive_integer_scalar("n_time", n_time),
            dt=dt,
            rng=rng,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
            state_space=state_space,
        )

    setattr(simulate_replay_event_with_validated_n_time, _REPLAY_EVENT_WRAPPER_ATTR, True)
    setattr(simulate_replay_event_with_validated_n_time, _ORIGINAL_ATTR, original)
    simulation_recovery.simulate_replay_event = simulate_replay_event_with_validated_n_time


def _patch_emissions_from_counts(simulation_recovery: Any) -> None:
    current = simulation_recovery.emissions_from_counts
    if getattr(current, _COUNT_WRAPPER_ATTR, False):
        return

    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def emissions_from_counts_with_validated_counts(
        encoding: Any,
        counts: Any,
        *,
        dt: float,
        spike_rate_scale: float = 1.0,
        likelihood_temperature: float = 1.0,
        negative_binomial_overdispersion: float = 0.0,
    ):
        validated_counts = _validated_count_matrix(
            counts,
            n_cells=int(getattr(encoding, "n_cells")),
        )
        return original(
            encoding,
            validated_counts,
            dt=dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )

    setattr(emissions_from_counts_with_validated_counts, _COUNT_WRAPPER_ATTR, True)
    setattr(emissions_from_counts_with_validated_counts, _ORIGINAL_ATTR, original)
    simulation_recovery.emissions_from_counts = emissions_from_counts_with_validated_counts


def _patch_valid_bins_and_prior(simulation_recovery: Any) -> None:
    current = simulation_recovery._valid_bins_and_prior
    if getattr(current, _PRIOR_WRAPPER_ATTR, False):
        return

    original = getattr(current, _ORIGINAL_ATTR, current)

    @wraps(original)
    def valid_bins_and_prior_with_validated_occupancy(encoding: Any):
        _validated_occupancy_vector(encoding)
        return original(encoding)

    setattr(valid_bins_and_prior_with_validated_occupancy, _PRIOR_WRAPPER_ATTR, True)
    setattr(valid_bins_and_prior_with_validated_occupancy, _ORIGINAL_ATTR, original)
    simulation_recovery._valid_bins_and_prior = valid_bins_and_prior_with_validated_occupancy


def _validated_count_matrix(counts: Any, *, n_cells: int) -> np.ndarray:
    if _contains_boolean_values(counts):
        raise ValueError("counts must contain numeric integer counts, not boolean values")
    if _contains_text_values(counts):
        raise ValueError("counts must contain numeric integer counts, not text values")
    try:
        raw_values = np.asarray(counts)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc

    if raw_values.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    if raw_values.shape[1] != int(n_cells):
        raise ValueError("counts columns must match encoding.n_cells")

    integer_info = np.iinfo(np.dtype(int))
    exact_counts = _exact_integer_count_array(raw_values, max_count=int(integer_info.max))
    if exact_counts is not None:
        return exact_counts

    try:
        values = raw_values.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("counts must contain numeric values") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    rounded = np.rint(values)
    if not np.all(np.isclose(values, rounded, rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    max_safe_float = np.nextafter(float(integer_info.max), 0.0)
    if np.any(rounded > max_safe_float):
        raise ValueError("counts must fit into integer count range")
    return np.asarray(rounded, dtype=int)


def _exact_integer_count_array(values: np.ndarray, *, max_count: int) -> np.ndarray | None:
    if np.issubdtype(values.dtype, np.integer):
        if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
            raise ValueError("counts must contain finite nonnegative values")
        if np.any(values > max_count):
            raise ValueError("counts must fit into integer count range")
        return values.astype(int, copy=False)
    if values.dtype != object:
        return None

    exact = np.empty(values.shape, dtype=int)
    for index, item in np.ndenumerate(values):
        try:
            integer_value = operator.index(item)
        except TypeError:
            try:
                integer_value = int(item)
            except (TypeError, ValueError, OverflowError):
                return None
            try:
                is_exact = item == integer_value
            except (TypeError, ValueError):
                return None
            if not isinstance(is_exact, (bool, np.bool_)) or not bool(is_exact):
                raise ValueError("counts must contain integer-valued counts")
        if integer_value < 0:
            raise ValueError("counts must contain finite nonnegative values")
        if integer_value > max_count:
            return None
        exact[index] = int(integer_value)
    return exact


def _validated_occupancy_vector(encoding: Any) -> np.ndarray:
    try:
        n_bins_raw = getattr(encoding, "n_bins")
    except AttributeError as exc:
        raise ValueError("encoding.n_bins is required") from exc
    n_bins = _positive_integer_scalar("encoding.n_bins", n_bins_raw)

    try:
        occupancy_raw = getattr(encoding, "occupancy_s")
    except AttributeError as exc:
        raise ValueError("encoding.occupancy_s is required") from exc
    if _contains_boolean_values(occupancy_raw):
        raise ValueError("occupancy_s must contain finite nonnegative values")
    if _contains_text_values(occupancy_raw):
        raise ValueError("occupancy_s must contain finite nonnegative values, not text values")
    try:
        occupancy = np.asarray(occupancy_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("occupancy_s must contain finite nonnegative values") from exc

    if occupancy.ndim != 1:
        raise ValueError("occupancy_s must be a one-dimensional vector")
    if occupancy.shape[0] != n_bins:
        raise ValueError("occupancy_s length must match encoding.n_bins")
    if not np.all(np.isfinite(occupancy)) or np.any(occupancy < 0.0):
        raise ValueError("occupancy_s must contain finite nonnegative values")
    return occupancy


def _contains_boolean_values(values: Any) -> bool:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))


def _contains_text_values(values: Any) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if raw.dtype.kind in {"U", "S"}:
        return True
    if raw.dtype == object:
        return any(isinstance(value, (str, bytes, np.str_, np.bytes_)) for value in raw.reshape(-1))
    return False


def _positive_integer_scalar(name: str, value: Any) -> int:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be an integer, not boolean")
    if _contains_text_values(value):
        raise ValueError(f"{name} must be a positive integer, not text")
    item = _strict_scalar_item(name, value)
    try:
        numeric = float(item)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _strict_scalar_item(name: str, value: Any) -> Any:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if array.shape != ():
        raise ValueError(f"{name} must be a positive integer")
    try:
        return array.item()
    except (AttributeError, IndexError, ValueError):
        return value


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    item = _scalar_item(value)
    return isinstance(item, (bool, np.bool_))


def _scalar_item(value: Any) -> Any:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if array.size != 1:
        return value
    try:
        return array.reshape(-1)[0].item()
    except (AttributeError, IndexError, ValueError):
        return value
