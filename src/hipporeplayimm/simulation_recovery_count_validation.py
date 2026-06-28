from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_COUNT_WRAPPER_ATTR = "_simulation_recovery_count_validation_wrapper"
_FINITE_SCALAR_WRAPPER_ATTR = "_simulation_recovery_positive_finite_scalar_validation_wrapper"
_LATENT_PATH_WRAPPER_ATTR = "_simulation_recovery_latent_path_n_time_validation_wrapper"
_REPLAY_EVENT_WRAPPER_ATTR = "_simulation_recovery_replay_event_n_time_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_simulation_recovery_count_validation_patch() -> None:
    from . import simulation_recovery

    _patch_positive_finite_scalar(simulation_recovery)
    _patch_simulate_latent_path(simulation_recovery)
    _patch_simulate_replay_event(simulation_recovery)
    _patch_emissions_from_counts(simulation_recovery)
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


def _validated_count_matrix(counts: Any, *, n_cells: int) -> np.ndarray:
    if _contains_boolean_values(counts):
        raise ValueError("counts must contain numeric integer counts, not boolean values")
    try:
        values = np.asarray(counts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc

    if values.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    if values.shape[1] != int(n_cells):
        raise ValueError("counts columns must match encoding.n_cells")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    rounded = np.rint(values)
    if not np.all(np.isclose(values, rounded, rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    return np.asarray(rounded, dtype=int)


def _contains_boolean_values(values: Any) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _positive_integer_scalar(name: str, value: Any) -> int:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be an integer, not boolean")
    item = _scalar_item(value)
    try:
        numeric = float(item)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


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
