"""Patch replay-gain cell mapping, Gamma-Poisson validation, and continuous-time cell IDs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
import operator
from typing import Any

import numpy as np
from scipy.special import gammaln

_PATCHED_FLAG = "_accuracy_replay_gain_gamma_patch_applied"
_ESTIMATE_WRAPPER_FLAG = "_accuracy_replay_gain_gamma_estimate_wrapper"
_GAMMA_WRAPPER_FLAG = "_accuracy_replay_gain_gamma_gamma_wrapper"
_CONTINUOUS_WRAPPER_FLAG = "_accuracy_replay_gain_gamma_continuous_wrapper"


def _contains_boolean_ids(values: np.ndarray) -> bool:
    raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _coerce_integral_ids(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        return np.empty(0, dtype=int)
    if _contains_boolean_ids(raw):
        raise ValueError(f"{name} must not contain boolean identifiers")

    integer_info = np.iinfo(np.dtype(int))
    coerced = [_coerce_integral_id(value, name, integer_info) for value in raw]
    return np.asarray(coerced, dtype=int)


def _coerce_integral_id(value: Any, name: str, integer_info: np.iinfo) -> int:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim != 0:
            raise ValueError(f"{name} must be one-dimensional")
        value = array.item()

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean identifiers")
    if isinstance(value, (int, np.integer)):
        identifier = int(value)
    elif isinstance(value, Decimal):
        identifier = _coerce_decimal_id(value, name)
    elif isinstance(value, (str, bytes)):
        identifier = _coerce_text_id(value, name)
    elif isinstance(value, (float, np.floating)):
        identifier = _coerce_float_id(float(value), name)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite integer identifiers") from exc
        identifier = _coerce_float_id(numeric, name)

    if identifier < int(integer_info.min) or identifier > int(integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return identifier


def _coerce_float_id(value: float, name: str) -> int:
    if not np.isfinite(value):
        raise ValueError(f"{name} must contain finite integer identifiers")
    if not value.is_integer():
        raise ValueError(f"{name} must be integer-valued")
    return int(value)


def _coerce_decimal_id(value: Decimal, name: str) -> int:
    if not value.is_finite():
        raise ValueError(f"{name} must contain finite integer identifiers")
    integer = value.to_integral_value()
    if value != integer:
        raise ValueError(f"{name} must be integer-valued")
    return int(integer)


def _coerce_text_id(value: str | bytes, name: str) -> int:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must contain finite integer identifiers") from exc
    else:
        text = value
    text = text.strip()
    if not text:
        raise ValueError(f"{name} must contain finite integer identifiers")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return _coerce_decimal_id(Decimal(text), name)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must contain finite integer identifiers") from exc


def _coerce_spike_counts(spike_counts: Any) -> np.ndarray:
    if _contains_boolean_ids(spike_counts):
        raise ValueError("spike_counts must be numeric counts, not boolean values")
    counts = np.asarray(spike_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("spike_counts must have shape (n_time, n_cells)")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("spike_counts must be finite and nonnegative")
    rounded = np.rint(counts)
    if not np.all(np.isclose(counts, rounded, rtol=0.0, atol=1e-9)):
        raise ValueError("spike_counts must contain integer counts")
    return rounded


def _coerce_positive_matrix(values: Any, name: str) -> np.ndarray:
    if _contains_boolean_ids(values):
        raise ValueError(f"{name} must be numeric, not boolean values")
    matrix = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return matrix


def _coerce_trial_exposure(dt: Any, n_time: int, spike_rate_scale: float) -> np.ndarray:
    if _contains_boolean_ids(dt):
        raise ValueError("dt must contain finite positive durations, not boolean values")
    dt_array = np.asarray(dt, dtype=float)
    if dt_array.ndim == 0:
        duration = float(dt_array)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("dt must contain finite positive durations")
        return np.full(n_time, duration * spike_rate_scale, dtype=float)

    if dt_array.shape != (n_time,):
        raise ValueError("dt must be scalar or one duration per time bin")
    if not np.all(np.isfinite(dt_array)) or np.any(dt_array <= 0.0):
        raise ValueError("dt must contain finite positive durations")
    return dt_array * spike_rate_scale


def _coerce_event_index(event_index: Any) -> int:
    if isinstance(event_index, (bool, np.bool_)):
        raise TypeError("event index must be an integer, not boolean")
    try:
        return int(operator.index(event_index))
    except TypeError as exc:
        raise TypeError("event index must be an integer") from exc


def _estimate_replay_cell_gains_impl(session, encoding, ripple_indices, config):
    from .accuracy_upgrades import ReplayGainConfig

    config = ReplayGainConfig() if config is None else config
    cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")
    if cell_ids.shape != (encoding.n_cells,):
        raise ValueError("encoding.cell_ids must contain one ID per encoding row")
    if np.unique(cell_ids).shape[0] != cell_ids.shape[0]:
        raise ValueError("encoding.cell_ids must be unique")

    cell_to_row = {int(cell_id): row for row, cell_id in enumerate(cell_ids)}
    observed = np.zeros(encoding.n_cells, dtype=float)
    total_duration = 0.0
    spikes = np.asarray(session.spikes)

    if spikes.size and (spikes.ndim != 2 or spikes.shape[1] < 2):
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")

    for event_index in ripple_indices:
        event = session.ripple(_coerce_event_index(event_index))
        total_duration += max(float(event.end) - float(event.start), 0.0)
        if spikes.size == 0:
            continue

        spike_times = np.asarray(spikes[:, 0], dtype=float)
        spike_cell_ids_raw = np.asarray(spikes[:, 1])
        in_window = (spike_times >= float(event.start)) & (spike_times < float(event.end))
        if not np.any(in_window):
            continue

        event_cell_ids = _coerce_integral_ids(spike_cell_ids_raw[in_window], "spike cell IDs")
        keep = np.isin(event_cell_ids, cell_ids)
        if not np.any(keep):
            continue

        matched_cell_ids = event_cell_ids[keep]
        rows = np.fromiter(
            (cell_to_row[int(cell_id)] for cell_id in matched_cell_ids),
            dtype=int,
            count=matched_cell_ids.shape[0],
        )
        np.add.at(observed, rows, 1.0)

    spatial_mean_rate = np.mean(np.asarray(encoding.rates_hz, dtype=float), axis=1)
    expected = spatial_mean_rate * max(total_duration, np.finfo(float).tiny)
    gains = (observed + config.prior_observed_spikes) / (expected + config.prior_expected_spikes)
    return np.clip(gains, float(config.min_gain), float(config.max_gain))


def _gamma_poisson_predictive_log_emissions_impl(
    spike_counts,
    rate_shape,
    rate_exposure_s,
    dt,
    *,
    spike_rate_scale: float = 1.0,
) -> np.ndarray:
    counts = _coerce_spike_counts(spike_counts)
    shape = _coerce_positive_matrix(rate_shape, "rate_shape")
    exposure = _coerce_positive_matrix(rate_exposure_s, "rate_exposure_s")
    if shape.shape != exposure.shape:
        raise ValueError("rate_shape and rate_exposure_s must have matching shapes")
    if shape.ndim != 2 or shape.shape[0] != counts.shape[1]:
        raise ValueError("rate prior arrays must have shape (n_cells, n_bins)")

    if isinstance(spike_rate_scale, (bool, np.bool_)):
        raise ValueError("spike_rate_scale must be finite and positive")
    spike_rate_scale = float(spike_rate_scale)
    if not np.isfinite(spike_rate_scale) or spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be finite and positive")

    trial_exposure = _coerce_trial_exposure(dt, counts.shape[0], spike_rate_scale)
    out = np.zeros((counts.shape[0], shape.shape[1]), dtype=float)
    beta = np.maximum(exposure, np.finfo(float).tiny)
    for time_index, trial_dt in enumerate(trial_exposure):
        k = counts[time_index][:, None]
        out[time_index] = np.sum(
            gammaln(shape + k)
            - gammaln(shape)
            - gammaln(k + 1.0)
            + shape * np.log(beta / (beta + trial_dt))
            + k * np.log(trial_dt / (beta + trial_dt)),
            axis=0,
        )
    return out


def _event_spikes_for_continuous_time(session, encoding, ripple_event) -> np.ndarray:
    """Return in-window spikes with validated IDs without touching out-of-window IDs."""

    cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")
    if cell_ids.shape != (encoding.n_cells,):
        raise ValueError("encoding.cell_ids must contain one ID per encoding row")
    if np.unique(cell_ids).shape[0] != cell_ids.shape[0]:
        raise ValueError("encoding.cell_ids must be unique")

    spikes = np.asarray(session.spikes)
    if spikes.size == 0:
        return np.empty((0, 2), dtype=object)
    if spikes.ndim != 2 or spikes.shape[1] < 2:
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")

    spike_times = np.asarray(spikes[:, 0], dtype=float)
    in_window = (spike_times >= float(ripple_event.start)) & (spike_times < float(ripple_event.end))
    if not np.any(in_window):
        return np.empty((0, 2), dtype=object)

    event_cell_ids = _coerce_integral_ids(spikes[in_window, 1], "spike cell IDs")
    keep = np.isin(event_cell_ids, cell_ids)
    if not np.any(keep):
        return np.empty((0, 2), dtype=object)

    event_times = spike_times[in_window][keep]
    event_ids = event_cell_ids[keep]
    event_spikes = np.empty((event_times.shape[0], 2), dtype=object)
    event_spikes[:, 0] = event_times
    event_spikes[:, 1] = event_ids
    order = np.argsort(event_times, kind="mergesort")
    return event_spikes[order]


def _build_continuous_time_emissions_impl(accuracy_module, session, encoding, ripple, config=None):
    config = accuracy_module.ContinuousTimeEmissionConfig() if config is None else config
    if not np.isfinite(config.spike_rate_scale) or config.spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be finite and positive")
    if not np.isfinite(config.min_interval_s) or config.min_interval_s <= 0.0:
        raise ValueError("min_interval_s must be finite and positive")

    ripple_event = accuracy_module._coerce_ripple_event(session, ripple)
    start = float(ripple_event.start)
    end = float(ripple_event.end)
    if end <= start:
        raise ValueError("ripple end must be greater than ripple start")

    event_spikes = _event_spikes_for_continuous_time(session, encoding, ripple_event)
    cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")
    cell_to_col = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    rows: list[np.ndarray] = []
    durations: list[float] = []
    times: list[float] = []
    cursor = start
    spike_index = 0
    while spike_index < event_spikes.shape[0]:
        spike_time = float(event_spikes[spike_index, 0])
        dt = max(spike_time - cursor, float(config.min_interval_s))
        counts = np.zeros(encoding.n_cells, dtype=int)
        while spike_index < event_spikes.shape[0] and np.isclose(float(event_spikes[spike_index, 0]), spike_time, rtol=0.0, atol=1e-12):
            col = cell_to_col.get(int(event_spikes[spike_index, 1]))
            if col is not None:
                counts[col] += 1
            spike_index += 1
        rows.append(counts)
        durations.append(dt)
        times.append(spike_time)
        cursor = spike_time

    terminal_dt = end - cursor
    if config.include_terminal_no_spike_interval or not rows:
        rows.append(np.zeros(encoding.n_cells, dtype=int))
        durations.append(max(terminal_dt, float(config.min_interval_s)))
        times.append(end)

    spike_counts = np.vstack(rows) if rows else np.zeros((0, encoding.n_cells), dtype=int)
    durations_arr = np.asarray(durations, dtype=float)
    log_likelihood = accuracy_module._poisson_log_emissions(
        spike_counts,
        encoding.rates_hz,
        durations_arr,
        spike_rate_scale=float(config.spike_rate_scale),
    )
    times_arr = np.asarray(times, dtype=float)
    transition_durations = np.maximum(np.diff(times_arr), float(config.min_interval_s)) if times_arr.shape[0] > 1 else np.empty(0, dtype=float)
    emissions = accuracy_module.LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=spike_counts,
        times=times_arr,
        dt=float(np.median(durations_arr)) if durations_arr.size else float(config.min_interval_s),
        cell_ids=cell_ids.copy(),
        n_spikes=int(spike_counts.sum()),
        bin_durations=durations_arr,
        transition_durations=transition_durations,
    )
    emissions.metadata = {
        "emission_model": "continuous-time-binned-at-spikes",
        "continuous_time_intervals": int(emissions.n_time),
        "continuous_time_min_interval_s": float(config.min_interval_s),
    }
    return emissions


def _validate_continuous_time_spike_cell_ids(accuracy_module, session, encoding, ripple) -> None:
    cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")
    if cell_ids.shape != (encoding.n_cells,):
        raise ValueError("encoding.cell_ids must contain one ID per encoding row")
    if np.unique(cell_ids).shape[0] != cell_ids.shape[0]:
        raise ValueError("encoding.cell_ids must be unique")

    spikes = np.asarray(session.spikes)
    if spikes.size == 0:
        return
    if spikes.ndim != 2 or spikes.shape[1] < 2:
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")

    ripple_event = accuracy_module._coerce_ripple_event(session, ripple)
    spike_times = np.asarray(spikes[:, 0], dtype=float)
    in_window = (spike_times >= float(ripple_event.start)) & (spike_times < float(ripple_event.end))
    if np.any(in_window):
        _coerce_integral_ids(spikes[in_window, 1], "spike cell IDs")


def apply_accuracy_replay_gain_gamma_patch() -> None:
    """Install robust replay-gain row mapping, Gamma-Poisson guards, and continuous-time ID validation."""

    from . import accuracy_upgrades as accuracy_module

    current_estimate_replay_cell_gains = accuracy_module.estimate_replay_cell_gains
    current_gamma_poisson_predictive_log_emissions = accuracy_module.gamma_poisson_predictive_log_emissions
    current_build_continuous_time_emissions = accuracy_module.build_continuous_time_emissions
    estimate_is_current = bool(getattr(current_estimate_replay_cell_gains, _ESTIMATE_WRAPPER_FLAG, False))
    gamma_is_current = bool(getattr(current_gamma_poisson_predictive_log_emissions, _GAMMA_WRAPPER_FLAG, False))
    continuous_is_current = bool(getattr(current_build_continuous_time_emissions, _CONTINUOUS_WRAPPER_FLAG, False))
    if getattr(accuracy_module, _PATCHED_FLAG, False) and estimate_is_current and gamma_is_current and continuous_is_current:
        return

    if not estimate_is_current:

        @wraps(current_estimate_replay_cell_gains)
        def estimate_replay_cell_gains(session, encoding, ripple_indices, config=None):
            return _estimate_replay_cell_gains_impl(session, encoding, ripple_indices, config)

        setattr(estimate_replay_cell_gains, _ESTIMATE_WRAPPER_FLAG, True)
        accuracy_module.estimate_replay_cell_gains = estimate_replay_cell_gains

    if not gamma_is_current:

        @wraps(current_gamma_poisson_predictive_log_emissions)
        def gamma_poisson_predictive_log_emissions(
            spike_counts,
            rate_shape,
            rate_exposure_s,
            dt,
            *,
            spike_rate_scale: float = 1.0,
        ):
            return _gamma_poisson_predictive_log_emissions_impl(
                spike_counts,
                rate_shape,
                rate_exposure_s,
                dt,
                spike_rate_scale=spike_rate_scale,
            )

        setattr(gamma_poisson_predictive_log_emissions, _GAMMA_WRAPPER_FLAG, True)
        accuracy_module.gamma_poisson_predictive_log_emissions = gamma_poisson_predictive_log_emissions

    if not continuous_is_current:

        @wraps(current_build_continuous_time_emissions)
        def build_continuous_time_emissions(session, encoding, ripple, config=None):
            _validate_continuous_time_spike_cell_ids(
                accuracy_module,
                session,
                encoding,
                ripple,
            )
            return _build_continuous_time_emissions_impl(accuracy_module, session, encoding, ripple, config)

        setattr(build_continuous_time_emissions, _CONTINUOUS_WRAPPER_FLAG, True)
        accuracy_module.build_continuous_time_emissions = build_continuous_time_emissions

    setattr(accuracy_module, _PATCHED_FLAG, True)


__all__ = ["apply_accuracy_replay_gain_gamma_patch"]
