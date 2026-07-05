"""Validate clusterless mark-group identifiers before integer coercion.

Clusterless grouping keys are identifiers, not boolean flags.  They are often
stored as floating-point MATLAB values, so integral floats remain supported, but
malformed boolean, fractional, non-finite, or out-of-range values must be rejected
before NumPy integer casts can alias them to unrelated groups or force a silent
fallback to the global mark likelihood.  Row-count validation is part of the same
contract: group/cell identifiers must stay aligned with the mark rows before they
are used as boolean masks during clusterless encoding.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
import sys
from typing import Any

import numpy as np
from scipy.special import gammaln

_PATCHED_FLAG = "_clusterless_mark_group_validation_patch_applied"
_WRAPPER_MARKER = "_clusterless_mark_group_validation_wrapper"
_BUILD_EMISSIONS_WRAPPER_MARKER = "_clusterless_mark_group_build_emissions_wrapper"


def _mark_group_guard(wrapper):
    setattr(wrapper, _WRAPPER_MARKER, True)
    return wrapper


def _mark_build_emissions_guard(wrapper):
    setattr(wrapper, _BUILD_EMISSIONS_WRAPPER_MARKER, True)
    return wrapper


def _is_current_group_guard(value: object) -> bool:
    return bool(getattr(value, _WRAPPER_MARKER, False))


def _is_current_build_emissions_guard(value: object) -> bool:
    return bool(getattr(value, _BUILD_EMISSIONS_WRAPPER_MARKER, False))


def _wrappers_are_current(clusterless) -> bool:
    try:
        return (
            _is_current_group_guard(clusterless._mark_group_ids_for_config)
            and _is_current_group_guard(clusterless.ClusterlessMarkEncoding._coerce_group_indices)
            and _is_current_build_emissions_guard(clusterless.build_clusterless_mark_emissions)
        )
    except AttributeError:
        return False


def _contains_boolean_ids(values: Any) -> bool:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        raw = np.asarray(values)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _coerce_integral_group_ids(
    values: Any,
    name: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Return integer group IDs without lossy bool/fraction/range coercion."""

    raw = np.asarray(values, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    else:
        raw = raw.reshape(-1)
    if expected_size is not None and raw.shape[0] != int(expected_size):
        raise ValueError(
            f"{name} must contain one value per spike mark row; "
            f"expected {int(expected_size)}, got {raw.shape[0]}"
        )
    if _contains_boolean_ids(raw):
        raise ValueError(f"{name} must not contain boolean identifiers")
    integer_info = np.iinfo(np.dtype(int))
    coerced = [
        _coerce_integral_group_id(value, name, integer_info)
        for value in raw
    ]
    return np.asarray(coerced, dtype=int)


def _coerce_integral_group_id(value: Any, name: str, integer_info: np.iinfo) -> int:
    """Coerce one group identifier without sending integer inputs through float."""

    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=object)
        if arr.ndim != 0:
            raise ValueError(f"{name} must be one-dimensional")
        value = arr.item()

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean identifiers")

    if isinstance(value, (int, np.integer)):
        identifier = int(value)
    elif isinstance(value, Decimal):
        identifier = _coerce_decimal_group_id(value, name)
    elif isinstance(value, (str, bytes)):
        identifier = _coerce_text_group_id(value, name)
    elif isinstance(value, (float, np.floating)):
        identifier = _coerce_float_group_id(float(value), name)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite integer identifiers") from exc
        identifier = _coerce_float_group_id(numeric, name)

    if identifier < int(integer_info.min) or identifier > int(integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return identifier


def _coerce_float_group_id(value: float, name: str) -> int:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite integer identifiers")
    if not value.is_integer():
        raise ValueError(f"{name} must be integer-valued")
    return int(value)


def _coerce_decimal_group_id(value: Decimal, name: str) -> int:
    if not value.is_finite():
        raise ValueError(f"{name} must be finite integer identifiers")
    integer = value.to_integral_value()
    if value != integer:
        raise ValueError(f"{name} must be integer-valued")
    return int(integer)


def _coerce_text_group_id(value: str | bytes, name: str) -> int:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must be finite integer identifiers") from exc
    else:
        text = value
    text = text.strip()
    if not text:
        raise ValueError(f"{name} must be finite integer identifiers")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return _coerce_decimal_group_id(Decimal(text), name)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be finite integer identifiers") from exc


def _preserve_group_ids(group_ids: Any, mask: np.ndarray) -> np.ndarray:
    """Filter group IDs without converting them through integer dtype first."""

    raw = np.asarray(group_ids, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    return raw.reshape(-1)[mask]


def _synchronize_build_emission_aliases(previous: object, patched: object) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "build_clusterless_mark_emissions", None) is previous:
            module.build_clusterless_mark_emissions = patched


def apply_clusterless_mark_group_validation_patch() -> None:
    """Install strict validation for clusterless mark group IDs."""

    from . import clusterless

    if getattr(clusterless, _PATCHED_FLAG, False) and _wrappers_are_current(clusterless):
        return

    @_mark_group_guard
    def mark_group_ids_for_config(session, config):
        marks = session.spike_marks
        if marks is None:
            raise ValueError("Session does not contain spike marks.")
        expected_size = int(marks.n_spikes)
        group_by = clusterless._normalize_mark_group_by(config.mark_group_by)
        if group_by == "none":
            return None
        if group_by == "cell":
            return None if marks.cell_ids is None else _coerce_integral_group_ids(
                marks.cell_ids,
                "clusterless cell group IDs",
                expected_size=expected_size,
            )
        if group_by == "tetrode":
            if marks.group_ids is None:
                raise ValueError("clusterless mark grouping by tetrode requires spike-mark group IDs from Tetrode_Cell_IDs")
            return _coerce_integral_group_ids(
                marks.group_ids,
                "clusterless tetrode group IDs",
                expected_size=expected_size,
            )
        if marks.group_ids is not None:
            return _coerce_integral_group_ids(
                marks.group_ids,
                "clusterless mark group IDs",
                expected_size=expected_size,
            )
        return None

    @_mark_group_guard
    def coerce_group_indices(self, group_ids, n_marks: int):
        if group_ids is None or self.group_ids is None:
            return None
        raw_group_ids = np.asarray(group_ids, dtype=object)
        if raw_group_ids.ndim == 0:
            raw_group_ids = np.full(int(n_marks), raw_group_ids.item(), dtype=object if raw_group_ids.dtype == object else raw_group_ids.dtype)
        raw_group_ids = raw_group_ids.reshape(-1)
        coerced = _coerce_integral_group_ids(
            raw_group_ids,
            "mark group IDs",
            expected_size=int(n_marks),
        )
        encoding_group_ids = _coerce_integral_group_ids(self.group_ids, "encoding mark group IDs")
        sorted_order = np.argsort(encoding_group_ids)
        sorted_groups = encoding_group_ids[sorted_order]
        positions = np.searchsorted(sorted_groups, coerced)
        in_bounds = positions < sorted_groups.shape[0]
        matches = np.zeros(int(n_marks), dtype=bool)
        matches[in_bounds] = sorted_groups[positions[in_bounds]] == coerced[in_bounds]
        group_indices = np.full(int(n_marks), -1, dtype=int)
        group_indices[matches] = sorted_order[positions[matches]]
        return group_indices

    current_build = clusterless.build_clusterless_mark_emissions
    if _is_current_build_emissions_guard(current_build):
        build_clusterless_mark_emissions = current_build
    else:
        @_mark_build_emissions_guard
        @wraps(current_build)
        def build_clusterless_mark_emissions(session, encoding, ripple, config=None):
            config = clusterless.EmissionConfig() if config is None else config
            if not np.isfinite(config.spike_rate_scale) or config.spike_rate_scale <= 0.0:
                raise ValueError("spike_rate_scale must be positive and finite")
            clusterless._validate_emission_calibration(
                likelihood_temperature=config.likelihood_temperature,
                negative_binomial_overdispersion=config.negative_binomial_overdispersion,
            )
            if config.cell_weights is not None:
                raise ValueError(
                    "cell_weights are only supported for sorted-spike emissions; "
                    "use likelihood_temperature to calibrate clusterless emissions"
                )
            if config.negative_binomial_overdispersion > 0.0:
                raise ValueError("negative_binomial_overdispersion is only implemented for sorted-spike emissions")
            marks = session.spike_marks
            if marks is None or marks.n_features == 0:
                raise ValueError("Session does not contain spike marks for clusterless emission scoring.")
            ripple_event = clusterless._coerce_ripple_event(session, ripple)
            edges = clusterless._time_bin_edges(ripple_event.start, ripple_event.end, config.time_bin_s)
            bin_durations = np.diff(edges)
            times = edges[:-1] + 0.5 * bin_durations
            dt = float(np.median(bin_durations))
            counts = np.zeros(times.shape[0], dtype=int)
            scaled_rate_hz = np.maximum(
                encoding.rate_hz * float(config.spike_rate_scale),
                np.finfo(float).tiny,
            )
            log_likelihood = -scaled_rate_hz[None, :] * bin_durations[:, None]

            mark_times, mark_values, mark_group_ids = clusterless._marks_for_config(session, encoding.config)
            keep = (
                (mark_times >= ripple_event.start)
                & (mark_times < ripple_event.end)
                & np.all(np.isfinite(mark_values), axis=1)
            )
            if np.any(keep):
                mark_times = mark_times[keep]
                mark_values = mark_values[keep]
                if mark_group_ids is not None:
                    mark_group_ids = _preserve_group_ids(mark_group_ids, keep)
                time_bins = np.searchsorted(edges, mark_times, side="right") - 1
                valid = (time_bins >= 0) & (time_bins < counts.shape[0])
                time_bins = time_bins[valid].astype(int)
                mark_values = mark_values[valid]
                if mark_group_ids is not None:
                    mark_group_ids = _preserve_group_ids(mark_group_ids, valid)
                log_rate = np.log(scaled_rate_hz)
                mark_log_likelihood = encoding.log_mark_likelihood(mark_values, mark_group_ids)
                group_indices = encoding._coerce_group_indices(mark_group_ids, mark_values.shape[0]) if mark_group_ids is not None else None
                for local_index, time_bin in enumerate(time_bins):
                    local_log_rate = log_rate
                    if group_indices is not None and encoding.group_rate_hz is not None and group_indices[local_index] >= 0:
                        local_log_rate = np.log(
                            np.maximum(encoding.group_rate_hz[int(group_indices[local_index])] * float(config.spike_rate_scale), np.finfo(float).tiny)
                        )
                    log_likelihood[time_bin] += local_log_rate + mark_log_likelihood[local_index]
                np.add.at(counts, time_bins, 1)
            log_likelihood += (counts * np.log(bin_durations) - gammaln(counts + 1))[:, None]
            log_likelihood = clusterless._apply_likelihood_temperature(log_likelihood, config.likelihood_temperature)

            return clusterless.LogEmissionTensor(
                log_likelihood=log_likelihood,
                spike_counts=counts[:, None],
                times=times,
                dt=dt,
                cell_ids=np.array([0], dtype=int),
                n_spikes=int(counts.sum()),
                bin_durations=bin_durations,
                transition_durations=np.diff(times) if times.shape[0] > 1 else np.empty(0, dtype=float),
                metadata={
                    "clusterless_mark_likelihood": encoding.mark_likelihood,
                    "clusterless_mark_kde_bandwidth": clusterless._format_float_array(clusterless._sqrt_optional(encoding.mark_kde_variance)),
                    "clusterless_mark_kde_max_neighbors": clusterless._kde_neighbor_count(encoding),
                    "clusterless_mark_group_by": clusterless._normalize_mark_group_by(encoding.config.mark_group_by),
                    "clusterless_mark_groups": encoding.n_mark_groups,
                },
            )

        setattr(build_clusterless_mark_emissions, "__hipporeplayimm_original__", current_build)
        clusterless.build_clusterless_mark_emissions = build_clusterless_mark_emissions
        _synchronize_build_emission_aliases(current_build, build_clusterless_mark_emissions)

    clusterless._mark_group_ids_for_config = mark_group_ids_for_config
    clusterless.ClusterlessMarkEncoding._coerce_group_indices = coerce_group_indices
    setattr(clusterless, _PATCHED_FLAG, True)
