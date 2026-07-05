"""Validate cell identifiers used while constructing emissions."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_emission_cell_id_validation_patch_applied"
_POISSON_INPUT_PATCHED_FLAG = "_poisson_boolean_input_validation_patch_applied"
_POISSON_INPUT_WRAPPER_MARKER = "_poisson_boolean_input_validation_wrapper"
_BUILD_EMISSIONS_WRAPPER_MARKER = "_emission_cell_id_build_emissions_wrapper"
_KD_BUILD_EMISSIONS_WRAPPER_MARKER = "_emission_cell_id_kd_build_emissions_wrapper"
_SORTED_SPIKE_COUNTS_WRAPPER_MARKER = "_emission_cell_id_sorted_spike_counts_wrapper"


def _mark_wrapper(wrapper: Any, marker: str) -> Any:
    setattr(wrapper, marker, True)
    return wrapper


def _is_marked_wrapper(value: Any, marker: str) -> bool:
    return bool(getattr(value, marker, False))


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


def _reject_boolean_poisson_inputs(spike_counts: Any, rates_hz: Any) -> None:
    if _contains_boolean_values(spike_counts):
        raise ValueError("spike_counts must contain numeric integer counts, not boolean values")
    if _contains_boolean_values(rates_hz):
        raise ValueError("rates_hz must contain numeric rates, not boolean values")


def _coerce_integral_ids(values: Any, name: str) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if ids.size == 0:
        return np.empty(0, dtype=int)
    if _contains_boolean_values(ids):
        raise ValueError(f"{name} must not contain boolean identifiers")
    integer_info = np.iinfo(np.dtype(int))
    return np.asarray(
        [_coerce_integral_id(value, name, integer_info) for value in ids.reshape(-1)],
        dtype=int,
    )


def _coerce_integral_id(value: Any, name: str, integer_info: np.iinfo) -> int:
    try:
        item = np.asarray(value).item()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer identifiers") from exc
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean identifiers")
    if isinstance(item, (int, np.integer)):
        identifier = int(item)
    else:
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite integer identifiers") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must contain finite integer identifiers")
        if not numeric.is_integer():
            raise ValueError(f"{name} must be integer-valued")
        identifier = int(numeric)
    if identifier < int(integer_info.min) or identifier > int(integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return identifier


def _cell_id_row_indices(cell_ids: np.ndarray, spike_cell_ids: np.ndarray) -> np.ndarray:
    """Map cell IDs to encoding rows without lossy integer casts."""

    available = _coerce_integral_ids(cell_ids, "encoding.cell_ids")
    if np.unique(available).shape[0] != available.shape[0]:
        raise ValueError("encoding.cell_ids must be unique")
    requested = _coerce_integral_ids(spike_cell_ids, "spike cell IDs")
    row_by_cell_id = {int(cell_id): index for index, cell_id in enumerate(available)}
    return np.fromiter(
        (row_by_cell_id.get(int(cell_id), -1) for cell_id in requested),
        dtype=int,
        count=requested.shape[0],
    )


def _validate_session_cell_ids(session: Any, encoding: Any, ripple_event: Any, *, source: str) -> None:
    encoding_cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")
    spikes = np.asarray(session.spikes)
    if spikes.size == 0 or encoding_cell_ids.size == 0:
        return
    if spikes.ndim != 2 or spikes.shape[1] < 2:
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")
    in_window = (spikes[:, 0] >= ripple_event.start) & (spikes[:, 0] < ripple_event.end)
    if np.any(in_window):
        _coerce_integral_ids(spikes[in_window, 1], source)


def _validate_edge_window_cell_ids(session: Any, encoding: Any, edges: Any) -> None:
    encoding_cell_ids = _coerce_integral_ids(encoding.cell_ids, "encoding.cell_ids")
    spikes = np.asarray(session.spikes)
    if spikes.size == 0 or encoding_cell_ids.size == 0:
        return
    if spikes.ndim != 2 or spikes.shape[1] < 2:
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")
    edge_values = np.asarray(edges, dtype=float)
    if edge_values.ndim == 1 and edge_values.shape[0] >= 2:
        in_window = (spikes[:, 0] >= edge_values[0]) & (spikes[:, 0] < edge_values[-1])
        if np.any(in_window):
            _coerce_integral_ids(spikes[in_window, 1], "spike cell IDs")


def _encoding_build_emissions_patch_current(encoding_module: Any) -> bool:
    return _is_marked_wrapper(getattr(encoding_module, "build_emissions", None), _BUILD_EMISSIONS_WRAPPER_MARKER)


def _kd_cell_id_patch_current(kd_module: Any) -> bool:
    return _is_marked_wrapper(getattr(kd_module, "build_kd_emissions", None), _KD_BUILD_EMISSIONS_WRAPPER_MARKER)


def _extensions_cell_id_patch_current(extensions_module: Any) -> bool:
    return _is_marked_wrapper(
        getattr(extensions_module, "_sorted_spike_counts_for_edges", None),
        _SORTED_SPIKE_COUNTS_WRAPPER_MARKER,
    )


def _apply_poisson_input_validation_patch(encoding_module: Any, kd_module: Any) -> None:
    if not _is_marked_wrapper(getattr(encoding_module, "_validate_poisson_inputs", None), _POISSON_INPUT_WRAPPER_MARKER):
        original_validate_poisson_inputs = encoding_module._validate_poisson_inputs

        @wraps(original_validate_poisson_inputs)
        def validate_poisson_inputs(spike_counts, rates_hz):
            _reject_boolean_poisson_inputs(spike_counts, rates_hz)
            return original_validate_poisson_inputs(spike_counts, rates_hz)

        _mark_wrapper(validate_poisson_inputs, _POISSON_INPUT_WRAPPER_MARKER)
        setattr(validate_poisson_inputs, _POISSON_INPUT_PATCHED_FLAG, True)
        encoding_module._validate_poisson_inputs = validate_poisson_inputs

    setattr(encoding_module, _POISSON_INPUT_PATCHED_FLAG, True)
    kd_module._validate_poisson_inputs = encoding_module._validate_poisson_inputs
    setattr(kd_module, _POISSON_INPUT_PATCHED_FLAG, True)


def apply_emission_cell_id_validation_patch() -> None:
    """Install integral-ID validation for emission row lookups."""

    from . import emission_timing_validation as timing_validation
    from . import encoding as encoding_module
    from . import kd_reference as kd_module
    from . import log_emission_n_spikes_validation as n_spikes_validation
    from . import result_improvement_extensions as extensions_module

    _apply_poisson_input_validation_patch(encoding_module, kd_module)
    timing_validation.apply_emission_timing_validation_patch()
    n_spikes_validation.apply_log_emission_n_spikes_validation_patch()

    if not _encoding_build_emissions_patch_current(encoding_module):
        original_build_emissions = encoding_module.build_emissions

        @wraps(original_build_emissions)
        def build_emissions(session, encoding, ripple, config=None):
            ripple_event = encoding_module._coerce_ripple_event(session, ripple)
            _validate_session_cell_ids(session, encoding, ripple_event, source="spike cell IDs")
            return original_build_emissions(session, encoding, ripple, config)

        _mark_wrapper(build_emissions, _BUILD_EMISSIONS_WRAPPER_MARKER)
        setattr(build_emissions, _PATCHED_FLAG, True)
        encoding_module.build_emissions = build_emissions

    encoding_module._cell_id_row_indices = _cell_id_row_indices
    setattr(encoding_module, _PATCHED_FLAG, True)

    if not _kd_cell_id_patch_current(kd_module):
        original_build_kd_emissions = kd_module.build_kd_emissions

        @wraps(original_build_kd_emissions)
        def build_kd_emissions(session, encoding, ripple, time_bin_s, spike_rate_scale=1.0):
            ripple_event = kd_module._coerce_ripple_event(session, ripple)
            _validate_session_cell_ids(session, encoding, ripple_event, source="spike cell IDs")
            return original_build_kd_emissions(
                session,
                encoding,
                ripple,
                time_bin_s,
                spike_rate_scale=spike_rate_scale,
            )

        _mark_wrapper(build_kd_emissions, _KD_BUILD_EMISSIONS_WRAPPER_MARKER)
        setattr(build_kd_emissions, _PATCHED_FLAG, True)
        kd_module.build_kd_emissions = build_kd_emissions

    setattr(kd_module, _PATCHED_FLAG, True)

    if not _extensions_cell_id_patch_current(extensions_module):
        original_sorted_spike_counts_for_edges = extensions_module._sorted_spike_counts_for_edges

        @wraps(original_sorted_spike_counts_for_edges)
        def _sorted_spike_counts_for_edges(session, encoding, edges):
            _validate_edge_window_cell_ids(session, encoding, edges)
            return original_sorted_spike_counts_for_edges(session, encoding, edges)

        _mark_wrapper(_sorted_spike_counts_for_edges, _SORTED_SPIKE_COUNTS_WRAPPER_MARKER)
        setattr(_sorted_spike_counts_for_edges, _PATCHED_FLAG, True)
        extensions_module._sorted_spike_counts_for_edges = _sorted_spike_counts_for_edges

    setattr(extensions_module, _PATCHED_FLAG, True)


__all__ = ["apply_emission_cell_id_validation_patch"]
