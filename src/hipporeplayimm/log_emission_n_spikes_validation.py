"""Validate ``LogEmissionTensor`` count summaries, durations, and cell identifiers."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from .encoding import LogEmissionTensor

_PATCH_FLAG = "_n_spikes_validation_applied"
_POST_INIT_WRAPPER_MARKER = "_n_spikes_validation_post_init_wrapper"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def _log_emission_n_spikes_patch_current() -> bool:
    return bool(getattr(LogEmissionTensor.__post_init__, _POST_INIT_WRAPPER_MARKER, False))


def apply_log_emission_n_spikes_validation_patch() -> None:
    """Install idempotent ``LogEmissionTensor`` post-construction guards."""

    if _log_emission_n_spikes_patch_current():
        setattr(LogEmissionTensor, _PATCH_FLAG, True)
        return

    original_post_init = LogEmissionTensor.__post_init__

    @wraps(original_post_init)
    def _validated_post_init(self: LogEmissionTensor) -> None:
        _validate_duration_inputs(self)
        _validate_raw_log_likelihood(self.log_likelihood)
        original_post_init(self)
        _validate_log_likelihood(self)
        _validate_n_spikes(self)
        _validate_cell_ids(self)

    setattr(_validated_post_init, _POST_INIT_WRAPPER_MARKER, True)
    LogEmissionTensor.__post_init__ = _validated_post_init  # type: ignore[method-assign]
    setattr(LogEmissionTensor, _PATCH_FLAG, True)


def _validate_duration_inputs(emissions: LogEmissionTensor) -> None:
    """Reject invalid duration inputs before dataclass coercion."""

    if _contains_text_values(emissions.dt):
        raise ValueError("dt must be a numeric duration, not text")
    if _contains_boolean_values(emissions.dt):
        raise ValueError("dt must be a numeric duration, not boolean")
    _require_scalar_duration(emissions.dt, "dt")

    if emissions.bin_durations is not None:
        if _contains_text_values(emissions.bin_durations):
            raise ValueError("bin_durations must contain numeric durations, not text values")
        if _contains_boolean_values(emissions.bin_durations):
            raise ValueError("bin_durations must contain numeric durations, not boolean values")
    if emissions.transition_durations is not None:
        if _contains_text_values(emissions.transition_durations):
            raise ValueError("transition_durations must contain numeric durations, not text values")
        if _contains_boolean_values(emissions.transition_durations):
            raise ValueError("transition_durations must contain numeric durations, not boolean values")


def _require_scalar_duration(value: Any, name: str) -> None:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar duration") from exc
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar duration")


def _validate_raw_log_likelihood(values: Any) -> None:
    """Reject values that float coercion would silently reinterpret."""

    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("log_likelihood must contain numeric real values") from exc

    if raw.dtype.kind == "b":
        raise ValueError("log_likelihood must contain numeric real values, not boolean values")
    if raw.dtype.kind in {"S", "U"}:
        raise ValueError("log_likelihood must contain numeric real values, not text values")
    if raw.dtype.kind == "c":
        raise ValueError("log_likelihood must contain numeric real values, not complex values")
    if raw.dtype.kind == "O":
        for item in raw.reshape(-1):
            if isinstance(item, (bool, np.bool_)):
                raise ValueError("log_likelihood must contain numeric real values, not boolean values")
            if isinstance(item, _STRING_TYPES):
                raise ValueError("log_likelihood must contain numeric real values, not text values")
            if isinstance(item, (complex, np.complexfloating)):
                raise ValueError("log_likelihood must contain numeric real values, not complex values")

    try:
        np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("log_likelihood must contain numeric real values") from exc


def _validate_log_likelihood(emissions: LogEmissionTensor) -> None:
    """Reject invalid numeric likelihood entries without checking model support."""

    values = np.asarray(emissions.log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must be a two-dimensional array")
    if values.shape[0] == 0:
        raise ValueError("log_likelihood must include at least one time bin")
    if values.shape[1] == 0:
        raise ValueError("log_likelihood must include at least one spatial bin")
    if np.any(np.isnan(values)):
        raise ValueError("log_likelihood must not contain NaN values")


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


def _contains_text_values(values: Any) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.str_) or np.issubdtype(raw.dtype, np.bytes_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, _STRING_TYPES) for value in raw.reshape(-1))
    return False


def _require_scalar_count(value: Any, name: str) -> None:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar count") from exc
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar count")


def _validate_n_spikes(emissions: LogEmissionTensor) -> None:
    if _contains_boolean_values(emissions.spike_counts):
        raise ValueError("spike_counts must be numeric counts, not boolean values")
    spike_counts = np.asarray(emissions.spike_counts, dtype=float)
    rounded_counts = np.rint(spike_counts)
    if not np.all(np.isclose(spike_counts, rounded_counts, rtol=0.0, atol=0.0)):
        raise ValueError("spike_counts must be integer-valued")
    total_spikes = float(rounded_counts.sum())
    _require_scalar_count(emissions.n_spikes, "n_spikes")
    if _contains_boolean_values(emissions.n_spikes):
        raise ValueError("n_spikes must be a numeric count, not boolean")
    try:
        n_spikes = float(emissions.n_spikes)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_spikes must be numeric") from exc
    if not np.isfinite(n_spikes) or n_spikes < 0.0:
        raise ValueError("n_spikes must be finite and nonnegative")
    rounded = float(np.rint(n_spikes))
    if not np.isclose(n_spikes, rounded, rtol=0.0, atol=0.0):
        raise ValueError("n_spikes must be integer-valued")
    if not np.isclose(rounded, total_spikes, rtol=0.0, atol=0.0):
        raise ValueError("n_spikes must equal the total spike_counts sum")
    emissions.spike_counts = rounded_counts.astype(int, copy=False)
    emissions.n_spikes = int(rounded)


def _validate_cell_ids(emissions: LogEmissionTensor) -> None:
    emissions.cell_ids = _coerce_cell_ids(emissions.cell_ids)


def _coerce_cell_ids(values: Any) -> np.ndarray:
    try:
        cell_ids = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell_ids must contain finite integer identifiers") from exc
    if cell_ids.ndim != 1:
        raise ValueError("cell_ids must be one-dimensional")
    if cell_ids.size == 0:
        return np.empty(0, dtype=int)
    if _contains_boolean_values(cell_ids):
        raise ValueError("cell_ids must be numeric integer identifiers, not boolean values")

    integer_info = np.iinfo(np.dtype(int))
    canonical = np.asarray(
        [_coerce_integer_identifier(value, "cell_ids", integer_info) for value in cell_ids.reshape(-1)],
        dtype=int,
    )
    if np.unique(canonical).shape[0] != canonical.shape[0]:
        raise ValueError("cell_ids must be unique")
    return canonical


def _coerce_integer_identifier(value: Any, name: str, integer_info: np.iinfo) -> int:
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


apply_log_emission_n_spikes_validation_patch()

__all__ = ["apply_log_emission_n_spikes_validation_patch"]
