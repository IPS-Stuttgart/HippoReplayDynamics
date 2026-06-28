"""Validate benchmark count configuration before event and split selection.

``BenchmarkConfig.max_events_per_session`` and ``BenchmarkConfig.n_cell_splits``
are counts, not flags. Python booleans are integers, so raw ``int(...)``
coercion can silently turn ``True`` into a one-event benchmark or a one-split
benchmark and ``False`` into an empty benchmark or an invalid zero-split
benchmark. Validate the counts explicitly while still accepting integer-valued
values such as ``1.0`` or ``"1"`` that can arise from notebooks or tabular
configuration files.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from functools import wraps

import numpy as np

_PATCHED_FLAG = "_benchmark_event_limit_validation_patch_applied"
_CELL_SPLIT_COUNT_PATCHED_FLAG = "_benchmark_cell_split_count_validation_patch_applied"
_CONFIG_METADATA_PATCHED_FLAG = "_benchmark_config_metadata_cell_split_count_validation_patch_applied"
_SPLIT_METADATA_PATCHED_FLAG = "_benchmark_split_metadata_cell_split_count_validation_patch_applied"


class _EventLimitConfigProxy:
    """Delegate a config object while overriding the validated event limit."""

    def __init__(self, config: object, max_events_per_session: int) -> None:
        self._config = config
        self.max_events_per_session = int(max_events_per_session)

    def __getattr__(self, name: str) -> object:
        return getattr(self._config, name)


def apply_benchmark_event_limit_validation_patch() -> None:
    """Install strict validation for benchmark count fields."""

    from . import benchmarks

    _patch_event_indices(benchmarks)
    _patch_n_cell_splits(benchmarks)
    _patch_benchmark_metadata(benchmarks)
    setattr(benchmarks, _PATCHED_FLAG, True)


def _patch_event_indices(benchmarks: object) -> None:
    current = benchmarks._event_indices
    if getattr(current, _PATCHED_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _event_indices(session, config, *, split_index: int = 0):
        max_events = _coerce_optional_nonnegative_integer(
            getattr(config, "max_events_per_session", None),
            "max_events_per_session",
        )
        if max_events is not None:
            config = _config_with_validated_event_limit(config, max_events)
        return previous(session, config, split_index=split_index)

    setattr(_event_indices, _PATCHED_FLAG, True)
    setattr(_event_indices, "__hipporeplayimm_original__", previous)
    benchmarks._event_indices = _event_indices


def _patch_n_cell_splits(benchmarks: object) -> None:
    current = benchmarks._n_cell_splits
    if getattr(current, _CELL_SPLIT_COUNT_PATCHED_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _n_cell_splits(config):
        return _coerce_positive_integer(
            getattr(config, "n_cell_splits", 1),
            "n_cell_splits",
        )

    setattr(_n_cell_splits, _CELL_SPLIT_COUNT_PATCHED_FLAG, True)
    setattr(_n_cell_splits, "__hipporeplayimm_original__", previous)
    benchmarks._n_cell_splits = _n_cell_splits


def _patch_benchmark_metadata(benchmarks: object) -> None:
    _patch_benchmark_config_metadata(benchmarks)
    _patch_benchmark_split_metadata(benchmarks)


def _patch_benchmark_config_metadata(benchmarks: object) -> None:
    current = benchmarks._benchmark_config_metadata
    if getattr(current, _CONFIG_METADATA_PATCHED_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _benchmark_config_metadata(config):
        n_cell_splits = _coerce_positive_integer(
            getattr(config, "n_cell_splits", 1),
            "n_cell_splits",
        )
        out = dict(previous(config))
        out["benchmark_n_cell_splits"] = n_cell_splits
        return out

    setattr(_benchmark_config_metadata, _CONFIG_METADATA_PATCHED_FLAG, True)
    setattr(_benchmark_config_metadata, "__hipporeplayimm_original__", previous)
    benchmarks._benchmark_config_metadata = _benchmark_config_metadata


def _patch_benchmark_split_metadata(benchmarks: object) -> None:
    current = benchmarks._benchmark_split_metadata
    if getattr(current, _SPLIT_METADATA_PATCHED_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _benchmark_split_metadata(config, split_index: int):
        n_cell_splits = _coerce_positive_integer(
            getattr(config, "n_cell_splits", 1),
            "n_cell_splits",
        )
        out = dict(previous(config, split_index))
        out["benchmark_cell_split_count"] = n_cell_splits
        return out

    setattr(_benchmark_split_metadata, _SPLIT_METADATA_PATCHED_FLAG, True)
    setattr(_benchmark_split_metadata, "__hipporeplayimm_original__", previous)
    benchmarks._benchmark_split_metadata = _benchmark_split_metadata


def _config_with_validated_event_limit(config: object, max_events: int) -> object:
    try:
        return replace(config, max_events_per_session=int(max_events))
    except TypeError:
        return _EventLimitConfigProxy(config, int(max_events))


def _coerce_optional_nonnegative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        _raise_invalid_nonnegative_integer(name)

    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        _raise_invalid_nonnegative_integer(name, exc)
    if scalar.ndim != 0:
        _raise_invalid_nonnegative_integer(name)
    if np.issubdtype(scalar.dtype, np.bool_):
        _raise_invalid_nonnegative_integer(name)

    return _coerce_scalar_nonnegative_integer(scalar.item(), name)


def _coerce_scalar_nonnegative_integer(item: object, name: str) -> int:
    if isinstance(item, (bool, np.bool_)):
        _raise_invalid_nonnegative_integer(name)
    if isinstance(item, (int, np.integer)):
        candidate = int(item)
        if candidate < 0:
            _raise_invalid_nonnegative_integer(name)
        return candidate
    if isinstance(item, Decimal):
        return _coerce_decimal_nonnegative_integer(item, name)
    if isinstance(item, (str, bytes)):
        text = item.decode().strip() if isinstance(item, bytes) else item.strip()
        return _coerce_text_nonnegative_integer(text, name)

    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_invalid_nonnegative_integer(name, exc)
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0.0:
        _raise_invalid_nonnegative_integer(name)
    return int(numeric)


def _coerce_text_nonnegative_integer(text: str, name: str) -> int:
    if not text:
        _raise_invalid_nonnegative_integer(name)
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        _raise_invalid_nonnegative_integer(name, exc)
    return _coerce_decimal_nonnegative_integer(decimal, name)


def _coerce_decimal_nonnegative_integer(decimal: Decimal, name: str) -> int:
    if not decimal.is_finite() or decimal < 0 or decimal != decimal.to_integral_value():
        _raise_invalid_nonnegative_integer(name)
    return int(decimal)


def _coerce_positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")

    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a positive integer")
    if np.issubdtype(scalar.dtype, np.bool_):
        raise ValueError(f"{name} must be a positive integer")

    item = scalar.item()
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(item, Decimal):
        numeric = item
    elif isinstance(item, (str, bytes)):
        text = item.decode().strip() if isinstance(item, bytes) else item.strip()
        try:
            numeric = Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not numeric.is_finite() or numeric < 1 or numeric != numeric.to_integral_value():
            raise ValueError(f"{name} must be a positive integer")
        return int(numeric)
    else:
        try:
            numeric_float = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not np.isfinite(numeric_float) or not numeric_float.is_integer() or numeric_float < 1.0:
            raise ValueError(f"{name} must be a positive integer")
        return int(numeric_float)
    if not numeric.is_finite() or numeric < 1 or numeric != numeric.to_integral_value():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _raise_invalid_nonnegative_integer(name: str, exc: Exception | None = None) -> None:
    if exc is None:
        raise ValueError(f"{name} must be a non-negative integer")
    raise ValueError(f"{name} must be a non-negative integer") from exc


__all__ = [
    "apply_benchmark_event_limit_validation_patch",
    "_coerce_optional_nonnegative_integer",
    "_coerce_positive_integer",
]
