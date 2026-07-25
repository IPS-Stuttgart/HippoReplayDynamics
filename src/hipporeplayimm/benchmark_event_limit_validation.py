"""Validate benchmark selection configuration before event and split selection.

``BenchmarkConfig.max_events_per_session`` and ``BenchmarkConfig.n_cell_splits``
are counts, not flags. Python booleans are integers, so raw ``int(...)``
coercion can silently turn ``True`` into a one-event benchmark or a one-split
benchmark and ``False`` into an empty benchmark or an invalid zero-split
benchmark. Validate the counts explicitly while still accepting integer-valued
values such as ``1.0`` or ``"1"`` that can arise from notebooks or tabular
configuration files.

``BenchmarkConfig.randomize_event_subset`` is a flag, not an arbitrary truthy
value. In particular, ``bool("false")`` is ``True`` and previously enabled
random event sampling while metadata also reported the flag as enabled. Require
a genuine Python/NumPy boolean scalar and pass a canonical ``bool`` to the
wrapped benchmark functions.
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


class _BenchmarkSelectionConfigProxy:
    """Delegate a config object while overriding validated selection fields."""

    def __init__(self, config: object, overrides: dict[str, object]) -> None:
        self._config = config
        self._overrides = dict(overrides)

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._config, name)


def apply_benchmark_event_limit_validation_patch() -> None:
    """Install strict validation for benchmark count and subset-selection fields."""

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
        randomize_event_subset = _coerce_boolean_scalar(
            getattr(config, "randomize_event_subset", False),
            "randomize_event_subset",
        )
        overrides: dict[str, object] = {
            "randomize_event_subset": randomize_event_subset,
        }
        if max_events is not None:
            overrides["max_events_per_session"] = max_events
        config = _config_with_validated_selection(config, overrides)
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
        randomize_event_subset = _coerce_boolean_scalar(
            getattr(config, "randomize_event_subset", False),
            "randomize_event_subset",
        )
        validated_config = _config_with_validated_selection(
            config,
            {"randomize_event_subset": randomize_event_subset},
        )
        out = dict(previous(validated_config))
        out["benchmark_n_cell_splits"] = n_cell_splits
        out["benchmark_randomize_event_subset"] = randomize_event_subset
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
        randomize_event_subset = _coerce_boolean_scalar(
            getattr(config, "randomize_event_subset", False),
            "randomize_event_subset",
        )
        validated_config = _config_with_validated_selection(
            config,
            {"randomize_event_subset": randomize_event_subset},
        )
        out = dict(previous(validated_config, split_index))
        out["benchmark_cell_split_count"] = n_cell_splits
        out["benchmark_randomize_event_subset"] = randomize_event_subset
        return out

    setattr(_benchmark_split_metadata, _SPLIT_METADATA_PATCHED_FLAG, True)
    setattr(_benchmark_split_metadata, "__hipporeplayimm_original__", previous)
    benchmarks._benchmark_split_metadata = _benchmark_split_metadata


def _config_with_validated_selection(
    config: object,
    overrides: dict[str, object],
) -> object:
    try:
        return replace(config, **overrides)
    except TypeError:
        return _BenchmarkSelectionConfigProxy(config, overrides)


def _coerce_boolean_scalar(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a boolean scalar") from exc
    if scalar.ndim != 0 or not np.issubdtype(scalar.dtype, np.bool_):
        raise ValueError(f"{name} must be a boolean scalar")

    try:
        item = scalar.item()
    except ValueError as exc:
        raise ValueError(f"{name} must be a boolean scalar") from exc
    if not isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(item)


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
    if isinstance(item, (float, np.floating)):
        if not np.isfinite(item) or not item.is_integer() or item < 0:
            _raise_invalid_nonnegative_integer(name)
        return int(item)

    try:
        candidate = int(item)
        exact = bool(item == candidate)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_invalid_nonnegative_integer(name, exc)
    if not exact or candidate < 0:
        _raise_invalid_nonnegative_integer(name)
    return candidate


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

    try:
        candidate = _coerce_scalar_nonnegative_integer(scalar.item(), name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if candidate < 1:
        raise ValueError(f"{name} must be a positive integer")
    return candidate


def _raise_invalid_nonnegative_integer(name: str, exc: Exception | None = None) -> None:
    if exc is None:
        raise ValueError(f"{name} must be a non-negative integer")
    raise ValueError(f"{name} must be a non-negative integer") from exc


__all__ = [
    "apply_benchmark_event_limit_validation_patch",
    "_coerce_boolean_scalar",
    "_coerce_optional_nonnegative_integer",
    "_coerce_positive_integer",
]
