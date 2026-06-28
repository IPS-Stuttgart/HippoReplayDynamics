"""Validate benchmark event-limit configuration before event selection.

``BenchmarkConfig.max_events_per_session`` is a count, not a flag.  Python
booleans are integers, so the raw ``int(...)`` coercion in benchmark event
selection could silently turn ``True`` into a one-event benchmark and ``False``
into an empty benchmark.  Validate the count explicitly while still accepting
integer-valued values such as ``1.0`` or ``"1"`` that can arise from notebooks
or tabular configuration files.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from functools import wraps

import numpy as np

_PATCHED_FLAG = "_benchmark_event_limit_validation_patch_applied"


class _EventLimitConfigProxy:
    """Delegate a config object while overriding the validated event limit."""

    def __init__(self, config: object, max_events_per_session: int) -> None:
        self._config = config
        self.max_events_per_session = int(max_events_per_session)

    def __getattr__(self, name: str) -> object:
        return getattr(self._config, name)


def apply_benchmark_event_limit_validation_patch() -> None:
    """Install strict validation for ``BenchmarkConfig.max_events_per_session``."""

    from . import benchmarks

    current = benchmarks._event_indices
    if getattr(current, _PATCHED_FLAG, False):
        setattr(benchmarks, _PATCHED_FLAG, True)
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
    setattr(benchmarks, _PATCHED_FLAG, True)


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


def _raise_invalid_nonnegative_integer(name: str, exc: Exception | None = None) -> None:
    if exc is None:
        raise ValueError(f"{name} must be a non-negative integer")
    raise ValueError(f"{name} must be a non-negative integer") from exc


__all__ = [
    "apply_benchmark_event_limit_validation_patch",
    "_coerce_optional_nonnegative_integer",
]
