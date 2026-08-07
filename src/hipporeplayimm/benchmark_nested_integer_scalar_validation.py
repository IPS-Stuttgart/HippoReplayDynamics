"""Reject lossy nested scalar wrappers in benchmark integer configuration.

Benchmark count and seed validators intentionally accept scalar numeric wrappers
that commonly arrive from NumPy, MATLAB, or tabular configuration.  A nested
zero-dimensional object array, however, could previously hide a Boolean or a
one-element array until Python's ``int(...)`` coercion.  That silently turned
``True`` into ``1`` and relied on deprecated array-to-scalar conversion for
singleton arrays.

Install a narrow preflight around the shared benchmark integer helpers so only
true scalar leaves reach the existing exact-integer validation.  The same active
helper is synchronized into benchmark seed validation, which imports it by
value.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_OPTIONAL_FLAG = "_benchmark_nested_optional_integer_scalar_wrapper"
_POSITIVE_FLAG = "_benchmark_nested_positive_integer_scalar_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_benchmark_nested_integer_scalar_validation_patch() -> None:
    """Install strict recursive scalar preflight for benchmark counts and seeds."""

    from . import benchmark_event_limit_validation as limits
    from . import benchmark_seed_validation as seeds

    current_optional = limits._coerce_optional_nonnegative_integer
    if not getattr(current_optional, _OPTIONAL_FLAG, False):
        original_optional = current_optional

        @wraps(original_optional)
        def coerce_optional_nonnegative_integer(value: object, name: str) -> int | None:
            if value is None:
                return original_optional(value, name)
            scalar = _unwrap_scalar_leaf(value, name, positive=False)
            return original_optional(scalar, name)

        setattr(coerce_optional_nonnegative_integer, _OPTIONAL_FLAG, True)
        setattr(coerce_optional_nonnegative_integer, _ORIGINAL_ATTR, original_optional)
        limits._coerce_optional_nonnegative_integer = coerce_optional_nonnegative_integer

    current_positive = limits._coerce_positive_integer
    if not getattr(current_positive, _POSITIVE_FLAG, False):
        original_positive = current_positive

        @wraps(original_positive)
        def coerce_positive_integer(value: object, name: str) -> int:
            scalar = _unwrap_scalar_leaf(value, name, positive=True)
            return original_positive(scalar, name)

        setattr(coerce_positive_integer, _POSITIVE_FLAG, True)
        setattr(coerce_positive_integer, _ORIGINAL_ATTR, original_positive)
        limits._coerce_positive_integer = coerce_positive_integer

    _synchronize_seed_integer_helper(seeds, limits._coerce_optional_nonnegative_integer)


def _unwrap_scalar_leaf(value: object, name: str, *, positive: bool) -> object:
    """Recursively unwrap zero-dimensional NumPy containers without lossy casts."""

    current = value
    seen: set[int] = set()
    while True:
        if isinstance(current, (bool, np.bool_)):
            _raise_invalid_integer(name, positive=positive)

        if isinstance(current, np.ndarray):
            if current.ndim != 0:
                _raise_invalid_integer(name, positive=positive)
            marker = id(current)
            if marker in seen:
                _raise_invalid_integer(name, positive=positive)
            seen.add(marker)
            try:
                current = current.item()
            except (TypeError, ValueError) as exc:
                _raise_invalid_integer(name, positive=positive, exc=exc)
            continue

        if isinstance(current, np.generic):
            current = current.item()
            continue

        return current


def _synchronize_seed_integer_helper(seed_module: Any, active: Any) -> None:
    """Refresh the helper alias imported by ``benchmark_seed_validation``."""

    current = getattr(seed_module, "_coerce_optional_nonnegative_integer", None)
    if current is active:
        return

    lineage: set[Any] = set()
    cursor = active
    while callable(cursor) and cursor not in lineage:
        lineage.add(cursor)
        cursor = getattr(cursor, _ORIGINAL_ATTR, None)
    if current in lineage:
        seed_module._coerce_optional_nonnegative_integer = active


def _raise_invalid_integer(
    name: str,
    *,
    positive: bool,
    exc: Exception | None = None,
) -> None:
    qualifier = "positive" if positive else "non-negative"
    message = f"{name} must be a {qualifier} integer"
    if exc is None:
        raise ValueError(message)
    raise ValueError(message) from exc


__all__ = ["apply_benchmark_nested_integer_scalar_validation_patch"]
