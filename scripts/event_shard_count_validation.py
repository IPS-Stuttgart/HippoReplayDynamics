"""Exact integer validation for event-shard workflow counts."""

from __future__ import annotations

import operator

import numpy as np


def integer_count(name: str, value: object, *, minimum: int) -> int:
    """Return a canonical Python integer without coercing booleans or floats."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if raw.ndim != 0:
        raise TypeError(f"{name} must be an integer scalar")
    item = raw.item()
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer scalar, not boolean")
    try:
        count = int(operator.index(item))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if count < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def optional_nonnegative_integer_count(name: str, value: object | None) -> int | None:
    """Return ``None`` or a canonical non-negative Python integer."""

    if value is None:
        return None
    return integer_count(name, value, minimum=0)


def positive_integer_count(name: str, value: object) -> int:
    """Return a canonical positive Python integer."""

    return integer_count(name, value, minimum=1)
