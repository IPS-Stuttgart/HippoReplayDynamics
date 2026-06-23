"""Strict numeric parsing guards for PyRecEst score metadata."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_pyrecest_numeric_metadata_guard_applied"
_RAW_FLOAT_ERROR = "could not convert string to float"


def apply_pyrecest_numeric_metadata_guard_patch() -> None:
    """Reject boolean and malformed values in PyRecEst numeric metadata."""

    from . import pyrecest_score_metadata as metadata

    current = metadata._metadata_float_from_value
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def metadata_float_from_value(value: Any, column: str) -> float | None:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{column} must contain finite numeric values")
        try:
            return current(value, column)
        except ValueError as exc:
            if _RAW_FLOAT_ERROR in str(exc):
                raise ValueError(f"{column} must contain finite numeric values") from exc
            raise

    setattr(metadata_float_from_value, _PATCHED_FLAG, True)
    metadata._metadata_float_from_value = metadata_float_from_value


__all__ = ["apply_pyrecest_numeric_metadata_guard_patch"]
