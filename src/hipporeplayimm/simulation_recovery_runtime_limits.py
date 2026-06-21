"""Strict validation for synthetic-recovery runtime limits."""

from __future__ import annotations

from typing import Any

import numpy as np


_PATCHED_FLAG = "_strict_simulation_recovery_runtime_limit_validation_applied"


def apply_simulation_recovery_runtime_limit_validation_patch() -> None:
    """Reject malformed runtime-limit values before recovery loops start."""

    from . import simulation_recovery as recovery

    if getattr(recovery, _PATCHED_FLAG, False):
        return

    def validate_recovery_runtime_limits(config: Any) -> None:
        max_synthetic_events = getattr(config, "max_synthetic_events", None)
        if max_synthetic_events is not None:
            _positive_integer_value("max_synthetic_events", max_synthetic_events)
        max_runtime_s = getattr(config, "max_runtime_s", None)
        if max_runtime_s is not None:
            _positive_finite_scalar("max_runtime_s", max_runtime_s)

    validate_recovery_runtime_limits.__name__ = recovery._validate_recovery_runtime_limits.__name__
    validate_recovery_runtime_limits.__doc__ = recovery._validate_recovery_runtime_limits.__doc__
    recovery._validate_recovery_runtime_limits = validate_recovery_runtime_limits
    setattr(recovery, _PATCHED_FLAG, True)


def _positive_integer_value(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive integer")
    if isinstance(value, str):
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive integer") from exc
        if integer <= 0:
            raise ValueError(f"{name} must be a finite positive integer")
        return integer
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive integer")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} must be a finite positive integer")
    return integer


def _positive_finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


__all__ = ["apply_simulation_recovery_runtime_limit_validation_patch"]
