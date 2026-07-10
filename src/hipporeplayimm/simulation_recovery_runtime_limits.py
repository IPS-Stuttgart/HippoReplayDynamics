"""Strict validation for synthetic-recovery runtime limits and boolean controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from .simulation_recovery_empty_csv import apply_simulation_recovery_empty_csv_patch


_PATCHED_FLAG = "_strict_simulation_recovery_runtime_limit_validation_applied"
_BOOLEAN_CONTROL_FIELDS = (
    "score_with_occupancy",
    "oracle_candidate_support",
    "continue_on_error",
    "progress_log",
)


def apply_simulation_recovery_runtime_limit_validation_patch() -> None:
    """Reject malformed runtime-limit values before recovery loops start."""

    from . import simulation_recovery as recovery

    apply_simulation_recovery_empty_csv_patch()
    if getattr(recovery, _PATCHED_FLAG, False):
        return

    def validate_recovery_runtime_limits(config: Any) -> None:
        events_per_model = getattr(config, "events_per_model", None)
        _positive_integer_value("events_per_model", events_per_model)

        max_template_events = getattr(config, "max_template_events", None)
        if max_template_events is not None:
            _positive_integer_value("max_template_events", max_template_events)

        max_synthetic_events = getattr(config, "max_synthetic_events", None)
        if max_synthetic_events is not None:
            _positive_integer_value("max_synthetic_events", max_synthetic_events)
        max_runtime_s = getattr(config, "max_runtime_s", None)
        if max_runtime_s is not None:
            _positive_finite_scalar("max_runtime_s", max_runtime_s)

        for field_name in _BOOLEAN_CONTROL_FIELDS:
            _strict_bool_value(field_name, getattr(config, field_name, None))

    validate_recovery_runtime_limits.__name__ = recovery._validate_recovery_runtime_limits.__name__
    validate_recovery_runtime_limits.__doc__ = recovery._validate_recovery_runtime_limits.__doc__
    recovery._validate_recovery_runtime_limits = validate_recovery_runtime_limits
    setattr(recovery, _PATCHED_FLAG, True)


def _positive_integer_value(name: str, value: Any) -> int:
    scalar = _scalar_non_boolean_value(name, value, message="must be a finite positive integer")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive integer")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} must be a finite positive integer")
    return integer


def _positive_finite_scalar(name: str, value: Any) -> float:
    scalar = _scalar_non_boolean_value(name, value, message="must be finite and positive")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _scalar_non_boolean_value(name: str, value: Any, *, message: str) -> Any:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} {message}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} {message}")
    scalar = raw.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} {message}")
    return scalar


def _strict_bool_value(name: str, value: Any) -> bool:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a boolean")
    scalar = raw.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    raise ValueError(f"{name} must be a boolean")


__all__ = ["apply_simulation_recovery_runtime_limit_validation_patch"]
