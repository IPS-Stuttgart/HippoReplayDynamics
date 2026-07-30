"""Strict validation for synthetic-recovery runtime limits and boolean controls."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

from .simulation_recovery_empty_csv import apply_simulation_recovery_empty_csv_patch
from .simulation_recovery_overdispersion import apply_simulation_recovery_overdispersion_patch


_PATCHED_FLAG = "_strict_simulation_recovery_runtime_limit_validation_applied"
_VALIDATOR_WRAPPER_ATTR = "_strict_simulation_recovery_runtime_limit_validator"
_RUN_WRAPPER_ATTR = "_strict_simulation_recovery_runtime_limit_preflight_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
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
    apply_simulation_recovery_overdispersion_patch()

    current_validator = recovery._validate_recovery_runtime_limits
    if not getattr(current_validator, _VALIDATOR_WRAPPER_ATTR, False):

        @wraps(current_validator)
        def validate_recovery_runtime_limits(config: Any) -> None:
            _normalized_runtime_config(config)

        setattr(validate_recovery_runtime_limits, _VALIDATOR_WRAPPER_ATTR, True)
        setattr(validate_recovery_runtime_limits, _ORIGINAL_ATTR, current_validator)
        recovery._validate_recovery_runtime_limits = validate_recovery_runtime_limits

    current_run = recovery.run_session_simulation_recovery
    if not getattr(current_run, _RUN_WRAPPER_ATTR, False):
        original_run = current_run

        @wraps(original_run)
        def run_session_simulation_recovery_with_runtime_limit_preflight(
            dataset_root: Any,
            session_id: str,
            config: Any,
        ) -> Any:
            normalized_config = _normalized_runtime_config(config)
            return original_run(dataset_root, session_id, normalized_config)

        setattr(
            run_session_simulation_recovery_with_runtime_limit_preflight,
            _RUN_WRAPPER_ATTR,
            True,
        )
        setattr(
            run_session_simulation_recovery_with_runtime_limit_preflight,
            _ORIGINAL_ATTR,
            original_run,
        )
        recovery.run_session_simulation_recovery = (
            run_session_simulation_recovery_with_runtime_limit_preflight
        )

    setattr(recovery, _PATCHED_FLAG, True)


def _normalized_runtime_config(config: Any) -> Any:
    updates: dict[str, Any] = {
        "events_per_model": _positive_integer_value(
            "events_per_model",
            getattr(config, "events_per_model", None),
        ),
    }

    for field_name in ("max_template_events", "max_synthetic_events"):
        value = getattr(config, field_name, None)
        updates[field_name] = (
            None if value is None else _positive_integer_value(field_name, value)
        )

    max_runtime_s = getattr(config, "max_runtime_s", None)
    updates["max_runtime_s"] = (
        None
        if max_runtime_s is None
        else _positive_finite_scalar("max_runtime_s", max_runtime_s)
    )

    for field_name in _BOOLEAN_CONTROL_FIELDS:
        updates[field_name] = _strict_bool_value(
            field_name,
            getattr(config, field_name, None),
        )

    return replace(config, **updates)


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
