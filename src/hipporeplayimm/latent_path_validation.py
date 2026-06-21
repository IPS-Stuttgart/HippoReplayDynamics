"""Validation helpers for synthetic recovery inputs."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np


def apply_latent_path_validation_patch() -> None:
    """Install synthetic-recovery input validation wrappers."""

    import hipporeplayimm.simulation_recovery as recovery

    if not getattr(recovery, "_latent_path_n_time_validation_applied", False):
        _patch_simulate_latent_path(recovery)
    if not getattr(recovery, "_emissions_from_counts_input_validation_applied", False):
        _patch_emissions_from_counts(recovery)
    if not getattr(recovery, "_recovery_event_count_validation_applied", False):
        _patch_run_session_simulation_recovery(recovery)


def _patch_simulate_latent_path(recovery: Any) -> None:
    original = recovery.simulate_latent_path

    @wraps(original)
    def checked_simulate_latent_path(*args: Any, **kwargs: Any) -> Any:
        if "n_time" in kwargs:
            kwargs = dict(kwargs)
            kwargs["n_time"] = _positive_integer_value("n_time", kwargs["n_time"])
        return original(*args, **kwargs)

    recovery.simulate_latent_path = checked_simulate_latent_path
    recovery._latent_path_n_time_validation_applied = True


def _patch_run_session_simulation_recovery(recovery: Any) -> None:
    original = recovery.run_session_simulation_recovery

    @wraps(original)
    def checked_run_session_simulation_recovery(
        dataset_root: Any,
        session_id: Any,
        config: Any,
    ) -> Any:
        validated_config = _config_with_validated_event_counts(config)
        return original(dataset_root, session_id, validated_config)

    recovery.run_session_simulation_recovery = checked_run_session_simulation_recovery
    recovery._recovery_event_count_validation_applied = True


def _config_with_validated_event_counts(config: Any) -> Any:
    updates = {
        "events_per_model": _positive_integer_value(
            "events_per_model",
            getattr(config, "events_per_model"),
        )
    }
    for name in ("max_template_events", "max_synthetic_events"):
        value = getattr(config, name, None)
        if value is not None:
            updates[name] = _positive_integer_value(name, value)
    return replace(config, **updates)


def _positive_integer_value(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive integer-valued")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive integer-valued") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive integer-valued")
    integer_value = int(round(numeric))
    if not np.isclose(numeric, integer_value, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} must be positive integer-valued")
    return integer_value


def _patch_emissions_from_counts(recovery: Any) -> None:
    original = recovery.emissions_from_counts

    @wraps(original)
    def checked_emissions_from_counts(encoding: Any, counts: Any, *args: Any, **kwargs: Any) -> Any:
        return original(encoding, _checked_count_array(counts), *args, **kwargs)

    recovery.emissions_from_counts = checked_emissions_from_counts
    recovery._emissions_from_counts_input_validation_applied = True


def _checked_count_array(counts: Any) -> np.ndarray:
    raw = np.asarray(counts)
    if raw.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    if raw.shape[0] == 0:
        raise ValueError("counts must contain at least one time bin")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    if not np.all(np.isclose(numeric, np.rint(numeric), rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    return numeric.astype(int)
