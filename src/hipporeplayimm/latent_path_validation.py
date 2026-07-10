"""Validation helpers for synthetic recovery inputs."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np


_MODEL_LIST_PATCHED_FLAG = "_simulation_recovery_model_list_validation_applied"
_MODEL_LIST_WRAPPER_MARKER = "_simulation_recovery_model_list_validation_wrapper"


def apply_latent_path_validation_patch() -> None:
    """Install synthetic-recovery input validation wrappers."""

    import hipporeplayimm.simulation_recovery as recovery

    if not getattr(recovery, "_latent_path_n_time_validation_applied", False):
        _patch_simulate_latent_path(recovery)
    if not getattr(recovery, "_replay_event_n_time_validation_applied", False):
        _patch_simulate_replay_event(recovery)
    if not getattr(recovery, "_emissions_from_counts_input_validation_applied", False):
        _patch_emissions_from_counts(recovery)
    if not getattr(recovery, "_recovery_event_count_validation_applied", False):
        _patch_run_session_simulation_recovery(recovery)
    if not getattr(recovery, _MODEL_LIST_PATCHED_FLAG, False):
        _patch_parse_model_list(recovery)


def _patch_parse_model_list(recovery: Any) -> None:
    current = recovery.parse_model_list
    if getattr(current, _MODEL_LIST_WRAPPER_MARKER, False):
        setattr(recovery, _MODEL_LIST_PATCHED_FLAG, True)
        return

    @wraps(current)
    def checked_parse_model_list(spec: Any) -> tuple[str, ...]:
        return current(_validated_model_list_spec(spec))

    setattr(checked_parse_model_list, _MODEL_LIST_WRAPPER_MARKER, True)
    setattr(checked_parse_model_list, "__hipporeplayimm_original__", current)
    recovery.parse_model_list = checked_parse_model_list
    setattr(recovery, _MODEL_LIST_PATCHED_FLAG, True)


def _validated_model_list_spec(spec: Any) -> Any:
    """Reject empty entries instead of silently changing the requested model set."""

    if isinstance(spec, str):
        if spec.strip() and "," in spec:
            comma_parts = spec.split(",")
            if any(not part.strip() for part in comma_parts):
                raise ValueError("model list must not contain empty comma-separated entries")
        return spec

    try:
        values = tuple(spec)
    except TypeError:
        return spec
    if any(not str(value).strip() for value in values):
        raise ValueError("model list must not contain empty entries")
    return values


def _patch_simulate_latent_path(recovery: Any) -> None:
    original = recovery.simulate_latent_path

    @wraps(original)
    def checked_simulate_latent_path(*args: Any, **kwargs: Any) -> Any:
        if "n_time" in kwargs:
            kwargs = dict(kwargs)
            kwargs["n_time"] = _positive_integer_value("n_time", kwargs["n_time"])
        if "state_space" in kwargs and kwargs["state_space"] is not None:
            kwargs = dict(kwargs)
            true_model = kwargs.get("true_model", "")
            _validate_latent_path_motion_sigmas(
                true_model,
                kwargs["state_space"],
            )
            kwargs["state_space"] = _state_space_with_unused_sigmas_neutralized(
                true_model,
                kwargs["state_space"],
            )
        return original(*args, **kwargs)

    recovery.simulate_latent_path = checked_simulate_latent_path
    recovery._latent_path_n_time_validation_applied = True


def _state_space_with_unused_sigmas_neutralized(true_model: Any, state_space: Any) -> Any:
    """Prevent inactive model parameters from being evaluated by the legacy simulator."""

    model = str(true_model).strip().lower()
    if model == "diffusion":
        return replace(
            state_space,
            momentum_sigma_cm_sqrt_s=0.0,
            momentum_initial_sigma_cm_sqrt_s=0.0,
        )
    if model == "momentum":
        return replace(state_space, diffusion_sigma_cm_sqrt_s=0.0)
    return state_space


def _patch_simulate_replay_event(recovery: Any) -> None:
    original = recovery.simulate_replay_event

    @wraps(original)
    def checked_simulate_replay_event(*args: Any, **kwargs: Any) -> Any:
        if "n_time" in kwargs:
            kwargs = dict(kwargs)
            kwargs["n_time"] = _positive_integer_value("n_time", kwargs["n_time"])
        return original(*args, **kwargs)

    recovery.simulate_replay_event = checked_simulate_replay_event
    recovery._replay_event_n_time_validation_applied = True


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
    max_template_events = getattr(config, "max_template_events", None)
    if max_template_events is not None:
        updates["max_template_events"] = _positive_integer_or_uncapped_value(
            "max_template_events",
            max_template_events,
        )
    max_synthetic_events = getattr(config, "max_synthetic_events", None)
    if max_synthetic_events is not None:
        updates["max_synthetic_events"] = _positive_integer_value(
            "max_synthetic_events",
            max_synthetic_events,
        )
    return replace(config, **updates)


def _positive_integer_value(name: str, value: Any) -> int:
    integer_value = _integer_valued_scalar(name, value)
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive integer-valued")
    return integer_value


def _positive_integer_or_uncapped_value(name: str, value: Any) -> int | None:
    integer_value = _integer_valued_scalar(name, value)
    return None if integer_value <= 0 else integer_value


def _integer_valued_scalar(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be positive integer-valued")
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be positive integer-valued")
    if _contains_text_values(raw):
        raise ValueError(f"{name} must be positive integer-valued, not text")
    if np.issubdtype(raw.dtype, np.bool_) or (
        raw.dtype == object and isinstance(raw.item(), (bool, np.bool_))
    ):
        raise TypeError(f"{name} must be positive integer-valued")
    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive integer-valued") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be positive integer-valued")
    integer_value = int(round(numeric))
    if not np.isclose(numeric, integer_value, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} must be positive integer-valued")
    return integer_value


def _validate_latent_path_motion_sigmas(true_model: Any, state_space: Any) -> None:
    model = str(true_model).strip().lower()
    if model == "diffusion":
        _finite_nonnegative_value(
            "state_space.diffusion_sigma_cm_sqrt_s",
            getattr(state_space, "diffusion_sigma_cm_sqrt_s"),
        )
    elif model == "momentum":
        _finite_nonnegative_value(
            "state_space.momentum_initial_sigma_cm_sqrt_s",
            getattr(state_space, "momentum_initial_sigma_cm_sqrt_s"),
        )
        _finite_nonnegative_value(
            "state_space.momentum_sigma_cm_sqrt_s",
            getattr(state_space, "momentum_sigma_cm_sqrt_s"),
        )


def _finite_nonnegative_value(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    raw = np.asarray(value)
    if raw.ndim != 0 or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


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
    if _contains_boolean_values(raw):
        raise ValueError("counts must contain numeric integer counts, not boolean values")
    if _contains_text_values(raw):
        raise ValueError("counts must contain numeric integer counts, not text values")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    if not np.all(np.isclose(numeric, np.rint(numeric), rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    return numeric.astype(int)


def _contains_boolean_values(values: Any) -> bool:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))


def _contains_text_values(values: Any) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if raw.dtype.kind in {"U", "S"}:
        return True
    if raw.dtype == object:
        return any(isinstance(value, (str, bytes, np.str_, np.bytes_)) for value in raw.reshape(-1))
    return False
