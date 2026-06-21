"""Validation helpers for synthetic recovery inputs."""

from __future__ import annotations

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


def _patch_simulate_latent_path(recovery: Any) -> None:
    original = recovery.simulate_latent_path

    @wraps(original)
    def checked_simulate_latent_path(*args: Any, **kwargs: Any) -> Any:
        if "n_time" in kwargs and int(kwargs["n_time"]) <= 0:
            raise ValueError("n_time must be positive")
        return original(*args, **kwargs)

    recovery.simulate_latent_path = checked_simulate_latent_path
    recovery._latent_path_n_time_validation_applied = True


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
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    if not np.all(np.isclose(numeric, np.rint(numeric), rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    return numeric.astype(int)
