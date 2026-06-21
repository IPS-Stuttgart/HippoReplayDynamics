"""Validation helpers for synthetic latent-path simulation."""

from __future__ import annotations

from functools import wraps
from typing import Any


def apply_latent_path_validation_patch() -> None:
    """Reject empty direct latent-path simulations with a clear ValueError."""

    import hipporeplayimm.simulation_recovery as recovery

    if getattr(recovery, "_latent_path_n_time_validation_applied", False):
        return

    original = recovery.simulate_latent_path

    @wraps(original)
    def checked_simulate_latent_path(*args: Any, **kwargs: Any) -> Any:
        if "n_time" in kwargs and int(kwargs["n_time"]) <= 0:
            raise ValueError("n_time must be positive")
        return original(*args, **kwargs)

    recovery.simulate_latent_path = checked_simulate_latent_path
    recovery._latent_path_n_time_validation_applied = True
