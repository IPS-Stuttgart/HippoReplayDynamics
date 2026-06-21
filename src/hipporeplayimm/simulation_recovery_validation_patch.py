"""Runtime validation patches for simulation recovery helpers."""

from __future__ import annotations

from functools import wraps
from typing import Any


def apply_simulation_recovery_validation_patch() -> None:
    """Make direct latent-path simulation reject empty paths clearly."""

    import hipporeplayimm.simulation_recovery as recovery

    if getattr(recovery, "_latent_path_n_time_validation_patch_applied", False):
        return

    original_simulate_latent_path = recovery.simulate_latent_path

    @wraps(original_simulate_latent_path)
    def simulate_latent_path_with_n_time_validation(*args: Any, **kwargs: Any) -> Any:
        if "n_time" in kwargs and int(kwargs["n_time"]) <= 0:
            raise ValueError("n_time must be positive")
        return original_simulate_latent_path(*args, **kwargs)

    simulate_latent_path_with_n_time_validation._validates_positive_n_time = True  # type: ignore[attr-defined]
    recovery.simulate_latent_path = simulate_latent_path_with_n_time_validation
    recovery._latent_path_n_time_validation_patch_applied = True
