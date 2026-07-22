"""Sample synthetic recovery counts from the configured count likelihood."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


_PATCHED_FLAG = "_simulation_recovery_overdispersion_sampling_applied"
_WRAPPER_MARKER = "_simulation_recovery_overdispersion_sampling_wrapper"


class _NegativeBinomialSamplingGenerator:
    """Delegate RNG operations while replacing Poisson count draws with NB draws."""

    def __init__(self, rng: Any, overdispersion: float) -> None:
        self._rng = rng
        self._overdispersion = float(overdispersion)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rng, name)

    def poisson(self, lam: Any, size: Any = None) -> Any:
        mean = np.asarray(lam, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            dispersion_size = 1.0 / self._overdispersion
            scaled_mean = self._overdispersion * mean
            success_probability = 1.0 / (1.0 + scaled_mean)
        if not np.isfinite(dispersion_size) or np.all(
            success_probability == 1.0
        ):
            return self._rng.poisson(lam, size=size)
        return self._rng.negative_binomial(
            dispersion_size,
            success_probability,
            size=size,
        )


def apply_simulation_recovery_overdispersion_patch() -> None:
    """Make synthetic count sampling match the configured emission likelihood."""

    from . import simulation_recovery as recovery

    current = recovery.simulate_replay_event
    if getattr(current, _WRAPPER_MARKER, False):
        setattr(recovery, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def simulate_replay_event(*args: Any, **kwargs: Any) -> Any:
        overdispersion = _finite_nonnegative_scalar(
            "negative_binomial_overdispersion",
            kwargs.get("negative_binomial_overdispersion", 0.0),
        )
        if overdispersion == 0.0 or "rng" not in kwargs:
            return current(*args, **kwargs)

        patched_kwargs = dict(kwargs)
        patched_kwargs["rng"] = _NegativeBinomialSamplingGenerator(
            kwargs["rng"],
            overdispersion,
        )
        return current(*args, **patched_kwargs)

    setattr(simulate_replay_event, _WRAPPER_MARKER, True)
    setattr(simulate_replay_event, "__hipporeplayimm_original__", current)
    recovery.simulate_replay_event = simulate_replay_event
    setattr(recovery, _PATCHED_FLAG, True)


def _finite_nonnegative_scalar(name: str, value: Any) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must be finite and nonnegative")
    scalar = raw.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


__all__ = ["apply_simulation_recovery_overdispersion_patch"]
