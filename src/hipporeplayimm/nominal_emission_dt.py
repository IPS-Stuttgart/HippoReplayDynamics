"""Preserve configured replay-bin width as the scalar emission ``dt``.

Partial final replay bins have their realized exposure in ``bin_durations`` and
their center-to-center dynamics interval in ``transition_durations``.  The
scalar ``dt`` remains the nominal decoder bin width because legacy per-step
parameters (for example momentum velocity decay) use it as their reference
interval.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any, Callable

import numpy as np

_PATCHED_ATTR = "_nominal_emission_dt_wrapper"


def _positive_finite_dt(value: object, *, name: str) -> float:
    """Return a positive finite scalar duration without Boolean coercion."""

    current = value
    seen: set[int] = set()
    while True:
        if isinstance(current, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite positive duration, not boolean")
        try:
            scalar = np.asarray(current)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive scalar duration") from exc
        if scalar.ndim != 0:
            raise ValueError(f"{name} must be a finite positive scalar duration")
        if np.issubdtype(scalar.dtype, np.bool_):
            raise ValueError(f"{name} must be a finite positive duration, not boolean")
        if scalar.dtype != object:
            current = scalar.item()
            break
        marker = id(scalar)
        if marker in seen:
            raise ValueError(f"{name} must be a finite positive scalar duration")
        seen.add(marker)
        item = scalar.item()
        if item is current:
            break
        current = item

    if isinstance(current, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive duration, not boolean")
    try:
        duration = float(current)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar duration") from exc
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return duration


def _configured_builder(
    builder: Callable[..., Any],
    *,
    default_config_factory: Callable[[], Any],
) -> Callable[..., Any]:
    """Return a builder wrapper that restores ``config.time_bin_s`` as ``dt``."""

    if getattr(builder, _PATCHED_ATTR, False):
        return builder

    @wraps(builder)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        emissions = builder(*args, **kwargs)
        config = kwargs.get("config")
        if config is None and len(args) >= 4:
            config = args[3]
        if config is None:
            config = default_config_factory()
        emissions.dt = _positive_finite_dt(config.time_bin_s, name="config.time_bin_s")
        return emissions

    setattr(wrapped, _PATCHED_ATTR, True)
    return wrapped


def _kd_builder(builder: Callable[..., Any]) -> Callable[..., Any]:
    """Return a KD builder wrapper that restores its explicit nominal bin width."""

    if getattr(builder, _PATCHED_ATTR, False):
        return builder

    @wraps(builder)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        emissions = builder(*args, **kwargs)
        if "time_bin_s" in kwargs:
            time_bin_s = kwargs["time_bin_s"]
        elif len(args) >= 4:
            time_bin_s = args[3]
        else:  # pragma: no cover - the wrapped builder raises before returning.
            raise TypeError("build_kd_emissions requires time_bin_s")
        emissions.dt = _positive_finite_dt(time_bin_s, name="time_bin_s")
        return emissions

    setattr(wrapped, _PATCHED_ATTR, True)
    return wrapped


def _synchronize_builder_aliases(
    *,
    build_emissions: Callable[..., Any],
    build_kd_emissions: Callable[..., Any],
    build_clusterless_mark_emissions: Callable[..., Any],
) -> None:
    """Refresh package modules that imported an emission builder by value."""

    active = {
        "build_emissions": build_emissions,
        "build_kd_emissions": build_kd_emissions,
        "build_clusterless_mark_emissions": build_clusterless_mark_emissions,
    }
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        for name, builder in active.items():
            if hasattr(module, name):
                setattr(module, name, builder)


def apply_nominal_emission_dt_patch() -> None:
    """Keep scalar ``dt`` nominal across all replay emission builders."""

    from . import clusterless
    from . import encoding
    from . import kd_reference

    encoding.build_emissions = _configured_builder(
        encoding.build_emissions,
        default_config_factory=encoding.EmissionConfig,
    )
    clusterless.build_clusterless_mark_emissions = _configured_builder(
        clusterless.build_clusterless_mark_emissions,
        default_config_factory=encoding.EmissionConfig,
    )
    kd_reference.build_kd_emissions = _kd_builder(kd_reference.build_kd_emissions)
    _synchronize_builder_aliases(
        build_emissions=encoding.build_emissions,
        build_kd_emissions=kd_reference.build_kd_emissions,
        build_clusterless_mark_emissions=clusterless.build_clusterless_mark_emissions,
    )


__all__ = ["apply_nominal_emission_dt_patch"]
