"""Preserve configured replay-bin width as the scalar emission ``dt``.

Partial final replay bins have their realized exposure in ``bin_durations`` and
their center-to-center dynamics interval in ``transition_durations``.  The
scalar ``dt`` remains the nominal decoder bin width because legacy per-step
parameters (for example momentum velocity decay) use it as their reference
interval.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_PATCHED_ATTR = "_nominal_emission_dt_wrapper"


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
        emissions.dt = float(config.time_bin_s)
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
        emissions.dt = float(time_bin_s)
        return emissions

    setattr(wrapped, _PATCHED_ATTR, True)
    return wrapped


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


__all__ = ["apply_nominal_emission_dt_patch"]
