"""Preserve continuous-time dynamics intervals from emission timestamps.

Continuous-time emissions use ``min_interval_s`` to keep Poisson bins numerically
well-defined. That lower bound belongs to the observation bins only. Applying
it again to center-to-center transition durations makes duration-aware dynamics
evolve over more time than the recorded timestamps contain.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_continuous_time_transition_duration_patch_applied"
_WRAPPER_FLAG = "_continuous_time_transition_duration_wrapper"
_ACCURACY_WRAPPER_FLAG = "_accuracy_replay_gain_gamma_continuous_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_continuous_time_transition_duration_patch() -> None:
    """Keep continuous-time transition durations equal to timestamp differences."""

    from . import accuracy_upgrades

    current = accuracy_upgrades.build_continuous_time_emissions
    if getattr(current, _WRAPPER_FLAG, False):
        setattr(accuracy_upgrades, _PATCHED_FLAG, True)
        _synchronize_builder_aliases(current)
        return

    @wraps(current)
    def build_continuous_time_emissions(
        session: Any,
        encoding: Any,
        ripple: Any,
        config: Any = None,
    ):
        emissions = current(session, encoding, ripple, config)
        times = np.asarray(emissions.times, dtype=float)
        if times.shape != (emissions.n_time,):
            raise ValueError("continuous-time emission times must contain one timestamp per row")
        if not np.all(np.isfinite(times)):
            raise ValueError("continuous-time emission timestamps must be finite")

        transition_durations = (
            np.diff(times)
            if times.shape[0] > 1
            else np.empty(0, dtype=float)
        )
        if not np.all(np.isfinite(transition_durations)) or np.any(
            transition_durations <= 0.0
        ):
            raise ValueError(
                "continuous-time emission timestamps must be finite and strictly increasing"
            )
        emissions.transition_durations = transition_durations
        return emissions

    setattr(build_continuous_time_emissions, _WRAPPER_FLAG, True)
    # The validation patch treats this marker as its idempotence contract. Keep
    # it on the outer wrapper so repeated apply_runtime_patches() calls do not
    # rebuild the wrapper stack.
    setattr(build_continuous_time_emissions, _ACCURACY_WRAPPER_FLAG, True)
    setattr(build_continuous_time_emissions, _ORIGINAL_ATTR, current)
    accuracy_upgrades.build_continuous_time_emissions = build_continuous_time_emissions
    setattr(accuracy_upgrades, _PATCHED_FLAG, True)
    _synchronize_builder_aliases(build_continuous_time_emissions)


def _synchronize_builder_aliases(active: Any) -> None:
    """Refresh package modules that imported the builder before patching."""

    lineage: set[Any] = set()
    current = active
    while callable(current) and current not in lineage:
        lineage.add(current)
        current = getattr(current, _ORIGINAL_ATTR, None)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        alias = getattr(module, "build_continuous_time_emissions", None)
        if callable(alias) and alias in lineage and alias is not active:
            setattr(module, "build_continuous_time_emissions", active)


__all__ = ["apply_continuous_time_transition_duration_patch"]
