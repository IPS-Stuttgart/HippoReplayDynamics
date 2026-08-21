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
_EMISSION_TIMESTAMP_WRAPPER_FLAG = "_explicit_transition_timestamp_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_TRAJECTORY_IMM_VALIDATION_FLAG = "_trajectory_imm_parameter_validation_patch_applied"
_TRAJECTORY_IMM_VALIDATION_WRAPPER_ATTR = (
    "_trajectory_imm_parameter_validation_wrapper"
)
_TRAJECTORY_IMM_VALIDATION_SOURCE = "model_parameter_validation.py"


def apply_continuous_time_transition_duration_patch() -> None:
    """Keep continuous-time transition durations equal to timestamp differences."""

    from . import accuracy_upgrades
    from .continuous_time_imm_transition_patch import (
        apply_continuous_time_imm_transition_patch,
    )

    _restore_trajectory_imm_reload_patches()
    apply_continuous_time_imm_transition_patch()
    _wrap_log_emission_timestamp_validation()
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


def _wrapper_chain(function: Any):
    """Yield one wrapper lineage without trusting copied marker attributes."""

    seen: set[int] = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        yield current
        original = getattr(current, _ORIGINAL_ATTR, None)
        wrapped = getattr(current, "__wrapped__", None)
        if callable(original) and original is not current:
            current = original
        elif callable(wrapped) and wrapped is not current:
            current = wrapped
        else:
            break


def _adopt_trajectory_imm_validation_wrapper(function: Any) -> bool:
    """Find and mark the live validation wrapper in a composed wrapper chain.

    The validation wrapper predates function-level sentinels. Identify it once
    by its defining source/code name and attach a self-owned marker. Later
    ``functools.wraps`` layers may copy that attribute, but only the actual
    validation wrapper owns a marker that points back to itself.
    """

    for wrapper in _wrapper_chain(function):
        if (
            getattr(wrapper, _TRAJECTORY_IMM_VALIDATION_WRAPPER_ATTR, None)
            is wrapper
        ):
            return True
        code = getattr(wrapper, "__code__", None)
        if code is None:
            continue
        source_name = str(getattr(code, "co_filename", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if (
            source_name == _TRAJECTORY_IMM_VALIDATION_SOURCE
            and getattr(code, "co_name", "") == "trajectory_imm_mode_stickiness"
        ):
            setattr(
                wrapper,
                _TRAJECTORY_IMM_VALIDATION_WRAPPER_ATTR,
                wrapper,
            )
            return True
    return False


def _restore_trajectory_imm_reload_patches() -> None:
    """Restore trajectory-IMM validation and diagnostics discarded by reload.

    ``importlib.reload`` executes a module in its existing namespace. Patch-only
    module flags can therefore survive while freshly defined functions replace
    their wrappers. Use the live wrapper chain as the source of truth, replay
    validation only when its wrapper is absent, and replay the independently
    idempotent diagnostic patch on every runtime refresh.
    """

    from . import model_parameter_validation
    from . import state_space_trajectory_imm as trajectory_imm
    from .trajectory_imm_single_bin_diagnostics import (
        apply_trajectory_imm_single_bin_diagnostics_patch,
    )

    validation_sentinel = trajectory_imm._trajectory_imm_mode_stickiness
    if not _adopt_trajectory_imm_validation_wrapper(validation_sentinel):
        setattr(trajectory_imm, _TRAJECTORY_IMM_VALIDATION_FLAG, False)
        model_parameter_validation.apply_model_parameter_validation_patch()
        validation_sentinel = trajectory_imm._trajectory_imm_mode_stickiness
        if not _adopt_trajectory_imm_validation_wrapper(validation_sentinel):
            raise RuntimeError(
                "failed to restore trajectory-IMM parameter validation after reload"
            )
    setattr(trajectory_imm, _TRAJECTORY_IMM_VALIDATION_FLAG, True)

    apply_trajectory_imm_single_bin_diagnostics_patch()


def _wrap_log_emission_timestamp_validation() -> None:
    """Reject non-monotone timestamps even when durations are supplied explicitly.

    ``LogEmissionTensor`` is also used internally for time-reversed inference,
    where timestamps are intentionally strictly decreasing.  Therefore the
    container-level invariant is strict monotonicity in either direction; the
    forward continuous-time emission builder retains its stronger increasing-
    time contract above.
    """

    from . import encoding

    current = encoding.LogEmissionTensor.__post_init__
    if getattr(current, _EMISSION_TIMESTAMP_WRAPPER_FLAG, False):
        return

    @wraps(current)
    def post_init(self) -> None:
        current(self)
        times = np.asarray(self.times, dtype=float)
        if times.shape == (0,) or times.size <= 1:
            return
        if times.shape != (self.n_time,):
            raise ValueError("times must contain one value per emission row")
        differences = np.diff(times)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError("times must be strictly monotonic")

    setattr(post_init, _EMISSION_TIMESTAMP_WRAPPER_FLAG, True)
    setattr(post_init, _ORIGINAL_ATTR, current)
    encoding.LogEmissionTensor.__post_init__ = post_init


def _synchronize_builder_aliases(active: Any) -> None:
    """Refresh package modules that imported the builder before patching."""

    lineage: set[Any] = set()
    current = active
    while callable(current) and current not in lineage:
        lineage.add(current)
        current = getattr(current, _ORIGINAL_ATTR, None)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name != "hipporeplayimm" and not module_name.startswith(
            "hipporeplayimm."
        ):
            continue
        alias = getattr(module, "build_continuous_time_emissions", None)
        if callable(alias) and alias in lineage and alias is not active:
            setattr(module, "build_continuous_time_emissions", active)


__all__ = ["apply_continuous_time_transition_duration_patch"]
