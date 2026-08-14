"""Refresh state-space momentum velocity-decay guards after helper replacement.

``model_parameter_validation`` installs the original numeric guards on the
state-space momentum helpers.  Its historical module-level sentinel can survive
``importlib.reload(state_space_model)`` or targeted helper replacement while the
wrapped functions themselves disappear.  This small compatibility layer marks
the live callables, so the public runtime-patch hook can repair that stale state
without relying on the coarse module flag.
"""

from __future__ import annotations

from functools import wraps

_PATCHED_FLAG = "_state_space_velocity_decay_runtime_refresh_patch_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_state_space_velocity_decay_runtime_refresh_patch() -> None:
    """Ensure both state-space momentum helpers validate scalar decay bounds."""

    from . import state_space_model
    from .model_parameter_validation import _validate_config_momentum_velocity_decay

    original_decays = state_space_model._momentum_velocity_decays
    if not getattr(original_decays, _PATCHED_FLAG, False):

        @wraps(original_decays)
        def momentum_velocity_decays(config, transition_durations):
            _validate_config_momentum_velocity_decay(config)
            return original_decays(config, transition_durations)

        setattr(momentum_velocity_decays, _PATCHED_FLAG, True)
        setattr(momentum_velocity_decays, _ORIGINAL_ATTR, original_decays)
        state_space_model._momentum_velocity_decays = momentum_velocity_decays

    original_multipliers = state_space_model._momentum_prediction_multipliers
    if not getattr(original_multipliers, _PATCHED_FLAG, False):

        @wraps(original_multipliers)
        def momentum_prediction_multipliers(
            config,
            transition_durations,
            *,
            fallback_dt,
        ):
            _validate_config_momentum_velocity_decay(config)
            return original_multipliers(
                config,
                transition_durations,
                fallback_dt=fallback_dt,
            )

        setattr(momentum_prediction_multipliers, _PATCHED_FLAG, True)
        setattr(momentum_prediction_multipliers, _ORIGINAL_ATTR, original_multipliers)
        state_space_model._momentum_prediction_multipliers = momentum_prediction_multipliers

    setattr(state_space_model, _PATCHED_FLAG, True)


__all__ = ["apply_state_space_velocity_decay_runtime_refresh_patch"]
