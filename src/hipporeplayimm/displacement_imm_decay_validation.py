"""Keep displacement-IMM velocity-decay validation synchronized."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_displacement_imm_velocity_decay_validation_patch_applied"
_DURATION_SCALE_PATCHED_FLAG = "_displacement_duration_scale_validation_patch_applied"


def apply_displacement_imm_decay_validation_patch() -> None:
    """Validate displacement-momentum helpers after direct submodule imports.

    ``state_space_displacement_imm`` imports duration helpers by value from
    ``state_space_displacement_momentum``. Runtime validation therefore has to
    rebind both the source helper and the displacement-IMM alias; otherwise stale
    aliases can bypass the package-level validation path.
    """

    from . import state_space_displacement_imm, state_space_displacement_momentum
    from .model_parameter_validation import (
        _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG,
        _validate_config_momentum_velocity_decay,
    )

    patched = state_space_displacement_momentum._duration_adjusted_decays
    if not getattr(state_space_displacement_momentum, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, False):
        original = patched

        @wraps(original)
        def duration_adjusted_decays(config, durations, reference_dt):
            _validate_config_momentum_velocity_decay(config)
            return original(config, durations, reference_dt)

        setattr(duration_adjusted_decays, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, True)
        setattr(duration_adjusted_decays, _PATCHED_FLAG, True)
        setattr(duration_adjusted_decays, "__hipporeplayimm_original__", original)
        state_space_displacement_momentum._duration_adjusted_decays = duration_adjusted_decays
        setattr(state_space_displacement_momentum, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, True)
        patched = duration_adjusted_decays
    else:
        setattr(patched, _PATCHED_FLAG, True)

    state_space_displacement_imm._duration_adjusted_decays = patched
    setattr(state_space_displacement_imm, _PATCHED_FLAG, True)

    scale_at = state_space_displacement_momentum._duration_scale_at
    if not getattr(scale_at, _DURATION_SCALE_PATCHED_FLAG, False):
        original_scale_at = scale_at

        @wraps(original_scale_at)
        def duration_scale_at(durations, transition_index, reference_dt):
            durations = state_space_displacement_momentum._validate_transition_duration_array(durations)
            reference_dt = float(reference_dt)
            if not np.isfinite(reference_dt) or reference_dt <= 0.0:
                raise ValueError("reference dt must be finite and positive")
            if durations.size == 0:
                return original_scale_at(durations, transition_index, reference_dt)
            if int(transition_index) != transition_index or int(transition_index) < 0 or int(transition_index) >= durations.size:
                raise ValueError("transition index must reference an existing transition duration")
            return original_scale_at(durations, int(transition_index), reference_dt)

        setattr(duration_scale_at, _DURATION_SCALE_PATCHED_FLAG, True)
        setattr(duration_scale_at, "__hipporeplayimm_original__", original_scale_at)
        state_space_displacement_momentum._duration_scale_at = duration_scale_at
        scale_at = duration_scale_at

    state_space_displacement_imm._duration_scale_at = scale_at
    setattr(state_space_displacement_imm, _DURATION_SCALE_PATCHED_FLAG, True)


__all__ = ["apply_displacement_imm_decay_validation_patch"]
