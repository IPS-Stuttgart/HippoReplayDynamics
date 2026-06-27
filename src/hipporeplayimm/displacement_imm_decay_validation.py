"""Keep displacement-IMM velocity-decay validation synchronized."""

from __future__ import annotations

from functools import wraps

_PATCHED_FLAG = "_displacement_imm_velocity_decay_validation_patch_applied"


def apply_displacement_imm_decay_validation_patch() -> None:
    """Validate displacement-IMM velocity decay even after direct submodule imports.

    ``state_space_displacement_imm`` imports ``_duration_adjusted_decays`` by value
    from ``state_space_displacement_momentum``. The displacement-momentum helper
    is patched at runtime to reject boolean and out-of-range velocity-decay
    settings, but a stale displacement-IMM alias can otherwise continue to call
    the unvalidated original helper. Rebinding the alias is idempotent and keeps
    the finite-displacement IMM scorer on the same validation path as the
    displacement-momentum scorer.
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


__all__ = ["apply_displacement_imm_decay_validation_patch"]
