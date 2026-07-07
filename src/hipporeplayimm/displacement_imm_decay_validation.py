"""Keep displacement/trajectory IMM velocity-decay validation synchronized."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_displacement_imm_velocity_decay_validation_patch_applied"
_DURATION_SCALE_PATCHED_FLAG = "_displacement_duration_scale_validation_patch_applied"
_TRAJECTORY_IMM_DECAY_PATCHED_FLAG = "_trajectory_imm_velocity_decay_validation_patch_applied"
_CANDIDATE_KINEMATIC_DECAY_PATCHED_FLAG = "_candidate_kinematic_velocity_decay_validation_patch_applied"


def _apply_candidate_kinematic_velocity_decay_validation_patch() -> None:
    """Install an explicit CandidateKinematicModel velocity-decay guard."""

    from . import models
    from .model_parameter_validation import _validate_unit_interval_parameter

    cls = models.CandidateKinematicModel
    current = cls.__post_init__
    if getattr(current, _CANDIDATE_KINEMATIC_DECAY_PATCHED_FLAG, False):
        return

    @wraps(current)
    def post_init(self):
        _validate_unit_interval_parameter("velocity_decay", self.velocity_decay)
        return current(self)

    setattr(post_init, _CANDIDATE_KINEMATIC_DECAY_PATCHED_FLAG, True)
    setattr(post_init, "__hipporeplayimm_original__", current)
    cls.__post_init__ = post_init


def apply_displacement_imm_decay_validation_patch() -> None:
    """Validate imported duration helpers after direct submodule imports.

    ``state_space_displacement_imm`` and ``state_space_trajectory_imm`` import
    duration helpers by value from the displacement/sparse momentum modules.
    Runtime validation therefore has to rebind those aliases; otherwise stale
    aliases can bypass the package-level validation path.
    """

    from . import (
        state_space_displacement_imm,
        state_space_displacement_momentum,
        state_space_sparse_momentum,
        state_space_trajectory_imm,
    )
    from .model_parameter_validation import (
        _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG,
        _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG,
        _validate_config_momentum_velocity_decay,
    )

    _apply_candidate_kinematic_velocity_decay_validation_patch()

    def validated_decay_helper(current, *, patch_flag: str, alias_flag: str):
        """Return a velocity-decay-validating wrapper for the current helper.

        The source modules also carry coarse "already patched" sentinels.  Those
        sentinels can survive module reloads or targeted helper replacement in
        tests/downstream notebooks, so wrapper freshness must be decided from the
        current callable itself rather than from the module-level flag alone.
        """

        if getattr(current, patch_flag, False):
            setattr(current, alias_flag, True)
            return current

        original = current

        @wraps(original)
        def duration_adjusted_decays(config, durations, reference_dt):
            _validate_config_momentum_velocity_decay(config)
            return original(config, durations, reference_dt)

        setattr(duration_adjusted_decays, patch_flag, True)
        setattr(duration_adjusted_decays, alias_flag, True)
        setattr(duration_adjusted_decays, "__hipporeplayimm_original__", original)
        return duration_adjusted_decays

    patched = validated_decay_helper(
        state_space_displacement_momentum._duration_adjusted_decays,
        patch_flag=_DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG,
        alias_flag=_PATCHED_FLAG,
    )
    state_space_displacement_momentum._duration_adjusted_decays = patched
    setattr(state_space_displacement_momentum, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, True)
    state_space_displacement_imm._duration_adjusted_decays = patched
    setattr(state_space_displacement_imm, _PATCHED_FLAG, True)

    sparse_patched = validated_decay_helper(
        state_space_sparse_momentum._duration_adjusted_decays,
        patch_flag=_SPARSE_MOMENTUM_DECAY_PATCHED_FLAG,
        alias_flag=_TRAJECTORY_IMM_DECAY_PATCHED_FLAG,
    )
    state_space_sparse_momentum._duration_adjusted_decays = sparse_patched
    setattr(state_space_sparse_momentum, _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG, True)
    state_space_trajectory_imm._duration_adjusted_decays = sparse_patched
    setattr(state_space_trajectory_imm, _TRAJECTORY_IMM_DECAY_PATCHED_FLAG, True)

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
