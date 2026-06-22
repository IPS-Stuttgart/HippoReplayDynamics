"""Runtime guards for duration-dependent momentum helpers.

The exact sparse and finite-displacement momentum decoders derive both velocity
decay and relative velocity scaling from transition durations.  Non-finite or
non-positive values should fail at the helper boundary instead of producing
NaN/inf transition parameters that later corrupt the dynamic program.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def apply_sparse_momentum_duration_validation_patch() -> None:
    """Install duration validation on momentum duration helper functions."""

    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum

    _patch_duration_helpers(sparse_momentum)
    _patch_duration_helpers(displacement_momentum)


def _patch_duration_helpers(module: Any) -> None:
    if getattr(module, "_duration_validation_patch_applied", False):
        return

    original_duration_adjusted_decays = module._duration_adjusted_decays
    original_time_scales = module._time_scales

    @wraps(original_duration_adjusted_decays)
    def duration_adjusted_decays(config: object, durations: Any, reference_dt: float) -> np.ndarray:
        return original_duration_adjusted_decays(
            config,
            _valid_transition_durations(durations),
            reference_dt,
        )

    @wraps(original_time_scales)
    def time_scales(durations: Any) -> np.ndarray:
        return original_time_scales(_valid_transition_durations(durations))

    module._duration_adjusted_decays = duration_adjusted_decays
    module._time_scales = time_scales
    module._duration_validation_patch_applied = True


def _valid_transition_durations(durations: Any) -> np.ndarray:
    values = np.asarray(durations, dtype=float)
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    return values
