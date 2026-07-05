"""Validate emission timing metadata before numeric coercion.

``LogEmissionTensor`` is a public data container used by tests, synthetic
benchmarks, and downstream scripts.  Its duration fields represent physical
seconds; accepting Python/NumPy booleans is almost always accidental, and would
otherwise silently coerce ``True`` to ``1.0`` or ``False`` to ``0.0`` before the
existing finite/positive checks run.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_emission_timing_validation_patch_applied"


def _contains_boolean_numeric(value: object) -> bool:
    """Return whether a scalar or array-like value contains boolean numerics."""

    try:
        values = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(values.dtype, np.bool_):
        return True
    if values.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in values.flat)
    return False


def _reject_boolean_numeric(name: str, value: object) -> None:
    if _contains_boolean_numeric(value):
        raise ValueError(f"{name} must be numeric, not boolean")


def _validate_log_emission_timing_fields(tensor: object) -> None:
    _reject_boolean_numeric("times", getattr(tensor, "times"))
    _reject_boolean_numeric("dt", getattr(tensor, "dt"))
    bin_durations = getattr(tensor, "bin_durations")
    if bin_durations is not None:
        _reject_boolean_numeric("bin_durations", bin_durations)
    transition_durations = getattr(tensor, "transition_durations")
    if transition_durations is not None:
        _reject_boolean_numeric("transition_durations", transition_durations)


def apply_emission_timing_validation_patch() -> None:
    """Install validation for direct ``LogEmissionTensor`` construction."""

    from . import encoding

    current = encoding.LogEmissionTensor.__post_init__
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def post_init(self):
        _validate_log_emission_timing_fields(self)
        return current(self)

    setattr(post_init, _PATCHED_FLAG, True)
    setattr(post_init, "__hipporeplayimm_original__", current)
    encoding.LogEmissionTensor.__post_init__ = post_init


__all__ = ["apply_emission_timing_validation_patch"]
