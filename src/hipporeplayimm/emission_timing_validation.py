"""Validate emission metadata before numeric coercion.

``LogEmissionTensor`` is a public data container used by tests, synthetic
benchmarks, and downstream scripts. Its duration fields and log-likelihood
entries are numeric; accepting Python/NumPy booleans is almost always accidental
and would otherwise silently coerce ``True`` to ``1.0`` or ``False`` to ``0.0``.

All numeric tensor fields are real-valued. NumPy's float coercion can silently
discard imaginary components from complex arrays and NumPy complex scalars, so
those inputs must be rejected before the wrapped constructor normalizes them.
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


def _contains_complex_numeric(value: object) -> bool:
    """Return whether a scalar or array-like value contains complex numerics."""

    try:
        values = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(values.dtype, np.complexfloating):
        return True
    if values.dtype == object:
        return any(isinstance(item, (complex, np.complexfloating)) for item in values.flat)
    return False


def _reject_boolean_numeric(name: str, value: object) -> None:
    if _contains_boolean_numeric(value):
        raise ValueError(f"{name} must be numeric, not boolean")


def _reject_complex_numeric(name: str, value: object) -> None:
    if _contains_complex_numeric(value):
        raise ValueError(f"{name} must contain real values, not complex values")


def _validate_log_emission_fields(tensor: object) -> None:
    _reject_boolean_numeric("log_likelihood", getattr(tensor, "log_likelihood"))
    _reject_boolean_numeric("times", getattr(tensor, "times"))
    _reject_boolean_numeric("dt", getattr(tensor, "dt"))
    bin_durations = getattr(tensor, "bin_durations")
    if bin_durations is not None:
        _reject_boolean_numeric("bin_durations", bin_durations)
    transition_durations = getattr(tensor, "transition_durations")
    if transition_durations is not None:
        _reject_boolean_numeric("transition_durations", transition_durations)

    for name in (
        "log_likelihood",
        "spike_counts",
        "times",
        "dt",
        "cell_ids",
        "n_spikes",
        "bin_durations",
        "transition_durations",
    ):
        value = getattr(tensor, name)
        if value is not None:
            _reject_complex_numeric(name, value)


def apply_emission_timing_validation_patch() -> None:
    """Install validation for direct ``LogEmissionTensor`` construction."""

    from . import encoding

    current = encoding.LogEmissionTensor.__post_init__
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def post_init(self):
        _validate_log_emission_fields(self)
        return current(self)

    setattr(post_init, _PATCHED_FLAG, True)
    setattr(post_init, "__hipporeplayimm_original__", current)
    encoding.LogEmissionTensor.__post_init__ = post_init


__all__ = ["apply_emission_timing_validation_patch"]
