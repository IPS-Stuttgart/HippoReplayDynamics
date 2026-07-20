"""Reject invalid finite-displacement decoder configuration values."""

from __future__ import annotations

from functools import wraps

import numpy as np

from .state_space_utils import _is_boolean_scalar

_PATCHED_FLAG = "_displacement_config_bool_validation_patch_applied"
_ORIGINALS_ATTR = "_displacement_config_bool_validation_originals"
_WRAPPER_MARKER = "_displacement_config_bool_validation_wrapper"


def _reject_array_shaped_scalar(name: str, value: object) -> None:
    """Reject values that NumPy/Python might coerce from an array to a scalar."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if array.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")


def _reject_boolean_scalar(name: str, value: object) -> None:
    _reject_array_shaped_scalar(name, value)
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be numeric, not boolean")


def _coerce_nonnegative_integer_scalar(name: str, value: object) -> int:
    _reject_boolean_scalar(name, value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a nonnegative integer scalar") from exc
    if array.ndim != 0:
        raise TypeError(f"{name} must be a nonnegative integer scalar")
    try:
        numeric = float(array)
    except OverflowError as exc:
        raise ValueError(f"{name} must fit into integer range") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a nonnegative integer scalar") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer")
    integer_info = np.iinfo(np.dtype(int))
    if numeric < integer_info.min or numeric > integer_info.max:
        raise ValueError(f"{name} must fit into integer range")
    return int(numeric)


def _mark_bool_guard(wrapper):
    setattr(wrapper, _WRAPPER_MARKER, True)
    return wrapper


def _is_current_bool_guard(value: object) -> bool:
    return bool(getattr(value, _WRAPPER_MARKER, False))


def _wrappers_are_current(displacement_imm, displacement_momentum) -> bool:
    return (
        _is_current_bool_guard(displacement_momentum._displacement_lattice)
        and _is_current_bool_guard(displacement_momentum._positive_config_value)
        and _is_current_bool_guard(displacement_momentum._displacement_transition_sigma_cm_sqrt_s)
        and _is_current_bool_guard(displacement_momentum._score_displacement_momentum_exact)
        and _is_current_bool_guard(displacement_imm._score_displacement_imm_exact)
    )


def _originals(displacement_imm, displacement_momentum) -> dict[str, object]:
    originals = getattr(displacement_momentum, _ORIGINALS_ATTR, None)
    if originals is None:
        originals = {
            "lattice": displacement_momentum._displacement_lattice,
            "positive_config_value": displacement_momentum._positive_config_value,
            "transition_sigma": displacement_momentum._displacement_transition_sigma_cm_sqrt_s,
            "momentum_score": displacement_momentum._score_displacement_momentum_exact,
            "imm_score": displacement_imm._score_displacement_imm_exact,
        }
        setattr(displacement_momentum, _ORIGINALS_ATTR, originals)
    return originals


def apply_displacement_config_bool_validation_patch() -> None:
    """Install value guards for finite-displacement decoder configuration.

    Python booleans are subclasses of ``int`` and NumPy booleans cast cleanly to
    ``int``/``float``.  Fractional numeric values also cast through ``int(...)``.
    Single-element NumPy arrays can likewise cast through ``float(...)`` for scale
    parameters.  The finite-displacement state-space models use explicit numeric
    casts for lattice radii and scale parameters, so malformed values can
    otherwise silently change the displacement lattice or transition scale instead
    of failing fast.

    The module-level flag is not sufficient by itself: tests or downstream code
    can replace the guarded helpers while leaving the flag set.  Re-checking the
    wrapper marker lets the public runtime patch hook refresh stale helpers.
    """

    from . import state_space
    from . import state_space_displacement_imm as displacement_imm
    from . import state_space_displacement_momentum as displacement_momentum

    originals = _originals(displacement_imm, displacement_momentum)
    if getattr(displacement_momentum, _PATCHED_FLAG, False) and _wrappers_are_current(
        displacement_imm,
        displacement_momentum,
    ):
        _synchronize_aliases(state_space, displacement_imm, displacement_momentum)
        return

    original_lattice = originals["lattice"]
    original_positive_config_value = originals["positive_config_value"]
    original_transition_sigma = originals["transition_sigma"]
    original_momentum_score = originals["momentum_score"]
    original_imm_score = originals["imm_score"]

    @_mark_bool_guard
    @wraps(original_lattice)
    def displacement_lattice(bin_centers, *, radius_bins):
        radius = _coerce_nonnegative_integer_scalar("displacement_radius_bins", radius_bins)
        return original_lattice(bin_centers, radius_bins=radius)

    @_mark_bool_guard
    @wraps(original_positive_config_value)
    def positive_config_value(config, name: str, *, default: float):
        _reject_boolean_scalar(str(name), getattr(config, name, 0.0))
        return original_positive_config_value(config, name, default=default)

    @_mark_bool_guard
    @wraps(original_transition_sigma)
    def displacement_transition_sigma_cm_sqrt_s(config):
        raw_value = getattr(config, "displacement_transition_sigma_cm_sqrt_s", 0.0)
        _reject_boolean_scalar("displacement_transition_sigma_cm_sqrt_s", raw_value)
        try:
            uses_default = float(raw_value) == 0.0
        except (TypeError, ValueError):
            uses_default = False
        if uses_default:
            _reject_boolean_scalar("momentum_sigma_cm_sqrt_s", getattr(config, "momentum_sigma_cm_sqrt_s", 85.0))
        return original_transition_sigma(config)

    @_mark_bool_guard
    @wraps(original_momentum_score)
    def score_displacement_momentum_exact(emissions, bin_centers, config, transition_durations_s, *args, **kwargs):
        _coerce_nonnegative_integer_scalar("displacement_radius_bins", getattr(config, "displacement_radius_bins", 2))
        return original_momentum_score(emissions, bin_centers, config, transition_durations_s, *args, **kwargs)

    @_mark_bool_guard
    @wraps(original_imm_score)
    def score_displacement_imm_exact(emissions, bin_centers, config, transition_durations_s, *args, **kwargs):
        _coerce_nonnegative_integer_scalar("displacement_radius_bins", getattr(config, "displacement_radius_bins", 2))
        return original_imm_score(emissions, bin_centers, config, transition_durations_s, *args, **kwargs)

    displacement_momentum._displacement_lattice = displacement_lattice
    displacement_momentum._positive_config_value = positive_config_value
    displacement_momentum._displacement_transition_sigma_cm_sqrt_s = displacement_transition_sigma_cm_sqrt_s
    displacement_momentum._score_displacement_momentum_exact = score_displacement_momentum_exact
    displacement_imm._score_displacement_imm_exact = score_displacement_imm_exact
    setattr(displacement_momentum, _PATCHED_FLAG, True)
    _synchronize_aliases(state_space, displacement_imm, displacement_momentum)


def _synchronize_aliases(state_space, displacement_imm, displacement_momentum) -> None:
    """Refresh modules that imported displacement helpers by value."""

    displacement_imm._displacement_lattice = displacement_momentum._displacement_lattice
    displacement_imm._positive_config_value = displacement_momentum._positive_config_value
    for name in (
        "_displacement_lattice",
        "_positive_config_value",
        "_displacement_transition_sigma_cm_sqrt_s",
        "_score_displacement_momentum_exact",
    ):
        if hasattr(state_space, name):
            setattr(state_space, name, getattr(displacement_momentum, name))
    if hasattr(state_space, "_score_displacement_imm_exact"):
        state_space._score_displacement_imm_exact = displacement_imm._score_displacement_imm_exact


__all__ = ["apply_displacement_config_bool_validation_patch"]
