"""Reject boolean finite-displacement decoder configuration values."""

from __future__ import annotations

from functools import wraps

from .state_space_utils import _is_boolean_scalar

_PATCHED_FLAG = "_displacement_config_bool_validation_patch_applied"


def _reject_boolean_scalar(name: str, value: object) -> None:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be numeric, not boolean")


def apply_displacement_config_bool_validation_patch() -> None:
    """Install bool guards for finite-displacement decoder configuration.

    Python booleans are subclasses of ``int`` and NumPy booleans cast cleanly to
    ``int``/``float``.  The finite-displacement state-space models use explicit
    numeric casts for lattice radii and scale parameters, so a misspecified
    boolean can otherwise silently change the displacement lattice or transition
    scale instead of failing fast.
    """

    from . import state_space
    from . import state_space_displacement_imm as displacement_imm
    from . import state_space_displacement_momentum as displacement_momentum

    if getattr(displacement_momentum, _PATCHED_FLAG, False):
        _synchronize_aliases(state_space, displacement_imm, displacement_momentum)
        return

    original_lattice = displacement_momentum._displacement_lattice
    original_positive_config_value = displacement_momentum._positive_config_value
    original_transition_sigma = displacement_momentum._displacement_transition_sigma_cm_sqrt_s
    original_momentum_score = displacement_momentum._score_displacement_momentum_exact
    original_imm_score = displacement_imm._score_displacement_imm_exact

    @wraps(original_lattice)
    def displacement_lattice(bin_centers, *, radius_bins):
        _reject_boolean_scalar("displacement_radius_bins", radius_bins)
        return original_lattice(bin_centers, radius_bins=radius_bins)

    @wraps(original_positive_config_value)
    def positive_config_value(config, name: str, *, default: float):
        _reject_boolean_scalar(str(name), getattr(config, name, 0.0))
        return original_positive_config_value(config, name, default=default)

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

    @wraps(original_momentum_score)
    def score_displacement_momentum_exact(emissions, bin_centers, config, transition_durations_s, *args, **kwargs):
        _reject_boolean_scalar("displacement_radius_bins", getattr(config, "displacement_radius_bins", 2))
        return original_momentum_score(emissions, bin_centers, config, transition_durations_s, *args, **kwargs)

    @wraps(original_imm_score)
    def score_displacement_imm_exact(emissions, bin_centers, config, transition_durations_s, *args, **kwargs):
        _reject_boolean_scalar("displacement_radius_bins", getattr(config, "displacement_radius_bins", 2))
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
