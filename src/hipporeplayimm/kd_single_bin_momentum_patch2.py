"""Handle one-bin and impossible-row KD evidence edge cases."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_kd_single_bin_momentum_patch2_applied"
_WRAPPER_ATTR = "_kd_single_bin_momentum_wrapper"
_VARIABLE_DURATION_DIFFUSION_WRAPPER_ATTR = (
    "_kd_variable_duration_diffusion_impossible_first_row_wrapper"
)
_VARIABLE_DURATION_MOMENTUM_WRAPPER_ATTR = (
    "_kd_variable_duration_momentum_impossible_first_row_wrapper"
)


def _current_patch_installed(kd: object) -> bool:
    current = getattr(kd, "_second_order_separable_log_evidence", None)
    return bool(getattr(current, _WRAPPER_ATTR, False))


def _first_emission_row_has_no_mass(kd: object, log_emissions: object) -> bool:
    emission, _ = kd._scaled_emission(log_emissions, 0)
    return not np.any(emission > 0.0)


def _patch_variable_duration_impossible_first_rows(kd: object) -> None:
    """Return exact impossible evidence before duration recursions divide by zero."""

    current_diffusion = kd.kd_diffusion_log_evidence_from_transition
    if not getattr(
        current_diffusion,
        _VARIABLE_DURATION_DIFFUSION_WRAPPER_ATTR,
        False,
    ):

        @wraps(current_diffusion)
        def diffusion(log_emissions, n_bins_x, n_bins_y, transition):
            if isinstance(transition, (list, tuple)) and _first_emission_row_has_no_mass(
                kd,
                log_emissions,
            ):
                return float("-inf")
            return current_diffusion(
                log_emissions,
                n_bins_x,
                n_bins_y,
                transition,
            )

        setattr(diffusion, _VARIABLE_DURATION_DIFFUSION_WRAPPER_ATTR, True)
        kd.kd_diffusion_log_evidence_from_transition = diffusion

    current_momentum = kd.kd_momentum_log_evidence_from_transitions
    if not getattr(
        current_momentum,
        _VARIABLE_DURATION_MOMENTUM_WRAPPER_ATTR,
        False,
    ):

        @wraps(current_momentum)
        def momentum(log_emissions, n_bins, initial, transition):
            if isinstance(transition, (list, tuple)) and _first_emission_row_has_no_mass(
                kd,
                log_emissions,
            ):
                return float("-inf")
            return current_momentum(
                log_emissions,
                n_bins,
                initial,
                transition,
            )

        setattr(momentum, _VARIABLE_DURATION_MOMENTUM_WRAPPER_ATTR, True)
        kd.kd_momentum_log_evidence_from_transitions = momentum


def apply_kd_single_bin_momentum_patch2() -> None:
    from . import kd_impossible_emission_patch as impossible_patch
    from . import kd_random_effects_validation as random_effects_validation
    from . import kd_reference as kd

    current = getattr(kd, "_second_order_separable_log_evidence", None)
    active_wrapper = current if getattr(current, _WRAPPER_ATTR, False) else None

    random_effects_validation.apply_kd_random_effects_validation_patch()
    impossible_patch.apply_kd_impossible_emission_patch()
    _patch_variable_duration_impossible_first_rows(kd)

    if active_wrapper is not None:
        kd._second_order_separable_log_evidence = active_wrapper
        setattr(kd, _PATCHED_FLAG, True)
        return

    original = kd._second_order_separable_log_evidence

    def second_order(log_emissions, n_bins, initial, transition):
        values = np.asarray(log_emissions, dtype=float)
        if values.ndim == 2 and values.shape[0] == 1:
            emission, offset = kd._scaled_emission(values, 0)
            alpha = emission.reshape(int(n_bins), int(n_bins)) / values.shape[1]
            mass = float(alpha.sum())
            return -np.inf if mass <= 0.0 else float(np.log(mass) + offset)
        return original(log_emissions, n_bins, initial, transition)

    setattr(second_order, _WRAPPER_ATTR, True)
    kd._second_order_separable_log_evidence = second_order
    setattr(kd, _PATCHED_FLAG, True)
