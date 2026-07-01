"""Handle one-bin KD second-order evidence without indexing a missing row."""

from __future__ import annotations

import numpy as np

_PATCHED_FLAG = "_kd_single_bin_momentum_patch2_applied"
_WRAPPER_ATTR = "_kd_single_bin_momentum_wrapper"


def _current_patch_installed(kd: object) -> bool:
    current = getattr(kd, "_second_order_separable_log_evidence", None)
    return bool(getattr(current, _WRAPPER_ATTR, False))


def apply_kd_single_bin_momentum_patch2() -> None:
    from . import kd_impossible_emission_patch as impossible_patch
    from . import kd_random_effects_validation as random_effects_validation
    from . import kd_reference as kd

    current = getattr(kd, "_second_order_separable_log_evidence", None)
    active_wrapper = current if getattr(current, _WRAPPER_ATTR, False) else None

    random_effects_validation.apply_kd_random_effects_validation_patch()
    impossible_patch.apply_kd_impossible_emission_patch()

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
