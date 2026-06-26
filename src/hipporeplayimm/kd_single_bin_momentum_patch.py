"""Handle one-bin KD momentum evidence without entering second-order recursion."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_kd_single_bin_momentum_patch_applied"


def apply_kd_single_bin_momentum_patch() -> None:
    """Patch KD momentum evidence to support single-bin emissions."""

    from . import kd_reference as kd

    if getattr(kd, _PATCHED_FLAG, False):
        return

    original = kd._second_order_separable_log_evidence

    @wraps(original)
    def second_order_separable_log_evidence(
        log_emissions: np.ndarray,
        n_bins: int,
        initial: np.ndarray,
        transition: np.ndarray,
    ) -> float:
        values = np.asarray(log_emissions, dtype=float)
        if values.shape[0] != 1:
            return original(log_emissions, n_bins, initial, transition)
        emission, offset = kd._scaled_emission(values, 0)
        alpha = emission.reshape(n_bins, n_bins) / values.shape[1]
        mass = float(alpha.sum())
        if mass <= 0.0:
            return float("-inf")
        return float(np.log(mass) + offset)

    kd._second_order_separable_log_evidence = second_order_separable_log_evidence
    setattr(kd, _PATCHED_FLAG, True)


__all__ = ["apply_kd_single_bin_momentum_patch"]
