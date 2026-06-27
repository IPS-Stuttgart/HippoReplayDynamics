"""Patch KD momentum evidence for one-bin events."""

from __future__ import annotations

import numpy as np

_PATCHED_FLAG = "_kd_single_bin_momentum_patch_applied"


def apply_kd_single_bin_momentum_patch() -> None:
    """Return the one-bin spatial marginal for second-order KD recursions."""

    from . import kd_reference as kd

    if getattr(kd, _PATCHED_FLAG, False):
        return

    original = kd._second_order_separable_log_evidence

    def second_order_separable_log_evidence(log_emissions, n_bins, initial, transition):
        values = np.asarray(log_emissions, dtype=float)
        if values.shape[0] != 1:
            return original(log_emissions, n_bins, initial, transition)
        emission, offset = kd._scaled_emission(values, 0)
        alpha = emission.reshape(n_bins, n_bins) / values.shape[1]
        normalizer = float(alpha.sum())
        if normalizer <= 0.0:
            return -np.inf
        return float(np.log(normalizer) + offset)

    kd._second_order_separable_log_evidence = second_order_separable_log_evidence
    setattr(kd, _PATCHED_FLAG, True)


__all__ = ["apply_kd_single_bin_momentum_patch"]
