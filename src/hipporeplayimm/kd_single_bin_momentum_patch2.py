"""Handle one-bin KD second-order evidence without indexing a missing row."""

from __future__ import annotations

import numpy as np

_PATCHED_FLAG = "_kd_single_bin_momentum_patch2_applied"


def apply_kd_single_bin_momentum_patch2() -> None:
    from . import kd_reference as kd

    if getattr(kd, _PATCHED_FLAG, False):
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

    kd._second_order_separable_log_evidence = second_order
    setattr(kd, _PATCHED_FLAG, True)
