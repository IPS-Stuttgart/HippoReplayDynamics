import importlib
import numpy as np

_PATCHED_FLAG = "_short_event_patch_applied"


def apply_short_event_patch() -> None:
    module = importlib.import_module("hipporeplayimm.kd_reference")
    if getattr(module, _PATCHED_FLAG, False):
        return
    original = module._second_order_separable_log_evidence

    def helper(log_emissions, n_bins, initial, transition):
        values = np.asarray(log_emissions, dtype=float)
        if values.ndim == 2 and values.shape[0] == 1:
            scaled, offset = module._scaled_emission(values, 0)
            weights = scaled.reshape(int(n_bins), int(n_bins)) / values.shape[1]
            mass = float(weights.sum())
            if mass <= 0.0:
                return -np.inf
            return float(np.log(mass) + offset)
        return original(log_emissions, n_bins, initial, transition)

    module._second_order_separable_log_evidence = helper
    setattr(module, _PATCHED_FLAG, True)
