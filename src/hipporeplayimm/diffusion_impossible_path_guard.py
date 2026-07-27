"""Clear undefined posterior outputs for impossible core diffusion paths."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_ATTR = "_diffusion_impossible_posterior_guard_applied"
_NO_FINITE_PATH = "no_finite_path"
_POSTERIOR_DIAGNOSTIC_KEYS = (
    "decoded_endpoint_x",
    "decoded_endpoint_y",
    "decoded_map_x",
    "decoded_map_y",
    "decoded_map_bin",
    "terminal_posterior_entropy",
)


def apply_diffusion_impossible_path_guard_patch() -> None:
    """Keep exact negative-infinite evidence without exposing NaN posteriors."""

    from . import models

    current = models.DiffusionModel.score
    if getattr(current, _PATCHED_ATTR, False):
        return

    @wraps(current)
    def score_with_impossible_posterior_guard(
        self: Any,
        emissions: Any,
        bin_centers: Any,
    ):
        with np.errstate(invalid="ignore"):
            result = current(self, emissions, bin_centers)
        if not np.isneginf(float(result.log_likelihood)):
            return result

        result.terminal_log_posterior = None
        result.trajectory_log_posterior = None
        result.diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        for key in _POSTERIOR_DIAGNOSTIC_KEYS:
            result.diagnostics.pop(key, None)
        result.diagnostics["diffusion_path_support"] = _NO_FINITE_PATH
        return result

    setattr(score_with_impossible_posterior_guard, _PATCHED_ATTR, True)
    setattr(
        score_with_impossible_posterior_guard,
        "__hipporeplayimm_original__",
        current,
    )
    models.DiffusionModel.score = score_with_impossible_posterior_guard


__all__ = ["apply_diffusion_impossible_path_guard_patch"]
