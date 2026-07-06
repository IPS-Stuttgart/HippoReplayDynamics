"""Compatibility patch for one-dimensional first-order IMM diagnostic grids."""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np

_PATCH_ATTR = "_first_order_imm_1d_diagnostics_patch"
_ORIGINAL_ATTR = "_first_order_imm_1d_diagnostics_original"


def _as_content_bin_centers(bin_centers: np.ndarray, n_bins: int) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[0] != n_bins or centers.shape[1] < 1:
        raise ValueError("bin_centers must contain one coordinate row per spatial bin")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must be finite")
    return centers


def _wrap_helper(helper: Callable[..., dict[str, float | int]]) -> Callable[..., dict[str, float | int]]:
    if getattr(helper, _PATCH_ATTR, False):
        return helper

    def one_dimensional_first_order_imm_content_diagnostics(
        mode_posterior: np.ndarray,
        trajectory_log_posterior: np.ndarray,
        bin_centers: np.ndarray,
        dt_s: float,
    ) -> dict[str, float | int]:
        trajectory = np.asarray(trajectory_log_posterior, dtype=float)
        n_bins = int(trajectory.shape[1]) if trajectory.ndim == 2 else -1
        centers = _as_content_bin_centers(bin_centers, n_bins)
        return helper(mode_posterior, trajectory_log_posterior, centers, dt_s)

    setattr(one_dimensional_first_order_imm_content_diagnostics, _PATCH_ATTR, True)
    setattr(one_dimensional_first_order_imm_content_diagnostics, _ORIGINAL_ATTR, helper)
    return one_dimensional_first_order_imm_content_diagnostics


def _patch_loaded_aliases(original: Callable[..., dict[str, float | int]], replacement: Callable[..., dict[str, float | int]]) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_first_order_imm_content_diagnostics", None) is original:
            module._first_order_imm_content_diagnostics = replacement


def apply_first_order_imm_1d_diagnostics_patch() -> None:
    """Allow compact ``(bins,)`` coordinate grids in first-order IMM diagnostics."""

    import hipporeplayimm.state_space_utils as state_space_utils

    current = state_space_utils._first_order_imm_content_diagnostics
    wrapped = _wrap_helper(current)
    state_space_utils._first_order_imm_content_diagnostics = wrapped
    if wrapped is not current:
        _patch_loaded_aliases(current, wrapped)
