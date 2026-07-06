"""Validate state-space decoder score inputs before scoring."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_bin_center_validation_patch_applied"
_SPARSE_MOMENTUM_PATCHED_FLAG = "_sparse_momentum_bin_center_validation_patch_applied"
_FIRST_ORDER_IMM_DIAGNOSTICS_PATCHED_FLAG = "_first_order_imm_bin_center_validation_patch_applied"


def _validate_state_space_log_likelihood(emissions: Any) -> None:
    """Reject malformed state-space emissions before candidate selection."""

    values = np.asarray(emissions.log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("emissions.log_likelihood must be two-dimensional")
    if values.shape[0] == 0:
        raise ValueError("emissions must contain at least one time bin")
    if values.shape[1] == 0:
        raise ValueError("emissions must contain at least one spatial bin")
    expected_shape = (int(emissions.n_time), int(emissions.n_bins))
    if values.shape != expected_shape:
        raise ValueError("emissions.log_likelihood shape must match emissions.n_time and emissions.n_bins")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("emissions.log_likelihood must not contain NaN or +inf")
    if not np.all(np.any(np.isfinite(values), axis=1)):
        raise ValueError("every emission row must contain at least one finite spatial-bin log likelihood")


def _coerce_state_space_bin_centers(bin_centers: Any, n_bins: int) -> np.ndarray:
    """Return finite 2D bin centers with one row per spatial bin."""

    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[1] == 0:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if centers.shape[0] != int(n_bins):
        raise ValueError("bin_centers must contain one row per emission spatial bin")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must be finite")
    return centers


def _patch_sparse_momentum_bin_center_validation() -> None:
    """Install the same validation on the direct exact-sparse momentum helper."""

    from . import state_space as ss
    from . import state_space_sparse_momentum as sparse_momentum

    previous_score = sparse_momentum._score_sparse_momentum_exact
    if getattr(previous_score, _SPARSE_MOMENTUM_PATCHED_FLAG, False):
        return

    def score_sparse_momentum_exact(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory: bool = True,
    ):
        _validate_state_space_log_likelihood(emissions)
        centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
        return previous_score(
            emissions,
            centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )

    score_sparse_momentum_exact.__name__ = getattr(
        previous_score,
        "__name__",
        "_score_sparse_momentum_exact",
    )
    score_sparse_momentum_exact.__doc__ = getattr(previous_score, "__doc__", None)
    score_sparse_momentum_exact.__module__ = getattr(previous_score, "__module__", __name__)
    setattr(score_sparse_momentum_exact, _SPARSE_MOMENTUM_PATCHED_FLAG, True)
    setattr(score_sparse_momentum_exact, "__hipporeplayimm_original__", previous_score)
    sparse_momentum._score_sparse_momentum_exact = score_sparse_momentum_exact
    if getattr(ss, "_score_sparse_momentum_exact", None) is previous_score:
        ss._score_sparse_momentum_exact = score_sparse_momentum_exact


def _wrap_first_order_imm_content_diagnostics(previous_helper):
    if getattr(previous_helper, _FIRST_ORDER_IMM_DIAGNOSTICS_PATCHED_FLAG, False):
        return previous_helper

    def first_order_imm_content_diagnostics(mode_posterior, trajectory_log_posterior, bin_centers, dt_s):
        trajectory = np.asarray(trajectory_log_posterior, dtype=float)
        n_bins = int(trajectory.shape[1]) if trajectory.ndim == 2 else -1
        centers = _coerce_state_space_bin_centers(bin_centers, n_bins)
        return previous_helper(mode_posterior, trajectory_log_posterior, centers, dt_s)

    first_order_imm_content_diagnostics.__name__ = getattr(previous_helper, "__name__", "_first_order_imm_content_diagnostics")
    first_order_imm_content_diagnostics.__doc__ = getattr(previous_helper, "__doc__", None)
    first_order_imm_content_diagnostics.__module__ = getattr(previous_helper, "__module__", __name__)
    setattr(first_order_imm_content_diagnostics, _FIRST_ORDER_IMM_DIAGNOSTICS_PATCHED_FLAG, True)
    setattr(first_order_imm_content_diagnostics, "__hipporeplayimm_original__", previous_helper)
    return first_order_imm_content_diagnostics


def _patch_first_order_imm_diagnostics_bin_centers() -> None:
    """Allow compact one-dimensional grids in first-order IMM content diagnostics."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        helper = getattr(module, "_first_order_imm_content_diagnostics", None)
        if helper is None or getattr(helper, _FIRST_ORDER_IMM_DIAGNOSTICS_PATCHED_FLAG, False):
            continue
        module._first_order_imm_content_diagnostics = _wrap_first_order_imm_content_diagnostics(helper)


def apply_state_space_bin_center_validation_patch() -> None:
    """Install state-space score input validation for ``StateSpaceReplayModel.score``."""

    from . import state_space as ss

    if not getattr(ss.StateSpaceReplayModel.score, _PATCHED_FLAG, False):
        previous_score = ss.StateSpaceReplayModel.score

        def score(self, emissions, bin_centers, *args, **kwargs):
            _validate_state_space_log_likelihood(emissions)
            centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
            return previous_score(self, emissions, centers, *args, **kwargs)

        score.__name__ = getattr(previous_score, "__name__", "score")
        score.__doc__ = getattr(previous_score, "__doc__", None)
        score.__module__ = getattr(previous_score, "__module__", __name__)
        setattr(score, _PATCHED_FLAG, True)
        if getattr(previous_score, "_native_duration_occupancy_aware", False):
            setattr(score, "_native_duration_occupancy_aware", True)
        setattr(score, "__hipporeplayimm_original__", previous_score)
        ss.StateSpaceReplayModel.score = score

    _patch_sparse_momentum_bin_center_validation()
    _patch_first_order_imm_diagnostics_bin_centers()


__all__ = ["apply_state_space_bin_center_validation_patch"]
