"""Keep trajectory-IMM single-bin evidence out of exact comparisons."""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

from .evidence_reporting import DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT

_PATCHED_ATTR = "_trajectory_imm_single_bin_diagnostics_patch_applied"


def apply_trajectory_imm_single_bin_diagnostics_patch() -> None:
    """Patch trajectory-family IMM diagnostics for degenerate one-bin events."""

    from . import state_space_trajectory_imm as trajectory_imm

    current = trajectory_imm._score_trajectory_imm_exact_sparse
    if getattr(current, _PATCHED_ATTR, False):
        _refresh_public_alias(current)
        return

    @wraps(current)
    def score_trajectory_imm_exact_sparse(*args: Any, **kwargs: Any) -> Any:
        logp, trajectory, terminal, mode_posterior, diagnostics = current(*args, **kwargs)
        emissions = args[0] if args else kwargs.get("emissions")
        if int(getattr(emissions, "n_time", 0)) == 1:
            updated = dict(diagnostics)
            updated.update(_single_bin_diagnostics())
            diagnostics = updated
        return logp, trajectory, terminal, mode_posterior, diagnostics

    setattr(score_trajectory_imm_exact_sparse, _PATCHED_ATTR, True)
    trajectory_imm._score_trajectory_imm_exact_sparse = score_trajectory_imm_exact_sparse
    _refresh_public_alias(score_trajectory_imm_exact_sparse)


def _refresh_public_alias(patched: Any) -> None:
    state_space = sys.modules.get("hipporeplayimm.state_space")
    if state_space is not None:
        setattr(state_space, "_score_trajectory_imm_exact_sparse", patched)


def _single_bin_diagnostics() -> dict[str, int | str]:
    return {
        "state_space_trajectory_imm_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        "state_space_trajectory_imm_state_support": "single_bin_fragmented_fallback",
        "state_space_trajectory_imm_transition_support": "none_single_bin",
        "state_space_trajectory_imm_degenerate_reason": "single_time_bin_fragmented_marginal",
        "state_space_trajectory_imm_required_min_time_bins": 2,
        "state_space_momentum_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        "state_space_momentum_candidate_support": "not_used_single_bin",
        "state_space_momentum_candidate_selection": "none_single_bin",
    }


__all__ = ["apply_trajectory_imm_single_bin_diagnostics_patch"]
