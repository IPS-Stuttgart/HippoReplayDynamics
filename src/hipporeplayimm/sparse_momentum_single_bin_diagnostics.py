"""Complete exact sparse momentum diagnostics for single-bin fallbacks.

The exact sparse momentum decoder falls back to the fragmented one-bin marginal
when an event contains only one emission bin: no pair state or backward transition
rows exist in that degenerate case.  Keep the diagnostic schema explicit so
single-bin rows are auditable and align with the multi-bin full/evidence-only
paths.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_SINGLE_BIN_EVIDENCE_MODE = "single_bin_fragmented_fallback"


def apply_sparse_momentum_single_bin_diagnostics_patch() -> None:
    """Install single-bin diagnostic completion for sparse momentum scoring."""

    import hipporeplayimm.state_space as state_space
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum

    score = sparse_momentum._score_sparse_momentum_exact
    if getattr(score, "_single_bin_diagnostics_patch_applied", False):
        return

    @wraps(score)
    def score_with_single_bin_diagnostics(
        emissions: Any,
        bin_centers: Any,
        config: Any,
        transition_durations_s: Any,
        *,
        valid_bin_mask: Any = None,
        return_trajectory: bool = True,
    ):
        logp, trajectory, terminal, diagnostics = score(
            emissions,
            bin_centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        if getattr(emissions, "n_time", None) != 1:
            return logp, trajectory, terminal, diagnostics

        diagnostics = dict(diagnostics)
        if return_trajectory:
            diagnostics["state_space_sparse_momentum_backward_transition_rows"] = "none_single_bin"
            diagnostics["state_space_sparse_momentum_evidence_mode"] = _SINGLE_BIN_EVIDENCE_MODE
            diagnostics["state_space_sparse_momentum_evidence_only"] = 0
            diagnostics["state_space_momentum_trajectory_posterior"] = _SINGLE_BIN_EVIDENCE_MODE
        else:
            diagnostics["state_space_sparse_momentum_backward_transition_rows"] = "skipped_evidence_only"
            diagnostics["state_space_sparse_momentum_evidence_mode"] = "evidence_only"
            diagnostics["state_space_sparse_momentum_evidence_only"] = 1
            diagnostics["state_space_momentum_trajectory_posterior"] = "not_returned_evidence_only"
        return logp, trajectory, terminal, diagnostics

    score_with_single_bin_diagnostics._single_bin_diagnostics_patch_applied = True  # type: ignore[attr-defined]
    sparse_momentum._score_sparse_momentum_exact = score_with_single_bin_diagnostics
    state_space._score_sparse_momentum_exact = score_with_single_bin_diagnostics
