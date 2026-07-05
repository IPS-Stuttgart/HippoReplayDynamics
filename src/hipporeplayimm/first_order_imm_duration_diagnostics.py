"""Preserve partial-bin duration metadata for first-order IMM diagnostics."""

from __future__ import annotations

from functools import wraps
from threading import RLock

from .duration_dynamics import DurationFloat, transition_durations_s

_DIAGNOSTIC_PATCH_LOCK = RLock()


def apply_first_order_imm_duration_diagnostics_patch() -> None:
    """Install a compatibility wrapper around the duration-aware scorer.

    The native scorer computes variable transition durations before it calls the
    first-order IMM content diagnostics, but older call sites pass only scalar
    ``dt`` into the diagnostic helper.  Wrap that helper during first-order IMM
    scoring so duration-dependent content diagnostics still see the explicit
    per-transition durations.
    """

    import hipporeplayimm.duration_occupancy as duration_occupancy

    current = duration_occupancy._score_state_space_duration_with_occupancy
    if getattr(current, "_first_order_imm_duration_diagnostics_aware", False):
        return

    original_score = current

    @wraps(original_score)
    def score_with_duration_diagnostics(
        self,
        emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory: bool = True,
    ):
        if getattr(self, "mode", None) != "first-order-imm":
            return original_score(
                self,
                emissions,
                bin_centers,
                candidate_indices=candidate_indices,
                occupancy_s=occupancy_s,
                return_trajectory=return_trajectory,
            )

        with _DIAGNOSTIC_PATCH_LOCK:
            original_diagnostics = duration_occupancy._first_order_imm_content_diagnostics

            def diagnostics_with_transition_durations(mode_post, trajectory, centers, dt_s):
                diagnostic_dt = dt_s
                if getattr(diagnostic_dt, "transition_durations", None) is None:
                    durations = transition_durations_s(emissions)
                    if durations.size:
                        diagnostic_dt = DurationFloat(float(diagnostic_dt), durations)
                return original_diagnostics(mode_post, trajectory, centers, diagnostic_dt)

            duration_occupancy._first_order_imm_content_diagnostics = diagnostics_with_transition_durations
            try:
                return original_score(
                    self,
                    emissions,
                    bin_centers,
                    candidate_indices=candidate_indices,
                    occupancy_s=occupancy_s,
                    return_trajectory=return_trajectory,
                )
            finally:
                duration_occupancy._first_order_imm_content_diagnostics = original_diagnostics

    score_with_duration_diagnostics._first_order_imm_duration_diagnostics_aware = True
    score_with_duration_diagnostics.__duration_occupancy_previous_score__ = original_score
    duration_occupancy._score_state_space_duration_with_occupancy = score_with_duration_diagnostics
