"""Guard reverse-time wrappers against time-misaligned evidence-only terminals.

A reverse-time score can only be mapped back to the original event endpoint when
the base model returns the full trajectory posterior.  If the base model returns
only a terminal posterior for the reversed multi-bin event, that posterior
belongs to the original start time rather than the original endpoint.  In that
evidence-only case, drop the terminal posterior and endpoint diagnostics instead
of exposing a misaligned endpoint.
"""

from __future__ import annotations

from functools import wraps
from typing import Any


_REVERSE_TIME_TERMINAL_UNAVAILABLE = "unavailable_without_trajectory"
_POSTERIOR_DIAGNOSTIC_KEYS = (
    "decoded_endpoint_x",
    "decoded_endpoint_y",
    "decoded_map_x",
    "decoded_map_y",
    "decoded_map_bin",
    "terminal_posterior_entropy",
)


def apply_reverse_time_terminal_guard_patch() -> None:
    """Install the evidence-only terminal guard on compatibility wrappers."""

    from . import result_improvement_extensions as extensions

    score = extensions.ReverseTimeReplayModel.score
    if getattr(score, "_reverse_time_terminal_guard_applied", False):
        return

    @wraps(score)
    def score_with_terminal_guard(
        self: Any,
        emissions: Any,
        bin_centers: Any,
        *,
        occupancy_s: Any = None,
        candidate_indices: Any = None,
    ):
        result = score(
            self,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
        )
        return _clear_unmappable_reverse_terminal(result)

    score_with_terminal_guard._reverse_time_terminal_guard_applied = True  # type: ignore[attr-defined]
    extensions.ReverseTimeReplayModel.score = score_with_terminal_guard


def _clear_unmappable_reverse_terminal(result: Any) -> Any:
    """Clear reverse-time terminal posterior when no trajectory can remap it."""

    if (
        result.trajectory_log_posterior is not None
        or result.terminal_log_posterior is None
        or int(getattr(result, "n_time", 0)) <= 1
    ):
        return result
    result.terminal_log_posterior = None
    result.diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    for key in _POSTERIOR_DIAGNOSTIC_KEYS:
        result.diagnostics.pop(key, None)
    result.diagnostics["reverse_time_terminal_posterior"] = _REVERSE_TIME_TERMINAL_UNAVAILABLE
    return result
