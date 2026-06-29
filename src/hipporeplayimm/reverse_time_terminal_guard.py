"""Guard reverse-time wrappers against time-misaligned evidence-only terminals.

A reverse-time score can only be mapped back to the original event endpoint when
the base model returns the full trajectory posterior.  If the base model returns
only a terminal posterior for the reversed event, that posterior belongs to the
original start time rather than the original endpoint.  In that evidence-only
case, drop the terminal posterior and endpoint diagnostics instead of exposing a
misaligned endpoint.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

import numpy as np


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
        return_trajectory: bool | None = None,
    ):
        result = _score_reverse_with_supported_return_trajectory(
            extensions,
            self,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=return_trajectory,
        )
        return _clear_unmappable_reverse_terminal(result)

    score_with_terminal_guard._reverse_time_terminal_guard_applied = True  # type: ignore[attr-defined]
    score_with_terminal_guard.__hipporeplayimm_original__ = score  # type: ignore[attr-defined]
    extensions.ReverseTimeReplayModel.score = score_with_terminal_guard


def _score_reverse_with_supported_return_trajectory(
    extensions: Any,
    self: Any,
    emissions: Any,
    bin_centers: Any,
    *,
    occupancy_s: Any = None,
    candidate_indices: Any = None,
    return_trajectory: bool | None = None,
) -> Any:
    """Score a reverse-time model while preserving the trajectory-return flag."""

    reversed_emissions = extensions.copy_emissions_with_log_likelihood(
        emissions,
        emissions.log_likelihood,
        reverse_time=True,
    )
    reversed_candidates = (
        None
        if candidate_indices is None
        else [np.asarray(curr, dtype=int).copy() for curr in candidate_indices[::-1]]
    )
    result = extensions.score_replay_model_compat(
        self.base_model,
        reversed_emissions,
        bin_centers,
        occupancy_s=occupancy_s,
        candidate_indices=reversed_candidates,
        return_trajectory=return_trajectory,
    )
    if result.trajectory_log_posterior is not None:
        trajectory = np.asarray(result.trajectory_log_posterior, dtype=float)[::-1].copy()
        result.trajectory_log_posterior = trajectory
        result.terminal_log_posterior = trajectory[-1].copy()
    result.model_name = str(self.name)
    result.diagnostics = dict(result.diagnostics)
    if result.terminal_log_posterior is not None:
        result.diagnostics.update(extensions._posterior_diagnostics(result.terminal_log_posterior, bin_centers))
    result.diagnostics["direction_model"] = "reverse"
    result.diagnostics["reverse_time_base_model"] = str(getattr(self.base_model, "name", "model"))
    return result


def _call_score_with_supported_kwargs(
    score: Any,
    self: Any,
    emissions: Any,
    bin_centers: Any,
    optional_kwargs: dict[str, Any],
) -> Any:
    """Call a score function without dropping newly supported wrapper kwargs."""

    supported_kwargs = _supported_score_kwargs(score, optional_kwargs)
    if supported_kwargs:
        return score(self, emissions, bin_centers, **supported_kwargs)
    return score(self, emissions, bin_centers)


def _supported_score_kwargs(score: Any, optional_kwargs: dict[str, Any]) -> dict[str, Any]:
    if not optional_kwargs:
        return {}
    try:
        signature = inspect.signature(score)
    except (TypeError, ValueError):
        return dict(optional_kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(optional_kwargs)
    supported: dict[str, Any] = {}
    for keyword, value in optional_kwargs.items():
        parameter = parameters.get(keyword)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            supported[keyword] = value
    return supported


def _clear_unmappable_reverse_terminal(result: Any) -> Any:
    """Clear reverse-time terminal posterior when no trajectory can remap it."""

    if result.trajectory_log_posterior is not None or result.terminal_log_posterior is None:
        return result
    result.terminal_log_posterior = None
    result.diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    for key in _POSTERIOR_DIAGNOSTIC_KEYS:
        result.diagnostics.pop(key, None)
    result.diagnostics["reverse_time_terminal_posterior"] = _REVERSE_TIME_TERMINAL_UNAVAILABLE
    return result
