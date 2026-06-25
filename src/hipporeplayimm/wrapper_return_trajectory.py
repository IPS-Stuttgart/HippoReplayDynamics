"""Propagate evidence-only scoring controls through replay wrapper models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp

from .models import EventScore, _posterior_diagnostics
from .reverse_time_terminal_guard import _clear_unmappable_reverse_terminal

_PATCHED_FLAG = "_wrapper_return_trajectory_patch_applied"


def apply_wrapper_return_trajectory_patch() -> None:
    """Install wrapper score compatibility patches."""

    from . import result_improvement_extensions as extensions
    from . import reverse_models

    if not getattr(extensions, _PATCHED_FLAG, False):
        _patch_result_improvement_wrappers(extensions)
        setattr(extensions, _PATCHED_FLAG, True)
    if not getattr(reverse_models, _PATCHED_FLAG, False):
        _patch_direct_reverse_wrappers(extensions, reverse_models)
        setattr(reverse_models, _PATCHED_FLAG, True)


def _score_kwargs(occupancy_s=None, candidate_indices=None, return_trajectory: bool | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if occupancy_s is not None:
        kwargs["occupancy_s"] = occupancy_s
    if candidate_indices is not None:
        kwargs["candidate_indices"] = candidate_indices
    if return_trajectory is not None:
        kwargs["return_trajectory"] = bool(return_trajectory)
    return kwargs


def _reverse_candidate_indices(candidate_indices):
    if candidate_indices is None:
        return None
    # Preserve dtype so the base model can reject non-integer candidate arrays.
    return [np.asarray(curr).copy() for curr in candidate_indices[::-1]]


def _finalize_reverse_result(result, model_name: str, base_name: str, bin_centers: np.ndarray, return_trajectory: bool | None) -> EventScore:
    trajectory = None
    terminal = result.terminal_log_posterior
    mapped_terminal = False
    if result.trajectory_log_posterior is not None:
        trajectory = np.asarray(result.trajectory_log_posterior, dtype=float)[::-1].copy()
        terminal = trajectory[-1]
        mapped_terminal = True
        if return_trajectory is False:
            trajectory = None
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    diagnostics["time_direction"] = "reverse"
    diagnostics["direction_model"] = "reverse"
    diagnostics["base_model"] = str(result.model_name)
    diagnostics["reverse_time_base_model"] = str(base_name)
    score = EventScore(
        model_name,
        result.log_likelihood,
        result.n_time,
        result.n_spikes,
        diagnostics=diagnostics,
        terminal_log_posterior=terminal,
        trajectory_log_posterior=trajectory,
    )
    if terminal is not None:
        score.diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
    if mapped_terminal:
        return score
    return _clear_unmappable_reverse_terminal(score)


def _mix_posteriors(mixture_fn, left, right, weights):
    return mixture_fn(left, right, weights)


def _patch_result_improvement_wrappers(extensions: Any) -> None:
    def score_replay_model_compat(
        model,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory: bool | None = None,
    ):
        candidates = candidate_indices
        if candidates is None and hasattr(model, "candidate_indices"):
            candidates = extensions._call_candidate_indices_compat(model.candidate_indices, emissions, bin_centers)  # noqa: SLF001
        return extensions._call_score_with_supported_kwargs(  # noqa: SLF001
            model.score,
            emissions,
            bin_centers,
            _score_kwargs(occupancy_s, candidates, return_trajectory),
        )

    def reverse_score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory: bool | None = None):
        reversed_emissions = extensions.copy_emissions_with_log_likelihood(emissions, emissions.log_likelihood, reverse_time=True)
        result = score_replay_model_compat(
            self.base_model,
            reversed_emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=_reverse_candidate_indices(candidate_indices),
            return_trajectory=return_trajectory,
        )
        return _finalize_reverse_result(result, str(self.name), str(getattr(self.base_model, "name", "model")), bin_centers, return_trajectory)

    def bidirectional_score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory: bool | None = None):
        forward = score_replay_model_compat(
            self.forward_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=return_trajectory,
        )
        reverse = score_replay_model_compat(
            self.reverse_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=True if return_trajectory is False else return_trajectory,
        )
        values = np.array([forward.log_likelihood, reverse.log_likelihood], dtype=float)
        weights = np.exp(values - logsumexp(values))
        terminal = extensions._mixture_log_posterior([forward.terminal_log_posterior, reverse.terminal_log_posterior], weights)  # noqa: SLF001
        trajectory = None
        if return_trajectory is not False and forward.trajectory_log_posterior is not None and reverse.trajectory_log_posterior is not None:
            trajectory = extensions._mixture_log_posterior([forward.trajectory_log_posterior, reverse.trajectory_log_posterior], weights)  # noqa: SLF001
            terminal = np.asarray(trajectory[-1], dtype=float).copy()
        diagnostics = dict(getattr(forward if weights[0] >= weights[1] else reverse, "diagnostics", {}) or {})
        diagnostics.update({"direction_model": "bidirectional", "direction_forward_probability": float(weights[0]), "direction_reverse_probability": float(weights[1])})
        if terminal is not None:
            diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(self.name, float(logsumexp(values) - np.log(2.0)), emissions.n_time, emissions.n_spikes, diagnostics=diagnostics, terminal_log_posterior=terminal, trajectory_log_posterior=trajectory)

    score_replay_model_compat.__name__ = "score_replay_model_compat"
    score_replay_model_compat.__doc__ = extensions.score_replay_model_compat.__doc__
    score_replay_model_compat.__module__ = extensions.__name__
    reverse_score.__name__ = "score"
    reverse_score.__doc__ = extensions.ReverseTimeReplayModel.score.__doc__
    reverse_score.__module__ = extensions.__name__
    reverse_score._reverse_time_terminal_guard_applied = True  # type: ignore[attr-defined]
    bidirectional_score.__name__ = "score"
    bidirectional_score.__doc__ = extensions.BidirectionalReplayModel.score.__doc__
    bidirectional_score.__module__ = extensions.__name__
    extensions.score_replay_model_compat = score_replay_model_compat
    extensions.ReverseTimeReplayModel.score = reverse_score
    extensions.BidirectionalReplayModel.score = bidirectional_score


def _patch_direct_reverse_wrappers(extensions: Any, reverse_models: Any) -> None:
    def reverse_score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory: bool | None = None) -> EventScore:
        result = extensions._call_score_with_supported_kwargs(  # noqa: SLF001
            self.base_model.score,
            reverse_models.reverse_emissions(emissions),
            bin_centers,
            _score_kwargs(occupancy_s, _reverse_candidate_indices(candidate_indices), return_trajectory),
        )
        name = self.name or f"{result.model_name}-reverse"
        return _finalize_reverse_result(result, name, str(result.model_name), bin_centers, return_trajectory)

    def bidirectional_score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory: bool | None = None) -> EventScore:
        forward = extensions._call_score_with_supported_kwargs(
            self.base_model.score,
            emissions,
            bin_centers,
            _score_kwargs(occupancy_s, candidate_indices, return_trajectory),
        )
        reverse = reverse_models.ReverseTimeReplayModel(self.base_model).score(
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=True if return_trajectory is False else return_trajectory,
        )
        weights = np.exp(np.array([forward.log_likelihood, reverse.log_likelihood]) - logsumexp([forward.log_likelihood, reverse.log_likelihood]))
        terminal = _mix_posteriors(reverse_models._mixture_log_posterior, forward.terminal_log_posterior, reverse.terminal_log_posterior, weights)  # noqa: SLF001
        trajectory = None
        if return_trajectory is not False and forward.trajectory_log_posterior is not None and reverse.trajectory_log_posterior is not None:
            trajectory = _mix_posteriors(reverse_models._mixture_log_posterior, forward.trajectory_log_posterior, reverse.trajectory_log_posterior, weights)  # noqa: SLF001
            terminal = np.asarray(trajectory[-1], dtype=float).copy()
        diagnostics = {"time_direction": "bidirectional-mixture", "base_model": str(forward.model_name), "forward_model_posterior_probability": float(weights[0]), "reverse_model_posterior_probability": float(weights[1])}
        if terminal is not None:
            diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(self.name or f"{forward.model_name}-bidirectional", float(logsumexp([forward.log_likelihood, reverse.log_likelihood]) - np.log(2.0)), forward.n_time, forward.n_spikes, diagnostics=diagnostics, terminal_log_posterior=terminal, trajectory_log_posterior=trajectory)

    reverse_score.__name__ = "score"
    reverse_score.__doc__ = reverse_models.ReverseTimeReplayModel.score.__doc__
    reverse_score.__module__ = reverse_models.__name__
    bidirectional_score.__name__ = "score"
    bidirectional_score.__doc__ = reverse_models.BidirectionalReplayModel.score.__doc__
    bidirectional_score.__module__ = reverse_models.__name__
    reverse_models.ReverseTimeReplayModel.score = reverse_score
    reverse_models.BidirectionalReplayModel.score = bidirectional_score


__all__ = ["apply_wrapper_return_trajectory_patch"]
