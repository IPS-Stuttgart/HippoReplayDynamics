"""Propagate evidence-only scoring through replay wrapper models.

State-space models accept optional scoring controls such as
``return_trajectory=False``, ``occupancy_s``, and ``candidate_indices``. Lightweight
reverse-time and bidirectional wrappers historically did not expose all of those
keywords. This runtime patch keeps wrapper calls compatible with the base
state-space interface and continues to guard reverse-time terminals that cannot be
mapped back without a full trajectory posterior.
"""

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
            candidates = extensions._call_candidate_indices_compat(  # noqa: SLF001
                model.candidate_indices,
                emissions,
                bin_centers,
            )

        kwargs: dict[str, object] = {}
        if candidates is not None:
            kwargs["candidate_indices"] = candidates
        if occupancy_s is not None:
            kwargs["occupancy_s"] = occupancy_s
        if return_trajectory is not None:
            kwargs["return_trajectory"] = bool(return_trajectory)
        return extensions._call_score_with_supported_kwargs(  # noqa: SLF001
            model.score,
            emissions,
            bin_centers,
            kwargs,
        )

    def reverse_score(
        self,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory: bool | None = None,
    ):
        reversed_emissions = extensions.copy_emissions_with_log_likelihood(
            emissions,
            emissions.log_likelihood,
            reverse_time=True,
        )
        reversed_candidates = (
            None
            if candidate_indices is None
            else [np.asarray(curr).copy() for curr in candidate_indices[::-1]]
        )
        result = score_replay_model_compat(
            self.base_model,
            reversed_emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=reversed_candidates,
            return_trajectory=return_trajectory,
        )
        mapped_terminal_from_trajectory = False
        if result.trajectory_log_posterior is not None:
            trajectory = np.asarray(result.trajectory_log_posterior, dtype=float)[::-1].copy()
            result.terminal_log_posterior = trajectory[-1].copy()
            result.trajectory_log_posterior = None if return_trajectory is False else trajectory
            mapped_terminal_from_trajectory = True
        result.model_name = str(self.name)
        result.diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        if result.terminal_log_posterior is not None:
            result.diagnostics.update(_posterior_diagnostics(result.terminal_log_posterior, bin_centers))
        result.diagnostics["direction_model"] = "reverse"
        result.diagnostics["reverse_time_base_model"] = str(getattr(self.base_model, "name", "model"))
        if mapped_terminal_from_trajectory:
            return result
        return _clear_unmappable_reverse_terminal(result)

    def bidirectional_score(
        self,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory: bool | None = None,
    ):
        forward = score_replay_model_compat(
            self.forward_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=return_trajectory,
        )
        reverse_return_trajectory = True if return_trajectory is False else return_trajectory
        reverse = score_replay_model_compat(
            self.reverse_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=reverse_return_trajectory,
        )
        values = np.array([forward.log_likelihood, reverse.log_likelihood], dtype=float)
        logp = float(logsumexp(values) - np.log(2.0))
        weights = np.exp(values - logsumexp(values))
        chosen = forward if weights[0] >= weights[1] else reverse
        diagnostics = dict(getattr(chosen, "diagnostics", {}) or {})
        diagnostics.update(
            {
                "direction_model": "bidirectional",
                "direction_forward_probability": float(weights[0]),
                "direction_reverse_probability": float(weights[1]),
                "direction_forward_log_evidence": float(forward.log_likelihood),
                "direction_reverse_log_evidence": float(reverse.log_likelihood),
            }
        )
        terminal = extensions._mixture_log_posterior(  # noqa: SLF001
            [forward.terminal_log_posterior, reverse.terminal_log_posterior],
            weights,
        )
        trajectory = None
        if (
            return_trajectory is not False
            and forward.trajectory_log_posterior is not None
            and reverse.trajectory_log_posterior is not None
        ):
            trajectory = extensions._mixture_log_posterior(  # noqa: SLF001
                [forward.trajectory_log_posterior, reverse.trajectory_log_posterior],
                weights,
            )
        if trajectory is not None:
            terminal = np.asarray(trajectory[-1], dtype=float).copy()
        if terminal is not None:
            diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name,
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )

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
    def reverse_score(
        self,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        reversed_emissions = reverse_models.reverse_emissions(emissions)
        kwargs: dict[str, object] = {}
        if occupancy_s is not None:
            kwargs["occupancy_s"] = occupancy_s
        if candidate_indices is not None:
            kwargs["candidate_indices"] = [
                np.asarray(curr).copy()
                for curr in candidate_indices[::-1]
            ]
        if return_trajectory is not None:
            kwargs["return_trajectory"] = bool(return_trajectory)
        score = extensions._call_score_with_supported_kwargs(  # noqa: SLF001
            self.base_model.score,
            reversed_emissions,
            bin_centers,
            kwargs,
        )
        model_name = self.name or f"{score.model_name}-reverse"
        trajectory = None
        terminal = score.terminal_log_posterior
        mapped_terminal_from_trajectory = False
        if score.trajectory_log_posterior is not None:
            trajectory = score.trajectory_log_posterior[::-1].copy()
            terminal = trajectory[-1]
            mapped_terminal_from_trajectory = True
            if return_trajectory is False:
                trajectory = None
        diagnostics = dict(getattr(score, "diagnostics", {}) or {})
        diagnostics["time_direction"] = "reverse"
        diagnostics["base_model"] = str(score.model_name)
        result = EventScore(
            model_name,
            score.log_likelihood,
            score.n_time,
            score.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )
        if terminal is not None:
            result.diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        if mapped_terminal_from_trajectory:
            return result
        return _clear_unmappable_reverse_terminal(result)

    def bidirectional_score(
        self,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        kwargs: dict[str, object] = {}
        if occupancy_s is not None:
            kwargs["occupancy_s"] = occupancy_s
        if candidate_indices is not None:
            kwargs["candidate_indices"] = candidate_indices
        if return_trajectory is not None:
            kwargs["return_trajectory"] = bool(return_trajectory)
        forward = extensions._call_score_with_supported_kwargs(  # noqa: SLF001
            self.base_model.score,
            emissions,
            bin_centers,
            kwargs,
        )
        reverse_return_trajectory = True if return_trajectory is False else return_trajectory
        reverse = reverse_models.ReverseTimeReplayModel(self.base_model).score(
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=reverse_return_trajectory,
        )
        weights = np.exp(
            np.array([forward.log_likelihood, reverse.log_likelihood])
            - logsumexp([forward.log_likelihood, reverse.log_likelihood])
        )
        logp = float(logsumexp([forward.log_likelihood, reverse.log_likelihood]) - np.log(2.0))
        terminal = reverse_models._mixture_log_posterior(  # noqa: SLF001
            forward.terminal_log_posterior,
            reverse.terminal_log_posterior,
            weights,
        )
        trajectory = None
        if (
            return_trajectory is not False
            and forward.trajectory_log_posterior is not None
            and reverse.trajectory_log_posterior is not None
        ):
            trajectory = reverse_models._mixture_log_posterior(  # noqa: SLF001
                forward.trajectory_log_posterior,
                reverse.trajectory_log_posterior,
                weights,
            )
        if trajectory is not None:
            terminal = np.asarray(trajectory[-1], dtype=float).copy()
        diagnostics = {
            "time_direction": "bidirectional-mixture",
            "base_model": str(forward.model_name),
            "forward_model_posterior_probability": float(weights[0]),
            "reverse_model_posterior_probability": float(weights[1]),
        }
        if terminal is not None:
            diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name or f"{forward.model_name}-bidirectional",
            logp,
            forward.n_time,
            forward.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )

    reverse_score.__name__ = "score"
    reverse_score.__doc__ = reverse_models.ReverseTimeReplayModel.score.__doc__
    reverse_score.__module__ = reverse_models.__name__
    bidirectional_score.__name__ = "score"
    bidirectional_score.__doc__ = reverse_models.BidirectionalReplayModel.score.__doc__
    bidirectional_score.__module__ = reverse_models.__name__

    reverse_models.ReverseTimeReplayModel.score = reverse_score
    reverse_models.BidirectionalReplayModel.score = bidirectional_score


__all__ = ["apply_wrapper_return_trajectory_patch"]
