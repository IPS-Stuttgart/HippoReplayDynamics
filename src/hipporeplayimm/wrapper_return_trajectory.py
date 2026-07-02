"""Propagate evidence-only scoring through replay wrapper models.

State-space models accept optional scoring controls such as
``return_trajectory=False``, ``occupancy_s``, and ``candidate_indices``. Lightweight
reverse-time and bidirectional wrappers historically did not expose all of those
keywords. This runtime patch keeps wrapper calls compatible with the base
state-space interface and continues to guard reverse-time terminals that cannot be
mapped back without a full trajectory posterior.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .models import EventScore, _posterior_diagnostics
from .reverse_time_terminal_guard import _clear_unmappable_reverse_terminal

_PATCHED_FLAG = "_wrapper_return_trajectory_patch_applied"
_RESULT_COMPAT_WRAPPER_ATTR = "_wrapper_return_trajectory_score_compat_wrapper"
_EXTENSION_REVERSE_WRAPPER_ATTR = "_wrapper_return_trajectory_extension_reverse_score_wrapper"
_EXTENSION_BIDIRECTIONAL_WRAPPER_ATTR = "_wrapper_return_trajectory_extension_bidirectional_score_wrapper"
_DIRECT_REVERSE_WRAPPER_ATTR = "_wrapper_return_trajectory_direct_reverse_score_wrapper"
_DIRECT_BIDIRECTIONAL_WRAPPER_ATTR = "_wrapper_return_trajectory_direct_bidirectional_score_wrapper"
_STATE_SPACE_DIAGNOSTIC_PATCH_ATTR = "_state_space_evidence_only_diagnostics_patch_applied"
_STATE_SPACE_DIAGNOSTIC_ORIGINAL_ATTR = "_state_space_evidence_only_diagnostics_original"


def apply_wrapper_return_trajectory_patch() -> None:
    """Install wrapper score compatibility patches."""

    from . import result_improvement_extensions as extensions
    from . import reverse_models

    if not _result_improvement_wrappers_current(extensions):
        _patch_result_improvement_wrappers(extensions)
    setattr(extensions, _PATCHED_FLAG, True)

    if not _direct_reverse_wrappers_current(reverse_models):
        _patch_direct_reverse_wrappers(extensions, reverse_models)
    setattr(reverse_models, _PATCHED_FLAG, True)

    _patch_state_space_evidence_only_diagnostics()


def _result_improvement_wrappers_current(extensions: Any) -> bool:
    return (
        getattr(extensions, _PATCHED_FLAG, False)
        and getattr(getattr(extensions, "score_replay_model_compat", None), _RESULT_COMPAT_WRAPPER_ATTR, False)
        and getattr(getattr(extensions.ReverseTimeReplayModel, "score", None), _EXTENSION_REVERSE_WRAPPER_ATTR, False)
        and getattr(getattr(extensions.BidirectionalReplayModel, "score", None), _EXTENSION_BIDIRECTIONAL_WRAPPER_ATTR, False)
    )


def _direct_reverse_wrappers_current(reverse_models: Any) -> bool:
    return (
        getattr(reverse_models, _PATCHED_FLAG, False)
        and getattr(getattr(reverse_models.ReverseTimeReplayModel, "score", None), _DIRECT_REVERSE_WRAPPER_ATTR, False)
        and getattr(getattr(reverse_models.BidirectionalReplayModel, "score", None), _DIRECT_BIDIRECTIONAL_WRAPPER_ATTR, False)
    )


def _patch_state_space_evidence_only_diagnostics() -> None:
    """Keep path-model trajectory diagnostics aligned with evidence-only scoring."""

    import sys

    duration_occupancy = sys.modules.get("hipporeplayimm.duration_occupancy")
    state_space = sys.modules.get("hipporeplayimm.state_space")
    if duration_occupancy is None or state_space is None:
        return

    original = getattr(duration_occupancy, "_score_state_space_duration_with_occupancy", None)
    if original is None:
        return

    if getattr(original, _STATE_SPACE_DIAGNOSTIC_PATCH_ATTR, False):
        patched = original
    else:

        @wraps(original)
        def score_state_space_duration_with_occupancy(
            self,
            emissions,
            bin_centers,
            candidate_indices=None,
            *,
            occupancy_s=None,
            return_trajectory: bool = True,
        ):
            result = original(
                self,
                emissions,
                bin_centers,
                candidate_indices=candidate_indices,
                occupancy_s=occupancy_s,
                return_trajectory=return_trajectory,
            )
            if return_trajectory is False:
                _mark_pruned_path_evidence_only(self, result)
            return result

        setattr(
            score_state_space_duration_with_occupancy,
            _STATE_SPACE_DIAGNOSTIC_PATCH_ATTR,
            True,
        )
        setattr(
            score_state_space_duration_with_occupancy,
            _STATE_SPACE_DIAGNOSTIC_ORIGINAL_ATTR,
            original,
        )
        patched = score_state_space_duration_with_occupancy
        duration_occupancy._score_state_space_duration_with_occupancy = patched

    state_space_model = getattr(state_space, "StateSpaceReplayModel", None)
    if state_space_model is None:
        return

    current_score = getattr(state_space_model, "score", None)
    original_score = getattr(patched, _STATE_SPACE_DIAGNOSTIC_ORIGINAL_ATTR, None)
    if (
        current_score is patched
        or current_score is original
        or (original_score is not None and current_score is original_score)
    ):
        state_space_model.score = patched


def _mark_pruned_path_evidence_only(model: Any, result: EventScore) -> None:
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    mode = str(getattr(model, "mode", diagnostics.get("state_space_mode", "")))
    if mode in {"momentum", "momentum-exact-sparse"}:
        diagnostics["state_space_momentum_trajectory_posterior"] = "not_returned_evidence_only"
    elif mode == "imm":
        diagnostics["state_space_imm_trajectory_posterior"] = "not_returned_evidence_only"
    else:
        return
    result.diagnostics = diagnostics


def _is_exact_sparse_momentum_model(model: Any) -> bool:
    return str(getattr(model, "mode", "")) == "momentum-exact-sparse"


def _default_return_trajectory(model: Any, requested: bool | None) -> bool | None:
    if requested is not None:
        return bool(requested)
    if _is_exact_sparse_momentum_model(model):
        return False
    return None


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
        resolved_return_trajectory = _default_return_trajectory(model, return_trajectory)
        if resolved_return_trajectory is not None:
            kwargs["return_trajectory"] = resolved_return_trajectory
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
        base_return_trajectory = True if return_trajectory is None else bool(return_trajectory)
        result = score_replay_model_compat(
            self.base_model,
            reversed_emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=reversed_candidates,
            return_trajectory=base_return_trajectory,
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
            return_trajectory=_default_return_trajectory(self.forward_model, return_trajectory),
        )
        reverse_return_trajectory = True if return_trajectory is None or return_trajectory is False else return_trajectory
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
    setattr(score_replay_model_compat, _RESULT_COMPAT_WRAPPER_ATTR, True)
    setattr(reverse_score, _EXTENSION_REVERSE_WRAPPER_ATTR, True)
    setattr(bidirectional_score, _EXTENSION_BIDIRECTIONAL_WRAPPER_ATTR, True)

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
        kwargs["return_trajectory"] = True if return_trajectory is None else bool(return_trajectory)
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
        resolved_return_trajectory = _default_return_trajectory(self.base_model, return_trajectory)
        if resolved_return_trajectory is not None:
            kwargs["return_trajectory"] = resolved_return_trajectory
        forward = extensions._call_score_with_supported_kwargs(  # noqa: SLF001
            self.base_model.score,
            emissions,
            bin_centers,
            kwargs,
        )
        reverse_return_trajectory = True if return_trajectory is None or return_trajectory is False else return_trajectory
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
    setattr(reverse_score, _DIRECT_REVERSE_WRAPPER_ATTR, True)
    setattr(bidirectional_score, _DIRECT_BIDIRECTIONAL_WRAPPER_ATTR, True)

    reverse_models.ReverseTimeReplayModel.score = reverse_score
    reverse_models.BidirectionalReplayModel.score = bidirectional_score


__all__ = ["apply_wrapper_return_trajectory_patch"]
