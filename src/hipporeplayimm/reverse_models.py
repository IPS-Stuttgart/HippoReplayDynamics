"""Reverse-time and bidirectional wrappers for replay models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import EventScore, _posterior_diagnostics
from .reverse_time_terminal_guard import _clear_unmappable_reverse_terminal


@dataclass
class ReverseTimeReplayModel:
    """Score a replay model on the reversed time sequence."""

    base_model: object
    name: str | None = None

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        occupancy_s: np.ndarray | None = None,
        candidate_indices: list[np.ndarray] | None = None,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        reversed_emissions = reverse_emissions(emissions)
        reversed_candidates = _reverse_candidate_indices(candidate_indices)
        score = _score_model_with_optional_kwargs(
            self.base_model,
            reversed_emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=reversed_candidates,
            return_trajectory=return_trajectory,
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
        diagnostics = dict(score.diagnostics)
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
        if result.terminal_log_posterior is not None:
            result.diagnostics.update(_posterior_diagnostics(result.terminal_log_posterior, bin_centers))
        if mapped_terminal_from_trajectory:
            return result
        return _clear_unmappable_reverse_terminal(result)


@dataclass
class BidirectionalReplayModel:
    """Equal-prior mixture of a forward model and its reverse-time counterpart."""

    base_model: object
    name: str | None = None

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        occupancy_s: np.ndarray | None = None,
        candidate_indices: list[np.ndarray] | None = None,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        forward = _score_model_with_optional_kwargs(
            self.base_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=return_trajectory,
        )
        reverse_return_trajectory = True if return_trajectory is False else return_trajectory
        reverse = ReverseTimeReplayModel(self.base_model).score(
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=reverse_return_trajectory,
        )
        weights = np.exp(np.array([forward.log_likelihood, reverse.log_likelihood]) - logsumexp([forward.log_likelihood, reverse.log_likelihood]))
        logp = float(logsumexp([forward.log_likelihood, reverse.log_likelihood]) - np.log(2.0))
        terminal = _mixture_log_posterior(forward.terminal_log_posterior, reverse.terminal_log_posterior, weights)
        trajectory = None
        if return_trajectory is not False:
            trajectory = _mixture_log_posterior(forward.trajectory_log_posterior, reverse.trajectory_log_posterior, weights)
        if trajectory is not None:
            terminal = trajectory[-1].copy()
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


def reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
    bin_durations = _reversed_optional_duration_vector(
        getattr(emissions, "bin_durations", None),
        expected_length=emissions.n_time,
        name="bin_durations",
    )
    transition_durations = _reversed_optional_duration_vector(
        getattr(emissions, "transition_durations", None),
        expected_length=max(emissions.n_time - 1, 0),
        name="transition_durations",
    )
    reversed_times = _reversed_time_vector(emissions, transition_durations)
    out = LogEmissionTensor(
        log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[::-1].copy(),
        spike_counts=np.asarray(emissions.spike_counts)[::-1].copy(),
        times=reversed_times,
        dt=emissions.dt,
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(emissions.n_spikes),
        bin_durations=bin_durations,
        transition_durations=transition_durations,
    )
    metadata = getattr(emissions, "metadata", None)
    if metadata is not None:
        out.metadata = dict(metadata)
    return out


def _reversed_optional_duration_vector(
    values: np.ndarray | None,
    *,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise ValueError(
            f"{name} must contain {expected_length} values; got shape {array.shape}"
        )
    return array[::-1].copy()


def _reversed_time_vector(
    emissions: LogEmissionTensor,
    reversed_transition_durations: np.ndarray | None,
) -> np.ndarray:
    """Return increasing timestamps for the reversed emission rows."""

    times = np.asarray(emissions.times, dtype=float)
    if times.shape == (0,):
        return times.copy()
    if times.shape != (emissions.n_time,):
        return times.copy()
    if reversed_transition_durations is None:
        return (float(times[-1]) - times[::-1] + float(times[0])).copy()

    durations = np.asarray(reversed_transition_durations, dtype=float)
    expected_length = max(emissions.n_time - 1, 0)
    if durations.shape != (expected_length,):
        raise ValueError(f"transition_durations must contain {expected_length} values; got shape {durations.shape}")
    output = np.empty_like(times, dtype=float)
    output[0] = float(times[0])
    if durations.size:
        output[1:] = output[0] + np.cumsum(durations)
    return output


def _reverse_candidate_indices(candidate_indices: list[np.ndarray] | None) -> list[np.ndarray] | None:
    if candidate_indices is None:
        return None
    return [np.asarray(curr).copy() for curr in candidate_indices[::-1]]


def _score_model_with_optional_kwargs(
    model: object,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    *,
    occupancy_s: np.ndarray | None = None,
    candidate_indices: list[np.ndarray] | None = None,
    return_trajectory: bool | None = None,
) -> EventScore:
    kwargs: dict[str, object] = {}
    if occupancy_s is not None:
        kwargs["occupancy_s"] = occupancy_s
    if candidate_indices is not None:
        kwargs["candidate_indices"] = candidate_indices
    if return_trajectory is not None:
        kwargs["return_trajectory"] = bool(return_trajectory)
    return _call_score_with_supported_kwargs(model.score, emissions, bin_centers, kwargs)  # type: ignore[attr-defined]


def _call_score_with_supported_kwargs(
    score: object,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    optional_kwargs: dict[str, object],
) -> EventScore:
    supported_kwargs = _supported_score_kwargs(score, optional_kwargs)
    if supported_kwargs is not None:
        if supported_kwargs:
            return score(emissions, bin_centers, **supported_kwargs)  # type: ignore[operator]
        return score(emissions, bin_centers)  # type: ignore[operator]

    try:
        if optional_kwargs:
            return score(emissions, bin_centers, **optional_kwargs)  # type: ignore[operator]
        return score(emissions, bin_centers)  # type: ignore[operator]
    except TypeError as exc:
        unsupported = [
            keyword
            for keyword in optional_kwargs
            if _looks_like_unexpected_keyword_type_error(exc, keyword)
        ]
        if not unsupported:
            raise
        reduced_kwargs = {key: value for key, value in optional_kwargs.items() if key not in unsupported}
        if reduced_kwargs:
            return score(emissions, bin_centers, **reduced_kwargs)  # type: ignore[operator]
        return score(emissions, bin_centers)  # type: ignore[operator]


def _supported_score_kwargs(score: object, optional_kwargs: dict[str, object]) -> dict[str, object] | None:
    if not optional_kwargs:
        return {}
    try:
        signature = inspect.signature(score)
    except (TypeError, ValueError):
        return None
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(optional_kwargs)
    supported: dict[str, object] = {}
    for keyword, value in optional_kwargs.items():
        parameter = parameters.get(keyword)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            supported[keyword] = value
    return supported


def _looks_like_unexpected_keyword_type_error(exc: TypeError, keyword: str) -> bool:
    text = str(exc)
    return keyword in text and (
        "unexpected keyword" in text
        or "got an unexpected" in text
        or "invalid keyword" in text
        or "takes no keyword" in text
    )


def _mixture_log_posterior(left: np.ndarray | None, right: np.ndarray | None, weights: np.ndarray) -> np.ndarray | None:
    valid = [
        (np.asarray(post, dtype=float), float(weight))
        for post, weight in ((left, weights[0]), (right, weights[1]))
        if post is not None
    ]
    if not valid:
        return None
    stacked = np.stack(
        [post + np.log(max(weight, np.finfo(float).tiny)) for post, weight in valid],
        axis=0,
    )
    out = logsumexp(stacked, axis=0)
    return out - logsumexp(out, axis=-1, keepdims=True)
