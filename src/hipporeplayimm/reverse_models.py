"""Reverse-time and bidirectional wrappers for replay models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import EventScore
from .reverse_time_terminal_guard import _clear_unmappable_reverse_terminal


@dataclass
class ReverseTimeReplayModel:
    """Score a replay model on the reversed time sequence."""

    base_model: object
    name: str | None = None

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        reversed_emissions = reverse_emissions(emissions)
        score = self.base_model.score(reversed_emissions, bin_centers)
        model_name = self.name or f"{score.model_name}-reverse"
        trajectory = None
        terminal = score.terminal_log_posterior
        if score.trajectory_log_posterior is not None:
            trajectory = score.trajectory_log_posterior[::-1].copy()
            terminal = trajectory[-1]
        diagnostics = dict(score.diagnostics)
        diagnostics["time_direction"] = "reverse"
        diagnostics["base_model"] = str(score.model_name)
        return _clear_unmappable_reverse_terminal(
            EventScore(
                model_name,
                score.log_likelihood,
                score.n_time,
                score.n_spikes,
                diagnostics=diagnostics,
                terminal_log_posterior=terminal,
                trajectory_log_posterior=trajectory,
            )
        )


@dataclass
class BidirectionalReplayModel:
    """Equal-prior mixture of a forward model and its reverse-time counterpart."""

    base_model: object
    name: str | None = None

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        forward = self.base_model.score(emissions, bin_centers)
        reverse = ReverseTimeReplayModel(self.base_model).score(emissions, bin_centers)
        weights = np.exp(np.array([forward.log_likelihood, reverse.log_likelihood]) - logsumexp([forward.log_likelihood, reverse.log_likelihood]))
        logp = float(logsumexp([forward.log_likelihood, reverse.log_likelihood]) - np.log(2.0))
        terminal = _mixture_log_posterior(forward.terminal_log_posterior, reverse.terminal_log_posterior, weights)
        trajectory = _mixture_log_posterior(forward.trajectory_log_posterior, reverse.trajectory_log_posterior, weights)
        diagnostics = {
            "time_direction": "bidirectional-mixture",
            "base_model": str(forward.model_name),
            "forward_model_posterior_probability": float(weights[0]),
            "reverse_model_posterior_probability": float(weights[1]),
        }
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
    reversed_times = np.asarray(emissions.times, dtype=float)[::-1].copy()
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
