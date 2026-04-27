"""Replay motion models and log-space scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor


LOG_ZERO = -1.0e300


@dataclass
class EventScore:
    model_name: str
    log_likelihood: float
    n_time: int
    n_spikes: int
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)


class ReplayModel(Protocol):
    name: str

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        ...


@dataclass
class RandomModel:
    name: str = "random"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        del bin_centers
        logp = float(np.sum(logsumexp(emissions.log_likelihood, axis=1) - np.log(emissions.n_bins)))
        return EventScore(self.name, logp, emissions.n_time, emissions.n_spikes)


@dataclass
class StationaryModel:
    name: str = "stationary"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        del bin_centers
        logp = float(logsumexp(np.sum(emissions.log_likelihood, axis=0) - np.log(emissions.n_bins)))
        return EventScore(self.name, logp, emissions.n_time, emissions.n_spikes)


@dataclass
class DiffusionModel:
    sigma_cm: float = 12.0
    max_step_sigma: float = 3.0
    name: str = "diffusion"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        transition = _log_transition_matrix(
            bin_centers,
            sigma_cm=self.sigma_cm,
            max_step_sigma=self.max_step_sigma,
        )
        log_alpha = emissions.log_likelihood[0] - np.log(emissions.n_bins)
        for time_index in range(1, emissions.n_time):
            log_alpha = emissions.log_likelihood[time_index] + _log_sparse_matvec(log_alpha, transition)
        logp = float(logsumexp(log_alpha))
        return EventScore(self.name, logp, emissions.n_time, emissions.n_spikes)


@dataclass
class CandidateKinematicModel:
    """Candidate-pruned kinematic scorer.

    `mode="diffusion"` or `mode="momentum"` gives a static candidate-pruned
    motion model. `mode="imm"` switches between stationary, diffusion,
    momentum, and jump dynamics.
    """

    mode: str = "imm"
    top_k: int = 64
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm: float = 12.0
    momentum_sigma_cm: float = 12.0
    velocity_decay: float = 0.95
    mode_stickiness: float = 0.94
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"stationary", "diffusion", "momentum", "jump", "imm"}:
            raise ValueError(
                "mode must be one of 'stationary', 'diffusion', 'momentum', 'jump', or 'imm'"
            )
        if self.name is None:
            self.name = self.mode

    def candidate_indices(self, emissions: LogEmissionTensor) -> list[np.ndarray]:
        return [_top_candidate_indices(row, self.top_k) for row in emissions.log_likelihood]

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        if emissions.n_time == 1:
            base = RandomModel(name=str(self.name)).score(emissions, bin_centers)
            return base
        candidates = self.candidate_indices(emissions) if candidate_indices is None else candidate_indices
        if len(candidates) != emissions.n_time:
            raise ValueError("candidate_indices must contain one array per emission time bin")
        if self.mode != "imm":
            logp, mass = self._score_static_pair(emissions, bin_centers, candidates, self.mode)
        else:
            logp, mass = self._score_imm(emissions, bin_centers, candidates)
        return EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={"mean_candidate_log_mass": float(np.mean(mass))},
        )

    def _score_static_pair(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidates: list[np.ndarray],
        mode: str,
    ) -> tuple[float, list[float]]:
        masses = _candidate_log_masses(emissions, candidates)
        first = candidates[0]
        second = candidates[1]
        log_pair = _init_pair_log_alpha(
            emissions,
            first,
            second,
            bin_centers,
            self.diffusion_sigma_cm,
            mode=mode,
            stationary_sigma_cm=self.stationary_sigma_cm,
        )
        prev_prev = first
        prev = second
        for time_index in range(2, emissions.n_time):
            curr = candidates[time_index]
            log_pair = _advance_pair_log_alpha(
                log_pair,
                prev_prev,
                prev,
                curr,
                emissions.log_likelihood[time_index, curr],
                bin_centers,
                mode=mode,
                stationary_sigma_cm=self.stationary_sigma_cm,
                diffusion_sigma_cm=self.diffusion_sigma_cm,
                momentum_sigma_cm=self.momentum_sigma_cm,
                velocity_decay=self.velocity_decay,
            )
            prev_prev, prev = prev, curr
        return float(logsumexp(log_pair)), masses

    def _score_imm(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidates: list[np.ndarray],
    ) -> tuple[float, list[float]]:
        masses = _candidate_log_masses(emissions, candidates)
        modes = ("stationary", "diffusion", "momentum", "jump")
        transition_modes = _mode_transition_matrix(len(modes), self.mode_stickiness)
        first = candidates[0]
        second = candidates[1]
        by_mode = []
        for mode in modes:
            by_mode.append(
                _init_pair_log_alpha(
                    emissions,
                    first,
                    second,
                    bin_centers,
                    self.diffusion_sigma_cm,
                    mode=mode,
                    stationary_sigma_cm=self.stationary_sigma_cm,
                )
            )
        log_alpha = np.stack(by_mode, axis=0) - np.log(len(modes))
        prev_prev = first
        prev = second
        for time_index in range(2, emissions.n_time):
            curr = candidates[time_index]
            next_alpha = []
            for dst_mode_index, dst_mode in enumerate(modes):
                mixed_prev = logsumexp(
                    log_alpha + np.log(transition_modes[:, dst_mode_index])[:, None, None],
                    axis=0,
                )
                next_alpha.append(
                    _advance_pair_log_alpha(
                        mixed_prev,
                        prev_prev,
                        prev,
                        curr,
                        emissions.log_likelihood[time_index, curr],
                        bin_centers,
                        mode=dst_mode,
                        stationary_sigma_cm=self.stationary_sigma_cm,
                        diffusion_sigma_cm=self.diffusion_sigma_cm,
                        momentum_sigma_cm=self.momentum_sigma_cm,
                        velocity_decay=self.velocity_decay,
                    )
                )
            log_alpha = np.stack(next_alpha, axis=0)
            prev_prev, prev = prev, curr
        return float(logsumexp(log_alpha)), masses


def score_model(model: ReplayModel, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
    return model.score(emissions, bin_centers)


def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= log_emission.shape[0]:
        return np.arange(log_emission.shape[0], dtype=int)
    selected = np.argpartition(log_emission, -top_k)[-top_k:]
    return selected[np.argsort(log_emission[selected])[::-1]]


def _candidate_log_masses(emissions: LogEmissionTensor, candidates: list[np.ndarray]) -> list[float]:
    masses = []
    for time_index, curr in enumerate(candidates):
        masses.append(
            float(logsumexp(emissions.log_likelihood[time_index, curr]) - logsumexp(emissions.log_likelihood[time_index]))
        )
    return masses


def _log_transition_matrix(bin_centers: np.ndarray, sigma_cm: float, max_step_sigma: float) -> list[tuple[np.ndarray, np.ndarray]]:
    radius = sigma_cm * max_step_sigma
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for center in bin_centers:
        delta = bin_centers - center[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius * radius
        if not np.any(keep):
            keep[np.argmin(dist2)] = True
        indices = np.flatnonzero(keep)
        log_weights = -0.5 * dist2[indices] / (sigma_cm * sigma_cm)
        log_weights -= logsumexp(log_weights)
        output.append((indices, log_weights))
    return output


def _log_sparse_matvec(log_alpha: np.ndarray, transition: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    result = np.full(log_alpha.shape, LOG_ZERO, dtype=float)
    for src, (dst_indices, log_weights) in enumerate(transition):
        values = log_alpha[src] + log_weights
        for dst, value in zip(dst_indices, values):
            result[dst] = np.logaddexp(result[dst], value)
    return result


def _init_pair_log_alpha(
    emissions: LogEmissionTensor,
    first: np.ndarray,
    second: np.ndarray,
    bin_centers: np.ndarray,
    sigma_cm: float,
    mode: str = "diffusion",
    stationary_sigma_cm: float = 2.0,
) -> np.ndarray:
    first_ll = emissions.log_likelihood[0, first]
    second_ll = emissions.log_likelihood[1, second]
    coords_first = bin_centers[first]
    coords_second = bin_centers[second]
    if mode == "stationary":
        sigma = stationary_sigma_cm
    else:
        sigma = sigma_cm
    log_kernel = _pairwise_gaussian_log_prob(coords_first, coords_second, sigma)
    log_kernel = log_kernel - logsumexp(log_kernel, axis=1, keepdims=True)
    return first_ll[:, None] - np.log(len(first)) + log_kernel + second_ll[None, :]


def _advance_pair_log_alpha(
    log_pair: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    mode: str,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    momentum_sigma_cm: float,
    velocity_decay: float,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev), len(curr)), LOG_ZERO, dtype=float)
    if mode == "jump":
        collapsed_by_prev = logsumexp(log_pair, axis=0)
        uniform = -np.log(len(curr))
        return collapsed_by_prev[:, None] + uniform + curr_emission[None, :]
    for prev_col in range(len(prev)):
        if mode == "stationary":
            predicted = coords_prev[prev_col][None, :]
            sigma = stationary_sigma_cm
            previous_mass = logsumexp(log_pair[:, prev_col])
            log_kernel = _gaussian_log_prob(predicted, coords_curr, sigma)
        elif mode == "diffusion":
            predicted = coords_prev[prev_col][None, :]
            sigma = diffusion_sigma_cm
            previous_mass = logsumexp(log_pair[:, prev_col])
            log_kernel = _gaussian_log_prob(predicted, coords_curr, sigma)
        elif mode == "momentum":
            predictions = coords_prev[prev_col][None, :] + velocity_decay * (
                coords_prev[prev_col][None, :] - coords_prev_prev
            )
            sigma = momentum_sigma_cm
            log_kernel_by_source = _pairwise_gaussian_log_prob(predictions, coords_curr, sigma)
            values = log_pair[:, prev_col][:, None] + (
                log_kernel_by_source - logsumexp(log_kernel_by_source, axis=1, keepdims=True)
            )
            output[prev_col] = logsumexp(values, axis=0) + curr_emission
            continue
        else:
            raise ValueError(f"Unknown kinematic mode: {mode}")
        log_kernel = log_kernel - logsumexp(log_kernel)
        output[prev_col] = previous_mass + log_kernel + curr_emission
    return output


def _gaussian_log_prob(predicted: np.ndarray, observed: np.ndarray, sigma: float) -> np.ndarray:
    delta = observed - predicted
    dist2 = np.sum(delta * delta, axis=1)
    return -0.5 * dist2 / (sigma * sigma)


def _pairwise_gaussian_log_prob(predicted: np.ndarray, observed: np.ndarray, sigma: float) -> np.ndarray:
    delta = predicted[:, None, :] - observed[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    return -0.5 * dist2 / (sigma * sigma)


def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("mode_stickiness must be in [0, 1]")
    off_diag = (1.0 - stickiness) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, stickiness)
    return matrix
