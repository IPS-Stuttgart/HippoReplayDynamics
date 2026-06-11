"""Replay motion models and log-space scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .state_space_utils import _validate_candidate_indices


LOG_ZERO = -1.0e300
DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT = "degenerate_single_bin"


def _validate_positive_parameter(name: str, value: float) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative_parameter(name: str, value: float) -> None:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_probability_parameter(name: str, value: float) -> None:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")


@dataclass
class EventScore:
    model_name: str
    log_likelihood: float
    n_time: int
    n_spikes: int
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)
    terminal_log_posterior: np.ndarray | None = None
    trajectory_log_posterior: np.ndarray | None = None


class ReplayModel(Protocol):
    name: str

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        ...


def _validate_score_inputs(emissions: LogEmissionTensor, bin_centers: np.ndarray) -> np.ndarray:
    """Validate shared replay-model score inputs and return numeric bin centers."""

    log_likelihood = np.asarray(emissions.log_likelihood, dtype=float)
    if log_likelihood.ndim != 2:
        raise ValueError("emissions.log_likelihood must be two-dimensional")
    if log_likelihood.shape[0] == 0:
        raise ValueError("emissions must contain at least one time bin")

    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] < 1:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if centers.shape[0] != log_likelihood.shape[1]:
        raise ValueError("bin_centers must contain one row per emission spatial bin")
    return centers


@dataclass
class RandomModel:
    name: str = "random"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        bin_centers = _validate_score_inputs(emissions, bin_centers)
        trajectory_log_posterior = _normalize_log_weights_by_row(emissions.log_likelihood)
        terminal_log_posterior = trajectory_log_posterior[-1]
        logp = float(np.sum(logsumexp(emissions.log_likelihood, axis=1) - np.log(emissions.n_bins)))
        return EventScore(
            self.name,
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=_posterior_diagnostics(terminal_log_posterior, bin_centers),
            terminal_log_posterior=terminal_log_posterior,
            trajectory_log_posterior=trajectory_log_posterior,
        )


@dataclass
class StationaryModel:
    name: str = "stationary"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        bin_centers = _validate_score_inputs(emissions, bin_centers)
        cumulative_log_likelihood = np.cumsum(emissions.log_likelihood, axis=0) - np.log(emissions.n_bins)
        trajectory_log_posterior = _normalize_log_weights_by_row(cumulative_log_likelihood)
        terminal_log_posterior = trajectory_log_posterior[-1]
        logp = float(logsumexp(cumulative_log_likelihood[-1]))
        return EventScore(
            self.name,
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=_posterior_diagnostics(terminal_log_posterior, bin_centers),
            terminal_log_posterior=terminal_log_posterior,
            trajectory_log_posterior=trajectory_log_posterior,
        )


@dataclass
class DiffusionModel:
    sigma_cm: float = 12.0
    max_step_sigma: float = 3.0
    name: str = "diffusion"

    def __post_init__(self) -> None:
        _validate_positive_parameter("sigma_cm", self.sigma_cm)
        _validate_positive_parameter("max_step_sigma", self.max_step_sigma)

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        bin_centers = _validate_score_inputs(emissions, bin_centers)
        transition = _log_transition_matrix(
            bin_centers,
            sigma_cm=self.sigma_cm,
            max_step_sigma=self.max_step_sigma,
        )
        log_alpha = emissions.log_likelihood[0] - np.log(emissions.n_bins)
        trajectory_log_posterior = [_normalize_log_weights(log_alpha)]
        for time_index in range(1, emissions.n_time):
            log_alpha = emissions.log_likelihood[time_index] + _log_sparse_matvec(log_alpha, transition)
            trajectory_log_posterior.append(_normalize_log_weights(log_alpha))
        logp = float(logsumexp(log_alpha))
        terminal_log_posterior = _normalize_log_weights(log_alpha)
        return EventScore(
            self.name,
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=_posterior_diagnostics(terminal_log_posterior, bin_centers),
            terminal_log_posterior=terminal_log_posterior,
            trajectory_log_posterior=np.stack(trajectory_log_posterior, axis=0),
        )


@dataclass
class CandidateKinematicModel:
    """Candidate-pruned kinematic scorer.

    `mode="diffusion"` or `mode="momentum"` gives a static candidate-pruned
    motion model. `mode="imm"` switches between stationary, diffusion,
    momentum, and jump dynamics.

    Candidate pruning truncates the path sum for tractability. Priors and
    transition kernels are still normalized over the full spatial grid, so
    returned log likelihoods are conservative truncated full-grid evidences,
    not support-conditioned evidences.
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
        _validate_positive_parameter("stationary_sigma_cm", self.stationary_sigma_cm)
        _validate_positive_parameter("diffusion_sigma_cm", self.diffusion_sigma_cm)
        _validate_positive_parameter("momentum_sigma_cm", self.momentum_sigma_cm)
        _validate_nonnegative_parameter("velocity_decay", self.velocity_decay)
        _validate_probability_parameter("mode_stickiness", self.mode_stickiness)
        top_k_value = np.asarray(self.top_k)
        if top_k_value.ndim != 0 or not np.issubdtype(top_k_value.dtype, np.integer):
            raise TypeError("top_k must be an integer")
        self.top_k = int(top_k_value)
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
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
        bin_centers = _validate_score_inputs(emissions, bin_centers)
        if emissions.n_time == 1:
            if candidate_indices is not None:
                _validate_candidate_support(candidate_indices, emissions.n_time, emissions.n_bins)
            return self._score_single_bin(emissions, bin_centers)
        candidates = self.candidate_indices(emissions) if candidate_indices is None else candidate_indices
        candidates = _validate_candidate_support(candidates, emissions.n_time, emissions.n_bins)
        if self.mode != "imm":
            logp, mass, terminal_log_posterior, trajectory_log_posterior = self._score_static_pair(
                emissions,
                bin_centers,
                candidates,
                self.mode,
            )
        else:
            logp, mass, terminal_log_posterior, trajectory_log_posterior = self._score_imm(
                emissions,
                bin_centers,
                candidates,
            )
        diagnostics = {
            "mean_candidate_log_mass": float(np.mean(mass)),
            "candidate_evidence_support": "truncated_full_grid",
        }
        diagnostics.update(_posterior_diagnostics(terminal_log_posterior, bin_centers))
        return EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal_log_posterior,
            trajectory_log_posterior=trajectory_log_posterior,
        )

    def _score_single_bin(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        """Score a one-bin event without letting it enter dynamic comparisons.

        Candidate kinematic models are path models: diffusion, momentum, and IMM
        need at least a pair of positions to express dynamics.  For a single
        time bin the only well-defined likelihood is the random one-bin marginal.
        Keep the requested model name for traceability, but tag the support so
        reporting code treats the row as non-comparable rather than as a genuine
        dynamic-model evidence.
        """

        base = RandomModel(name=str(self.name)).score(emissions, bin_centers)
        diagnostics = dict(base.diagnostics)
        diagnostics.update(
            {
                "candidate_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
                "candidate_degenerate_reason": "single_time_bin_random_marginal",
                "candidate_required_min_time_bins": 2,
            }
        )
        return EventScore(
            str(self.name),
            base.log_likelihood,
            base.n_time,
            base.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=base.terminal_log_posterior,
            trajectory_log_posterior=base.trajectory_log_posterior,
        )

    def _score_static_pair(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidates: list[np.ndarray],
        mode: str,
    ) -> tuple[float, list[float], np.ndarray, np.ndarray]:
        masses = _candidate_log_masses(emissions, candidates)
        first = candidates[0]
        second = candidates[1]
        log_pair = _init_pair_log_alpha(
            emissions,
            first,
            second,
            bin_centers,
            mode=mode,
            stationary_sigma_cm=self.stationary_sigma_cm,
            diffusion_sigma_cm=self.diffusion_sigma_cm,
            momentum_sigma_cm=self.momentum_sigma_cm,
        )
        n_bins = bin_centers.shape[0]
        prev_prev = first
        prev = second
        trajectory_log_posterior = [
            _pair_previous_posterior(log_pair, first, n_bins),
            _pair_terminal_posterior(log_pair, second, n_bins),
        ]
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
            trajectory_log_posterior.append(
                _pair_terminal_posterior(log_pair, curr, n_bins)
            )
        terminal_log_posterior = trajectory_log_posterior[-1]
        return (
            float(logsumexp(log_pair)),
            masses,
            terminal_log_posterior,
            np.stack(trajectory_log_posterior, axis=0),
        )

    def _score_imm(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidates: list[np.ndarray],
    ) -> tuple[float, list[float], np.ndarray, np.ndarray]:
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
                    mode=mode,
                    stationary_sigma_cm=self.stationary_sigma_cm,
                    diffusion_sigma_cm=self.diffusion_sigma_cm,
                    momentum_sigma_cm=self.momentum_sigma_cm,
                )
            )
        log_alpha = np.stack(by_mode, axis=0) - np.log(len(modes))
        n_bins = bin_centers.shape[0]
        prev_prev = first
        prev = second
        trajectory_log_posterior = [
            _pair_previous_posterior(log_alpha, first, n_bins),
            _pair_terminal_posterior(log_alpha, second, n_bins),
        ]
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
            trajectory_log_posterior.append(
                _pair_terminal_posterior(log_alpha, curr, n_bins)
            )
        terminal_log_posterior = trajectory_log_posterior[-1]
        return (
            float(logsumexp(log_alpha)),
            masses,
            terminal_log_posterior,
            np.stack(trajectory_log_posterior, axis=0),
        )


def score_model(model: ReplayModel, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
    return model.score(emissions, bin_centers)


def _validate_candidate_support(
    candidates: list[np.ndarray],
    n_time: int,
    n_bins: int,
) -> list[np.ndarray]:
    try:
        return _validate_candidate_indices(candidates, n_time, n_bins)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


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


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    return log_weights - logsumexp(log_weights)


def _normalize_log_weights_by_row(log_weights: np.ndarray) -> np.ndarray:
    return log_weights - logsumexp(log_weights, axis=1, keepdims=True)


def _posterior_diagnostics(
    terminal_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
) -> dict[str, float | int]:
    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] < 1:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    posterior = np.exp(terminal_log_posterior)
    endpoint = posterior @ centers
    map_bin = int(np.argmax(terminal_log_posterior))
    with np.errstate(invalid="ignore"):
        entropy_terms = np.where(posterior > 0.0, posterior * terminal_log_posterior, 0.0)
    entropy = float(-np.sum(entropy_terms))
    endpoint_y = float(endpoint[1]) if centers.shape[1] > 1 else 0.0
    map_y = float(centers[map_bin, 1]) if centers.shape[1] > 1 else 0.0
    return {
        "decoded_endpoint_x": float(endpoint[0]),
        "decoded_endpoint_y": endpoint_y,
        "decoded_map_x": float(centers[map_bin, 0]),
        "decoded_map_y": map_y,
        "decoded_map_bin": map_bin,
        "terminal_posterior_entropy": entropy,
    }


def _pair_terminal_posterior(
    log_pair_or_modes: np.ndarray,
    current_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    if log_pair_or_modes.ndim == 3:
        collapsed = logsumexp(log_pair_or_modes, axis=(0, 1))
    else:
        collapsed = logsumexp(log_pair_or_modes, axis=0)
    log_posterior = np.full(n_bins, LOG_ZERO, dtype=float)
    log_posterior[current_indices] = collapsed
    return _normalize_log_weights(log_posterior)


def _pair_previous_posterior(
    log_pair_or_modes: np.ndarray,
    previous_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    if log_pair_or_modes.ndim == 3:
        collapsed = logsumexp(log_pair_or_modes, axis=(0, 2))
    else:
        collapsed = logsumexp(log_pair_or_modes, axis=1)
    log_posterior = np.full(n_bins, LOG_ZERO, dtype=float)
    log_posterior[previous_indices] = collapsed
    return _normalize_log_weights(log_posterior)


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
    *,
    mode: str = "diffusion",
    stationary_sigma_cm: float = 2.0,
    diffusion_sigma_cm: float = 12.0,
    momentum_sigma_cm: float = 12.0,
) -> np.ndarray:
    first_ll = emissions.log_likelihood[0, first]
    second_ll = emissions.log_likelihood[1, second]
    coords_first = bin_centers[first]
    coords_second = bin_centers[second]
    if mode == "jump":
        log_kernel = np.full((len(first), len(second)), -np.log(emissions.n_bins), dtype=float)
        return first_ll[:, None] - np.log(emissions.n_bins) + log_kernel + second_ll[None, :]
    if mode == "stationary":
        sigma = stationary_sigma_cm
    elif mode == "diffusion":
        sigma = diffusion_sigma_cm
    elif mode == "momentum":
        sigma = momentum_sigma_cm
    else:
        raise ValueError(f"Unknown kinematic mode: {mode}")
    log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
        coords_first,
        coords_second,
        bin_centers,
        sigma,
    )
    return first_ll[:, None] - np.log(emissions.n_bins) + log_kernel + second_ll[None, :]


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
        uniform = -np.log(bin_centers.shape[0])
        return collapsed_by_prev[:, None] + uniform + curr_emission[None, :]
    for prev_col in range(len(prev)):
        if mode == "stationary":
            predicted = coords_prev[prev_col][None, :]
            sigma = stationary_sigma_cm
            previous_mass = logsumexp(log_pair[:, prev_col])
            log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
                predicted,
                coords_curr,
                bin_centers,
                sigma,
            )[0]
        elif mode == "diffusion":
            predicted = coords_prev[prev_col][None, :]
            sigma = diffusion_sigma_cm
            previous_mass = logsumexp(log_pair[:, prev_col])
            log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
                predicted,
                coords_curr,
                bin_centers,
                sigma,
            )[0]
        elif mode == "momentum":
            predictions = coords_prev[prev_col][None, :] + velocity_decay * (
                coords_prev[prev_col][None, :] - coords_prev_prev
            )
            log_kernel_by_source = _full_grid_normalized_pairwise_gaussian_log_prob(
                predictions,
                coords_curr,
                bin_centers,
                momentum_sigma_cm,
            )
            values = log_pair[:, prev_col][:, None] + log_kernel_by_source
            output[prev_col] = logsumexp(values, axis=0) + curr_emission
            continue
        else:
            raise ValueError(f"Unknown kinematic mode: {mode}")
        output[prev_col] = previous_mass + log_kernel + curr_emission
    return output


def _full_grid_normalized_pairwise_gaussian_log_prob(
    predicted: np.ndarray,
    observed: np.ndarray,
    all_observed: np.ndarray,
    sigma: float,
) -> np.ndarray:
    log_kernel = _pairwise_gaussian_log_prob(predicted, observed, sigma)
    log_normalizer = logsumexp(
        _pairwise_gaussian_log_prob(predicted, all_observed, sigma),
        axis=1,
        keepdims=True,
    )
    return log_kernel - log_normalizer


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
