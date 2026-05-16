"""State-space replay decoder baselines with full trajectory posteriors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import EventScore, LOG_ZERO, _normalize_log_weights, _posterior_diagnostics


@dataclass(frozen=True)
class StateSpaceDecoderConfig:
    """Configuration for state-space replay baselines.

    Diffusion and momentum noise are specified in cm/sqrt(s). They are converted
    to per-bin standard deviations using the emission tensor's ``dt`` so that the
    same model can be evaluated with 1--3 ms replay bins.
    """

    mode: str = "diffusion"
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm_sqrt_s: float = 85.0
    max_step_sigma: float = 4.0
    imm_mode_stickiness: float = 0.95
    momentum_sigma_cm_sqrt_s: float = 85.0
    momentum_initial_sigma_cm_sqrt_s: float = 85.0
    momentum_velocity_decay: float = 0.95
    momentum_candidate_top_k: int = 128


@dataclass
class StateSpaceReplayModel:
    """Replay decoder baseline returning a posterior for every replay bin.

    Supported modes are ``stationary``, ``diffusion``, ``fragmented``/``jump``,
    ``imm``, and ``momentum``. The first-order models use exact full-grid
    forward-backward recursions. The momentum model uses candidate-pruned
    second-order dynamics for scalability. Its candidate recursion keeps
    full-grid prior and transition normalizers and drops off-support paths, so
    its evidence is a conservative truncated full-grid evidence. It returns
    candidate-supported per-bin posterior marginals.
    """

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        allowed = {"stationary", "diffusion", "fragmented", "jump", "imm", "momentum"}
        if self.mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        if self.name is None:
            self.name = f"state-space-{self.mode}"
        if self.config is None:
            self.config = StateSpaceDecoderConfig(mode=self.mode)
        elif self.config.mode != self.mode:
            self.config = replace(self.config, mode=self.mode)

    def candidate_indices(self, emissions: LogEmissionTensor) -> list[np.ndarray]:
        """Return the candidate support used by the pruned momentum recursion."""

        assert self.config is not None
        return [_top_candidate_indices(row, self.config.momentum_candidate_top_k) for row in emissions.log_likelihood]

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if emissions.n_bins != bin_centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        assert self.config is not None

        if self.mode == "stationary":
            logp, trajectory = _score_stationary(emissions)
            transition_sigma_cm = 0.0
            extra: dict[str, float | int | str] = {}
        elif self.mode in {"fragmented", "jump"}:
            logp, trajectory = _score_fragmented(emissions)
            transition_sigma_cm = float("inf")
            extra = {}
        elif self.mode == "diffusion":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            transition = _gaussian_transition_matrix(bin_centers, transition_sigma_cm, self.config.max_step_sigma)
            logp, trajectory = _forward_backward_first_order(emissions.log_likelihood, transition)
            extra = {}
        elif self.mode == "imm":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            logp, trajectory, mode_post = _score_imm(
                emissions.log_likelihood,
                bin_centers,
                stationary_sigma_cm=self.config.stationary_sigma_cm,
                diffusion_sigma_cm=transition_sigma_cm,
                max_step_sigma=self.config.max_step_sigma,
                mode_stickiness=self.config.imm_mode_stickiness,
            )
            names = ("stationary", "diffusion", "fragmented")
            extra = {
                f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
                for idx, name in enumerate(names)
            }
        else:
            transition_sigma_cm = _per_bin_sigma(self.config.momentum_sigma_cm_sqrt_s, emissions.dt)
            candidates = self.candidate_indices(emissions) if candidate_indices is None else candidate_indices
            _validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)
            logp, trajectory, masses = _score_momentum_candidates(
                emissions,
                bin_centers,
                candidates,
                sigma_cm=transition_sigma_cm,
                initial_sigma_cm=_per_bin_sigma(self.config.momentum_initial_sigma_cm_sqrt_s, emissions.dt),
                velocity_decay=self.config.momentum_velocity_decay,
            )
            extra = {
                "mean_candidate_log_mass": float(np.mean(masses)),
                "state_space_momentum_candidate_top_k": int(self.config.momentum_candidate_top_k),
                "state_space_momentum_candidate_support": "derived" if candidate_indices is None else "provided",
                "state_space_momentum_trajectory_posterior": "smoothed_pair_marginal",
                "state_space_momentum_evidence_support": "truncated_full_grid",
            }

        terminal = trajectory[-1]
        diagnostics: dict[str, float | int | str] = {
            "state_space_mode": str(self.mode),
            "state_space_time_bin_s": float(emissions.dt),
            "state_space_trajectory_posterior": 1,
            "state_space_trajectory_time_bins": int(emissions.n_time),
            "state_space_stationary_sigma_cm": float(self.config.stationary_sigma_cm),
            "state_space_diffusion_sigma_cm_sqrt_s": float(self.config.diffusion_sigma_cm_sqrt_s),
            "state_space_max_step_sigma": float(self.config.max_step_sigma),
            "state_space_imm_mode_stickiness": float(self.config.imm_mode_stickiness),
            "state_space_momentum_sigma_cm_sqrt_s": float(self.config.momentum_sigma_cm_sqrt_s),
            "state_space_momentum_initial_sigma_cm_sqrt_s": float(self.config.momentum_initial_sigma_cm_sqrt_s),
            "state_space_momentum_velocity_decay": float(self.config.momentum_velocity_decay),
            "state_space_transition_sigma_cm": float(transition_sigma_cm),
            "mean_trajectory_posterior_entropy": _mean_entropy(trajectory),
            **extra,
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    return max(float(sigma_cm_sqrt_s) * np.sqrt(max(float(dt_s), np.finfo(float).tiny)), np.finfo(float).eps)


def _score_stationary(emissions: LogEmissionTensor) -> tuple[float, np.ndarray]:
    log_weights = np.sum(emissions.log_likelihood, axis=0) - np.log(emissions.n_bins)
    logp = float(logsumexp(log_weights))
    posterior = _normalize_log_weights(log_weights)
    return logp, np.repeat(posterior[None, :], emissions.n_time, axis=0)


def _score_fragmented(emissions: LogEmissionTensor) -> tuple[float, np.ndarray]:
    scaled, offsets = _scaled_emissions(emissions.log_likelihood)
    row_sums = scaled.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("at least one emission row has no finite likelihood mass")
    logp = float(np.sum(np.log(row_sums / emissions.n_bins) + offsets))
    return logp, _as_log_probs(scaled / row_sums[:, None])


def _forward_backward_first_order(log_likelihood: np.ndarray, transition: csr_matrix) -> tuple[float, np.ndarray]:
    n_time, n_bins = log_likelihood.shape
    scaled, offsets = _scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = scaled[0] / n_bins
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        alpha = np.asarray(transition @ alpha, dtype=float) * scaled[time_index]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones(n_bins, dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta = np.asarray(transition.T @ (scaled[time_index] * beta), dtype=float) / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]
    return logp, _as_log_probs(smoothed)


def _score_imm(
    log_likelihood: np.ndarray,
    bin_centers: np.ndarray,
    *,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    max_step_sigma: float,
    mode_stickiness: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    modes = ("stationary", "diffusion", "fragmented")
    n_modes = len(modes)
    n_time, n_bins = log_likelihood.shape
    transitions = {
        "stationary": _gaussian_transition_matrix(bin_centers, stationary_sigma_cm, max_step_sigma),
        "diffusion": _gaussian_transition_matrix(bin_centers, diffusion_sigma_cm, max_step_sigma),
        "fragmented": None,
    }
    mode_transition = _mode_transition_matrix(n_modes, mode_stickiness)
    scaled, offsets = _scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_modes, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = np.tile(scaled[0] / (n_bins * n_modes), (n_modes, 1))
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        predicted = np.zeros_like(alpha)
        for dst_idx, dst_mode in enumerate(modes):
            dst = np.zeros(n_bins, dtype=float)
            for src_idx in range(n_modes):
                dst += mode_transition[src_idx, dst_idx] * _apply_transition(transitions[dst_mode], alpha[src_idx])
            predicted[dst_idx] = dst
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    # Backward smoothing over the joint mode-position state.
    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_modes, n_bins), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta_prev = np.zeros_like(beta)
        for src_idx in range(n_modes):
            for dst_idx, dst_mode in enumerate(modes):
                beta_prev[src_idx] += mode_transition[src_idx, dst_idx] * _apply_transition_backward(
                    transitions[dst_mode], scaled[time_index] * beta[dst_idx]
                )
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    return logp, _as_log_probs(smoothed.sum(axis=1)), smoothed.sum(axis=2)


def _apply_transition(transition: csr_matrix | None, weights: np.ndarray) -> np.ndarray:
    if transition is None:
        return np.full(weights.shape, float(weights.sum()) / weights.shape[0], dtype=float)
    return np.asarray(transition @ weights, dtype=float)


def _apply_transition_backward(transition: csr_matrix | None, values: np.ndarray) -> np.ndarray:
    if transition is None:
        return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
    return np.asarray(transition.T @ values, dtype=float)


def _score_momentum_candidates(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
    *,
    sigma_cm: float,
    initial_sigma_cm: float,
    velocity_decay: float,
) -> tuple[float, np.ndarray, list[float]]:
    if emissions.n_time == 1:
        logp, trajectory = _score_fragmented(emissions)
        return logp, trajectory, [0.0]

    masses = _candidate_log_masses(emissions.log_likelihood, candidates)
    log_pair = _init_pair_log_alpha(
        emissions.log_likelihood,
        candidates[0],
        candidates[1],
        bin_centers,
        sigma_cm=initial_sigma_cm,
    )
    pair_alphas = [log_pair]
    for time_index in range(2, emissions.n_time):
        log_pair = _advance_momentum_pair(
            log_pair,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            sigma_cm=sigma_cm,
            velocity_decay=velocity_decay,
        )
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        curr_time = pair_index + 2
        pair_betas[pair_index] = _backward_momentum_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            sigma_cm=sigma_cm,
            velocity_decay=velocity_decay,
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=1)
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=0)
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    return logp, trajectory, masses


def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= log_emission.shape[0]:
        return np.arange(log_emission.shape[0], dtype=int)
    selected = np.argpartition(log_emission, -top_k)[-top_k:]
    return selected[np.argsort(log_emission[selected])[::-1]]


def _validate_candidate_indices(candidates: list[np.ndarray], n_time: int, n_bins: int) -> None:
    if len(candidates) != n_time:
        raise ValueError("candidate_indices must contain one array per emission time bin")
    for time_index, curr in enumerate(candidates):
        arr = np.asarray(curr)
        if arr.ndim != 1:
            raise ValueError(f"candidate_indices[{time_index}] must be one-dimensional")
        if arr.size == 0:
            raise ValueError(f"candidate_indices[{time_index}] must not be empty")
        if np.any((arr < 0) | (arr >= n_bins)):
            raise ValueError(f"candidate_indices[{time_index}] contains an out-of-range bin")


def _candidate_log_masses(log_likelihood: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
    return [
        float(logsumexp(log_likelihood[time_index, curr]) - logsumexp(log_likelihood[time_index]))
        for time_index, curr in enumerate(candidates)
    ]


def _init_pair_log_alpha(
    log_likelihood: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
) -> np.ndarray:
    log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
        bin_centers[first],
        bin_centers[second],
        bin_centers,
        sigma_cm,
    )
    n_bins = log_likelihood.shape[1]
    return log_likelihood[0, first][:, None] - np.log(n_bins) + log_kernel + log_likelihood[1, second][None, :]


def _advance_momentum_pair(
    log_pair: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    velocity_decay: float,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev), len(curr)), LOG_ZERO, dtype=float)
    for prev_col in range(len(prev)):
        predictions = coords_prev[prev_col][None, :] + velocity_decay * (
            coords_prev[prev_col][None, :] - coords_prev_prev
        )
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            predictions,
            coords_curr,
            bin_centers,
            sigma_cm,
        )
        output[prev_col] = logsumexp(log_pair[:, prev_col][:, None] + log_kernel, axis=0) + curr_emission
    return output


def _backward_momentum_pair(
    next_beta: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    velocity_decay: float,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev_prev), len(prev)), LOG_ZERO, dtype=float)
    for prev_col in range(len(prev)):
        predictions = coords_prev[prev_col][None, :] + velocity_decay * (
            coords_prev[prev_col][None, :] - coords_prev_prev
        )
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            predictions,
            coords_curr,
            bin_centers,
            sigma_cm,
        )
        continuation = curr_emission[None, :] + next_beta[prev_col][None, :]
        output[:, prev_col] = logsumexp(log_kernel + continuation, axis=1)
    return output


def _gaussian_transition_matrix(bin_centers: np.ndarray, sigma_cm: float, max_step_sigma: float) -> csr_matrix:
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    n_bins = bin_centers.shape[0]
    radius2 = (sigma_cm * max_step_sigma) ** 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(bin_centers):
        delta = bin_centers - center[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius2
        if not np.any(keep):
            keep[int(np.argmin(dist2))] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma_cm * sigma_cm))
        weights /= float(weights.sum())
        rows.extend(int(idx) for idx in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))


def _full_grid_normalized_pairwise_gaussian_log_prob(
    predicted: np.ndarray,
    observed: np.ndarray,
    all_observed: np.ndarray,
    sigma_cm: float,
) -> np.ndarray:
    """Evaluate candidate log transitions with full-grid normalization."""

    log_kernel = _pairwise_gaussian_log_prob(predicted, observed, sigma_cm)
    log_normalizer = logsumexp(
        _pairwise_gaussian_log_prob(predicted, all_observed, sigma_cm),
        axis=1,
        keepdims=True,
    )
    return log_kernel - log_normalizer


def _pairwise_gaussian_log_prob(predicted: np.ndarray, observed: np.ndarray, sigma_cm: float) -> np.ndarray:
    delta = predicted[:, None, :] - observed[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    return -0.5 * dist2 / (sigma_cm * sigma_cm)


def _scaled_emissions(log_likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)
    if not np.all(np.any(finite, axis=1)):
        raise ValueError("every emission row must contain at least one finite value")
    offsets = np.max(np.where(finite, values, -np.inf), axis=1)
    shifted = np.where(finite, values - offsets[:, None], -np.inf)
    scaled = np.exp(np.clip(shifted, -745.0, 0.0))
    scaled[~finite] = 0.0
    return scaled, offsets


def _as_log_probs(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    out = np.full(probs.shape, LOG_ZERO, dtype=float)
    positive = probs > 0.0
    out[positive] = np.log(probs[positive])
    return out - logsumexp(out, axis=1)[:, None]


def _mean_entropy(trajectory_log_posterior: np.ndarray) -> float:
    posterior = np.exp(trajectory_log_posterior)
    return float(np.mean(-np.sum(posterior * trajectory_log_posterior, axis=1)))


def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    if n_modes < 2:
        return np.ones((n_modes, n_modes), dtype=float)
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("mode_stickiness must be in [0, 1]")
    off_diag = (1.0 - stickiness) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, stickiness)
    return matrix
