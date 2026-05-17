import itertools

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _kernel_log_prob(centers: np.ndarray, predicted: np.ndarray, dst: int, sigma_cm: float) -> float:
    weights = np.exp(-0.5 * np.sum((centers - predicted) ** 2, axis=1) / (sigma_cm * sigma_cm))
    return float(np.log(weights[dst] / weights.sum()))


def _bruteforce_duration_imm_log_evidence(
    emissions: LogEmissionTensor,
    centers: np.ndarray,
    config: StateSpaceDecoderConfig,
    transition_durations_s: np.ndarray,
    *,
    nominal_dt_s: float,
) -> float:
    modes = ("stationary", "diffusion", "momentum", "jump")
    n_modes = len(modes)
    n_bins = centers.shape[0]
    mode_transition = np.full((n_modes, n_modes), (1.0 - config.imm_mode_stickiness) / (n_modes - 1), dtype=float)
    np.fill_diagonal(mode_transition, config.imm_mode_stickiness)

    def kernel_log(mode: str, src: int, prev: int, dst: int, transition_index: int, *, initial: bool = False) -> float:
        if mode == "jump":
            return -np.log(n_bins)
        if initial:
            predicted = centers[src]
            if mode == "stationary":
                sigma_cm = config.stationary_sigma_cm
            elif mode == "diffusion":
                sigma_cm = config.diffusion_sigma_cm_sqrt_s * np.sqrt(transition_durations_s[0])
            elif mode == "momentum":
                sigma_cm = config.momentum_initial_sigma_cm_sqrt_s * np.sqrt(transition_durations_s[0])
            else:  # pragma: no cover - modes tuple controls this.
                raise AssertionError(mode)
            return _kernel_log_prob(centers, predicted, dst, float(sigma_cm))
        if mode == "stationary":
            predicted = centers[prev]
            sigma_cm = config.stationary_sigma_cm
        elif mode == "diffusion":
            predicted = centers[prev]
            sigma_cm = config.diffusion_sigma_cm_sqrt_s * np.sqrt(transition_durations_s[transition_index])
        elif mode == "momentum":
            duration = float(transition_durations_s[transition_index])
            previous_duration = float(transition_durations_s[transition_index - 1])
            decay = config.momentum_velocity_decay ** (duration / nominal_dt_s)
            duration_scale = duration / previous_duration
            predicted = centers[prev] + decay * duration_scale * (centers[prev] - centers[src])
            sigma_cm = config.momentum_sigma_cm_sqrt_s * np.sqrt(duration)
        else:  # pragma: no cover - modes tuple controls this.
            raise AssertionError(mode)
        return _kernel_log_prob(centers, predicted, dst, float(sigma_cm))

    terms: list[float] = []
    for x0, x1, x2 in itertools.product(range(n_bins), repeat=3):
        for first_mode_index, first_mode in enumerate(modes):
            for second_mode_index, second_mode in enumerate(modes):
                terms.append(
                    -np.log(n_bins)
                    + emissions.log_likelihood[0, x0]
                    - np.log(n_modes)
                    + kernel_log(first_mode, x0, x0, x1, 0, initial=True)
                    + emissions.log_likelihood[1, x1]
                    + np.log(mode_transition[first_mode_index, second_mode_index])
                    + kernel_log(second_mode, x0, x1, x2, 1)
                    + emissions.log_likelihood[2, x2]
                )
    return float(logsumexp(terms))


def test_four_mode_imm_uses_transition_durations_for_partial_bins():
    nominal_dt_s = 1.0
    transition_durations_s = np.array([0.25, 2.0], dtype=float)
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.90, 0.09, 0.01],
                    [0.05, 0.90, 0.05],
                    [0.01, 0.09, 0.90],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, transition_durations_s[0], transition_durations_s.sum()]),
        dt=nominal_dt_s,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="imm",
        stationary_sigma_cm=0.4,
        diffusion_sigma_cm_sqrt_s=1.2,
        imm_mode_stickiness=0.65,
        momentum_sigma_cm_sqrt_s=0.8,
        momentum_initial_sigma_cm_sqrt_s=1.2,
        momentum_velocity_decay=0.7,
        momentum_candidate_top_k=3,
    )

    score = StateSpaceReplayModel(mode="imm", config=config).score(emissions, centers)

    expected = _bruteforce_duration_imm_log_evidence(
        emissions,
        centers,
        config,
        transition_durations_s,
        nominal_dt_s=nominal_dt_s,
    )
    representative_dt_expected = _bruteforce_duration_imm_log_evidence(
        emissions,
        centers,
        config,
        np.full_like(transition_durations_s, nominal_dt_s),
        nominal_dt_s=nominal_dt_s,
    )

    assert np.allclose(score.log_likelihood, expected, rtol=1e-8, atol=1e-8)
    assert not np.allclose(score.log_likelihood, representative_dt_expected, rtol=1e-4, atol=1e-4)
    assert score.diagnostics["state_space_transition_durations"] == "0.25,2"
    assert score.diagnostics["state_space_imm_modes"] == "stationary,diffusion,momentum,jump"
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
