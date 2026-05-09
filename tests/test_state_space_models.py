import itertools

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig


def _synthetic_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.15, 0.65, 0.15, 0.05],
                    [0.05, 0.15, 0.65, 0.15],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.003, 0.006]),
        dt=0.003,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_state_space_diffusion_matches_bruteforce_tiny_grid():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="diffusion",
        diffusion_sigma_cm_sqrt_s=1.0,
        max_step_sigma=10.0,
    )
    score = SortedSpikeStateSpaceReplayModel(mode="diffusion", config=config).score(emissions, centers)

    transition = np.empty((2, 2))
    for src in range(2):
        weights = np.exp(-0.5 * np.sum((centers - centers[src]) ** 2, axis=1))
        transition[:, src] = weights / weights.sum()
    brute_terms = []
    for path in itertools.product(range(2), repeat=3):
        logp = -np.log(2.0) + emissions.log_likelihood[0, path[0]]
        logp += np.log(transition[path[1], path[0]]) + emissions.log_likelihood[1, path[1]]
        logp += np.log(transition[path[2], path[1]]) + emissions.log_likelihood[2, path[2]]
        brute_terms.append(logp)

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))
    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == (3, 2)
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)


def test_state_space_modes_return_full_trajectory_posteriors():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    for mode in ("stationary", "fragmented", "jump", "diffusion", "imm", "momentum"):
        config = StateSpaceDecoderConfig(mode=mode, momentum_candidate_top_k=4)
        score = SortedSpikeStateSpaceReplayModel(mode=mode, config=config).score(emissions, centers)

        assert np.isfinite(score.log_likelihood)
        assert score.model_name == f"sorted-spike-state-space-{mode}"
        assert score.trajectory_log_posterior is not None
        assert score.trajectory_log_posterior.shape == (emissions.n_time, emissions.n_bins)
        assert score.terminal_log_posterior is not None
        assert np.allclose(score.terminal_log_posterior, score.trajectory_log_posterior[-1])
        assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
        assert score.diagnostics["state_space_trajectory_posterior"] == 1
        assert score.diagnostics["state_space_observation_model"] == "sorted-spike-poisson"
