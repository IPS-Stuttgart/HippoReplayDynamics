import itertools

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel


def test_diffusion_matches_bruteforce_tiny_grid():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = DiffusionModel(sigma_cm=1.0, max_step_sigma=10.0)
    score = model.score(emissions, centers)

    transition = np.empty((2, 2))
    for i in range(2):
        weights = np.exp(-0.5 * np.sum((centers - centers[i]) ** 2, axis=1))
        transition[i] = weights / weights.sum()
    brute_terms = []
    for path in itertools.product(range(2), repeat=3):
        logp = -np.log(2.0) + emissions.log_likelihood[0, path[0]]
        logp += np.log(transition[path[0], path[1]]) + emissions.log_likelihood[1, path[1]]
        logp += np.log(transition[path[1], path[2]]) + emissions.log_likelihood[2, path[2]]
        brute_terms.append(logp)
    brute = logsumexp(brute_terms)

    assert np.allclose(score.log_likelihood, brute)


def test_imm_scores_stationary_to_momentum_synthetic_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.90, 0.05, 0.03, 0.02],
                [0.85, 0.10, 0.03, 0.02],
                [0.05, 0.80, 0.10, 0.05],
                [0.02, 0.08, 0.80, 0.10],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.arange(4),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = CandidateKinematicModel(mode="imm", top_k=4, diffusion_sigma_cm=1.0, momentum_sigma_cm=1.0)
    score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["mean_candidate_log_mass"] == 0.0


def test_pyrecest_goal_particle_model_scores_synthetic_event():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    log_likelihood = np.log(
        np.array(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.15, 0.65, 0.15, 0.05],
                [0.05, 0.15, 0.65, 0.15],
            ]
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.02, 0.04]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([[0.0, 0.0], [3.0, 0.0]]),
        n_particles=128,
        random_seed=0,
        jump_probability=0.0,
        goal_reset_probability=0.0,
    )

    score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert np.allclose(logsumexp(score.terminal_log_posterior), 0.0)
    assert score.diagnostics["pyrecest_candidate_goals"] == 2
