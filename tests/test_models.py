import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import (
    DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel
from hipporeplayimm.pyrecest_models import (
    PyRecEstGoalParticleIMMModel,
    PyRecEstGoalParticleModel,
    _grid_proposal_weights,
)


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


def test_candidate_jump_initial_pair_is_full_grid_uniform():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.98, 0.01, 0.01],
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = CandidateKinematicModel(
        mode="jump",
        top_k=3,
        diffusion_sigma_cm=0.1,
    )

    score = model.score(emissions, centers)

    assert np.allclose(
        np.exp(score.terminal_log_posterior),
        np.full(centers.shape[0], 1.0 / centers.shape[0]),
    )


def test_candidate_momentum_initial_pair_uses_momentum_sigma_not_diffusion_sigma():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.80, 0.10, 0.10],
                    [0.10, 0.45, 0.45],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    narrow_diffusion = CandidateKinematicModel(
        mode="momentum",
        top_k=3,
        diffusion_sigma_cm=0.2,
        momentum_sigma_cm=2.0,
    )
    broad_diffusion = CandidateKinematicModel(
        mode="momentum",
        top_k=3,
        diffusion_sigma_cm=10.0,
        momentum_sigma_cm=2.0,
    )

    narrow_score = narrow_diffusion.score(emissions, centers)
    broad_score = broad_diffusion.score(emissions, centers)

    assert np.allclose(narrow_score.log_likelihood, broad_score.log_likelihood)
    assert np.allclose(narrow_score.terminal_log_posterior, broad_score.terminal_log_posterior)


def test_candidate_diffusion_full_support_matches_exact_diffusion():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.60, 0.30, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.10, 0.30, 0.60],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )

    exact = DiffusionModel(sigma_cm=1.0, max_step_sigma=10.0).score(emissions, centers)
    candidate = CandidateKinematicModel(mode="diffusion", top_k=centers.shape[0], diffusion_sigma_cm=1.0).score(emissions, centers)

    assert np.allclose(candidate.log_likelihood, exact.log_likelihood)
    assert candidate.diagnostics["mean_candidate_log_mass"] == 0.0
    assert candidate.diagnostics["candidate_evidence_support"] == "truncated_full_grid"


def test_candidate_single_bin_is_marked_degenerate_and_not_comparable():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    log_likelihood = np.log(np.array([[0.2, 0.7, 0.1]]))
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = CandidateKinematicModel(mode="imm", top_k=3)

    score = model.score(emissions, centers)

    expected_random_marginal = float(logsumexp(log_likelihood[0]) - np.log(centers.shape[0]))
    assert score.model_name == "imm"
    assert np.allclose(score.log_likelihood, expected_random_marginal)
    assert score.diagnostics["candidate_evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert score.diagnostics["candidate_degenerate_reason"] == "single_time_bin_random_marginal"

    scored = ensure_evidence_support_columns(
        pd.DataFrame(
            [
                {
                    "status": "success",
                    "model": score.model_name,
                    "diagnostic_candidate_evidence_support": score.diagnostics["candidate_evidence_support"],
                }
            ]
        )
    )
    assert scored.loc[0, "evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_candidate_diffusion_pruned_support_uses_full_grid_normalization():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -10.0],
                [0.0, -10.0],
                [0.0, -10.0],
            ]
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = CandidateKinematicModel(mode="diffusion", top_k=1, diffusion_sigma_cm=1.0)
    candidates = model.candidate_indices(emissions)

    score = model.score(emissions, centers)

    src0 = int(candidates[0][0])
    dst1 = int(candidates[1][0])
    dst2 = int(candidates[2][0])
    weights01 = np.exp(-0.5 * np.sum((centers - centers[src0]) ** 2, axis=1))
    weights12 = np.exp(-0.5 * np.sum((centers - centers[dst1]) ** 2, axis=1))
    expected = (
        emissions.log_likelihood[0, src0]
        - np.log(emissions.n_bins)
        + np.log(weights01[dst1] / weights01.sum())
        + emissions.log_likelihood[1, dst1]
        + np.log(weights12[dst2] / weights12.sum())
        + emissions.log_likelihood[2, dst2]
    )
    assert np.allclose(score.log_likelihood, expected)


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
    assert score.diagnostics["pyrecest_position_proposal_probability"] == 0.0


def test_grid_proposal_weights_normalize_finite_likelihoods():
    weights = _grid_proposal_weights(np.array([0.0, -np.inf, np.log(3.0)]))

    assert np.allclose(weights, np.array([0.25, 0.0, 0.75]))


def test_pyrecest_goal_particle_model_uses_position_proposal_when_available():
    filters = pytest.importorskip("pyrecest.filters")
    particle_filter = getattr(filters, "GoalConditionedReplayParticleFilter", None)
    if particle_filter is None or not hasattr(
        particle_filter,
        "update_position_likelihood_with_proposal",
    ):
        pytest.skip("PyRecEst position proposal rejuvenation is not installed")

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
        position_proposal_probability=1.0,
    )

    score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["pyrecest_position_proposal_probability"] == 1.0
    assert score.diagnostics["pyrecest_last_position_proposal_fraction"] == 1.0


def test_pyrecest_goal_particle_imm_model_scores_synthetic_event():
    filters = pytest.importorskip("pyrecest.filters")
    if not hasattr(filters, "GoalConditionedReplayParticleIMMFilter"):
        pytest.skip("GoalConditionedReplayParticleIMMFilter is not installed")

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
    model = PyRecEstGoalParticleIMMModel(
        candidate_goals=np.array([[0.0, 0.0], [3.0, 0.0]]),
        n_particles=128,
        random_seed=0,
        jump_probability=0.0,
        goal_reset_probability=0.0,
        mode_stickiness=0.9,
    )

    score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert np.allclose(logsumexp(score.terminal_log_posterior), 0.0)
    assert score.diagnostics["pyrecest_candidate_goals"] == 2
    assert "pyrecest_most_likely_mode" in score.diagnostics
