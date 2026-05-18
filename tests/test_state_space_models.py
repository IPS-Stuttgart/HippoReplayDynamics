import itertools

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _augment_candidates_with_momentum_predictions,
)


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

    for mode in ("stationary", "fragmented", "jump", "diffusion", "first-order-imm", "imm", "momentum"):
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
        assert score.diagnostics["clusterless_mark_likelihood"] == "not_implemented"
        if mode == "momentum":
            assert score.diagnostics["state_space_momentum_trajectory_posterior"] == "smoothed_pair_marginal"
            assert score.diagnostics["state_space_momentum_predicted_candidate_top_k"] == 8
            assert score.diagnostics["mean_candidate_count"] == 4.0
        if mode == "imm":
            assert score.diagnostics["state_space_imm_modes"] == "stationary,diffusion,momentum,jump"
            assert score.diagnostics["state_space_imm_evidence_support"] == "truncated_full_grid"
            assert score.diagnostics["state_space_imm_predicted_candidate_top_k"] == 8
            assert "state_space_mode_momentum_terminal_probability" in score.diagnostics
            assert "state_space_mode_jump_terminal_probability" in score.diagnostics
        if mode == "first-order-imm":
            assert score.diagnostics["state_space_imm_modes"] == "stationary,diffusion,fragmented"
            assert score.diagnostics["state_space_imm_evidence_support"] == "exact_full_grid"


def test_adaptive_candidate_support_adds_forward_and_backward_predictions():
    centers = np.arange(7.0)[:, None]
    base = [np.array([0]), np.array([1]), np.array([5]), np.array([6])]

    candidates = _augment_candidates_with_momentum_predictions(
        base,
        centers,
        predicted_top_k=1,
        velocity_decay=1.0,
    )

    assert set(candidates[0]) == {0, 3}
    assert set(candidates[1]) == {1, 4}
    assert set(candidates[2]) == {2, 5}
    assert set(candidates[3]) == {6}


def test_state_space_model_uses_adaptive_candidate_support_when_bin_centers_given():
    centers = np.arange(7.0)[:, None]
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                [-5.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0, -5.0, 0.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 0.0],
            ]
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.arange(4, dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=1,
        momentum_predicted_candidate_top_k=1,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)

    emission_only = model.candidate_indices(emissions)
    adaptive = model.candidate_indices(emissions, centers)
    score = model.score(emissions, centers)

    assert [list(row) for row in emission_only] == [[0], [1], [5], [6]]
    assert 2 in adaptive[2]
    assert 3 in adaptive[0]
    assert score.diagnostics["state_space_momentum_predicted_candidate_top_k"] == 1
    assert score.diagnostics["mean_candidate_count"] > 1.0


def test_state_space_momentum_pruned_support_uses_full_grid_normalization():
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
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=1,
        momentum_predicted_candidate_top_k=0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)
    candidates = model.candidate_indices(emissions)

    score = model.score(emissions, centers)

    src0 = int(candidates[0][0])
    dst1 = int(candidates[1][0])
    dst2 = int(candidates[2][0])
    weights01 = np.exp(-0.5 * np.sum((centers - centers[src0]) ** 2, axis=1))
    predicted = centers[dst1] + config.momentum_velocity_decay * (centers[dst1] - centers[src0])
    weights12 = np.exp(-0.5 * np.sum((centers - predicted) ** 2, axis=1))
    expected = (
        emissions.log_likelihood[0, src0]
        - np.log(emissions.n_bins)
        + np.log(weights01[dst1] / weights01.sum())
        + emissions.log_likelihood[1, dst1]
        + np.log(weights12[dst2] / weights12.sum())
        + emissions.log_likelihood[2, dst2]
    )

    assert np.allclose(score.log_likelihood, expected)
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
    assert score.diagnostics["state_space_momentum_evidence_support"] == "truncated_full_grid"


def test_state_space_momentum_can_reuse_external_candidate_support():
    train = _synthetic_emissions()
    joint = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.02, 0.08, 0.20, 0.70],
                    [0.05, 0.15, 0.65, 0.15],
                    [0.70, 0.20, 0.08, 0.02],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.003, 0.006]),
        dt=0.003,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(mode="momentum", momentum_candidate_top_k=2)
    model = SortedSpikeStateSpaceReplayModel(mode="momentum", config=config)
    train_candidates = model.candidate_indices(train)

    derived_joint_candidates = model.candidate_indices(joint)
    provided_score = model.score(joint, centers, candidate_indices=train_candidates)
    derived_score = model.score(joint, centers)

    assert any(not np.array_equal(a, b) for a, b in zip(train_candidates, derived_joint_candidates, strict=True))
    assert np.isfinite(provided_score.log_likelihood)
    assert np.isfinite(derived_score.log_likelihood)
    assert provided_score.diagnostics["state_space_momentum_candidate_support"] == "provided"
    assert derived_score.diagnostics["state_space_momentum_candidate_support"] == "derived"


def test_state_space_four_mode_imm_matches_bruteforce_tiny_grid():
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
        mode="imm",
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        imm_mode_stickiness=0.8,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.9,
        momentum_candidate_top_k=2,
    )
    score = StateSpaceReplayModel(mode="imm", config=config).score(emissions, centers)

    modes = ("stationary", "diffusion", "momentum", "jump")
    transition = np.full((4, 4), (1.0 - config.imm_mode_stickiness) / 3.0)
    np.fill_diagonal(transition, config.imm_mode_stickiness)

    def kernel_log(mode: str, src: int, prev: int, dst: int, *, initial: bool = False) -> float:
        if mode == "jump":
            return -np.log(centers.shape[0])
        if initial:
            sigma = (
                config.stationary_sigma_cm
                if mode == "stationary"
                else config.diffusion_sigma_cm_sqrt_s
                if mode == "diffusion"
                else config.momentum_initial_sigma_cm_sqrt_s
            )
            predicted = centers[src]
        elif mode == "stationary":
            sigma = config.stationary_sigma_cm
            predicted = centers[prev]
        elif mode == "diffusion":
            sigma = config.diffusion_sigma_cm_sqrt_s
            predicted = centers[prev]
        elif mode == "momentum":
            sigma = config.momentum_sigma_cm_sqrt_s
            predicted = centers[prev] + config.momentum_velocity_decay * (centers[prev] - centers[src])
        else:
            raise AssertionError(mode)
        weights = np.exp(-0.5 * np.sum((centers - predicted) ** 2, axis=1) / (sigma * sigma))
        return float(np.log(weights[dst] / weights.sum()))

    brute_terms = []
    for x0, x1, x2 in itertools.product(range(2), repeat=3):
        for m1_idx, mode1 in enumerate(modes):
            for m2_idx, mode2 in enumerate(modes):
                brute_terms.append(
                    -np.log(2.0)
                    + emissions.log_likelihood[0, x0]
                    - np.log(len(modes))
                    + kernel_log(mode1, x0, x0, x1, initial=True)
                    + emissions.log_likelihood[1, x1]
                    + np.log(transition[m1_idx, m2_idx])
                    + kernel_log(mode2, x0, x1, x2)
                    + emissions.log_likelihood[2, x2]
                )

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))
    assert score.diagnostics["state_space_imm_modes"] == "stationary,diffusion,momentum,jump"
    terminal_probs = [score.diagnostics[f"state_space_mode_{mode}_terminal_probability"] for mode in modes]
    assert np.allclose(sum(terminal_probs), 1.0)
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
