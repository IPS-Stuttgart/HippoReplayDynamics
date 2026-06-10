import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

import hipporeplayimm.state_space_trajectory_imm as trajectory_imm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _augment_candidates_with_momentum_predictions,
    _candidate_evidence_support_label,
    _displacement_lattice,
    _mass_retaining_candidate_indices,
    _score_trajectory_imm_exact_sparse,
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

    for mode in (
        "stationary",
        "fragmented",
        "jump",
        "diffusion",
        "first-order-imm",
        "trajectory-imm-exact-sparse",
        "imm",
        "momentum",
        "momentum-exact-sparse",
        "displacement-momentum",
        "displacement-imm",
    ):
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
            assert score.diagnostics["state_space_momentum_evidence_support"] == "exact_full_grid"
            assert score.diagnostics["state_space_momentum_candidate_support"] == "full_grid"
            assert score.diagnostics["state_space_momentum_predicted_candidate_top_k"] == 8
            assert score.diagnostics["state_space_momentum_evidence_support"] == "exact_full_grid"
            assert score.diagnostics["mean_candidate_count"] == 4.0
        if mode == "momentum-exact-sparse":
            assert score.diagnostics["state_space_sparse_momentum_evidence_support"] == "exact_full_grid"
            assert score.diagnostics["state_space_sparse_momentum_state_support"] == "finite_radius_pair_grid"
            assert score.diagnostics["state_space_momentum_candidate_support"] == "not_used_exact_sparse"
            assert score.diagnostics["state_space_sparse_momentum_max_pair_count"] > 0
            assert score.diagnostics["state_space_sparse_momentum_evidence_mode"] == "full_smoothing"
            assert score.diagnostics["state_space_sparse_momentum_evidence_only"] == 0
        if mode == "trajectory-imm-exact-sparse":
            assert score.diagnostics["state_space_trajectory_imm_evidence_support"] == "exact_full_grid"
            assert (
                score.diagnostics["state_space_trajectory_imm_state_support"]
                == "exact_first_order_plus_finite_radius_pair_grid"
            )
            assert score.diagnostics["state_space_trajectory_imm_modes"] == (
                "stationary,diffusion,fragmented,momentum-exact-sparse"
            )
            assert score.diagnostics["state_space_trajectory_imm_mode_stickiness"] == 0.95
            assert score.diagnostics["state_space_trajectory_imm_momentum_initial_probability"] == 0.25
            terminal_probs = [
                score.diagnostics[f"state_space_mode_{name}_terminal_probability"]
                for name in ("stationary", "diffusion", "fragmented", "momentum_exact_sparse")
            ]
            event_probs = [
                score.diagnostics[f"state_space_mode_{name}_event_probability"]
                for name in ("stationary", "diffusion", "fragmented", "momentum_exact_sparse")
            ]
            assert np.allclose(sum(terminal_probs), 1.0)
            assert np.allclose(sum(event_probs), 1.0)
            assert 0.0 <= score.diagnostics["state_space_trajectory_family_event_probability"] <= 1.0
        if mode == "displacement-momentum":
            assert score.diagnostics["state_space_displacement_momentum_evidence_support"] == "exact_full_grid"
            assert score.diagnostics["state_space_displacement_momentum_state_support"] == "finite_displacement_grid"
            assert score.diagnostics["state_space_displacement_state_count"] == 25
            assert score.diagnostics["state_space_displacement_joint_state_count"] == 100
            assert "state_space_displacement_transition_sigma_cm" in score.diagnostics
        if mode == "displacement-imm":
            assert score.diagnostics["state_space_displacement_imm_evidence_support"] == "exact_full_grid"
            assert score.diagnostics["state_space_displacement_imm_state_support"] == "finite_displacement_grid"
            assert score.diagnostics["state_space_displacement_imm_modes"] == "stationary,diffusion,displacement-momentum,jump"
            assert score.diagnostics["state_space_displacement_imm_mode_count"] == 4
            assert score.diagnostics["state_space_displacement_imm_state_count"] == 400
            terminal_probs = [
                score.diagnostics[f"state_space_mode_{name}_terminal_probability"]
                for name in ("stationary", "diffusion", "displacement_momentum", "jump")
            ]
            assert np.allclose(sum(terminal_probs), 1.0)
        if mode == "imm":
            assert score.diagnostics["state_space_imm_modes"] == "stationary,diffusion,momentum,jump"
            assert score.diagnostics["state_space_imm_evidence_support"] == "exact_full_grid"
            assert score.diagnostics["state_space_imm_candidate_support"] == "full_grid"
            assert score.diagnostics["state_space_imm_predicted_candidate_top_k"] == 8
            assert "state_space_mode_momentum_terminal_probability" in score.diagnostics
            assert "state_space_mode_jump_terminal_probability" in score.diagnostics
        if mode == "first-order-imm":
            assert score.diagnostics["state_space_imm_modes"] == "stationary,diffusion,fragmented"
            assert score.diagnostics["state_space_imm_evidence_support"] == "exact_full_grid"
            terminal_probs = [
                score.diagnostics[f"state_space_mode_{name}_terminal_probability"]
                for name in ("stationary", "diffusion", "fragmented")
            ]
            event_probs = [
                score.diagnostics[f"state_space_mode_{name}_event_probability"]
                for name in ("stationary", "diffusion", "fragmented")
            ]
            assert np.allclose(sum(terminal_probs), 1.0)
            assert np.allclose(sum(event_probs), 1.0)
            assert score.diagnostics["state_space_imm_nonstationary_terminal_probability"] == pytest.approx(
                terminal_probs[1] + terminal_probs[2]
            )
            assert score.diagnostics["state_space_imm_nonstationary_event_probability"] == pytest.approx(
                event_probs[1] + event_probs[2]
            )
            assert score.diagnostics["state_space_imm_mean_mode_entropy"] >= 0.0


def test_exact_sparse_momentum_evidence_only_delegates_to_pyrecest_mode():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(mode="momentum-exact-sparse")
    model = SortedSpikeStateSpaceReplayModel(mode="momentum-exact-sparse", config=config)

    full = model.score(emissions, centers, return_trajectory=True)
    evidence_only = model.score(emissions, centers, return_trajectory=False)

    assert np.isfinite(evidence_only.log_likelihood)
    assert evidence_only.log_likelihood == pytest.approx(full.log_likelihood, abs=1e-12)
    assert full.trajectory_log_posterior is not None
    assert evidence_only.trajectory_log_posterior is None
    assert evidence_only.terminal_log_posterior is not None
    assert evidence_only.diagnostics["state_space_sparse_momentum_evidence_mode"] == "evidence_only"
    assert evidence_only.diagnostics["state_space_sparse_momentum_evidence_only"] == 1
    assert evidence_only.diagnostics["state_space_sparse_momentum_backward_transition_rows"] == "skipped_evidence_only"
    assert evidence_only.diagnostics["state_space_momentum_trajectory_posterior"] == "not_returned_evidence_only"


def test_trajectory_imm_helper_returns_log_terminal_for_full_smoothing():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(mode="trajectory-imm-exact-sparse")

    logp, trajectory, terminal, mode_post, diagnostics = _score_trajectory_imm_exact_sparse(
        emissions,
        centers,
        config,
        emissions.transition_durations,
        return_trajectory=True,
    )

    assert np.isfinite(logp)
    assert trajectory is not None
    assert mode_post is not None
    assert diagnostics["state_space_trajectory_imm_mode_posterior"] == "smoothed_heterogeneous_state"
    assert np.allclose(terminal, trajectory[-1])
    assert np.allclose(logsumexp(terminal), 0.0)


def test_trajectory_imm_uses_duration_specific_momentum_entry_sigma(monkeypatch):
    centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, centers.shape[0]), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.01, 0.05]),
        dt=0.01,
        cell_ids=np.array([1]),
        n_spikes=0,
        transition_durations=np.array([0.01, 0.04]),
    )
    config = StateSpaceDecoderConfig(
        mode="trajectory-imm-exact-sparse",
        trajectory_imm_mode_stickiness=0.5,
        trajectory_imm_momentum_initial_probability=0.0,
        trajectory_imm_momentum_switch_probability=0.2,
        momentum_initial_sigma_cm_sqrt_s=10.0,
        max_step_sigma=10.0,
    )
    forward_entry_sigmas: list[float] = []
    backward_entry_sigmas: list[float] = []
    original_forward = trajectory_imm._advance_position_to_momentum
    original_backward = trajectory_imm._backward_position_to_momentum

    def record_forward_entry_sigma(
        source_position,
        centers_arg,
        valid_indices,
        tree,
        emission,
        *,
        sigma_cm,
        max_step_sigma,
    ):
        forward_entry_sigmas.append(float(sigma_cm))
        return original_forward(
            source_position,
            centers_arg,
            valid_indices,
            tree,
            emission,
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )

    def record_backward_entry_sigma(
        dest_prev,
        dest_curr,
        dest_values,
        centers_arg,
        valid_indices,
        tree,
        *,
        sigma_cm,
        max_step_sigma,
    ):
        backward_entry_sigmas.append(float(sigma_cm))
        return original_backward(
            dest_prev,
            dest_curr,
            dest_values,
            centers_arg,
            valid_indices,
            tree,
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )

    monkeypatch.setattr(
        trajectory_imm,
        "_advance_position_to_momentum",
        record_forward_entry_sigma,
    )
    monkeypatch.setattr(
        trajectory_imm,
        "_backward_position_to_momentum",
        record_backward_entry_sigma,
    )

    _, _, _, _, diagnostics = trajectory_imm._score_trajectory_imm_exact_sparse(
        emissions,
        centers,
        config,
        emissions.transition_durations,
        return_trajectory=True,
    )

    assert any(np.isclose(value, 1.0) for value in forward_entry_sigmas)
    assert any(np.isclose(value, 2.0) for value in forward_entry_sigmas)
    assert any(np.isclose(value, 1.0) for value in backward_entry_sigmas)
    assert any(np.isclose(value, 2.0) for value in backward_entry_sigmas)
    assert diagnostics["state_space_momentum_initial_transition_sigma_cm"] == pytest.approx(1.5)
    assert diagnostics["state_space_momentum_initial_transition_sigma_cm_per_step"] == "1,2"


def test_adaptive_candidate_support_adds_forward_and_backward_predictions():
    centers = np.arange(7.0)[:, None]
    base = [np.array([0]), np.array([4]), np.array([5]), np.array([6])]

    candidates = _augment_candidates_with_momentum_predictions(
        base,
        centers,
        predicted_top_k=1,
        velocity_decay=1.0,
    )

    assert set(candidates[0]) == {0, 3}
    assert set(candidates[1]) == {4}
    assert set(candidates[2]) == {5, 6}
    assert set(candidates[3]) == {6}


def test_mass_retaining_candidate_support_tracks_emission_mass_with_bounds():
    log_emission = np.log(np.array([0.60, 0.25, 0.10, 0.05]))

    adaptive = _mass_retaining_candidate_indices(
        log_emission,
        top_k=1,
        mass_threshold=0.80,
        min_k=0,
        max_k=0,
    )
    assert list(adaptive) == [0, 1]

    min_limited = _mass_retaining_candidate_indices(
        log_emission,
        top_k=1,
        mass_threshold=0.50,
        min_k=3,
        max_k=0,
    )
    assert list(min_limited) == [0, 1, 2]

    capped = _mass_retaining_candidate_indices(
        log_emission,
        top_k=1,
        mass_threshold=0.99,
        min_k=0,
        max_k=2,
    )
    assert list(capped) == [0, 1]


def test_state_space_model_uses_mass_retaining_candidate_support():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=1,
        momentum_candidate_mass_threshold=0.85,
        momentum_candidate_max_k=3,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)

    candidates = model.candidate_indices(emissions)
    score = model.score(emissions, centers)

    assert [len(row) for row in candidates] == [2, 3, 3]
    masses = [
        float(np.exp(logsumexp(emissions.log_likelihood[time_index, row]) - logsumexp(emissions.log_likelihood[time_index])))
        for time_index, row in enumerate(candidates)
    ]
    assert all(mass >= 0.85 for mass in masses)
    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["state_space_momentum_candidate_selection"] == "adaptive_mass"
    assert score.diagnostics["state_space_momentum_candidate_mass_threshold"] == 0.85
    assert score.diagnostics["state_space_momentum_candidate_max_k"] == 3
    assert score.diagnostics["min_candidate_log_mass"] >= np.log(0.85) - 1e-12


def test_state_space_model_uses_adaptive_candidate_support_when_bin_centers_given():
    centers = np.arange(7.0)[:, None]
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0, 0.0, -5.0, -5.0],
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

    assert [list(row) for row in emission_only] == [[0], [4], [5], [6]]
    assert 3 in adaptive[0]
    assert 6 in adaptive[2]
    assert score.diagnostics["state_space_momentum_predicted_candidate_top_k"] == 1
    assert score.diagnostics["mean_candidate_count"] > 1.0


def test_mass_retaining_candidate_support_respects_threshold_and_bounds():
    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.04, 0.01]))

    selected = _mass_retaining_candidate_indices(log_emission, 0.95)
    capped = _mass_retaining_candidate_indices(log_emission, 0.999, max_k=3)
    forced_minimum = _mass_retaining_candidate_indices(log_emission, 0.50, min_k=4)

    assert list(selected) == [0, 1, 2]
    assert list(capped) == [0, 1, 2]
    assert list(forced_minimum) == [0, 1, 2, 3]


def test_state_space_model_can_use_mass_retaining_candidate_support():
    centers = np.arange(4.0)[:, None]
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.09, 0.01],
                    [0.40, 0.35, 0.15, 0.10],
                    [0.97, 0.01, 0.01, 0.01],
                ]
            )
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
        momentum_candidate_mass_threshold=0.90,
        momentum_candidate_min_k=1,
        momentum_candidate_max_k=3,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)

    candidates = model.candidate_indices(emissions, centers)
    score = model.score(emissions, centers)

    assert [list(row) for row in candidates] == [[0, 1], [0, 1, 2], [0]]
    assert score.diagnostics["state_space_momentum_candidate_top_k"] == 1
    assert score.diagnostics["state_space_momentum_candidate_mass_threshold"] == 0.90
    assert score.diagnostics["state_space_momentum_candidate_max_k"] == 3
    assert score.diagnostics["mean_candidate_count"] == 2.0
    assert score.diagnostics["min_candidate_log_mass"] <= score.diagnostics["mean_candidate_log_mass"]


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


def test_state_space_momentum_full_candidate_support_is_marked_exact():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=0,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)

    candidates = model.candidate_indices(emissions, centers)
    score = model.score(emissions, centers)

    assert all(
        np.array_equal(row, np.arange(emissions.n_bins, dtype=int))
        for row in candidates
    )
    assert _candidate_evidence_support_label(candidates, emissions.n_bins) == "exact_full_grid"
    assert score.diagnostics["state_space_momentum_candidate_selection"] == "full_grid"
    assert score.diagnostics["state_space_momentum_candidate_support"] == "full_grid"
    assert score.diagnostics["state_space_momentum_evidence_support"] == "exact_full_grid"


def test_displacement_imm_returns_exact_finite_state_mode_posterior():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="displacement-imm",
        displacement_radius_bins=1,
        displacement_position_sigma_cm=1.0,
        displacement_transition_sigma_cm_sqrt_s=1.0,
        displacement_prior_sigma_cm=1.0,
        imm_mode_stickiness=0.8,
    )

    score = SortedSpikeStateSpaceReplayModel(mode="displacement-imm", config=config).score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
    assert score.diagnostics["state_space_displacement_imm_evidence_support"] == "exact_full_grid"
    assert score.diagnostics["state_space_displacement_state_count"] == 9
    assert score.diagnostics["state_space_displacement_imm_state_count"] == 144
    terminal_probs = [
        score.diagnostics[f"state_space_mode_{name}_terminal_probability"]
        for name in ("stationary", "diffusion", "displacement_momentum", "jump")
    ]
    assert np.allclose(sum(terminal_probs), 1.0)


def test_displacement_momentum_uses_declared_finite_lattice():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="displacement-momentum",
        displacement_radius_bins=1,
        displacement_position_sigma_cm=1.0,
        displacement_transition_sigma_cm_sqrt_s=1.0,
        displacement_prior_sigma_cm=1.0,
    )

    score = SortedSpikeStateSpaceReplayModel(mode="displacement-momentum", config=config).score(emissions, centers)
    lattice = _displacement_lattice(centers, radius_bins=1)

    assert lattice.shape == (9, 2)
    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == (emissions.n_time, emissions.n_bins)
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
    assert score.diagnostics["state_space_displacement_state_count"] == 9
    assert score.diagnostics["state_space_displacement_joint_state_count"] == 36
    assert score.diagnostics["state_space_displacement_momentum_evidence_support"] == "exact_full_grid"


def test_finite_displacement_modes_honor_return_trajectory_false():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    for mode in ("displacement-momentum", "displacement-imm"):
        config = StateSpaceDecoderConfig(
            mode=mode,
            displacement_radius_bins=1,
            displacement_position_sigma_cm=1.0,
            displacement_transition_sigma_cm_sqrt_s=1.0,
            displacement_prior_sigma_cm=1.0,
            imm_mode_stickiness=0.8,
        )
        model = SortedSpikeStateSpaceReplayModel(mode=mode, config=config)

        full = model.score(emissions, centers, return_trajectory=True)
        evidence_only = model.score(emissions, centers, return_trajectory=False)

        assert evidence_only.log_likelihood == pytest.approx(
            full.log_likelihood,
            abs=1e-12,
        )
        assert full.trajectory_log_posterior is not None
        assert evidence_only.trajectory_log_posterior is None
        assert evidence_only.terminal_log_posterior is not None
        np.testing.assert_allclose(
            evidence_only.terminal_log_posterior,
            full.trajectory_log_posterior[-1],
        )
        assert np.allclose(logsumexp(evidence_only.terminal_log_posterior), 0.0)
        assert evidence_only.diagnostics["state_space_trajectory_posterior"] == 0


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


def test_state_space_model_rejects_duplicate_or_non_integer_candidate_support():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=2,
        momentum_predicted_candidate_top_k=0,
    )
    model = StateSpaceReplayModel(mode="momentum", config=config)

    duplicate_candidates = [np.array([0, 1]), np.array([1, 1]), np.array([2, 3])]
    with pytest.raises(ValueError, match="duplicate"):
        model.score(emissions, centers, candidate_indices=duplicate_candidates)

    float_candidates = [np.array([0, 1]), np.array([1.0, 2.0]), np.array([2, 3])]
    with pytest.raises(TypeError, match="integer"):
        model.score(emissions, centers, candidate_indices=float_candidates)


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
    assert score.diagnostics["state_space_imm_evidence_support"] == "exact_full_grid"
    terminal_probs = [score.diagnostics[f"state_space_mode_{mode}_terminal_probability"] for mode in modes]
    assert np.allclose(sum(terminal_probs), 1.0)
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)


def test_state_space_momentum_exact_sparse_matches_bruteforce_tiny_grid():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.45, 0.55],
                ]
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="momentum-exact-sparse",
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.9,
        max_step_sigma=10.0,
    )
    score = StateSpaceReplayModel(mode="momentum-exact-sparse", config=config).score(emissions, centers)

    def kernel_log(predicted: np.ndarray, dst: int, sigma: float) -> float:
        weights = np.exp(-0.5 * np.sum((centers - predicted[None, :]) ** 2, axis=1) / (sigma * sigma))
        return float(np.log(weights[dst] / weights.sum()))

    brute_terms = []
    for x0, x1, x2, x3 in itertools.product(range(2), repeat=4):
        predicted = centers[x1] + config.momentum_velocity_decay * (centers[x1] - centers[x0])
        predicted_next = centers[x2] + config.momentum_velocity_decay * (centers[x2] - centers[x1])
        brute_terms.append(
            -np.log(2.0)
            + emissions.log_likelihood[0, x0]
            + kernel_log(centers[x0], x1, 1.0)
            + emissions.log_likelihood[1, x1]
            + kernel_log(predicted, x2, 1.0)
            + emissions.log_likelihood[2, x2]
            + kernel_log(predicted_next, x3, 1.0)
            + emissions.log_likelihood[3, x3]
        )

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
    assert score.diagnostics["state_space_sparse_momentum_evidence_support"] == "exact_full_grid"
    assert score.diagnostics["state_space_momentum_evidence_support"] == "exact_full_grid"
    assert score.diagnostics["state_space_momentum_candidate_selection"] == "none_exact_sparse"
    assert score.diagnostics["state_space_sparse_momentum_max_pair_count"] == 4
    assert score.diagnostics["state_space_sparse_momentum_transition_support"] == "finite_radius_gaussian"
    assert score.diagnostics["state_space_sparse_momentum_backward_transition_rows"] == "forward_cached"
    assert score.diagnostics["state_space_sparse_momentum_transition_row_cache_hits"] == 4
    assert score.diagnostics["state_space_sparse_momentum_transition_row_cache_entries"] == 4


def test_state_space_trajectory_imm_exact_sparse_matches_bruteforce_tiny_grid():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                    [0.4, 0.6],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="trajectory-imm-exact-sparse",
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        imm_mode_stickiness=0.8,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.9,
        max_step_sigma=10.0,
    )
    score = StateSpaceReplayModel(mode="trajectory-imm-exact-sparse", config=config).score(emissions, centers)

    modes = ("stationary", "diffusion", "fragmented", "momentum-exact-sparse")
    transition = np.full((4, 4), (1.0 - config.imm_mode_stickiness) / 3.0)
    np.fill_diagonal(transition, config.imm_mode_stickiness)

    def kernel_log(
        dst_mode: str,
        src_mode: str,
        x_prev_prev: int,
        x_prev: int,
        x_dst: int,
        *,
        initial: bool = False,
    ) -> float:
        if dst_mode == "fragmented":
            return -np.log(centers.shape[0])
        if dst_mode == "stationary":
            sigma = config.stationary_sigma_cm
            predicted = centers[x_prev]
        elif dst_mode == "diffusion":
            sigma = config.diffusion_sigma_cm_sqrt_s
            predicted = centers[x_prev]
        elif dst_mode == "momentum-exact-sparse":
            if initial or src_mode != "momentum-exact-sparse":
                sigma = config.momentum_initial_sigma_cm_sqrt_s
                predicted = centers[x_prev]
            else:
                sigma = config.momentum_sigma_cm_sqrt_s
                predicted = centers[x_prev] + config.momentum_velocity_decay * (
                    centers[x_prev] - centers[x_prev_prev]
                )
        else:
            raise AssertionError(dst_mode)
        weights = np.exp(-0.5 * np.sum((centers - predicted[None, :]) ** 2, axis=1) / (sigma * sigma))
        return float(np.log(weights[x_dst] / weights.sum()))

    brute_terms = []
    for x0, x1, x2 in itertools.product(range(2), repeat=3):
        for m0_idx, mode0 in enumerate(modes):
            for m1_idx, mode1 in enumerate(modes):
                for m2_idx, mode2 in enumerate(modes):
                    brute_terms.append(
                        -np.log(2.0)
                        - np.log(len(modes))
                        + emissions.log_likelihood[0, x0]
                        + np.log(transition[m0_idx, m1_idx])
                        + kernel_log(mode1, mode0, x0, x0, x1, initial=True)
                        + emissions.log_likelihood[1, x1]
                        + np.log(transition[m1_idx, m2_idx])
                        + kernel_log(mode2, mode1, x0, x1, x2)
                        + emissions.log_likelihood[2, x2]
                    )

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
    assert score.diagnostics["state_space_trajectory_imm_evidence_support"] == "exact_full_grid"
    assert score.diagnostics["state_space_trajectory_imm_modes"] == (
        "stationary,diffusion,fragmented,momentum-exact-sparse"
    )
    assert score.diagnostics["state_space_trajectory_imm_mode_stickiness"] == 0.8
    assert score.diagnostics["state_space_trajectory_imm_momentum_initial_probability"] == 0.25
    terminal_probs = [
        score.diagnostics[f"state_space_mode_{name}_terminal_probability"]
        for name in ("stationary", "diffusion", "fragmented", "momentum_exact_sparse")
    ]
    assert np.allclose(sum(terminal_probs), 1.0)


def test_state_space_trajectory_imm_can_use_specific_persistence_prior():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="trajectory-imm-exact-sparse",
        imm_mode_stickiness=0.8,
        trajectory_imm_mode_stickiness=0.985,
    )

    score = StateSpaceReplayModel(mode="trajectory-imm-exact-sparse", config=config).score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["state_space_trajectory_imm_mode_stickiness"] == 0.985
    assert score.diagnostics["state_space_imm_mode_stickiness"] == 0.8


def test_state_space_trajectory_imm_can_use_anchored_momentum_prior():
    emissions = _synthetic_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="trajectory-imm-exact-sparse",
        imm_mode_stickiness=0.95,
        trajectory_imm_momentum_initial_probability=0.05,
        trajectory_imm_momentum_switch_probability=0.005,
    )

    score = StateSpaceReplayModel(mode="trajectory-imm-exact-sparse", config=config).score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["state_space_trajectory_imm_momentum_initial_probability"] == 0.05
    assert score.diagnostics["state_space_trajectory_imm_momentum_switch_probability"] == 0.005


def test_sparse_momentum_evidence_only_skips_smoothed_trajectory():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.45, 0.55],
                ]
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="momentum-exact-sparse",
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.9,
        max_step_sigma=10.0,
    )
    model = StateSpaceReplayModel(mode="momentum-exact-sparse", config=config)

    smoothed = model.score(emissions, centers)
    evidence_only = model.score(emissions, centers, return_trajectory=False)

    assert np.allclose(evidence_only.log_likelihood, smoothed.log_likelihood)
    assert evidence_only.trajectory_log_posterior is None
    assert evidence_only.terminal_log_posterior is not None
    assert smoothed.terminal_log_posterior is not None
    assert np.allclose(evidence_only.terminal_log_posterior, smoothed.terminal_log_posterior)
    assert evidence_only.diagnostics["state_space_trajectory_posterior"] == 0
    assert evidence_only.diagnostics["state_space_momentum_trajectory_posterior"] == "not_returned_evidence_only"
    assert (
        evidence_only.diagnostics["state_space_sparse_momentum_backward_transition_rows"]
        == "skipped_evidence_only"
    )
    assert evidence_only.diagnostics["decoded_map_bin"] == smoothed.diagnostics["decoded_map_bin"]


def test_exact_sparse_momentum_matches_dense_finite_radius_reference_and_heldout_identity():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    train = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.72, 0.20, 0.08],
                    [0.12, 0.72, 0.16],
                    [0.08, 0.18, 0.74],
                    [0.10, 0.24, 0.66],
                ]
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    test_log_likelihood = np.log(
        np.array(
            [
                [0.55, 0.35, 0.10],
                [0.18, 0.64, 0.18],
                [0.12, 0.32, 0.56],
                [0.14, 0.30, 0.56],
            ]
        )
    )
    joint = LogEmissionTensor(
        log_likelihood=train.log_likelihood + test_log_likelihood,
        spike_counts=np.zeros((4, 1), dtype=int),
        times=train.times.copy(),
        dt=1.0,
        cell_ids=train.cell_ids.copy(),
        n_spikes=0,
    )
    config = StateSpaceDecoderConfig(
        mode="momentum-exact-sparse",
        momentum_sigma_cm_sqrt_s=0.9,
        momentum_initial_sigma_cm_sqrt_s=0.9,
        momentum_velocity_decay=0.65,
        max_step_sigma=1.25,
    )
    exact_model = StateSpaceReplayModel(mode="momentum-exact-sparse", config=config)

    train_score = exact_model.score(train, centers)
    joint_score = exact_model.score(joint, centers)
    dense_train_logz = _dense_finite_radius_momentum_log_evidence(train, centers, config)
    dense_joint_logz = _dense_finite_radius_momentum_log_evidence(joint, centers, config)

    assert abs(train_score.log_likelihood - dense_train_logz) < 1e-8
    assert abs(joint_score.log_likelihood - dense_joint_logz) < 1e-8
    assert abs(
        (joint_score.log_likelihood - train_score.log_likelihood)
        - (dense_joint_logz - dense_train_logz)
    ) < 1e-8
    assert train_score.diagnostics["state_space_momentum_candidate_selection"] == "none_exact_sparse"
    assert joint_score.diagnostics["state_space_momentum_candidate_selection"] == "none_exact_sparse"

    pruned_config = StateSpaceDecoderConfig(
        mode="momentum",
        momentum_sigma_cm_sqrt_s=config.momentum_sigma_cm_sqrt_s,
        momentum_initial_sigma_cm_sqrt_s=config.momentum_initial_sigma_cm_sqrt_s,
        momentum_velocity_decay=config.momentum_velocity_decay,
        max_step_sigma=config.max_step_sigma,
        momentum_candidate_top_k=1,
        momentum_predicted_candidate_top_k=0,
    )
    pruned_score = StateSpaceReplayModel(mode="momentum", config=pruned_config).score(train, centers)

    assert pruned_score.log_likelihood <= train_score.log_likelihood + 1e-8
    assert pruned_score.diagnostics["state_space_momentum_evidence_support"] == "truncated_full_grid"


def _dense_finite_radius_momentum_log_evidence(
    emissions: LogEmissionTensor,
    centers: np.ndarray,
    config: StateSpaceDecoderConfig,
) -> float:
    terms = []
    prior = -np.log(emissions.n_bins)
    sigma = float(config.momentum_sigma_cm_sqrt_s)
    initial_sigma = float(config.momentum_initial_sigma_cm_sqrt_s)
    radius = float(config.max_step_sigma)
    decay = float(config.momentum_velocity_decay)
    for path in itertools.product(range(emissions.n_bins), repeat=emissions.n_time):
        initial = _finite_radius_log_transition(
            centers,
            predicted=centers[path[0]],
            dst=path[1],
            sigma=initial_sigma,
            radius=radius,
        )
        if not np.isfinite(initial):
            continue
        logp = prior + emissions.log_likelihood[0, path[0]]
        logp += initial + emissions.log_likelihood[1, path[1]]
        ok = True
        for time_index in range(2, emissions.n_time):
            prev_prev = path[time_index - 2]
            prev = path[time_index - 1]
            dst = path[time_index]
            predicted = centers[prev] + decay * (centers[prev] - centers[prev_prev])
            transition = _finite_radius_log_transition(
                centers,
                predicted=predicted,
                dst=dst,
                sigma=sigma,
                radius=radius,
            )
            if not np.isfinite(transition):
                ok = False
                break
            logp += transition + emissions.log_likelihood[time_index, dst]
        if ok:
            terms.append(logp)
    return float(logsumexp(terms))


def _finite_radius_log_transition(
    centers: np.ndarray,
    *,
    predicted: np.ndarray,
    dst: int,
    sigma: float,
    radius: float,
) -> float:
    distances = np.linalg.norm(centers - predicted[None, :], axis=1)
    support = np.flatnonzero(distances <= radius * sigma)
    if support.size == 0:
        support = np.asarray([int(np.argmin(distances))], dtype=int)
    if dst not in set(int(index) for index in support):
        return float("-inf")
    weights = np.exp(-0.5 * np.sum((centers[support] - predicted[None, :]) ** 2, axis=1) / (sigma * sigma))
    local = int(np.flatnonzero(support == dst)[0])
    return float(np.log(weights[local] / weights.sum()))
