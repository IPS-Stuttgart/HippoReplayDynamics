from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_candidates import _score_imm_candidates
from hipporeplayimm.state_space_candidates_momentum import _score_momentum_candidates


def _candidate_inputs() -> tuple[LogEmissionTensor, np.ndarray, list[np.ndarray]]:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.80, 0.15, 0.05],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.15, 0.80],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=float,
    )
    candidates = [
        np.array([0], dtype=int),
        np.array([1], dtype=int),
        np.array([2], dtype=int),
    ]
    return emissions, bin_centers, candidates


def _assert_exact_candidate_support(
    trajectory: np.ndarray,
    candidates: list[np.ndarray],
) -> None:
    for time_index, candidate in enumerate(candidates):
        row = trajectory[time_index]
        active = np.zeros(row.shape[0], dtype=bool)
        active[candidate] = True
        assert np.all(np.isfinite(row[active]))
        assert np.all(np.isneginf(row[~active]))
        assert logsumexp(row) == 0.0


def test_momentum_candidate_trajectory_uses_exact_zero_outside_support() -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    logp, trajectory, _masses = _score_momentum_candidates(
        emissions,
        bin_centers,
        candidates,
        sigma_cm=1.0,
        initial_sigma_cm=1.0,
        velocity_decay=0.9,
    )

    assert np.isfinite(logp)
    _assert_exact_candidate_support(trajectory, candidates)


def test_imm_candidate_trajectory_uses_exact_zero_outside_support() -> None:
    emissions, bin_centers, candidates = _candidate_inputs()

    logp, trajectory, mode_post, _masses = _score_imm_candidates(
        emissions,
        bin_centers,
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm=1.0,
        momentum_sigma_cm=1.0,
        momentum_initial_sigma_cm=1.0,
        velocity_decay=0.9,
        mode_stickiness=0.9,
        candidate_indices=candidates,
    )

    assert np.isfinite(logp)
    np.testing.assert_allclose(mode_post.sum(axis=1), 1.0)
    _assert_exact_candidate_support(trajectory, candidates)
