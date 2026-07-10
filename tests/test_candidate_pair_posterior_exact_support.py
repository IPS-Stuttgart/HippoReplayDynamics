from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import (
    CandidateKinematicModel,
    _pair_previous_posterior,
    _pair_terminal_posterior,
)


def test_pair_posterior_helpers_use_negative_infinity_outside_candidates() -> None:
    log_pair = np.array([[0.0]])

    previous = _pair_previous_posterior(
        log_pair,
        np.array([2], dtype=int),
        4,
    )
    terminal = _pair_terminal_posterior(
        log_pair,
        np.array([1], dtype=int),
        4,
    )

    np.testing.assert_array_equal(
        previous,
        np.array([-np.inf, -np.inf, 0.0, -np.inf]),
    )
    np.testing.assert_array_equal(
        terminal,
        np.array([-np.inf, 0.0, -np.inf, -np.inf]),
    )


def test_candidate_model_trajectory_has_exact_candidate_support() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.80, 0.15, 0.05],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.15, 0.80],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    candidates = [
        np.array([0], dtype=int),
        np.array([1], dtype=int),
        np.array([2], dtype=int),
    ]

    score = CandidateKinematicModel(
        mode="diffusion",
        top_k=1,
        diffusion_sigma_cm=1.0,
    ).score(
        emissions,
        centers,
        candidate_indices=candidates,
    )

    assert score.trajectory_log_posterior is not None
    for time_index, candidate in enumerate(candidates):
        row = score.trajectory_log_posterior[time_index]
        active = np.zeros(emissions.n_bins, dtype=bool)
        active[candidate] = True
        assert np.all(np.isfinite(row[active]))
        assert np.all(np.isneginf(row[~active]))
        assert logsumexp(row) == 0.0

    assert score.terminal_log_posterior is not None
    np.testing.assert_array_equal(
        np.isfinite(score.terminal_log_posterior),
        np.array([False, False, True]),
    )
