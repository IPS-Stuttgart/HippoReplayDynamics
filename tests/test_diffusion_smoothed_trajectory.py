from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import DiffusionModel


def test_diffusion_model_returns_full_event_smoothed_trajectory() -> None:
    centers = np.array([[0.0], [1.0]])
    emission_probabilities = np.array(
        [
            [0.50, 0.50],
            [0.90, 0.10],
            [0.01, 0.99],
        ]
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.log(emission_probabilities),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = DiffusionModel(sigma_cm=1.0, max_step_sigma=10.0)

    score = model.score(emissions, centers)

    transition = np.empty((2, 2), dtype=float)
    for source, center in enumerate(centers):
        weights = np.exp(-0.5 * np.sum((centers - center) ** 2, axis=1))
        transition[source] = weights / weights.sum()

    paths = list(itertools.product(range(centers.shape[0]), repeat=emissions.n_time))
    path_log_probabilities = []
    for path in paths:
        log_probability = -np.log(centers.shape[0])
        for time_index, spatial_bin in enumerate(path):
            log_probability += emissions.log_likelihood[time_index, spatial_bin]
            if time_index:
                log_probability += np.log(
                    transition[path[time_index - 1], spatial_bin]
                )
        path_log_probabilities.append(log_probability)

    log_evidence = float(logsumexp(path_log_probabilities))
    expected = np.empty((emissions.n_time, centers.shape[0]), dtype=float)
    for time_index in range(emissions.n_time):
        for spatial_bin in range(centers.shape[0]):
            expected[time_index, spatial_bin] = (
                logsumexp(
                    [
                        log_probability
                        for path, log_probability in zip(
                            paths,
                            path_log_probabilities,
                            strict=True,
                        )
                        if path[time_index] == spatial_bin
                    ]
                )
                - log_evidence
            )

    assert score.trajectory_log_posterior is not None
    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(score.log_likelihood, log_evidence)
    np.testing.assert_allclose(score.trajectory_log_posterior, expected)
    np.testing.assert_allclose(score.terminal_log_posterior, expected[-1])
    assert not np.allclose(
        score.trajectory_log_posterior[0],
        np.log(emission_probabilities[0]),
    )
