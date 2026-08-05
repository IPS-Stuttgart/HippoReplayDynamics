from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel

_MODES = ("stationary", "diffusion", "momentum", "jump")


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.01, 0.09, 0.90],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def _transition_log_probability(
    model: CandidateKinematicModel,
    mode: str,
    centers: np.ndarray,
    previous_previous: int,
    previous: int,
    current: int,
    *,
    initial: bool,
) -> float:
    if mode == "jump":
        return -np.log(centers.shape[0])
    if initial or mode != "momentum":
        predicted = centers[previous]
    else:
        predicted = centers[previous] + model.velocity_decay * (
            centers[previous] - centers[previous_previous]
        )
    sigma = {
        "stationary": model.stationary_sigma_cm,
        "diffusion": model.diffusion_sigma_cm,
        "momentum": model.momentum_sigma_cm,
    }[mode]
    squared_distance = np.sum((centers - predicted[None, :]) ** 2, axis=1)
    log_kernel = -0.5 * squared_distance / (sigma * sigma)
    return float(log_kernel[current] - logsumexp(log_kernel))


def _exact_static_trajectory(
    model: CandidateKinematicModel,
    emissions: LogEmissionTensor,
    centers: np.ndarray,
) -> tuple[float, np.ndarray]:
    paths = list(itertools.product(range(emissions.n_bins), repeat=emissions.n_time))
    log_weights = []
    for first, second, third in paths:
        log_weights.append(
            emissions.log_likelihood[0, first]
            - np.log(emissions.n_bins)
            + _transition_log_probability(
                model,
                model.mode,
                centers,
                first,
                first,
                second,
                initial=True,
            )
            + emissions.log_likelihood[1, second]
            + _transition_log_probability(
                model,
                model.mode,
                centers,
                first,
                second,
                third,
                initial=False,
            )
            + emissions.log_likelihood[2, third]
        )
    return _trajectory_from_enumeration(paths, np.asarray(log_weights), emissions.n_bins)


def _exact_imm_trajectory(
    model: CandidateKinematicModel,
    emissions: LogEmissionTensor,
    centers: np.ndarray,
) -> tuple[float, np.ndarray]:
    n_modes = len(_MODES)
    off_diagonal = (1.0 - model.mode_stickiness) / (n_modes - 1)
    mode_transition = np.full((n_modes, n_modes), off_diagonal, dtype=float)
    np.fill_diagonal(mode_transition, model.mode_stickiness)

    paths: list[tuple[int, int, int]] = []
    log_weights: list[float] = []
    for first_mode, second_mode in itertools.product(range(n_modes), repeat=2):
        for first, second, third in itertools.product(
            range(emissions.n_bins),
            repeat=emissions.n_time,
        ):
            paths.append((first, second, third))
            log_weights.append(
                emissions.log_likelihood[0, first]
                - np.log(emissions.n_bins)
                - np.log(n_modes)
                + _transition_log_probability(
                    model,
                    _MODES[first_mode],
                    centers,
                    first,
                    first,
                    second,
                    initial=True,
                )
                + emissions.log_likelihood[1, second]
                + np.log(mode_transition[first_mode, second_mode])
                + _transition_log_probability(
                    model,
                    _MODES[second_mode],
                    centers,
                    first,
                    second,
                    third,
                    initial=False,
                )
                + emissions.log_likelihood[2, third]
            )
    return _trajectory_from_enumeration(paths, np.asarray(log_weights), emissions.n_bins)


def _trajectory_from_enumeration(
    paths: list[tuple[int, int, int]],
    log_weights: np.ndarray,
    n_bins: int,
) -> tuple[float, np.ndarray]:
    log_evidence = float(logsumexp(log_weights))
    trajectory = []
    for time_index in range(3):
        marginal = np.full(n_bins, -np.inf, dtype=float)
        for path, log_weight in zip(paths, log_weights, strict=True):
            marginal[path[time_index]] = np.logaddexp(
                marginal[path[time_index]],
                log_weight,
            )
        trajectory.append(marginal - log_evidence)
    return log_evidence, np.stack(trajectory, axis=0)


@pytest.mark.parametrize("mode", ["stationary", "diffusion", "momentum", "jump"])
def test_candidate_static_trajectory_matches_full_event_posterior(mode: str) -> None:
    emissions = _emissions()
    centers = np.array([[0.0], [1.0], [2.0]])
    model = CandidateKinematicModel(
        mode=mode,
        top_k=0,
        stationary_sigma_cm=0.4,
        diffusion_sigma_cm=0.8,
        momentum_sigma_cm=0.6,
        velocity_decay=0.9,
    )

    score = model.score(emissions, centers)
    expected_evidence, expected_trajectory = _exact_static_trajectory(
        model,
        emissions,
        centers,
    )

    assert score.trajectory_log_posterior is not None
    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(score.log_likelihood, expected_evidence)
    np.testing.assert_allclose(
        score.trajectory_log_posterior,
        expected_trajectory,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        score.terminal_log_posterior,
        expected_trajectory[-1],
        atol=1e-12,
    )


def test_candidate_imm_trajectory_matches_full_event_posterior() -> None:
    emissions = _emissions()
    centers = np.array([[0.0], [1.0], [2.0]])
    model = CandidateKinematicModel(
        mode="imm",
        top_k=0,
        stationary_sigma_cm=0.4,
        diffusion_sigma_cm=0.8,
        momentum_sigma_cm=0.6,
        velocity_decay=0.9,
        mode_stickiness=0.7,
    )

    score = model.score(emissions, centers)
    expected_evidence, expected_trajectory = _exact_imm_trajectory(
        model,
        emissions,
        centers,
    )

    assert score.trajectory_log_posterior is not None
    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(score.log_likelihood, expected_evidence)
    np.testing.assert_allclose(
        score.trajectory_log_posterior,
        expected_trajectory,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        score.terminal_log_posterior,
        expected_trajectory[-1],
        atol=1e-12,
    )
