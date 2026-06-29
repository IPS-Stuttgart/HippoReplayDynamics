from __future__ import annotations

import numpy as np

from hipporeplayimm.bidirectional_infinite_evidence_patch import (
    _equal_prior_logp_and_weights,
    _safe_mixture_log_posterior,
)
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import BidirectionalReplayModel
from hipporeplayimm.reverse_models import (
    BidirectionalReplayModel as DirectBidirectionalReplayModel,
)


def _duration_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -1.0],
                [-2.0, -0.5],
                [-0.25, -3.0],
            ],
            dtype=float,
        ),
        spike_counts=np.array([[0], [2], [1]], dtype=int),
        times=np.array([0.005, 0.020, 0.055], dtype=float),
        dt=0.02,
        cell_ids=np.array([7], dtype=int),
        n_spikes=3,
        bin_durations=np.array([0.010, 0.020, 0.030], dtype=float),
        transition_durations=np.array([0.015, 0.035], dtype=float),
    )


class _ImpossibleTrajectoryModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray, **_kwargs: object) -> EventScore:
        del bin_centers
        trajectory = np.full(
            (emissions.n_time, emissions.n_bins),
            1.0 / emissions.n_bins,
            dtype=float,
        )
        log_trajectory = np.log(trajectory)
        return EventScore(
            self.name,
            float("-inf"),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={},
            terminal_log_posterior=log_trajectory[-1].copy(),
            trajectory_log_posterior=log_trajectory,
        )


def test_bidirectional_wrapper_keeps_finite_weights_when_both_directions_impossible() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    bidirectional = BidirectionalReplayModel(
        forward_model=_ImpossibleTrajectoryModel("forward"),
        reverse_model=_ImpossibleTrajectoryModel("reverse"),
        name="bidirectional",
    )

    result = bidirectional.score(emissions, bin_centers)

    assert np.isneginf(result.log_likelihood)
    np.testing.assert_allclose(result.diagnostics["direction_forward_probability"], 0.5)
    np.testing.assert_allclose(result.diagnostics["direction_reverse_probability"], 0.5)
    assert result.terminal_log_posterior is not None
    assert np.all(np.isfinite(result.terminal_log_posterior))
    np.testing.assert_allclose(np.exp(result.terminal_log_posterior).sum(), 1.0)


def test_direct_bidirectional_wrapper_keeps_finite_weights_when_both_directions_impossible() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    bidirectional = DirectBidirectionalReplayModel(
        _ImpossibleTrajectoryModel("base"),
        name="direct-bidirectional",
    )

    result = bidirectional.score(emissions, bin_centers)

    assert np.isneginf(result.log_likelihood)
    np.testing.assert_allclose(result.diagnostics["forward_model_posterior_probability"], 0.5)
    np.testing.assert_allclose(result.diagnostics["reverse_model_posterior_probability"], 0.5)
    assert result.terminal_log_posterior is not None
    assert np.all(np.isfinite(result.terminal_log_posterior))
    np.testing.assert_allclose(np.exp(result.terminal_log_posterior).sum(), 1.0)


def test_zero_weight_direction_does_not_leak_terminal_support() -> None:
    logp, weights = _equal_prior_logp_and_weights([float("inf"), 0.0])
    forward = np.array([0.0, -np.inf], dtype=float)
    reverse = np.array([-np.inf, 0.0], dtype=float)

    mixed = _safe_mixture_log_posterior([forward, reverse], weights)

    assert np.isposinf(logp)
    np.testing.assert_allclose(weights, np.array([1.0, 0.0]))
    assert mixed is not None
    np.testing.assert_allclose(mixed[0], 0.0)
    assert np.isneginf(mixed[1])
