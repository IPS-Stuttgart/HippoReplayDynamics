import numpy as np
import pytest

pytest.importorskip("pyrecest")

from pyrecest.filters import (
    adaptive_position_proposal_probability,
    build_replay_grid_likelihood_lookup,
    effective_sample_size_fraction,
    replay_grid_log_likelihood_values,
)


class _DummyState:
    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float)


class _DummyFilter:
    def __init__(self, weights):
        self.filter_state = _DummyState(weights)


def test_linear_grid_likelihood_interpolates_rectilinear_log_values():
    bin_centers = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    values = 20.0 * bin_centers[:, 0] + 10.0 * bin_centers[:, 1]
    lookup = build_replay_grid_likelihood_lookup(bin_centers, "linear")
    result = replay_grid_log_likelihood_values(
        np.asarray([[0.25, 0.50]]),
        values,
        bin_centers,
        lookup=lookup,
    )
    assert lookup.method == "linear"
    assert np.allclose(result, [10.0])


def test_linear_grid_likelihood_falls_back_to_nearest_outside_grid():
    bin_centers = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    values = np.asarray([0.0, 1.0, 2.0, 3.0])
    lookup = build_replay_grid_likelihood_lookup(bin_centers, "linear")
    result = replay_grid_log_likelihood_values(
        np.asarray([[2.0, 2.0]]),
        values,
        bin_centers,
        lookup=lookup,
    )
    assert np.allclose(result, [3.0])


def test_linear_grid_likelihood_uses_nearest_for_irregular_bins():
    bin_centers = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    lookup = build_replay_grid_likelihood_lookup(bin_centers, "linear")
    assert lookup.method == "nearest"


def test_position_proposal_probability_is_ess_adaptive():
    assert np.isclose(effective_sample_size_fraction(np.ones(4)), 1.0)
    assert adaptive_position_proposal_probability(_DummyFilter(np.ones(4)), 0.5, 0.5) == (0.0, 1.0)
    probability, ess_fraction = adaptive_position_proposal_probability(
        _DummyFilter([1.0, 0.0, 0.0, 0.0]),
        0.5,
        0.5,
    )
    assert np.isclose(probability, 0.5)
    assert np.isclose(ess_fraction, 0.25)
