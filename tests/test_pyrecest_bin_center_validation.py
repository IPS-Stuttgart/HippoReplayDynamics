from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.pyrecest_models as pyrecest_models
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import (
    PyRecEstGoalParticleModel,
    _coerce_candidate_goals,
    _farthest_point_subset,
)


def _two_bin_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.2, 0.8]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_pyrecest_goal_helpers_accept_one_dimensional_centers():
    centers = np.array([0.0, 10.0, 20.0], dtype=float)

    explicit_goals = _coerce_candidate_goals(np.array([0.0, 20.0], dtype=float), centers)
    np.testing.assert_allclose(explicit_goals, np.array([[0.0], [20.0]], dtype=float))

    automatic_goals = _coerce_candidate_goals(None, centers)
    np.testing.assert_allclose(automatic_goals, centers[:, None])

    subset = _farthest_point_subset(np.arange(40, dtype=float), max_points=4)
    assert subset.shape == (4, 1)
    assert np.all(np.isfinite(subset))


def test_pyrecest_goal_particle_score_coerces_one_dimensional_centers(monkeypatch):
    captured: dict[str, object] = {}

    class FakeLookup:
        method = "linear"

    class FakeGridLikelihood:
        @staticmethod
        def build_replay_grid_likelihood_lookup(bin_centers, interpolation):
            captured["lookup_centers_shape"] = np.asarray(bin_centers).shape
            assert interpolation == "linear"
            return FakeLookup()

        @staticmethod
        def adaptive_position_proposal_probability(filter_, probability, threshold):
            return float(probability), 1.0

        @staticmethod
        def update_position_grid_likelihood(filter_, log_likelihood, bin_centers, **kwargs):
            captured.setdefault("update_centers_shape", np.asarray(bin_centers).shape)
            return float(np.max(log_likelihood))

        @staticmethod
        def particle_position_log_posterior(position_particles, weights, bin_centers):
            captured.setdefault("posterior_centers_shape", np.asarray(bin_centers).shape)
            probabilities = np.asarray(weights, dtype=float)
            probabilities = probabilities / probabilities.sum()
            return np.log(probabilities)

    class FakeFilterState:
        def __init__(self):
            self.w = np.array([0.5, 0.5], dtype=float)

    class FakeFilter:
        def __init__(self):
            self.position_particles = np.array([[0.0], [1.0]], dtype=float)
            self.filter_state = FakeFilterState()
            self.last_jump_fraction = 0.0
            self.last_goal_remap_fraction = 0.0
            self.last_position_proposal_fraction = 0.0

        def predict_replay(self, *, dt, use_semi_implicit_position_update):
            assert use_semi_implicit_position_update is True
            captured.setdefault("predict_dts", []).append(float(dt))

        def get_goal_posterior_weights(self, goals):
            return np.full(goals.shape[0], 1.0 / goals.shape[0], dtype=float)

    def fake_build_filter(self, bin_centers, candidate_goals, dt):
        captured["filter_centers_shape"] = np.asarray(bin_centers).shape
        captured["goal_shape"] = np.asarray(candidate_goals).shape
        captured["dt"] = float(dt)
        return FakeFilter()

    monkeypatch.setattr(pyrecest_models, "_import_replay_grid_likelihood", lambda: FakeGridLikelihood)
    monkeypatch.setattr(PyRecEstGoalParticleModel, "_build_filter", fake_build_filter)

    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([0.0, 1.0], dtype=float),
        n_particles=2,
        random_seed=0,
    )
    score = model.score(_two_bin_emissions(), np.array([0.0, 1.0], dtype=float))

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == (2, 2)
    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(np.exp(score.trajectory_log_posterior).sum(axis=1), np.ones(2))
    assert captured["lookup_centers_shape"] == (2, 1)
    assert captured["update_centers_shape"] == (2, 1)
    assert captured["posterior_centers_shape"] == (2, 1)
    assert captured["filter_centers_shape"] == (2, 1)
    assert captured["goal_shape"] == (2, 1)
    assert captured["predict_dts"] == [0.01]


def test_pyrecest_goal_particle_score_rejects_bin_count_mismatch():
    model = PyRecEstGoalParticleModel(n_particles=2)

    with pytest.raises(ValueError, match="emissions.n_bins"):
        model.score(_two_bin_emissions(), np.array([0.0, 1.0, 2.0], dtype=float))
