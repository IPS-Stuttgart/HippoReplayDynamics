import sys
import types

import numpy as np
from scipy.spatial import cKDTree

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import (
    PyRecEstGoalParticleIMMModel,
    PyRecEstGoalParticleModel,
    _interpolated_grid_values,
    _update_filter_from_grid_likelihood,
)


def test_pyrecest_goal_particle_model_passes_transition_durations(monkeypatch):
    goal_filter, _ = _install_dummy_pyrecest(monkeypatch)
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    emissions = _pyrecest_duration_test_emissions()
    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([[0.0, 0.0], [2.0, 0.0]]),
        n_particles=centers.shape[0],
        random_seed=0,
        jump_probability=0.0,
        goal_reset_probability=0.0,
    )

    score = model.score(emissions, centers)

    assert np.allclose(goal_filter.prediction_dts, emissions.transition_durations)
    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["pyrecest_time_bin_s"] == emissions.dt
    assert score.diagnostics["pyrecest_transition_durations"] == "0.01,0.04"


def test_pyrecest_goal_particle_imm_model_passes_transition_durations(monkeypatch):
    _, imm_filter = _install_dummy_pyrecest(monkeypatch)
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    emissions = _pyrecest_duration_test_emissions()
    model = PyRecEstGoalParticleIMMModel(
        candidate_goals=np.array([[0.0, 0.0], [2.0, 0.0]]),
        n_particles=centers.shape[0],
        random_seed=0,
        jump_probability=0.0,
        goal_reset_probability=0.0,
        mode_stickiness=0.9,
    )

    score = model.score(emissions, centers)

    assert np.allclose(imm_filter.prediction_dts, emissions.transition_durations)
    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["pyrecest_transition_durations"] == "0.01,0.04"


def test_grid_likelihood_update_interpolates_particle_positions():
    centers = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    log_likelihood = np.array([0.0, 2.0, 1.0, 3.0])
    filter_ = _DummyLikelihoodFilter(np.array([[0.5, 0.5]]))

    log_marginal = _update_filter_from_grid_likelihood(
        filter_,
        log_likelihood,
        centers,
        cKDTree(centers),
    )

    assert np.isclose(log_marginal, 1.5)


def test_interpolated_grid_values_falls_back_for_irregular_grids():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    values = np.array([0.0, 1.0, 3.0])

    interpolated = _interpolated_grid_values(
        np.array([[2.6, 0.0]]),
        values,
        centers,
        cKDTree(centers),
    )

    assert np.allclose(interpolated, [3.0])


class _DummyLikelihoodFilter:
    def __init__(self, positions):
        self.position_particles = np.asarray(positions, dtype=float)
        self.filter_state = types.SimpleNamespace(
            w=np.full(
                self.position_particles.shape[0],
                1.0 / self.position_particles.shape[0],
            )
        )

    def update_position_likelihood(self, likelihood, *, return_log_marginal=False):
        values = np.asarray(likelihood(self.position_particles), dtype=float)
        marginal = float(np.average(values, weights=self.filter_state.w))
        if return_log_marginal:
            return float(np.log(marginal))
        return self


def _pyrecest_duration_test_emissions():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.10, 0.20, 0.70],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.005, 0.020, 0.060]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    emissions.transition_durations = np.array([0.01, 0.04])
    return emissions


def _install_dummy_pyrecest(monkeypatch):
    class DummyGaussianDistribution:
        def __init__(self, mean, covariance):
            self.mean = np.asarray(mean, dtype=float)
            self.covariance = np.asarray(covariance, dtype=float)
            self.dim = int(self.mean.shape[0])

    class DummyLinearDiracDistribution:
        def __init__(self, d, w=None):
            self.d = np.asarray(d, dtype=float)
            if w is None:
                self.w = np.full(self.d.shape[0], 1.0 / self.d.shape[0])
            else:
                self.w = np.asarray(w, dtype=float)

    class BaseDummyFilter:
        mode_names = ("stationary", "diffusion", "momentum", "goal_directed", "jump")

        def __init__(
            self,
            *,
            n_particles,
            position_dim,
            dt,
            initial_position_distribution,
            candidate_goals,
            **_kwargs,
        ):
            self.n_particles = int(n_particles)
            self.position_dim = int(position_dim)
            self.dt = float(dt)
            self.candidate_goals = np.asarray(candidate_goals, dtype=float)
            positions = np.asarray(initial_position_distribution.d, dtype=float)
            if positions.shape[0] < self.n_particles:
                reps = int(np.ceil(self.n_particles / positions.shape[0]))
                positions = np.tile(positions, (reps, 1))
            self._position_particles = positions[: self.n_particles].copy()
            self.filter_state = types.SimpleNamespace(
                w=np.full(self.n_particles, 1.0 / self.n_particles)
            )
            self.__class__.init_dts.append(self.dt)

        @property
        def position_particles(self):
            return self._position_particles

        @property
        def last_jump_fraction(self):
            return 0.0

        @property
        def last_goal_remap_fraction(self):
            return 0.0

        @property
        def last_position_proposal_fraction(self):
            return 0.0

        @property
        def mode_probabilities(self):
            return np.full(len(self.mode_names), 1.0 / len(self.mode_names))

        @property
        def last_mode_transition_fraction(self):
            return 0.0

        def predict_replay(self, dt=None, **_kwargs):
            self.__class__.prediction_dts.append(self.dt if dt is None else float(dt))
            return self

        def update_position_likelihood(self, likelihood, *, return_log_marginal=False):
            values = np.asarray(likelihood(self.position_particles), dtype=float)
            marginal = float(np.average(values, weights=self.filter_state.w))
            if return_log_marginal:
                return float(np.log(marginal))
            return self

        def update_position_likelihood_with_proposal(self, likelihood, **kwargs):
            return self.update_position_likelihood(
                likelihood,
                return_log_marginal=kwargs.get("return_log_marginal", False),
            )

        def get_goal_posterior_weights(self, goals):
            return np.full(np.asarray(goals).shape[0], 1.0 / np.asarray(goals).shape[0])

        def most_likely_mode(self):
            return self.mode_names[0]

    class DummyGoalFilter(BaseDummyFilter):
        init_dts = []
        prediction_dts = []

    class DummyIMMFilter(BaseDummyFilter):
        init_dts = []
        prediction_dts = []

    DummyGoalFilter.init_dts = []
    DummyGoalFilter.prediction_dts = []
    DummyIMMFilter.init_dts = []
    DummyIMMFilter.prediction_dts = []

    pyrecest_module = types.ModuleType("pyrecest")
    pyrecest_module.__path__ = []
    filters_module = types.ModuleType("pyrecest.filters")
    filters_module.GoalConditionedReplayParticleFilter = DummyGoalFilter
    filters_module.GoalConditionedReplayParticleIMMFilter = DummyIMMFilter
    distributions_module = types.ModuleType("pyrecest.distributions")
    distributions_module.GaussianDistribution = DummyGaussianDistribution
    distributions_module.LinearDiracDistribution = DummyLinearDiracDistribution
    pyrecest_module.filters = filters_module
    pyrecest_module.distributions = distributions_module

    monkeypatch.setitem(sys.modules, "pyrecest", pyrecest_module)
    monkeypatch.setitem(sys.modules, "pyrecest.filters", filters_module)
    monkeypatch.setitem(sys.modules, "pyrecest.distributions", distributions_module)
    return DummyGoalFilter, DummyIMMFilter
