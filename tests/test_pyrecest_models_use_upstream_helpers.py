from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def test_replay_grid_helpers_are_loaded_from_pyrecest(monkeypatch) -> None:
    pyrecest_module = types.ModuleType("pyrecest")
    filters_module = types.ModuleType("pyrecest.filters")
    calls: list[str] = []

    def build_replay_grid_likelihood_lookup(bin_centers, method):
        calls.append("build_replay_grid_likelihood_lookup")
        return {"method": method, "n": len(bin_centers)}

    def update_position_grid_likelihood(filter_, values, bin_centers, **kwargs):
        del filter_, values, bin_centers, kwargs
        calls.append("update_position_grid_likelihood")
        return 0.0

    def particle_position_log_posterior(positions, weights, bin_centers):
        del positions, weights
        calls.append("particle_position_log_posterior")
        n_bins = np.asarray(bin_centers).shape[0]
        return np.full(n_bins, -np.log(n_bins))

    def adaptive_position_proposal_probability(filter_, base_probability, ess_threshold):
        del filter_, ess_threshold
        calls.append("adaptive_position_proposal_probability")
        return base_probability, 1.0

    filters_module.build_replay_grid_likelihood_lookup = build_replay_grid_likelihood_lookup
    filters_module.update_position_grid_likelihood = update_position_grid_likelihood
    filters_module.particle_position_log_posterior = particle_position_log_posterior
    filters_module.adaptive_position_proposal_probability = adaptive_position_proposal_probability
    pyrecest_module.filters = filters_module
    monkeypatch.setitem(sys.modules, "pyrecest", pyrecest_module)
    monkeypatch.setitem(sys.modules, "pyrecest.filters", filters_module)

    pyrecest_models = importlib.import_module("hipporeplayimm.pyrecest_models")
    helpers = pyrecest_models._import_replay_grid_likelihood()

    bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    helpers.build_replay_grid_likelihood_lookup(bin_centers, "nearest")
    helpers.update_position_grid_likelihood(object(), np.asarray([0.0, -1.0]), bin_centers)
    helpers.particle_position_log_posterior(bin_centers, np.asarray([0.5, 0.5]), bin_centers)
    helpers.adaptive_position_proposal_probability(object(), 0.5, 0.5)

    assert calls == [
        "build_replay_grid_likelihood_lookup",
        "update_position_grid_likelihood",
        "particle_position_log_posterior",
        "adaptive_position_proposal_probability",
    ]
