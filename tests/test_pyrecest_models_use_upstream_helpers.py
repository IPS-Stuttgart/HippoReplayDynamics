from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def test_replay_grid_helpers_are_loaded_from_pyrecest(monkeypatch) -> None:
    module_name = "pyrecest.filters.replay_grid_likelihood"
    fake = types.ModuleType(module_name)
    calls: list[str] = []

    def build_grid_likelihood_lookup(bin_centers, method):
        calls.append("build_grid_likelihood_lookup")
        return {"method": method, "n": len(bin_centers)}

    def grid_log_likelihood_values(positions, values, bin_tree, lookup):
        calls.append("grid_log_likelihood_values")
        return np.zeros(np.asarray(positions).shape[0], dtype=float)

    def grid_proposal_weights(values):
        calls.append("grid_proposal_weights")
        values = np.asarray(values, dtype=float)
        return np.full(values.shape, 1.0 / values.size, dtype=float)

    def particle_position_log_posterior(positions, weights, bin_centers, bin_tree):
        calls.append("particle_position_log_posterior")
        return np.full(np.asarray(bin_centers).shape[0], -np.log(np.asarray(bin_centers).shape[0]))

    def effective_sample_size_fraction(weights):
        calls.append("effective_sample_size_fraction")
        return 1.0

    fake.build_grid_likelihood_lookup = build_grid_likelihood_lookup
    fake.grid_log_likelihood_values = grid_log_likelihood_values
    fake.grid_proposal_weights = grid_proposal_weights
    fake.particle_position_log_posterior = particle_position_log_posterior
    fake.effective_sample_size_fraction = effective_sample_size_fraction
    monkeypatch.setitem(sys.modules, module_name, fake)

    pyrecest_models = importlib.import_module("hipporeplayimm.pyrecest_models")
    helpers = pyrecest_models._load_pyrecest_replay_grid_helpers()

    bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    lookup = helpers.build_grid_likelihood_lookup(bin_centers, "nearest")
    helpers.grid_log_likelihood_values(bin_centers, np.asarray([0.0, -1.0]), object(), lookup)
    helpers.grid_proposal_weights(np.asarray([0.0, -1.0]))
    helpers.particle_position_log_posterior(bin_centers, np.asarray([0.5, 0.5]), bin_centers, object())
    helpers.effective_sample_size_fraction(np.asarray([0.5, 0.5]))

    assert calls == [
        "build_grid_likelihood_lookup",
        "grid_log_likelihood_values",
        "grid_proposal_weights",
        "particle_position_log_posterior",
        "effective_sample_size_fraction",
    ]
