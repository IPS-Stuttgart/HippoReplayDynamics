from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel


def test_pyrecest_model_reports_install_hint_when_extra_is_missing() -> None:
    if importlib.util.find_spec("pyrecest") is not None:
        pytest.skip("PyRecEst is installed in this environment")

    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.3, 0.7]])),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        n_particles=8,
        random_seed=0,
    )

    with pytest.raises(RuntimeError, match="hipporeplayimm\\[pyrecest\\]"):
        model.score(emissions, np.array([[0.0, 0.0], [1.0, 0.0]]))
