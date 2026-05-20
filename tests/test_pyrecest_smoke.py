from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel


def test_pyrecest_particle_model_with_position_proposals_smoke() -> None:
    pytest.importorskip("pyrecest.filters")
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0, -2.0, -2.0, -4.0], [-4.0, -2.0, -2.0, 0.0]], dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        n_particles=16,
        position_proposal_probability=1.0,
        random_seed=11,
    )

    score = model.score(emissions, bin_centers)

    assert np.isfinite(score.log_likelihood)
    assert "pyrecest_last_position_proposal_fraction" in score.diagnostics
