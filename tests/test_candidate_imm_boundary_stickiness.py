from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel


@pytest.mark.parametrize("stickiness", [0.0, 1.0])
def test_candidate_imm_boundary_stickiness_scores_without_runtime_warnings(
    stickiness: float,
) -> None:
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                    [0.6, 0.4],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = CandidateKinematicModel(
        mode="imm",
        top_k=2,
        mode_stickiness=stickiness,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        score = model.score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert np.all(np.isfinite(score.trajectory_log_posterior))
