import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import (
    StateSpaceReplayModel,
    _gaussian_transition_matrix,
    _score_fragmented,
    _score_imm_candidates,
)


def test_state_space_split_keeps_legacy_helper_exports():
    assert callable(_gaussian_transition_matrix)
    assert callable(_score_fragmented)
    assert callable(_score_imm_candidates)


def test_state_space_model_scores_after_split():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.6, 0.2],
                    [0.1, 0.2, 0.7],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.01, 0.03, 0.05]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])

    score = StateSpaceReplayModel(mode="diffusion").score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == (3, 3)
