import numpy as np
from scipy.sparse import csr_matrix

from hipporeplayimm.empirical_transition import (
    EmpiricalTransitionStateSpaceReplayModel,
)
from hipporeplayimm.encoding import LogEmissionTensor


def test_empirical_transition_model_accepts_vector_shaped_1d_grid() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([], dtype=int),
        n_spikes=0,
    )

    score = EmpiricalTransitionStateSpaceReplayModel(
        csr_matrix(np.eye(2)),
    ).score(emissions, np.array([0.0, 1.0], dtype=float))

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert score.terminal_log_posterior.shape == (2,)
    assert score.diagnostics["decoded_endpoint_y"] == 0.0


def test_empirical_transition_model_uses_stable_endpoint_diagnostics() -> None:
    n_bins = 7
    max_float = np.finfo(float).max
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, n_bins), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.tile(np.array([max_float, -max_float]), (n_bins, 1))

    score = EmpiricalTransitionStateSpaceReplayModel(
        csr_matrix(np.eye(n_bins)),
    ).score(emissions, centers)

    assert np.isfinite(score.diagnostics["decoded_endpoint_x"])
    assert np.isfinite(score.diagnostics["decoded_endpoint_y"])
    assert score.diagnostics["decoded_endpoint_x"] == max_float
    assert score.diagnostics["decoded_endpoint_y"] == -max_float
