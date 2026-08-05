import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import StationaryModel


def test_stationary_model_repeats_full_event_posterior_across_time() -> None:
    log_likelihood = np.array(
        [
            [0.0, -10.0],
            [-9.0, 0.0],
        ]
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0], [1.0]])

    score = StationaryModel().score(emissions, centers)

    assert score.trajectory_log_posterior is not None
    assert score.terminal_log_posterior is not None
    expected = np.repeat(
        score.terminal_log_posterior[None, :],
        emissions.n_time,
        axis=0,
    )
    np.testing.assert_allclose(score.trajectory_log_posterior, expected)
    np.testing.assert_allclose(
        np.exp(score.trajectory_log_posterior).sum(axis=1),
        1.0,
    )
