import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import RandomModel


def test_core_model_endpoint_mean_stays_finite_at_float64_coordinate_limit() -> None:
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

    score = RandomModel().score(emissions, centers)

    assert np.isfinite(score.diagnostics["decoded_endpoint_x"])
    assert np.isfinite(score.diagnostics["decoded_endpoint_y"])
    assert score.diagnostics["decoded_endpoint_x"] == max_float
    assert score.diagnostics["decoded_endpoint_y"] == -max_float
