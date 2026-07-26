import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import StationaryModel


def _emissions(log_likelihood: np.ndarray) -> LogEmissionTensor:
    n_time = log_likelihood.shape[0]
    return LogEmissionTensor(
        log_likelihood=np.asarray(log_likelihood, dtype=float),
        spike_counts=np.zeros((n_time, 1), dtype=int),
        times=np.arange(n_time, dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_stationary_model_rejects_disjoint_time_bin_support() -> None:
    emissions = _emissions(np.array([[0.0, -np.inf], [-np.inf, 0.0]]))
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="stationary model has no finite path mass"):
        StationaryModel().score(emissions, centers)


def test_stationary_model_keeps_overlapping_finite_support() -> None:
    emissions = _emissions(np.array([[0.0, -1.0], [-np.inf, 0.0]]))
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])

    score = StationaryModel().score(emissions, centers)

    assert np.isfinite(score.log_likelihood)
    assert not np.any(np.isnan(score.terminal_log_posterior))
    np.testing.assert_allclose(np.exp(score.terminal_log_posterior).sum(), 1.0)
