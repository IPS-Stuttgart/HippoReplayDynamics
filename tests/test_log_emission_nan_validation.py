"""Regression tests for undefined log-emission likelihoods."""

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _tensor(log_likelihood: np.ndarray) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_log_emission_tensor_rejects_nan_log_likelihood() -> None:
    with pytest.raises(ValueError, match="log_likelihood must not contain NaN values"):
        _tensor(np.array([[0.0, np.nan]]))


def test_log_emission_tensor_preserves_negative_infinity_support_mask() -> None:
    tensor = _tensor(np.array([[0.0, -np.inf]]))

    assert np.isneginf(tensor.log_likelihood[0, 1])
