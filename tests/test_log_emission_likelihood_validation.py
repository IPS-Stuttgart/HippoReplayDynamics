from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _tensor_with_log_likelihood(log_likelihood: np.ndarray) -> LogEmissionTensor:
    values = np.asarray(log_likelihood, dtype=float)
    return LogEmissionTensor(
        log_likelihood=values,
        spike_counts=np.zeros((values.shape[0], 1), dtype=int),
        times=np.arange(values.shape[0], dtype=float) * 0.1,
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_log_emission_tensor_rejects_nan_likelihood_entries_at_construction():
    with pytest.raises(ValueError, match="NaN"):
        _tensor_with_log_likelihood(np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float))


def test_log_emission_tensor_rejects_rows_without_finite_likelihood_mass():
    with pytest.raises(ValueError, match="at least one finite spatial-bin"):
        _tensor_with_log_likelihood(np.array([[0.0, -1.0], [-np.inf, -np.inf]], dtype=float))


def test_log_emission_tensor_allows_individual_impossible_bins():
    tensor = _tensor_with_log_likelihood(np.array([[0.0, -np.inf], [-1.0, 0.0]], dtype=float))

    assert tensor.n_time == 2
    assert tensor.n_bins == 2
    assert np.isneginf(tensor.log_likelihood[0, 1])
