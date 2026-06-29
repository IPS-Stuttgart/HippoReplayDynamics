from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def test_log_emission_tensor_rejects_nan_log_likelihood() -> None:
    with pytest.raises(ValueError, match="NaN"):
        LogEmissionTensor(
            log_likelihood=np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float),
            spike_counts=np.zeros((2, 1), dtype=int),
            times=np.arange(2, dtype=float),
            dt=1.0,
            cell_ids=np.array([1], dtype=int),
            n_spikes=0,
        )


def test_log_emission_tensor_rejects_mismatched_n_spikes() -> None:
    with pytest.raises(ValueError, match="total spike_counts sum"):
        LogEmissionTensor(
            log_likelihood=np.zeros((2, 2), dtype=float),
            spike_counts=np.array([[1, 0], [0, 2]], dtype=int),
            times=np.arange(2, dtype=float),
            dt=1.0,
            cell_ids=np.array([1, 2], dtype=int),
            n_spikes=2,
        )


def test_log_emission_tensor_canonicalizes_integral_counts() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.array([["1", "0"], ["0", "2"]], dtype=object),
        times=np.arange(2, dtype=float),
        dt=1.0,
        cell_ids=np.array([1, 2], dtype=int),
        n_spikes=3.0,
    )

    assert emissions.n_spikes == 3
    assert isinstance(emissions.n_spikes, int)
    assert np.issubdtype(emissions.spike_counts.dtype, np.integer)
    np.testing.assert_array_equal(emissions.spike_counts, np.array([[1, 0], [0, 2]], dtype=int))
