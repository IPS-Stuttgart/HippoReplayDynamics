from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _emissions(spike_counts: np.ndarray, n_spikes: int) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, spike_counts.shape[1]), dtype=float),
        spike_counts=spike_counts,
        times=np.array([0.0], dtype=float),
        dt=1.0,
        cell_ids=np.arange(1, spike_counts.shape[1] + 1, dtype=int),
        n_spikes=n_spikes,
    )


def test_log_emission_preserves_exact_large_spike_total() -> None:
    first = 2**53
    counts = np.array([[first, 1]], dtype=object)

    emissions = _emissions(counts, first + 1)

    assert emissions.spike_counts.dtype == np.dtype(int)
    assert emissions.spike_counts.tolist() == [[first, 1]]
    assert emissions.n_spikes == first + 1


def test_log_emission_rejects_spike_count_outside_integer_range_without_warning() -> None:
    too_large = int(np.iinfo(np.dtype(int)).max) + 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="spike_counts must fit into integer count range"):
            _emissions(np.array([[too_large]], dtype=object), too_large)
