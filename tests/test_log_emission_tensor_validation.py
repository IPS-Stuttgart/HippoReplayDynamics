from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _make_tensor(
    log_likelihood: np.ndarray,
    *,
    spike_counts: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> LogEmissionTensor:
    values = np.asarray(log_likelihood, dtype=float)
    n_time = int(values.shape[0])
    counts = np.zeros((n_time, 1), dtype=int) if spike_counts is None else np.asarray(spike_counts)
    timestamps = np.arange(n_time, dtype=float) * 0.02 if times is None else np.asarray(times, dtype=float)
    return LogEmissionTensor(
        log_likelihood=values,
        spike_counts=counts,
        times=timestamps,
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=int(np.sum(counts)),
    )


def test_log_emission_tensor_rejects_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="at least one time bin"):
        _make_tensor(
            np.empty((0, 2), dtype=float),
            spike_counts=np.empty((0, 1), dtype=int),
            times=np.empty(0, dtype=float),
        )


def test_log_emission_tensor_rejects_nan_likelihoods() -> None:
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        _make_tensor(np.array([[0.0, np.nan]], dtype=float))


def test_log_emission_tensor_rejects_fractional_spike_counts() -> None:
    with pytest.raises(ValueError, match="integer-valued counts"):
        _make_tensor(
            np.zeros((1, 2), dtype=float),
            spike_counts=np.array([[0.5]], dtype=float),
        )
