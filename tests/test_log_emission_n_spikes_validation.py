from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.log_emission_n_spikes_validation import apply_log_emission_n_spikes_validation_patch


def _tensor_with_counts(counts: np.ndarray, n_spikes: object) -> LogEmissionTensor:
    counts = np.asarray(counts, dtype=int)
    return LogEmissionTensor(
        log_likelihood=np.zeros((counts.shape[0], 2), dtype=float),
        spike_counts=counts,
        times=np.arange(counts.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.arange(counts.shape[1], dtype=int),
        n_spikes=n_spikes,
    )


def test_log_emission_n_spikes_patch_refreshes_stale_post_init_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def stale_post_init(self: LogEmissionTensor) -> None:
        self.log_likelihood = np.asarray(self.log_likelihood, dtype=float)

    monkeypatch.setattr(LogEmissionTensor, "__post_init__", stale_post_init)
    monkeypatch.setattr(LogEmissionTensor, "_n_spikes_validation_applied", True, raising=False)

    apply_log_emission_n_spikes_validation_patch()

    with pytest.raises(ValueError, match="total spike_counts sum"):
        _tensor_with_counts(np.array([[1, 0], [0, 2]], dtype=int), 2)


def test_log_emission_tensor_rejects_nan_log_likelihood() -> None:
    counts = np.zeros((2, 1), dtype=int)

    with pytest.raises(ValueError, match="NaN"):
        LogEmissionTensor(
            log_likelihood=np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float),
            spike_counts=counts,
            times=np.arange(counts.shape[0], dtype=float),
            dt=1.0,
            cell_ids=np.array([1], dtype=int),
            n_spikes=0,
        )


def test_log_emission_tensor_rejects_mismatched_n_spikes() -> None:
    with pytest.raises(ValueError, match="total spike_counts sum"):
        _tensor_with_counts(np.array([[1, 0], [0, 2]], dtype=int), 2)


def test_log_emission_tensor_rejects_invalid_n_spikes_summary_values() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _tensor_with_counts(np.array([[1, 0]], dtype=int), float("nan"))

    with pytest.raises(ValueError, match="integer-valued"):
        _tensor_with_counts(np.array([[1, 0]], dtype=int), 1.5)


def test_log_emission_tensor_canonicalizes_integral_n_spikes() -> None:
    emissions = _tensor_with_counts(np.array([[1, 0], [0, 2]], dtype=int), 3.0)

    assert emissions.n_spikes == 3
    assert isinstance(emissions.n_spikes, int)


def test_log_emission_tensor_canonicalizes_integral_spike_counts() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.array([["1", "0"], ["0", "2"]], dtype=object),
        times=np.arange(2, dtype=float),
        dt=1.0,
        cell_ids=np.array([1, 2], dtype=int),
        n_spikes=3.0,
    )

    assert np.issubdtype(emissions.spike_counts.dtype, np.integer)
    np.testing.assert_array_equal(
        emissions.spike_counts,
        np.array([[1, 0], [0, 2]], dtype=int),
    )
    assert emissions.n_spikes == 3
