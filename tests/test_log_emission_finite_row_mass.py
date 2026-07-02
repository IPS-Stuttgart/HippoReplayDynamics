from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm  # noqa: F401
from hipporeplayimm.encoding import LogEmissionTensor


def _make_tensor(log_likelihood: np.ndarray) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((log_likelihood.shape[0], 1), dtype=int),
        times=np.arange(log_likelihood.shape[0], dtype=float) * 0.01,
        dt=0.01,
        cell_ids=np.asarray([1], dtype=int),
        n_spikes=0,
    )


def test_log_emission_tensor_rejects_rows_without_finite_likelihood_mass() -> None:
    with pytest.raises(ValueError, match="at least one finite spatial-bin likelihood"):
        _make_tensor(
            np.asarray(
                [
                    [0.0, -np.inf],
                    [-np.inf, -np.inf],
                ],
                dtype=float,
            )
        )


def test_log_emission_tensor_allows_partial_infinite_likelihood_rows() -> None:
    tensor = _make_tensor(
        np.asarray(
            [
                [0.0, -np.inf],
                [-np.inf, -1.0],
            ],
            dtype=float,
        )
    )

    assert tensor.n_time == 2
    assert np.isneginf(tensor.log_likelihood[0, 1])
