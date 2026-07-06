from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def test_log_emission_tensor_rejects_rows_without_finite_support() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        LogEmissionTensor(
            log_likelihood=np.array(
                [
                    [0.0, -np.inf],
                    [-np.inf, -np.inf],
                ],
                dtype=float,
            ),
            spike_counts=np.zeros((2, 1), dtype=int),
            times=np.array([0.0, 1.0]),
            dt=1.0,
            cell_ids=np.array([1]),
            n_spikes=0,
        )


def test_log_emission_tensor_still_allows_impossible_bins_with_row_support() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )

    assert np.isneginf(emissions.log_likelihood[0, 1])
    assert np.isneginf(emissions.log_likelihood[1, 0])
