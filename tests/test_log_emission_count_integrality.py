from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def test_log_emission_tensor_rejects_fractional_spike_counts_with_integer_total() -> None:
    with pytest.raises(ValueError, match="spike_counts.*integer-valued"):
        LogEmissionTensor(
            log_likelihood=np.zeros((2, 1), dtype=float),
            spike_counts=np.array([[0.5], [0.5]], dtype=float),
            times=np.array([0.0, 0.02], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=1,
        )


def test_log_emission_tensor_rejects_boolean_spike_counts() -> None:
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        LogEmissionTensor(
            log_likelihood=np.zeros((1, 1), dtype=float),
            spike_counts=np.array([[True]], dtype=bool),
            times=np.array([0.0], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=1,
        )
