from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def test_log_emission_tensor_rejects_nan_log_likelihood() -> None:
    with pytest.raises(ValueError, match="log_likelihood.*NaN"):
        LogEmissionTensor(
            log_likelihood=np.array([[0.0, np.nan]]),
            spike_counts=np.zeros((1, 1), dtype=int),
            times=np.array([0.0]),
            dt=0.02,
            cell_ids=np.array([1]),
            n_spikes=0,
        )
