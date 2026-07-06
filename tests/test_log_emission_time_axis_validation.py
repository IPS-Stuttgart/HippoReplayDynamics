from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def test_log_emission_tensor_rejects_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="at least one time bin"):
        LogEmissionTensor(
            log_likelihood=np.empty((0, 1), dtype=float),
            spike_counts=np.empty((0, 1), dtype=int),
            times=np.empty(0, dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=0,
        )
