from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import RandomModel


def test_replay_model_rejects_empty_emission_time_axis() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.empty((0, 1), dtype=float),
        spike_counts=np.empty((0, 1), dtype=int),
        times=np.empty(0, dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )

    with pytest.raises(ValueError, match="at least one time bin"):
        RandomModel().score(emissions, np.zeros((1, 2), dtype=float))
