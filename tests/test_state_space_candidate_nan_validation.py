from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_state_space_momentum_rejects_nan_emissions_before_candidate_scoring() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, np.nan],
                [0.0, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=0,
        ),
    )

    with pytest.raises(ValueError, match="NaN"):
        model.score(emissions, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float))
