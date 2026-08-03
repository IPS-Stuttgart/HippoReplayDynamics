from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _uniform_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    "field",
    [
        "momentum_candidate_top_k",
        "momentum_candidate_min_k",
        "momentum_candidate_max_k",
        "momentum_predicted_candidate_top_k",
    ],
)
def test_provided_candidate_support_still_validates_candidate_counts(
    field: str,
) -> None:
    config_values: dict[str, object] = {
        "mode": "momentum",
        "momentum_candidate_top_k": 0,
        "momentum_candidate_min_k": 1,
        "momentum_candidate_max_k": 0,
        "momentum_predicted_candidate_top_k": 0,
    }
    config_values[field] = 1.5
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(**config_values),
    )
    candidates = [
        np.array([0, 1], dtype=int),
        np.array([0, 1], dtype=int),
    ]

    with pytest.raises(TypeError, match=field):
        model.score(
            _uniform_emissions(),
            np.array([[0.0], [1.0]], dtype=float),
            candidate_indices=candidates,
        )
