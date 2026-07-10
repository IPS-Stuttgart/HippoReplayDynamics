from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import RandomModel
from hipporeplayimm.state_space import StateSpaceReplayModel


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    "model_factory",
    [
        RandomModel,
        lambda: StateSpaceReplayModel(mode="fragmented"),
    ],
    ids=["core-random", "state-space"],
)
@pytest.mark.parametrize(
    "bin_centers",
    [
        np.array([[False, False], [True, False]]),
        np.array([["0.0", "0.0"], ["1.0", "0.0"]]),
        np.array([[0.0 + 1.0j, 0.0], [1.0, 0.0]]),
        np.array([[0.0, False], [1.0, 0.0]], dtype=object),
    ],
    ids=["boolean", "text", "complex", "object-boolean"],
)
def test_replay_models_reject_lossy_state_space_coordinate_coercions(
    model_factory: Callable[[], object],
    bin_centers: np.ndarray,
) -> None:
    model = model_factory()

    with pytest.raises(ValueError, match="bin_centers.*numeric real coordinates"):
        model.score(_emissions(), bin_centers)
