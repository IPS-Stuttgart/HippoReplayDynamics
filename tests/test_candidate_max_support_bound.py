from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _emissions() -> LogEmissionTensor:
    probabilities = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
    return LogEmissionTensor(
        log_likelihood=np.log(probabilities)[None, :],
        spike_counts=np.empty((1, 0), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.003,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    ("top_k", "min_k"),
    [
        (3, 1),
        (1, 3),
    ],
)
def test_mass_retaining_support_rejects_max_below_effective_lower_bound(
    top_k: int,
    min_k: int,
) -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=top_k,
            momentum_candidate_mass_threshold=0.9,
            momentum_candidate_min_k=min_k,
            momentum_candidate_max_k=2,
            momentum_predicted_candidate_top_k=0,
        ),
    )

    with pytest.raises(ValueError, match="max_k.*candidate lower bound"):
        model.candidate_indices(_emissions())


def test_mass_retaining_support_respects_valid_positive_max() -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=1,
            momentum_candidate_mass_threshold=0.7,
            momentum_candidate_min_k=1,
            momentum_candidate_max_k=2,
            momentum_predicted_candidate_top_k=0,
        ),
    )

    candidates = model.candidate_indices(_emissions())

    assert len(candidates) == 1
    assert candidates[0].size == 2
