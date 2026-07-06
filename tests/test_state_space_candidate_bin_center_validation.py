from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.05, 0.80, 0.10, 0.05],
                    [0.02, 0.10, 0.80, 0.08],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.00, 0.02, 0.04], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_candidate_indices_accepts_one_dimensional_bin_centers() -> None:
    model = StateSpaceReplayModel(
        "momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=1,
            momentum_predicted_candidate_top_k=1,
        ),
    )

    candidates = model.candidate_indices(_emissions(), np.arange(4, dtype=float))

    assert len(candidates) == 3
    for candidate in candidates:
        assert candidate.ndim == 1
        assert np.all(candidate >= 0)
        assert np.all(candidate < 4)


def test_candidate_indices_rejects_mismatched_bin_center_rows() -> None:
    model = StateSpaceReplayModel(
        "momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=1,
            momentum_predicted_candidate_top_k=1,
        ),
    )

    with pytest.raises(ValueError, match="one row per emission spatial bin"):
        model.candidate_indices(_emissions(), np.array([0.0, 1.0, 2.0], dtype=float))
