from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _candidate_inputs() -> tuple[LogEmissionTensor, np.ndarray, list[np.ndarray]]:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.80, 0.15, 0.05],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.15, 0.80],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=float,
    )
    candidates = [
        np.array([0], dtype=int),
        np.array([1], dtype=int),
        np.array([2], dtype=int),
    ]
    return emissions, bin_centers, candidates


@pytest.mark.parametrize("mode", ["momentum", "imm"])
def test_duration_aware_candidate_trajectory_uses_exact_support(mode: str) -> None:
    emissions, bin_centers, candidates = _candidate_inputs()
    model = StateSpaceReplayModel(
        mode=mode,
        config=StateSpaceDecoderConfig(mode=mode),
    )

    score = model.score(
        emissions,
        bin_centers,
        candidate_indices=candidates,
        return_trajectory=True,
    )

    assert np.isfinite(score.log_evidence)
    trajectory = score.trajectory_log_posterior
    assert trajectory is not None
    for time_index, candidate in enumerate(candidates):
        row = trajectory[time_index]
        active = np.zeros(row.shape[0], dtype=bool)
        active[candidate] = True
        assert np.all(np.isfinite(row[active]))
        assert np.all(np.isneginf(row[~active]))
        assert logsumexp(row) == 0.0
