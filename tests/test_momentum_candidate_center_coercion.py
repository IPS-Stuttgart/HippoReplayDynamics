from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_momentum_candidate_indices_accept_flat_1d_bin_centers() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.15, 0.65, 0.15, 0.05],
                    [0.05, 0.15, 0.65, 0.15],
                    [0.03, 0.12, 0.25, 0.60],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    flat_centers = np.arange(emissions.n_bins, dtype=float)
    column_centers = flat_centers[:, None]
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_sigma_cm_sqrt_s=1.0,
            momentum_initial_sigma_cm_sqrt_s=1.0,
            momentum_velocity_decay=0.9,
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=2,
        ),
    )

    flat_candidates = model.candidate_indices(emissions, flat_centers)
    column_candidates = model.candidate_indices(emissions, column_centers)

    assert len(flat_candidates) == emissions.n_time
    for flat, column in zip(flat_candidates, column_candidates):
        np.testing.assert_array_equal(flat, column)
