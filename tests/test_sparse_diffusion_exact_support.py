from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import DiffusionModel


def test_sparse_diffusion_preserves_unreachable_state_support() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 1.0], dtype=float),
        dt=1.0,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
        ],
        dtype=float,
    )

    with np.errstate(invalid="ignore"):
        score = DiffusionModel(sigma_cm=1.0, max_step_sigma=3.0).score(
            emissions,
            bin_centers,
        )

    assert np.isneginf(score.log_likelihood)
