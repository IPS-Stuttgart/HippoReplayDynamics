import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_candidate_source_rejects_rows_without_active_support() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
                [0.0, -np.inf],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.02, 0.04], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_source="emission",
        ),
    )

    with pytest.raises(ValueError, match="active support"):
        model.candidate_indices(
            emissions,
            centers,
            valid_bin_mask=np.array([False, True], dtype=bool),
        )
