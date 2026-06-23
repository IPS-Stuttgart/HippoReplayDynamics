import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel
from hipporeplayimm.state_space_model import _normalize_log_rows


def test_normalize_log_rows_rejects_rows_without_finite_mass():
    with pytest.raises(ValueError, match="row 1"):
        _normalize_log_rows(np.array([[0.0, -np.inf], [-np.inf, -np.inf]]))


def test_posterior_candidate_source_rejects_masked_out_finite_mass():
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0, -np.inf]]),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    valid_bin_mask = np.array([False, True])
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_source="posterior",
            momentum_candidate_top_k=1,
        ),
    )

    with pytest.raises(ValueError, match="row 0"):
        model.candidate_indices(emissions, centers, valid_bin_mask=valid_bin_mask)
