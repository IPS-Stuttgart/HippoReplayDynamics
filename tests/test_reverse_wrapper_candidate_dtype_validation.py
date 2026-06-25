from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.result_improvement_extensions import ReverseTimeReplayModel as CompatReverseTimeReplayModel
from hipporeplayimm.reverse_models import ReverseTimeReplayModel as DirectReverseTimeReplayModel
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def _synthetic_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.30],
                    [0.20, 0.80],
                    [0.40, 0.60],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.01, 0.02], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _momentum_model() -> StateSpaceReplayModel:
    return StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=0,
        ),
    )


def test_reverse_wrappers_preserve_candidate_index_dtype_for_base_validation() -> None:
    emissions = _synthetic_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    float_candidates = [
        np.array([0.0], dtype=float),
        np.array([1.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    ]

    for wrapper_type in (DirectReverseTimeReplayModel, CompatReverseTimeReplayModel):
        wrapped = wrapper_type(_momentum_model())

        with pytest.raises((TypeError, ValueError), match="integer"):
            wrapped.score(
                emissions,
                bin_centers,
                candidate_indices=float_candidates,
                return_trajectory=False,
            )
