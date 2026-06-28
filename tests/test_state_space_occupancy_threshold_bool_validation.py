from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.benchmarks import _score_train_joint_model
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _valid_bin_mask_from_occupancy,
)


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.1], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _bin_centers() -> np.ndarray:
    return np.column_stack((np.arange(3.0), np.zeros(3, dtype=float)))


@pytest.mark.parametrize("value", [True, np.bool_(True), np.asarray(True, dtype=object)])
def test_valid_bin_mask_rejects_boolean_occupancy_threshold(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="min_occupancy_s"):
        _valid_bin_mask_from_occupancy(np.array([0.0, 0.5, 1.0], dtype=float), value, 3)


def test_valid_bin_mask_keeps_numeric_occupancy_threshold() -> None:
    mask = _valid_bin_mask_from_occupancy(np.array([0.0, 0.5, 1.0], dtype=float), 0.5, 3)

    np.testing.assert_array_equal(mask, np.array([False, True, True]))


@pytest.mark.parametrize("value", [True, np.bool_(True), np.asarray(True, dtype=object)])
def test_candidate_support_rejects_boolean_valid_occupancy_threshold(value: object) -> None:
    hipporeplayimm.apply_runtime_patches()
    emissions = _emissions()
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=0,
            valid_occupancy_threshold_s=value,
        ),
    )

    with pytest.raises(TypeError, match="min_occupancy_s"):
        _score_train_joint_model(
            model,
            emissions,
            emissions,
            _bin_centers(),
            occupancy_s=np.array([0.0, 0.5, 1.0], dtype=float),
        )
