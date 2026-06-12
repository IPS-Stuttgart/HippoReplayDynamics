from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.benchmarks import _score_train_joint_model
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, LogEmissionTensor
from hipporeplayimm.simulation_recovery import _score_recovery_model
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _masked_candidate_fixture() -> tuple[LogEmissionTensor, np.ndarray, np.ndarray]:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [20.0, 19.0, 1.0, 0.0],
                [20.0, 19.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.1], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.column_stack((np.arange(4.0), np.zeros(4, dtype=float)))
    occupancy_s = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    return emissions, bin_centers, occupancy_s


def _occupancy_aware_momentum_config() -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode="momentum",
        momentum_candidate_top_k=2,
        momentum_predicted_candidate_top_k=0,
        valid_occupancy_threshold_s=0.5,
    )


def _encoding(bin_centers: np.ndarray, occupancy_s: np.ndarray) -> EncodingModel:
    return EncodingModel(
        x_edges=np.arange(5.0, dtype=float),
        y_edges=np.array([-0.5, 0.5], dtype=float),
        bin_centers=bin_centers,
        rates_hz=np.ones((1, bin_centers.shape[0]), dtype=float),
        occupancy_s=occupancy_s,
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


def test_benchmark_derives_state_space_candidates_on_valid_occupancy_support() -> None:
    emissions, bin_centers, occupancy_s = _masked_candidate_fixture()
    model = StateSpaceReplayModel(
        mode="momentum",
        config=_occupancy_aware_momentum_config(),
    )

    _, joint_score = _score_train_joint_model(
        model,
        emissions,
        emissions,
        bin_centers,
        occupancy_s=occupancy_s,
    )

    assert joint_score.diagnostics["state_space_valid_bin_count"] == 2
    assert joint_score.diagnostics["mean_candidate_count"] == pytest.approx(2.0)


def test_simulation_recovery_rescores_state_space_candidates_on_valid_occupancy_support() -> None:
    emissions, bin_centers, occupancy_s = _masked_candidate_fixture()
    encoding = _encoding(bin_centers, occupancy_s)
    model = SortedSpikeStateSpaceReplayModel(
        mode="momentum",
        config=_occupancy_aware_momentum_config(),
    )
    stale_unmasked_candidates = [
        np.array([0, 1], dtype=int),
        np.array([0, 1], dtype=int),
    ]

    score = _score_recovery_model(
        model,
        emissions,
        encoding,
        candidate_indices=stale_unmasked_candidates,
        score_with_occupancy=True,
    )

    assert score.diagnostics["state_space_valid_bin_count"] == 2
    assert score.diagnostics["mean_candidate_count"] == pytest.approx(2.0)
