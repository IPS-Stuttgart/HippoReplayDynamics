from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel as CompatBidirectionalReplayModel,
    ReverseTimeReplayModel as CompatReverseTimeReplayModel,
)
from hipporeplayimm.reverse_models import (
    BidirectionalReplayModel as DirectBidirectionalReplayModel,
    ReverseTimeReplayModel as DirectReverseTimeReplayModel,
)
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


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


def _state_space_model() -> StateSpaceReplayModel:
    return StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(mode="diffusion"),
    )


def test_reverse_wrappers_accept_evidence_only_return_trajectory() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()

    for wrapper_type in (DirectReverseTimeReplayModel, CompatReverseTimeReplayModel):
        wrapped = wrapper_type(_state_space_model())
        result = wrapped.score(emissions, bin_centers, return_trajectory=False)

        assert np.isfinite(result.log_likelihood)
        assert result.trajectory_log_posterior is None
        assert result.terminal_log_posterior is None
        assert result.diagnostics["reverse_time_terminal_posterior"] == "unavailable_without_trajectory"


def test_bidirectional_wrappers_forward_evidence_only_return_trajectory() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    wrappers = [
        DirectBidirectionalReplayModel(_state_space_model()),
        CompatBidirectionalReplayModel(
            forward_model=_state_space_model(),
            reverse_model=CompatReverseTimeReplayModel(_state_space_model()),
            name="compat-bidirectional",
        ),
    ]

    for wrapped in wrappers:
        result = wrapped.score(emissions, bin_centers, return_trajectory=False)

        assert np.isfinite(result.log_likelihood)
        assert result.trajectory_log_posterior is None
        assert result.terminal_log_posterior is not None
        assert np.isclose(np.exp(result.terminal_log_posterior).sum(), 1.0)
