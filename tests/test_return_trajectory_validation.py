from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import score_replay_model_compat
from hipporeplayimm.reverse_models import BidirectionalReplayModel, ReverseTimeReplayModel

BIN_CENTERS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


class AlwaysTrajectoryModel:
    name = "always-trajectory"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        trajectory = np.log(
            np.array(
                [
                    [0.65, 0.35],
                    [0.25, 0.75],
                ],
                dtype=float,
            )
        )
        return EventScore(
            self.name,
            -1.0,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={},
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=trajectory.copy(),
        )


class ReturnTrajectoryAwareModel(AlwaysTrajectoryModel):
    name = "return-trajectory-aware"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        result = super().score(emissions, bin_centers, return_trajectory=return_trajectory)
        if return_trajectory is False:
            result.trajectory_log_posterior = None
        return result


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.6, 0.4],
                    [0.7, 0.3],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    "wrapper_cls",
    [ReverseTimeReplayModel, BidirectionalReplayModel],
)
@pytest.mark.parametrize("value", ["False", "true", 0, 1])
def test_replay_wrappers_reject_non_boolean_return_trajectory(wrapper_cls, value):
    wrapper = wrapper_cls(AlwaysTrajectoryModel())

    with pytest.raises(TypeError, match="return_trajectory"):
        wrapper.score(_emissions(), BIN_CENTERS, return_trajectory=value)


@pytest.mark.parametrize(
    "wrapper_cls",
    [ReverseTimeReplayModel, BidirectionalReplayModel],
)
def test_replay_wrappers_normalize_numpy_false_return_trajectory(wrapper_cls):
    wrapper = wrapper_cls(AlwaysTrajectoryModel())

    result = wrapper.score(
        _emissions(),
        BIN_CENTERS,
        return_trajectory=np.bool_(False),
    )

    assert result.trajectory_log_posterior is None


@pytest.mark.parametrize("value", ["False", "true", 0, 1])
def test_score_replay_model_compat_rejects_non_boolean_return_trajectory(value):
    with pytest.raises(TypeError, match="return_trajectory"):
        score_replay_model_compat(
            AlwaysTrajectoryModel(),
            _emissions(),
            BIN_CENTERS,
            return_trajectory=value,
        )


def test_score_replay_model_compat_normalizes_numpy_false_return_trajectory():
    result = score_replay_model_compat(
        ReturnTrajectoryAwareModel(),
        _emissions(),
        BIN_CENTERS,
        return_trajectory=np.bool_(False),
    )

    assert result.trajectory_log_posterior is None
