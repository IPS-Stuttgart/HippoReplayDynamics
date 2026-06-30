from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel as CompatBidirectionalReplayModel,
    ReverseTimeReplayModel as CompatReverseTimeReplayModel,
)
from hipporeplayimm.reverse_models import BidirectionalReplayModel as DirectBidirectionalReplayModel


class _TrajectoryOnlyDirectionModel:
    name = "trajectory-only-direction"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers
        base = np.array(
            [
                [0.90, 0.10],
                [0.40, 0.60],
                [0.20, 0.80],
            ],
            dtype=float,
        )[: emissions.n_time]
        trajectory = np.log(base)
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=None,
            trajectory_log_posterior=None if return_trajectory is False else trajectory,
        )


class _DefaultTerminalDirectionModel:
    name = "default-terminal-direction"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers
        base = np.array(
            [
                [0.90, 0.10],
                [0.40, 0.60],
                [0.20, 0.80],
            ],
            dtype=float,
        )[: emissions.n_time]
        trajectory = np.log(base)
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=trajectory if return_trajectory is True else None,
        )


def _synthetic_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.70, 0.30], [0.20, 0.80], [0.40, 0.60]], dtype=float)),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.01, 0.02], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_evidence_only_bidirectional_terminal_uses_forward_trajectory_when_needed() -> None:
    emissions = _synthetic_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    wrappers = [
        DirectBidirectionalReplayModel(_TrajectoryOnlyDirectionModel()),
        CompatBidirectionalReplayModel(
            forward_model=_TrajectoryOnlyDirectionModel(),
            reverse_model=CompatReverseTimeReplayModel(_TrajectoryOnlyDirectionModel()),
            name="compat-bidirectional",
        ),
    ]

    for wrapped in wrappers:
        full = wrapped.score(emissions, bin_centers, return_trajectory=True)
        evidence_only = wrapped.score(emissions, bin_centers, return_trajectory=False)

        assert evidence_only.trajectory_log_posterior is None
        assert evidence_only.terminal_log_posterior is not None
        assert full.terminal_log_posterior is not None
        np.testing.assert_allclose(evidence_only.terminal_log_posterior, full.terminal_log_posterior)
        np.testing.assert_allclose(np.exp(evidence_only.terminal_log_posterior).sum(), 1.0)


def test_default_direct_bidirectional_terminal_maps_reverse_terminal_when_needed() -> None:
    emissions = _synthetic_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    wrapped = DirectBidirectionalReplayModel(_DefaultTerminalDirectionModel())

    full = wrapped.score(emissions, bin_centers, return_trajectory=True)
    default = wrapped.score(emissions, bin_centers)

    assert default.trajectory_log_posterior is None
    assert default.terminal_log_posterior is not None
    assert full.terminal_log_posterior is not None
    np.testing.assert_allclose(default.terminal_log_posterior, full.terminal_log_posterior)
    np.testing.assert_allclose(np.exp(default.terminal_log_posterior).sum(), 1.0)
