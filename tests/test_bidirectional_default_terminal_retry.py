from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import BidirectionalReplayModel as CompatBidirectionalReplayModel
from hipporeplayimm.reverse_models import BidirectionalReplayModel as DirectBidirectionalReplayModel


class _TerminalOnlyWithTrajectoryRequest:
    def __init__(self, name: str, terminal: np.ndarray | None = None) -> None:
        self.name = name
        self.terminal = None if terminal is None else np.asarray(terminal, dtype=float)

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool = False,
    ) -> EventScore:
        del bin_centers
        terminal = self.terminal
        if terminal is None:
            terminal = np.array([0.0, -np.inf], dtype=float) if emissions.log_likelihood[0, 0] == 0.0 else np.array([-np.inf, 0.0], dtype=float)
        trajectory = np.asarray([terminal], dtype=float) if return_trajectory else None
        terminal_log_posterior = trajectory[-1].copy() if trajectory is not None else None
        return EventScore(
            model_name=self.name,
            log_likelihood=0.0,
            n_time=emissions.n_time,
            n_spikes=emissions.n_spikes,
            diagnostics={},
            terminal_log_posterior=terminal_log_posterior,
            trajectory_log_posterior=trajectory,
        )


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array([[0.0, -1.0], [10.0, -2.0]], dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_compat_bidirectional_default_retries_forward_terminal_posterior() -> None:
    emissions = _emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    forward = _TerminalOnlyWithTrajectoryRequest("forward", np.array([0.0, -np.inf], dtype=float))
    reverse = _TerminalOnlyWithTrajectoryRequest("reverse", np.array([-np.inf, 0.0], dtype=float))

    score = CompatBidirectionalReplayModel(forward, reverse, name="bidirectional").score(emissions, bin_centers)

    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(np.exp(score.terminal_log_posterior), np.array([0.5, 0.5], dtype=float))
    assert score.trajectory_log_posterior is not None


def test_direct_bidirectional_default_retries_forward_terminal_posterior() -> None:
    emissions = _emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    base_model = _TerminalOnlyWithTrajectoryRequest("base")

    score = DirectBidirectionalReplayModel(base_model, name="bidirectional").score(emissions, bin_centers)

    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(np.exp(score.terminal_log_posterior), np.array([0.5, 0.5], dtype=float))
    assert score.trajectory_log_posterior is not None
