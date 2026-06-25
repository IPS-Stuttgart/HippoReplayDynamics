from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    ReverseTimeReplayModel,
    score_replay_model_compat,
)


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.1], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


class _ExactSparseMomentumModel:
    mode = "momentum-exact-sparse"
    name = "exact-sparse"

    def __init__(self) -> None:
        self.return_trajectory_calls: list[bool] = []

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        candidate_indices=None,
        occupancy_s=None,
        return_trajectory: bool = True,
    ) -> EventScore:
        del bin_centers, candidate_indices, occupancy_s
        self.return_trajectory_calls.append(bool(return_trajectory))
        terminal = np.log(np.array([0.75, 0.25], dtype=float))
        trajectory = None
        if return_trajectory:
            trajectory = np.repeat(terminal[None, :], emissions.n_time, axis=0)
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={"state_space_mode": self.mode},
            terminal_log_posterior=terminal.copy(),
            trajectory_log_posterior=trajectory,
        )


def test_compat_suppresses_exact_sparse_momentum_trajectory_by_default():
    model = _ExactSparseMomentumModel()

    result = score_replay_model_compat(model, _emissions(), _bin_centers())

    assert model.return_trajectory_calls == [False]
    assert result.trajectory_log_posterior is None


def test_compat_respects_explicit_exact_sparse_trajectory_request():
    model = _ExactSparseMomentumModel()

    result = score_replay_model_compat(
        model,
        _emissions(),
        _bin_centers(),
        return_trajectory=True,
    )

    assert model.return_trajectory_calls == [True]
    assert result.trajectory_log_posterior is not None


def test_reverse_wrapper_keeps_default_trajectory_for_terminal_mapping():
    model = _ExactSparseMomentumModel()
    reverse = ReverseTimeReplayModel(model, name="exact-sparse-reverse")

    result = reverse.score(_emissions(), _bin_centers())

    assert model.return_trajectory_calls == [True]
    assert result.terminal_log_posterior is not None
    assert result.trajectory_log_posterior is not None
