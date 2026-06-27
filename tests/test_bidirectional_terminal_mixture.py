from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.reverse_models import BidirectionalReplayModel

_ORIGINAL_FIRST_ROW = np.log(np.array([0.64, 0.36], dtype=float))
_FORWARD_TERMINAL = np.log(np.array([0.70, 0.30], dtype=float))
_REVERSE_TERMINAL = np.log(np.array([0.20, 0.80], dtype=float))
_EXPECTED_MIXED_TERMINAL = np.array([0.45, 0.55], dtype=float)


class _OneSidedTrajectoryModel:
    name = "one-sided-trajectory"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers, return_trajectory
        log_likelihood = np.asarray(emissions.log_likelihood, dtype=float)
        if np.allclose(log_likelihood[0], _ORIGINAL_FIRST_ROW):
            return EventScore(
                self.name,
                0.0,
                emissions.n_time,
                emissions.n_spikes,
                terminal_log_posterior=_FORWARD_TERMINAL.copy(),
                trajectory_log_posterior=None,
            )

        reverse_time_trajectory = np.vstack(
            [
                _REVERSE_TERMINAL,
                np.log(np.array([0.50, 0.50], dtype=float)),
                np.log(np.array([0.90, 0.10], dtype=float)),
            ]
        )
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=None,
            trajectory_log_posterior=reverse_time_trajectory,
        )


def test_bidirectional_keeps_mixed_terminal_when_only_one_direction_has_trajectory() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.vstack(
            [
                _ORIGINAL_FIRST_ROW,
                np.log(np.array([0.50, 0.50], dtype=float)),
                np.log(np.array([0.25, 0.75], dtype=float)),
            ]
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.01, 0.02], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    result = BidirectionalReplayModel(_OneSidedTrajectoryModel()).score(
        emissions,
        bin_centers,
        return_trajectory=True,
    )

    assert result.trajectory_log_posterior is None
    assert result.terminal_log_posterior is not None
    np.testing.assert_allclose(
        np.exp(result.terminal_log_posterior),
        _EXPECTED_MIXED_TERMINAL,
        atol=1e-12,
    )
    assert result.diagnostics["forward_model_posterior_probability"] == 0.5
    assert result.diagnostics["reverse_model_posterior_probability"] == 0.5
