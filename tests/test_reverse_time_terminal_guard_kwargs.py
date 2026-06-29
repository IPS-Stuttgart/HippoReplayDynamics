from __future__ import annotations

import numpy as np

from hipporeplayimm import result_improvement_extensions as extensions
from hipporeplayimm import reverse_time_terminal_guard
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore


def _single_bin_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array([[0.0]], dtype=float),
        spike_counts=np.zeros((1, 0), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.01,
        cell_ids=np.array([], dtype=int),
        n_spikes=0,
    )


def test_reverse_time_terminal_guard_preserves_supported_return_trajectory(monkeypatch) -> None:
    calls: list[bool | None] = []

    def score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory=None):
        del self, bin_centers, occupancy_s, candidate_indices
        calls.append(return_trajectory)
        trajectory = None
        if return_trajectory is not False:
            trajectory = np.zeros((emissions.n_time, emissions.n_bins), dtype=float)
        return EventScore(
            "reverse-test",
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=np.zeros(emissions.n_bins, dtype=float),
            trajectory_log_posterior=trajectory,
        )

    monkeypatch.setattr(extensions.ReverseTimeReplayModel, "score", score)
    reverse_time_terminal_guard.apply_reverse_time_terminal_guard_patch()

    wrapped = extensions.ReverseTimeReplayModel(base_model=object())
    wrapped.score(
        _single_bin_emissions(),
        np.array([[0.0, 0.0]], dtype=float),
        return_trajectory=False,
    )

    assert calls == [False]
