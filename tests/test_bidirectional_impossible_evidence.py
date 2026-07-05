from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.reverse_models import BidirectionalReplayModel


class _ImpossibleEvidenceModel:
    name = "impossible"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        del bin_centers
        trajectory = np.full_like(np.asarray(emissions.log_likelihood, dtype=float), -np.log(emissions.n_bins))
        return EventScore(
            self.name,
            float("-inf"),
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=trajectory[-1],
            trajectory_log_posterior=trajectory,
        )


def test_bidirectional_weights_remain_finite_when_both_directions_impossible() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    result = BidirectionalReplayModel(_ImpossibleEvidenceModel()).score(
        emissions,
        bin_centers,
        return_trajectory=True,
    )

    assert np.isneginf(result.log_likelihood)
    assert result.diagnostics["forward_model_posterior_probability"] == 0.5
    assert result.diagnostics["reverse_model_posterior_probability"] == 0.5
    assert not np.isnan(result.terminal_log_posterior).any()
    assert not np.isnan(result.trajectory_log_posterior).any()
    np.testing.assert_allclose(np.exp(result.terminal_log_posterior).sum(), 1.0)
