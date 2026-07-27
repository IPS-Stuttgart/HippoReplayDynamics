from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.reverse_models import BidirectionalReplayModel


class _DirectionalDiagnosticModel:
    name = "directional-diagnostic"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers, return_trajectory
        direction = (
            "forward"
            if emissions.log_likelihood[0, 0] > emissions.log_likelihood[-1, 0]
            else "reverse"
        )
        log_evidence = 0.0 if direction == "forward" else 5.0
        trajectory = np.full(
            (emissions.n_time, emissions.n_bins),
            -np.log(emissions.n_bins),
            dtype=float,
        )
        return EventScore(
            self.name,
            log_evidence,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={
                "direction_specific_marker": direction,
                "state_space_momentum_evidence_support": "exact_full_grid",
                "state_space_momentum_candidate_support": "full_grid",
            },
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=trajectory,
        )


def test_direct_bidirectional_preserves_evidence_dominant_diagnostics() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -1.0],
                [-2.0, -0.5],
                [-0.25, -3.0],
            ],
            dtype=float,
        ),
        spike_counts=np.array([[0], [2], [1]], dtype=int),
        times=np.array([0.005, 0.020, 0.055], dtype=float),
        dt=0.02,
        cell_ids=np.array([7], dtype=int),
        n_spikes=3,
        bin_durations=np.array([0.010, 0.020, 0.030], dtype=float),
        transition_durations=np.array([0.015, 0.035], dtype=float),
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    result = BidirectionalReplayModel(
        _DirectionalDiagnosticModel(),
        name="bidirectional",
    ).score(emissions, bin_centers)

    assert result.diagnostics["direction_specific_marker"] == "reverse"
    assert result.diagnostics["state_space_momentum_evidence_support"] == "exact_full_grid"
    assert result.diagnostics["state_space_momentum_candidate_support"] == "full_grid"
    assert result.diagnostics["time_direction"] == "bidirectional-mixture"
