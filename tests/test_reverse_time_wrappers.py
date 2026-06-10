from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel,
    ReverseTimeReplayModel as CompatReverseTimeReplayModel,
    copy_emissions_with_log_likelihood,
)
from hipporeplayimm.reverse_models import reverse_emissions


def _duration_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
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
        metadata={"source": "unit-test"},
    )


def test_reverse_emissions_preserves_reversed_duration_metadata() -> None:
    emissions = _duration_emissions()

    reversed_emissions = reverse_emissions(emissions)

    np.testing.assert_allclose(
        reversed_emissions.log_likelihood,
        emissions.log_likelihood[::-1],
    )
    np.testing.assert_allclose(
        reversed_emissions.bin_durations,
        emissions.bin_durations[::-1],
    )
    np.testing.assert_allclose(
        reversed_emissions.transition_durations,
        emissions.transition_durations[::-1],
    )
    assert reversed_emissions.metadata == emissions.metadata
    assert reversed_emissions.metadata is not emissions.metadata


def test_copy_emissions_with_log_likelihood_preserves_duration_metadata_when_reversing() -> None:
    emissions = _duration_emissions()
    log_likelihood = emissions.log_likelihood - 1.0

    copied = copy_emissions_with_log_likelihood(
        emissions,
        log_likelihood,
        reverse_time=True,
    )

    np.testing.assert_allclose(copied.log_likelihood, log_likelihood[::-1])
    np.testing.assert_allclose(copied.bin_durations, emissions.bin_durations[::-1])
    np.testing.assert_allclose(
        copied.transition_durations,
        emissions.transition_durations[::-1],
    )
    assert copied.metadata == emissions.metadata
    assert copied.metadata is not emissions.metadata


class _CandidateRecordingModel:
    def __init__(self, name: str, log_likelihood: float) -> None:
        self.name = name
        self.log_likelihood = float(log_likelihood)
        self.seen_candidate_indices: list[np.ndarray] | None = None

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        candidate_indices: list[np.ndarray] | None = None,
        occupancy_s: np.ndarray | None = None,
    ) -> EventScore:
        del occupancy_s
        self.seen_candidate_indices = (
            None
            if candidate_indices is None
            else [np.asarray(current, dtype=int).copy() for current in candidate_indices]
        )
        trajectory = np.full(
            (emissions.n_time, emissions.n_bins),
            -np.log(emissions.n_bins),
            dtype=float,
        )
        return EventScore(
            self.name,
            self.log_likelihood,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={},
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=trajectory,
        )


def test_bidirectional_wrapper_passes_candidate_indices_to_reverse_model() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    forward = _CandidateRecordingModel("forward", 0.0)
    reverse_base = _CandidateRecordingModel("reverse-base", -1.0)
    bidirectional = BidirectionalReplayModel(
        forward_model=forward,
        reverse_model=CompatReverseTimeReplayModel(reverse_base),
        name="bidirectional",
    )
    candidates = [np.array([0]), np.array([0, 1]), np.array([1])]

    bidirectional.score(emissions, bin_centers, candidate_indices=candidates)

    assert forward.seen_candidate_indices is not None
    assert reverse_base.seen_candidate_indices is not None
    for seen, expected in zip(forward.seen_candidate_indices, candidates, strict=True):
        np.testing.assert_array_equal(seen, expected)
    for seen, expected in zip(reverse_base.seen_candidate_indices, candidates[::-1], strict=True):
        np.testing.assert_array_equal(seen, expected)
