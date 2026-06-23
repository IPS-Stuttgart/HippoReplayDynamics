from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel,
    ReverseTimeReplayModel as CompatReverseTimeReplayModel,
    copy_emissions_with_log_likelihood,
)
from hipporeplayimm.reverse_models import (
    BidirectionalReplayModel as DirectBidirectionalReplayModel,
    ReverseTimeReplayModel as DirectReverseTimeReplayModel,
    reverse_emissions,
)


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


class _EvidenceOnlyPosteriorModel:
    def __init__(self, name: str, log_likelihood: float, posterior: np.ndarray) -> None:
        self.name = name
        self.log_likelihood = float(log_likelihood)
        self.posterior = np.asarray(posterior, dtype=float)

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        del bin_centers
        terminal = np.log(self.posterior / np.sum(self.posterior))
        return EventScore(
            self.name,
            self.log_likelihood,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={
                "decoded_endpoint_x": 123.0,
                "decoded_map_bin": 1,
                "terminal_posterior_entropy": 0.5,
            },
            terminal_log_posterior=terminal,
            trajectory_log_posterior=None,
        )


class _TrajectoryPosteriorModel:
    def __init__(self, name: str, log_likelihood: float, trajectory: np.ndarray) -> None:
        self.name = name
        self.log_likelihood = float(log_likelihood)
        self.trajectory = np.asarray(trajectory, dtype=float)

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        del bin_centers
        trajectory = self.trajectory[: emissions.n_time]
        if trajectory.shape != (emissions.n_time, emissions.n_bins):
            raise ValueError("test trajectory shape does not match emissions")
        trajectory = trajectory / trajectory.sum(axis=1, keepdims=True)
        log_trajectory = np.log(trajectory)
        return EventScore(
            self.name,
            self.log_likelihood,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={
                "decoded_endpoint_x": 123.0,
                "decoded_map_bin": 1,
                "terminal_posterior_entropy": 0.5,
            },
            terminal_log_posterior=log_trajectory[-1].copy(),
            trajectory_log_posterior=log_trajectory,
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


def test_reverse_time_wrappers_clear_unmappable_evidence_only_terminal() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    for wrapper_type in (DirectReverseTimeReplayModel, CompatReverseTimeReplayModel):
        wrapped = wrapper_type(_EvidenceOnlyPosteriorModel("evidence-only", 1.0, np.array([0.9, 0.1])))
        result = wrapped.score(emissions, bin_centers)

        assert result.trajectory_log_posterior is None
        assert result.terminal_log_posterior is None
        assert result.diagnostics["reverse_time_terminal_posterior"] == "unavailable_without_trajectory"
        assert "decoded_endpoint_x" not in result.diagnostics
        assert "decoded_map_bin" not in result.diagnostics
        assert "terminal_posterior_entropy" not in result.diagnostics


def test_direct_reverse_wrapper_recomputes_mapped_terminal_diagnostics() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    base = _TrajectoryPosteriorModel(
        "trajectory",
        0.0,
        np.array([[0.9, 0.1], [0.45, 0.55], [0.2, 0.8]], dtype=float),
    )

    result = DirectReverseTimeReplayModel(base, name="direct-reverse").score(emissions, bin_centers)

    posterior = np.array([0.9, 0.1], dtype=float)
    np.testing.assert_allclose(result.terminal_log_posterior, np.log(posterior))
    assert result.trajectory_log_posterior is not None
    assert result.diagnostics["decoded_map_bin"] == 0
    assert result.diagnostics["decoded_endpoint_x"] == 0.1
    expected_entropy = -float(np.sum(posterior * np.log(posterior)))
    np.testing.assert_allclose(result.diagnostics["terminal_posterior_entropy"], expected_entropy)


def test_direct_bidirectional_wrapper_keeps_forward_terminal_when_reverse_unmappable() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    base = _EvidenceOnlyPosteriorModel("base", 0.0, np.array([0.8, 0.2]))
    bidirectional = DirectBidirectionalReplayModel(base, name="direct-bidirectional")

    result = bidirectional.score(emissions, bin_centers)

    posterior = np.array([0.8, 0.2], dtype=float)
    np.testing.assert_allclose(result.terminal_log_posterior, np.log(posterior))
    assert result.trajectory_log_posterior is None
    assert result.diagnostics["decoded_map_bin"] == 0
    assert result.diagnostics["decoded_endpoint_x"] == 0.2
    expected_entropy = -float(np.sum(posterior * np.log(posterior)))
    np.testing.assert_allclose(result.diagnostics["terminal_posterior_entropy"], expected_entropy)


def test_bidirectional_wrapper_does_not_mix_unmappable_reverse_terminal() -> None:
    emissions = _duration_emissions()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    forward = _EvidenceOnlyPosteriorModel("forward", 0.0, np.array([0.8, 0.2]))
    reverse_base = _EvidenceOnlyPosteriorModel("reverse-base", 5.0, np.array([0.1, 0.9]))
    bidirectional = BidirectionalReplayModel(
        forward_model=forward,
        reverse_model=CompatReverseTimeReplayModel(reverse_base),
        name="bidirectional",
    )

    result = bidirectional.score(emissions, bin_centers)

    np.testing.assert_allclose(result.terminal_log_posterior, np.log(np.array([0.8, 0.2])))
