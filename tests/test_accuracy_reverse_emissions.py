from __future__ import annotations

import numpy as np

from hipporeplayimm.accuracy_upgrades import TimeReversedReplayModel, reverse_emissions
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore


def _variable_duration_emissions() -> LogEmissionTensor:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.asarray(
                [
                    [0.70, 0.30],
                    [0.20, 0.80],
                    [0.55, 0.45],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.asarray([0.01, 0.03, 0.10], dtype=float),
        dt=0.02,
        cell_ids=np.asarray([1], dtype=int),
        n_spikes=0,
        bin_durations=np.asarray([0.01, 0.02, 0.07], dtype=float),
        transition_durations=np.asarray([0.02, 0.07], dtype=float),
        metadata={"emission_model": "variable-duration-test"},
    )
    return emissions


def test_accuracy_reverse_emissions_preserves_reversed_duration_metadata() -> None:
    emissions = _variable_duration_emissions()

    reversed_emissions = reverse_emissions(emissions)

    np.testing.assert_allclose(reversed_emissions.log_likelihood, emissions.log_likelihood[::-1])
    np.testing.assert_allclose(reversed_emissions.spike_counts, emissions.spike_counts[::-1])
    np.testing.assert_allclose(reversed_emissions.times, emissions.times[::-1])
    np.testing.assert_allclose(reversed_emissions.bin_durations, [0.07, 0.02, 0.01])
    np.testing.assert_allclose(reversed_emissions.transition_durations, [0.07, 0.02])
    assert reversed_emissions.metadata == {"emission_model": "variable-duration-test"}
    assert reversed_emissions.metadata is not emissions.metadata


class _RecordingUniformModel:
    name = "recording-uniform"

    def __init__(self) -> None:
        self.seen_emissions: LogEmissionTensor | None = None

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        self.seen_emissions = emissions
        trajectory = np.log(
            np.full(
                (emissions.n_time, emissions.n_bins),
                1.0 / emissions.n_bins,
                dtype=float,
            )
        )
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=trajectory,
        )


def test_accuracy_time_reversed_wrapper_accepts_variable_duration_emissions() -> None:
    emissions = _variable_duration_emissions()
    base_model = _RecordingUniformModel()
    wrapped = TimeReversedReplayModel(base_model)

    result = wrapped.score(
        emissions,
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )

    assert result.n_time == emissions.n_time
    assert base_model.seen_emissions is not None
    np.testing.assert_allclose(base_model.seen_emissions.transition_durations, [0.07, 0.02])
