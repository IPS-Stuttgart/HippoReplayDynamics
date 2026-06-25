from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel as CompatBidirectionalReplayModel,
    ReverseTimeReplayModel as CompatReverseTimeReplayModel,
)
from hipporeplayimm.reverse_models import (
    BidirectionalReplayModel as DirectBidirectionalReplayModel,
    ReverseTimeReplayModel as DirectReverseTimeReplayModel,
)
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


class _LegacyTrajectoryOnlyModel:
    name = "legacy-trajectory"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        trajectory = np.asarray(emissions.log_likelihood, dtype=float).copy()
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=trajectory,
        )


class _TrajectoryWithoutTerminalModel:
    name = "trajectory-without-terminal"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers
        trajectory = np.asarray(emissions.log_likelihood, dtype=float).copy()
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=None,
            trajectory_log_posterior=None if return_trajectory is False else trajectory,
        )


class _KwargRecordingModel:
    name = "kwarg-recorder"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        occupancy_s: np.ndarray | None = None,
        candidate_indices: list[np.ndarray] | None = None,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers
        self.calls.append(
            {
                "log_likelihood": np.asarray(emissions.log_likelihood, dtype=float).copy(),
                "occupancy_s": None if occupancy_s is None else np.asarray(occupancy_s, dtype=float).copy(),
                "candidate_indices": None
                if candidate_indices is None
                else [np.asarray(curr, dtype=int).copy() for curr in candidate_indices],
                "return_trajectory": return_trajectory,
            }
        )
        trajectory = np.asarray(emissions.log_likelihood, dtype=float).copy()
        return EventScore(
            self.name,
            float(len(self.calls)),
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=None if return_trajectory is False else trajectory,
        )


def _synthetic_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.30],
                    [0.20, 0.80],
                    [0.40, 0.60],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.01, 0.02], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


def _state_space_model() -> StateSpaceReplayModel:
    return StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(mode="diffusion"),
    )


def _momentum_state_space_model() -> StateSpaceReplayModel:
    return StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=0,
        ),
    )


def _candidate_indices() -> list[np.ndarray]:
    return [
        np.array([0], dtype=int),
        np.array([1], dtype=int),
        np.array([0, 1], dtype=int),
    ]


def _float_candidate_indices() -> list[np.ndarray]:
    return [
        np.array([0.0], dtype=float),
        np.array([1.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    ]


def _candidate_lists(candidates: object) -> list[list[int]]:
    assert candidates is not None
    return [np.asarray(curr, dtype=int).tolist() for curr in candidates]  # type: ignore[union-attr]


def test_reverse_wrappers_accept_evidence_only_return_trajectory() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()

    for wrapper_type in (DirectReverseTimeReplayModel, CompatReverseTimeReplayModel):
        wrapped = wrapper_type(_state_space_model())
        result = wrapped.score(emissions, bin_centers, return_trajectory=False)

        assert np.isfinite(result.log_likelihood)
        assert result.trajectory_log_posterior is None
        assert result.terminal_log_posterior is None
        assert result.diagnostics["reverse_time_terminal_posterior"] == "unavailable_without_trajectory"


def test_reverse_wrappers_keep_remapped_terminal_when_only_return_suppresses_trajectory() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    expected_terminal = emissions.log_likelihood[-1]

    for wrapper_type in (DirectReverseTimeReplayModel, CompatReverseTimeReplayModel):
        wrapped = wrapper_type(_LegacyTrajectoryOnlyModel())
        result = wrapped.score(emissions, bin_centers, return_trajectory=False)

        assert result.trajectory_log_posterior is None
        assert result.terminal_log_posterior is not None
        np.testing.assert_allclose(result.terminal_log_posterior, expected_terminal)
        assert "reverse_time_terminal_posterior" not in result.diagnostics


def test_reverse_wrappers_preserve_candidate_index_dtype_for_base_validation() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    candidates = _float_candidate_indices()

    for wrapper_type in (DirectReverseTimeReplayModel, CompatReverseTimeReplayModel):
        wrapped = wrapper_type(_momentum_state_space_model())

        with pytest.raises((TypeError, ValueError), match="integer"):
            wrapped.score(
                emissions,
                bin_centers,
                candidate_indices=candidates,
                return_trajectory=False,
            )


def test_direct_reverse_wrapper_forwards_occupancy_and_reverses_candidates() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    occupancy_s = np.array([0.25, 1.00], dtype=float)
    candidates = _candidate_indices()
    base_model = _KwargRecordingModel()
    wrapped = DirectReverseTimeReplayModel(base_model)

    wrapped.score(
        emissions,
        bin_centers,
        occupancy_s=occupancy_s,
        candidate_indices=candidates,
        return_trajectory=False,
    )

    assert len(base_model.calls) == 1
    call = base_model.calls[0]
    np.testing.assert_allclose(call["log_likelihood"], emissions.log_likelihood[::-1])
    np.testing.assert_allclose(call["occupancy_s"], occupancy_s)
    assert _candidate_lists(call["candidate_indices"]) == [[0, 1], [1], [0]]
    assert call["return_trajectory"] is False


def test_direct_bidirectional_wrapper_forwards_occupancy_and_candidates_to_both_directions() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    occupancy_s = np.array([0.25, 1.00], dtype=float)
    candidates = _candidate_indices()
    base_model = _KwargRecordingModel()
    wrapped = DirectBidirectionalReplayModel(base_model)

    wrapped.score(
        emissions,
        bin_centers,
        occupancy_s=occupancy_s,
        candidate_indices=candidates,
        return_trajectory=False,
    )

    assert len(base_model.calls) == 2
    forward_call, reverse_call = base_model.calls
    np.testing.assert_allclose(forward_call["log_likelihood"], emissions.log_likelihood)
    np.testing.assert_allclose(reverse_call["log_likelihood"], emissions.log_likelihood[::-1])
    np.testing.assert_allclose(forward_call["occupancy_s"], occupancy_s)
    np.testing.assert_allclose(reverse_call["occupancy_s"], occupancy_s)
    assert _candidate_lists(forward_call["candidate_indices"]) == [[0], [1], [0, 1]]
    assert _candidate_lists(reverse_call["candidate_indices"]) == [[0, 1], [1], [0]]
    assert forward_call["return_trajectory"] is False
    assert reverse_call["return_trajectory"] is True


def test_bidirectional_wrappers_forward_evidence_only_return_trajectory() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    wrappers = [
        DirectBidirectionalReplayModel(_state_space_model()),
        CompatBidirectionalReplayModel(
            forward_model=_state_space_model(),
            reverse_model=CompatReverseTimeReplayModel(_state_space_model()),
            name="compat-bidirectional",
        ),
    ]

    for wrapped in wrappers:
        full = wrapped.score(emissions, bin_centers, return_trajectory=True)
        result = wrapped.score(emissions, bin_centers, return_trajectory=False)

        assert np.isfinite(result.log_likelihood)
        assert result.trajectory_log_posterior is None
        assert result.terminal_log_posterior is not None
        assert full.terminal_log_posterior is not None
        assert np.isclose(np.exp(result.terminal_log_posterior).sum(), 1.0)
        np.testing.assert_allclose(result.terminal_log_posterior, full.terminal_log_posterior)


def test_bidirectional_wrappers_use_mixed_trajectory_terminal_when_base_terminal_missing() -> None:
    emissions = _synthetic_emissions()
    bin_centers = _bin_centers()
    wrappers = [
        DirectBidirectionalReplayModel(_TrajectoryWithoutTerminalModel()),
        CompatBidirectionalReplayModel(
            forward_model=_TrajectoryWithoutTerminalModel(),
            reverse_model=CompatReverseTimeReplayModel(_TrajectoryWithoutTerminalModel()),
            name="compat-trajectory-without-terminal-bidirectional",
        ),
    ]

    for wrapped in wrappers:
        result = wrapped.score(emissions, bin_centers, return_trajectory=True)

        assert result.trajectory_log_posterior is not None
        assert result.terminal_log_posterior is not None
        np.testing.assert_allclose(
            result.terminal_log_posterior,
            result.trajectory_log_posterior[-1],
        )
