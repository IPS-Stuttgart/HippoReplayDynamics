from __future__ import annotations

import importlib
import inspect

import numpy as np

import hipporeplayimm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore


class _RecordingModel:
    name = "recording"
    mode = "momentum-exact-sparse"

    def __init__(self) -> None:
        self.return_trajectory_calls: list[bool | None] = []

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        occupancy_s: np.ndarray | None = None,
        candidate_indices: list[np.ndarray] | None = None,
        return_trajectory: bool | None = None,
    ) -> EventScore:
        del bin_centers, occupancy_s, candidate_indices
        self.return_trajectory_calls.append(return_trajectory)
        trajectory = np.asarray(emissions.log_likelihood, dtype=float).copy()
        return EventScore(
            self.name,
            0.0,
            emissions.n_time,
            emissions.n_spikes,
            terminal_log_posterior=trajectory[-1].copy(),
            trajectory_log_posterior=None if return_trajectory is False else trajectory,
        )


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01], dtype=float),
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _bin_centers() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)


def test_result_improvement_wrappers_restore_after_reload() -> None:
    from hipporeplayimm import result_improvement_extensions as extensions

    importlib.reload(extensions)
    assert getattr(extensions, "_wrapper_return_trajectory_patch_applied", False)
    assert "return_trajectory" not in inspect.signature(
        extensions.ReverseTimeReplayModel.score
    ).parameters

    hipporeplayimm.apply_runtime_patches()

    assert "return_trajectory" in inspect.signature(
        extensions.ReverseTimeReplayModel.score
    ).parameters
    assert "return_trajectory" in inspect.signature(
        extensions.BidirectionalReplayModel.score
    ).parameters

    base_model = _RecordingModel()
    result = extensions.ReverseTimeReplayModel(base_model).score(
        _emissions(),
        _bin_centers(),
        return_trajectory=False,
    )

    assert base_model.return_trajectory_calls == [False]
    assert result.trajectory_log_posterior is None


def test_direct_wrappers_restore_exact_sparse_defaults_after_reload() -> None:
    from hipporeplayimm import reverse_models

    importlib.reload(reverse_models)
    assert getattr(reverse_models, "_wrapper_return_trajectory_patch_applied", False)

    base_model = _RecordingModel()
    reverse_models.BidirectionalReplayModel(base_model).score(
        _emissions(),
        _bin_centers(),
    )
    assert base_model.return_trajectory_calls == [None, None]

    base_model.return_trajectory_calls.clear()
    hipporeplayimm.apply_runtime_patches()
    reverse_models.BidirectionalReplayModel(base_model).score(
        _emissions(),
        _bin_centers(),
    )

    assert base_model.return_trajectory_calls == [False, True]
