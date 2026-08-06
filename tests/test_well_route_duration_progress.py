from __future__ import annotations

import numpy as np
from scipy.sparse import eye

from hipporeplayimm.encoding import LogEmissionTensor
import hipporeplayimm.well_route_state_space as route_module
from hipporeplayimm.well_route_state_space import (
    WellRouteStateSpaceReplayModel,
    _transition_progress_fractions,
)


def test_route_targets_follow_elapsed_time_for_irregular_bins(monkeypatch) -> None:
    captured_targets: list[np.ndarray] = []

    def capture_transition(
        bin_centers,
        goal,
        *,
        drift_step_cm,
        sigma_cm,
        max_step_sigma,
    ):
        del drift_step_cm, sigma_cm, max_step_sigma
        captured_targets.append(np.asarray(goal, dtype=float).copy())
        return eye(np.asarray(bin_centers).shape[0], format="csr")

    monkeypatch.setattr(route_module, "_goal_transition_matrix", capture_transition)

    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 2), dtype=float),
        spike_counts=np.empty((3, 0), dtype=int),
        times=np.array([0.0, 0.75, 1.0], dtype=float),
        dt=0.25,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
        bin_durations=np.array([0.75, 0.25, 0.25], dtype=float),
        transition_durations=np.array([0.75, 0.25], dtype=float),
    )
    centers = np.array([[0.0], [10.0]], dtype=float)
    routes = np.array([[[0.0], [10.0]]], dtype=float)

    WellRouteStateSpaceReplayModel(
        candidate_routes=routes,
        transition_sigma_cm_sqrt_s=1.0,
        drift_speed_cm_s=0.0,
    ).score(emissions, centers)

    np.testing.assert_allclose(
        np.asarray(captured_targets, dtype=float)[:, 0],
        np.array([7.5, 10.0], dtype=float),
    )


def test_uniform_transition_durations_preserve_equal_step_progress() -> None:
    progress = _transition_progress_fractions(np.ones(4, dtype=float))

    np.testing.assert_allclose(progress, np.array([0.25, 0.5, 0.75, 1.0]))
