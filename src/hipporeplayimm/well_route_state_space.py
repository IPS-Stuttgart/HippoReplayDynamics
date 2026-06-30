"""Simple exact route-conditioned state-space replay model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .duration_dynamics import transition_durations_s
from .encoding import LogEmissionTensor
from .goal_state_space import _forward_backward_goal_mixture, _goal_transition_matrix
from .models import EventScore, _posterior_diagnostics
from .state_space_utils import _mean_entropy, _per_bin_sigma


@dataclass
class WellRouteStateSpaceReplayModel:
    """Exact mixture over deterministic well-to-well route templates.

    Each route is a sequence of waypoints.  During an event the active waypoint
    target progresses deterministically along the route, while position follows
    a drift-diffusion transition toward the current route target.  This is a
    compact route/progress baseline rather than a full latent route-progress HMM.
    """

    candidate_routes: np.ndarray | None = None
    transition_sigma_cm_sqrt_s: float = 85.0
    drift_speed_cm_s: float = 400.0
    max_step_sigma: float = 4.0
    max_default_points: int = 8
    name: str = "sorted-spike-state-space-route"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        centers = np.asarray(bin_centers, dtype=float)
        if centers.ndim != 2:
            raise ValueError("bin_centers must have shape (n_bins, position_dim)")
        if emissions.n_bins != centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        if centers.shape[1] == 0:
            raise ValueError("bin_centers must contain at least one coordinate column")
        if not np.all(np.isfinite(centers)):
            raise ValueError("bin_centers must be finite")

        transition_sigma_cm_sqrt_s = float(self.transition_sigma_cm_sqrt_s)
        drift_speed_cm_s = float(self.drift_speed_cm_s)
        max_step_sigma = float(self.max_step_sigma)
        if not np.isfinite(transition_sigma_cm_sqrt_s) or transition_sigma_cm_sqrt_s <= 0.0:
            raise ValueError("transition_sigma_cm_sqrt_s must be finite and positive")
        if not np.isfinite(drift_speed_cm_s) or drift_speed_cm_s < 0.0:
            raise ValueError("drift_speed_cm_s must be finite and non-negative")
        if not np.isfinite(max_step_sigma) or max_step_sigma <= 0.0:
            raise ValueError("max_step_sigma must be finite and positive")

        routes = _coerce_candidate_routes(self.candidate_routes, centers, self.max_default_points)
        durations = transition_durations_s(emissions)
        transitions = tuple(
            tuple(
                _goal_transition_matrix(
                    centers,
                    _route_target(route, transition_index + 1, max(emissions.n_time - 1, 1)),
                    drift_step_cm=drift_speed_cm_s * float(duration),
                    sigma_cm=_per_bin_sigma(transition_sigma_cm_sqrt_s, float(duration)),
                    max_step_sigma=max_step_sigma,
                )
                for transition_index, duration in enumerate(durations)
            )
            for route in routes
        )
        logp, trajectory, route_posteriors = _forward_backward_goal_mixture(emissions.log_likelihood, transitions)
        terminal = trajectory[-1]
        terminal_route_posterior = route_posteriors[-1]
        best_route = int(np.argmax(terminal_route_posterior))
        diagnostics = {
            "state_space_mode": "route",
            "state_space_observation_model": "sorted-spike-poisson",
            "state_space_trajectory_posterior": 1,
            "state_space_trajectory_time_bins": int(emissions.n_time),
            "route_state_space_candidate_routes": int(routes.shape[0]),
            "route_state_space_waypoints_per_route": int(routes.shape[1]),
            "route_state_space_transition_sigma_cm_sqrt_s": float(transition_sigma_cm_sqrt_s),
            "route_state_space_drift_speed_cm_s": float(drift_speed_cm_s),
            "route_state_space_max_step_sigma": float(max_step_sigma),
            "route_state_space_evidence_support": "exact_full_grid",
            "route_state_space_most_likely_route_index": best_route,
            "route_state_space_most_likely_route_probability": float(terminal_route_posterior[best_route]),
            "mean_trajectory_posterior_entropy": _mean_entropy(trajectory),
        }
        diagnostics.update(_posterior_diagnostics(terminal, centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def routes_from_wells(well_locations: np.ndarray, *, include_reverse: bool = True) -> np.ndarray:
    wells = np.asarray(well_locations, dtype=float)
    if wells.ndim != 2 or wells.shape[0] < 2:
        raise ValueError("well_locations must have shape (n_wells, position_dim) with at least two wells")
    if wells.shape[1] == 0:
        raise ValueError("well_locations must contain at least one coordinate column")
    if not np.all(np.isfinite(wells)):
        raise ValueError("well_locations must be finite")
    routes = []
    for start in range(wells.shape[0]):
        for end in range(wells.shape[0]):
            if start == end:
                continue
            if not include_reverse and end < start:
                continue
            routes.append(np.stack([wells[start], wells[end]], axis=0))
    return np.asarray(routes, dtype=float)


def _coerce_candidate_routes(routes: np.ndarray | None, centers: np.ndarray, max_default_points: int) -> np.ndarray:
    if routes is not None:
        arr = np.asarray(routes, dtype=float)
        if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] < 2 or arr.shape[2] != centers.shape[1]:
            raise ValueError("candidate_routes must have shape (n_routes, n_waypoints>=2, position_dim)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("candidate_routes must be finite")
        return arr
    max_points = _coerce_max_default_points(max_default_points)
    points = _farthest_point_subset(centers, max_points=max_points)
    return routes_from_wells(points)


def _coerce_max_default_points(value: int) -> int:
    arr = np.asarray(value)
    if arr.shape != () or arr.dtype.kind == "b" or arr.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("max_default_points must be an integer >= 2 when candidate_routes is not provided")
    numeric = float(arr)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("max_default_points must be an integer >= 2 when candidate_routes is not provided")
    max_points = int(numeric)
    if max_points < 2:
        raise ValueError("max_default_points must be at least 2 when candidate_routes is not provided")
    return max_points


def _route_target(route: np.ndarray, transition_number: int, n_transitions: int) -> np.ndarray:
    if n_transitions <= 0:
        return route[-1]
    progress = min(max(float(transition_number) / float(n_transitions), 0.0), 1.0)
    segment_float = progress * (route.shape[0] - 1)
    segment = min(int(np.floor(segment_float)), route.shape[0] - 2)
    local = segment_float - segment
    return (1.0 - local) * route[segment] + local * route[segment + 1]


def _farthest_point_subset(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.copy()
    selected = [int(np.argmin(np.sum(points, axis=1)))]
    min_dist2 = np.full(points.shape[0], np.inf, dtype=float)
    for _ in range(1, max_points):
        last = points[selected[-1]]
        dist2 = np.sum((points - last[None, :]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected.append(int(np.argmax(min_dist2)))
    return points[np.asarray(selected, dtype=int)]
