"""PyRecEst-backed replay models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import EventScore, LOG_ZERO, _posterior_diagnostics


@dataclass
class PyRecEstGoalParticleModel:
    """Goal-conditioned particle replay scorer backed by PyRecEst."""

    candidate_goals: np.ndarray | None = None
    n_particles: int = 512
    initial_velocity_sigma_cm_s: float = 120.0
    alpha: float = 0.80
    beta: float = 1.00
    process_noise_sigma_cm_s: float = 60.0
    position_jump_sigma_cm: float = 25.0
    jump_probability: float = 0.03
    goal_reset_probability: float = 0.02
    random_seed: int = 1
    name: str = "pyrecest-goal-particle"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")

        seed = _event_seed(self.random_seed, emissions)
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        bin_tree = cKDTree(bin_centers)
        goals = _coerce_candidate_goals(self.candidate_goals, bin_centers)
        filter_ = self._build_filter(bin_centers, goals, emissions.dt, rng)
        process_noise, position_jump = self._build_noise_distributions(bin_centers.shape[1])

        logp = 0.0
        for time_index in range(emissions.n_time):
            if time_index > 0:
                filter_.predict_replay(
                    dt=emissions.dt,
                    alpha=self.alpha,
                    beta=self.beta,
                    process_noise=process_noise,
                    position_jump_distribution=position_jump,
                    jump_probability=self.jump_probability,
                    goal_reset_probability=self.goal_reset_probability,
                    use_semi_implicit_position_update=True,
                )
            logp += _update_filter_from_grid_likelihood(
                filter_,
                emissions.log_likelihood[time_index],
                bin_tree,
            )

        terminal_log_posterior = _particle_position_log_posterior(
            filter_.position_particles,
            np.asarray(filter_.filter_state.w, dtype=float),
            bin_centers,
            bin_tree,
        )
        diagnostics = {
            "pyrecest_particles": int(self.n_particles),
            "pyrecest_candidate_goals": int(goals.shape[0]),
            "pyrecest_last_jump_fraction": float(filter_.last_jump_fraction),
            "pyrecest_last_goal_remap_fraction": float(filter_.last_goal_remap_fraction),
        }
        diagnostics.update(_goal_diagnostics(filter_, goals))
        diagnostics.update(_posterior_diagnostics(terminal_log_posterior, bin_centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal_log_posterior,
        )

    def _build_filter(
        self,
        bin_centers: np.ndarray,
        candidate_goals: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ):
        from pyrecest.filters import GoalConditionedReplayParticleFilter

        positions = _sample_rows(bin_centers, self.n_particles, rng)
        velocities = rng.normal(
            0.0,
            self.initial_velocity_sigma_cm_s,
            size=(self.n_particles, bin_centers.shape[1]),
        )
        goals = _sample_rows(candidate_goals, self.n_particles, rng)
        filter_ = GoalConditionedReplayParticleFilter(
            n_particles=self.n_particles,
            position_dim=bin_centers.shape[1],
            dt=dt,
            alpha=self.alpha,
            beta=self.beta,
            candidate_goals=candidate_goals,
        )
        filter_.set_state_components(
            positions=positions,
            velocities=velocities,
            goals=goals,
        )
        return filter_

    def _build_noise_distributions(self, position_dim: int):
        from pyrecest.distributions import GaussianDistribution

        zeros = np.zeros(position_dim, dtype=float)
        process_noise = GaussianDistribution(
            zeros,
            np.eye(position_dim) * self.process_noise_sigma_cm_s**2,
        )
        position_jump = GaussianDistribution(
            zeros,
            np.eye(position_dim) * self.position_jump_sigma_cm**2,
        )
        return process_noise, position_jump


def _event_seed(random_seed: int, emissions: LogEmissionTensor) -> int:
    event_offset = 0
    if emissions.times.size:
        event_offset = int(round(float(emissions.times[0]) * 1000.0))
    return int((random_seed + event_offset + 1009 * emissions.n_time) % (2**32 - 1))


def _coerce_candidate_goals(
    candidate_goals: np.ndarray | None,
    bin_centers: np.ndarray,
) -> np.ndarray:
    if candidate_goals is not None:
        goals = np.asarray(candidate_goals, dtype=float)
        invalid_shape = (
            goals.ndim != 2
            or goals.shape[1] != bin_centers.shape[1]
            or goals.shape[0] == 0
        )
        if invalid_shape:
            raise ValueError(
                "candidate_goals must have shape (n_goals, position_dim) "
                "and contain at least one row"
            )
        return goals
    return _farthest_point_subset(np.asarray(bin_centers, dtype=float), max_points=32)


def _farthest_point_subset(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.copy()
    selected = [int(np.argmin(points[:, 0] + points[:, 1]))]
    min_dist2 = np.full(points.shape[0], np.inf, dtype=float)
    for _ in range(1, max_points):
        last = points[selected[-1]]
        dist2 = np.sum((points - last[None, :]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected.append(int(np.argmax(min_dist2)))
    return points[np.asarray(selected, dtype=int)]


def _sample_rows(rows: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, rows.shape[0], size=n_samples)
    return np.asarray(rows[indices], dtype=float)


def _update_filter_from_grid_likelihood(
    filter_,
    log_likelihood: np.ndarray,
    bin_tree: cKDTree,
) -> float:
    particle_log_likelihood = _nearest_grid_values(
        filter_.position_particles,
        log_likelihood,
        bin_tree,
    )
    finite = np.isfinite(particle_log_likelihood)
    if not np.any(finite):
        raise ValueError("all particle log-likelihoods are non-finite")
    max_log = float(np.max(particle_log_likelihood[finite]))
    scaled = np.exp(np.clip(particle_log_likelihood - max_log, -745.0, 0.0))
    update_log = filter_.update_position_likelihood(
        lambda _positions: scaled,
        return_log_marginal=True,
    )
    return max_log + float(update_log)


def _nearest_grid_values(
    positions: np.ndarray,
    values: np.ndarray,
    bin_tree: cKDTree,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    indices = _nearest_bin_indices(positions, bin_tree)
    output = values[indices]
    if not np.all(np.isfinite(output)):
        finite_values = values[np.isfinite(values)]
        replacement = float(np.min(finite_values)) if finite_values.size else LOG_ZERO
        output = np.where(np.isfinite(output), output, replacement)
    return output


def _nearest_bin_indices(
    positions: np.ndarray,
    bin_tree: cKDTree,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    _, indices = bin_tree.query(positions, k=1)
    return np.asarray(indices, dtype=int)


def _particle_position_log_posterior(
    positions: np.ndarray,
    weights: np.ndarray,
    bin_centers: np.ndarray,
    bin_tree: cKDTree,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("particle weights must have positive total mass")
    weights = weights / total
    indices = _nearest_bin_indices(positions, bin_tree)
    masses = np.zeros(bin_centers.shape[0], dtype=float)
    np.add.at(masses, indices, weights)
    if not np.any(masses > 0.0):
        raise ValueError("particle posterior has no mass")
    log_posterior = np.full(bin_centers.shape[0], LOG_ZERO, dtype=float)
    positive = masses > 0.0
    log_posterior[positive] = np.log(masses[positive])
    return log_posterior - logsumexp(log_posterior)


def _goal_diagnostics(filter_, goals: np.ndarray) -> dict[str, float | int]:
    try:
        goal_weights = np.asarray(filter_.get_goal_posterior_weights(goals), dtype=float)
    except ValueError:
        return {}
    idx = int(np.argmax(goal_weights))
    return {
        "pyrecest_most_likely_goal_index": idx,
        "pyrecest_most_likely_goal_x": float(goals[idx, 0]),
        "pyrecest_most_likely_goal_y": (
            float(goals[idx, 1]) if goals.shape[1] > 1 else 0.0
        ),
        "pyrecest_most_likely_goal_probability": float(goal_weights[idx]),
    }
