'''Exact goal-conditioned state-space replay decoder.'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .duration_dynamics import transition_durations_s
from .encoding import LogEmissionTensor
from .evidence_reporting import EXACT_EVIDENCE_SUPPORT
from .models import EventScore, _posterior_diagnostics
from .state_space_utils import _as_log_probs, _mean_entropy, _per_bin_sigma, _scaled_emissions


@dataclass
class GoalStateSpaceReplayModel:
    '''Exact first-order state-space mixture over fixed candidate goals.

    The latent goal is fixed within an event and has a uniform prior over the
    candidate goals. For each goal, position follows a first-order drift-diffusion
    transition that moves the predicted position toward the goal by at most
    drift_speed_cm_s * dt before adding spatial Gaussian diffusion. Evidence,
    trajectory posteriors, and goal posteriors are computed by exact full-grid
    forward-backward recursions for every candidate goal and then marginalized.
    '''

    candidate_goals: np.ndarray | None = None
    transition_sigma_cm_sqrt_s: float = 85.0
    drift_speed_cm_s: float = 400.0
    max_step_sigma: float = 4.0
    name: str = 'sorted-spike-state-space-goal'

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError('emissions must contain at least one time bin')
        centers = np.asarray(bin_centers, dtype=float)
        if centers.ndim != 2:
            raise ValueError('bin_centers must have shape (n_bins, position_dim)')
        if emissions.n_bins != centers.shape[0]:
            raise ValueError('emissions.n_bins must match bin_centers rows')
        if float(self.transition_sigma_cm_sqrt_s) <= 0.0:
            raise ValueError('transition_sigma_cm_sqrt_s must be positive')
        if float(self.drift_speed_cm_s) < 0.0:
            raise ValueError('drift_speed_cm_s must be non-negative')
        if float(self.max_step_sigma) <= 0.0:
            raise ValueError('max_step_sigma must be positive')

        goals = _coerce_candidate_goals(self.candidate_goals, centers)
        transition_durations = transition_durations_s(emissions)
        transition_sigmas_cm = _goal_transition_sigmas_cm(
            self.transition_sigma_cm_sqrt_s,
            transition_durations,
        )
        drift_steps_cm = _goal_drift_steps_cm(self.drift_speed_cm_s, transition_durations)
        transitions = tuple(
            tuple(
                _goal_transition_matrix(
                    centers,
                    goal,
                    drift_step_cm=float(drift_steps_cm[transition_index]),
                    sigma_cm=float(transition_sigmas_cm[transition_index]),
                    max_step_sigma=self.max_step_sigma,
                )
                for transition_index in range(len(transition_durations))
            )
            for goal in goals
        )
        logp, trajectory, goal_posteriors = _forward_backward_goal_mixture(
            emissions.log_likelihood,
            transitions,
        )
        terminal = trajectory[-1]
        terminal_goal_posterior = goal_posteriors[-1]
        best_goal_index = int(np.argmax(terminal_goal_posterior))
        transition_sigma_cm = _representative_transition_parameter(
            transition_sigmas_cm,
            fallback=_per_bin_sigma(self.transition_sigma_cm_sqrt_s, float(emissions.dt)),
        )
        drift_step_cm = _representative_transition_parameter(
            drift_steps_cm,
            fallback=float(self.drift_speed_cm_s) * float(emissions.dt),
        )
        diagnostics: dict[str, float | int | str] = {
            'state_space_mode': 'goal',
            'state_space_observation_model': 'sorted-spike-poisson',
            'state_space_time_bin_s': float(emissions.dt),
            'state_space_trajectory_posterior': 1,
            'state_space_trajectory_time_bins': int(emissions.n_time),
            'mean_trajectory_posterior_entropy': _mean_entropy(trajectory),
            'goal_state_space_candidate_goals': int(goals.shape[0]),
            'goal_state_space_transition_sigma_cm_sqrt_s': float(self.transition_sigma_cm_sqrt_s),
            'goal_state_space_transition_sigma_cm': float(transition_sigma_cm),
            'goal_state_space_transition_sigma_cm_per_step': _format_float_series(
                transition_sigmas_cm
            ),
            'goal_state_space_transition_durations': _format_float_series(
                transition_durations
            ),
            'goal_state_space_drift_speed_cm_s': float(self.drift_speed_cm_s),
            'goal_state_space_drift_step_cm': float(drift_step_cm),
            'goal_state_space_drift_step_cm_per_step': _format_float_series(drift_steps_cm),
            'goal_state_space_max_step_sigma': float(self.max_step_sigma),
            'goal_state_space_evidence_support': EXACT_EVIDENCE_SUPPORT,
            'goal_state_space_most_likely_goal_index': best_goal_index,
            'goal_state_space_most_likely_goal_x': float(goals[best_goal_index, 0]),
            'goal_state_space_most_likely_goal_y': (
                float(goals[best_goal_index, 1]) if goals.shape[1] > 1 else 0.0
            ),
            'goal_state_space_most_likely_goal_probability': float(
                terminal_goal_posterior[best_goal_index]
            ),
            'goal_state_space_terminal_goal_entropy': _entropy_from_probabilities(
                terminal_goal_posterior
            ),
        }
        diagnostics.update(_posterior_diagnostics(terminal, centers))
        return EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def _forward_backward_goal_mixture(
    log_likelihood: np.ndarray,
    transitions: tuple[tuple[csr_matrix, ...], ...],
) -> tuple[float, np.ndarray, np.ndarray]:
    '''Run exact forward-backward recursion on the augmented goal-position state.'''

    if not transitions:
        raise ValueError('at least one goal transition sequence is required')
    n_time, n_bins = log_likelihood.shape
    n_goals = len(transitions)
    n_transitions = max(n_time - 1, 0)
    for goal_index, goal_transitions in enumerate(transitions):
        if len(goal_transitions) != n_transitions:
            raise ValueError(
                f'transitions[{goal_index}] must contain one matrix per transition'
            )
    scaled, offsets = _scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_goals, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = np.tile(scaled[0] / (n_bins * n_goals), (n_goals, 1))
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError('first emission row has no finite likelihood mass')
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        predicted = np.empty_like(alpha)
        for goal_index, goal_transitions in enumerate(transitions):
            predicted[goal_index] = np.asarray(
                goal_transitions[time_index - 1] @ alpha[goal_index],
                dtype=float,
            )
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f'emission row {time_index} has no finite predicted mass')
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_goals, n_bins), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        values = scaled[time_index][None, :] * beta
        beta_prev = np.empty_like(beta)
        for goal_index, goal_transitions in enumerate(transitions):
            beta_prev[goal_index] = np.asarray(
                goal_transitions[time_index - 1].T @ values[goal_index],
                dtype=float,
            )
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    position_posteriors = smoothed.sum(axis=1)
    goal_posteriors = smoothed.sum(axis=2)
    row_sums = goal_posteriors.sum(axis=1, keepdims=True)
    goal_posteriors = np.divide(
        goal_posteriors,
        row_sums,
        out=np.full_like(goal_posteriors, 1.0 / n_goals),
        where=row_sums > 0.0,
    )
    return float(logp), _as_log_probs(position_posteriors), goal_posteriors


def _goal_transition_matrix(
    bin_centers: np.ndarray,
    goal: np.ndarray,
    *,
    drift_step_cm: float,
    sigma_cm: float,
    max_step_sigma: float,
) -> csr_matrix:
    '''Build a column-stochastic drift-diffusion transition toward one goal.'''

    if sigma_cm <= 0.0:
        raise ValueError('sigma_cm must be positive')
    if drift_step_cm < 0.0:
        raise ValueError('drift_step_cm must be non-negative')
    if max_step_sigma <= 0.0:
        raise ValueError('max_step_sigma must be positive')
    centers = np.asarray(bin_centers, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (centers.shape[1],):
        raise ValueError('goal must have one coordinate per position dimension')

    n_bins = centers.shape[0]
    radius2 = (sigma_cm * max_step_sigma) ** 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(centers):
        predicted = _goal_drift_prediction(center, goal, drift_step_cm)
        delta = centers - predicted[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius2
        if not np.any(keep):
            keep[int(np.argmin(dist2))] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma_cm * sigma_cm))
        weights /= float(weights.sum())
        rows.extend(int(idx) for idx in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))


def _goal_drift_prediction(position: np.ndarray, goal: np.ndarray, drift_step_cm: float) -> np.ndarray:
    vector = goal - position
    distance = float(np.linalg.norm(vector))
    if drift_step_cm <= 0.0 or distance <= np.finfo(float).eps:
        return np.asarray(position, dtype=float)
    step = min(float(drift_step_cm), distance)
    return np.asarray(position, dtype=float) + (step / distance) * vector


def _goal_transition_sigmas_cm(
    transition_sigma_cm_sqrt_s: float,
    transition_durations_s: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _per_bin_sigma(transition_sigma_cm_sqrt_s, float(duration_s))
            for duration_s in transition_durations_s
        ],
        dtype=float,
    )


def _goal_drift_steps_cm(
    drift_speed_cm_s: float,
    transition_durations_s: np.ndarray,
) -> np.ndarray:
    return float(drift_speed_cm_s) * np.asarray(transition_durations_s, dtype=float)


def _representative_transition_parameter(values: np.ndarray, *, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(fallback)
    return float(np.median(arr))


def _format_float_series(values: np.ndarray) -> str:
    return ','.join(f'{float(value):.12g}' for value in np.asarray(values, dtype=float))


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
                'candidate_goals must have shape (n_goals, position_dim) '
                'and contain at least one row'
            )
        if not np.all(np.isfinite(goals)):
            raise ValueError('candidate_goals must be finite')
        return goals
    return _farthest_point_subset(np.asarray(bin_centers, dtype=float), max_points=32)


def _farthest_point_subset(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError('bin_centers must contain at least one position')
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


def _entropy_from_probabilities(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    positive = probs > 0.0
    return float(-np.sum(probs[positive] * np.log(probs[positive])))
