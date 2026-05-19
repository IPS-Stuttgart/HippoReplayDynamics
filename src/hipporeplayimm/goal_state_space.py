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

    The latent goal is fixed within an event and has either a uniform prior over
    candidate goals or caller-provided prior weights. The initial latent
    position is likewise uniform unless prior weights are provided. For each
    goal, position follows a first-order drift-diffusion
    transition that moves the predicted position toward the goal by at most
    drift_speed_cm_s * dt before adding spatial Gaussian diffusion. Evidence,
    trajectory posteriors, and goal posteriors are computed by exact full-grid
    forward-backward recursions for every candidate goal and then marginalized.
    '''

    candidate_goals: np.ndarray | None = None
    goal_prior_weights: np.ndarray | None = None
    initial_position_prior_weights: np.ndarray | None = None
    initial_position_prior_direction_mode: str = 'all'
    reverse_terminal_position_prior_weights: np.ndarray | None = None
    transition_sigma_cm_sqrt_s: float = 85.0
    lateral_sigma_scale: float = 1.0
    diffusion_mixture_weight: float = 0.0
    drift_speed_cm_s: float = 400.0
    max_step_sigma: float = 4.0
    reset_probability: float = 0.0
    reset_initial_position_prior_weight: float = 0.0
    component_switch_probability: float = 0.0
    direction_mode: str = 'toward'
    terminal_goal_prior_sigma_cm: float = 0.0
    terminal_goal_prior_weight: float = 1.0
    initial_goal_prior_sigma_cm: float = 0.0
    initial_goal_prior_weight: float = 1.0
    toward_direction_prior_weight: float = 0.5
    reverse_terminal_position_prior_weight: float = 1.0
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
        lateral_sigma_scale = float(self.lateral_sigma_scale)
        if lateral_sigma_scale <= 0.0:
            raise ValueError('lateral_sigma_scale must be positive')
        diffusion_mixture_weight = float(self.diffusion_mixture_weight)
        if diffusion_mixture_weight < 0.0 or diffusion_mixture_weight > 1.0:
            raise ValueError('diffusion_mixture_weight must be in [0, 1]')
        if float(self.drift_speed_cm_s) < 0.0:
            raise ValueError('drift_speed_cm_s must be non-negative')
        if float(self.max_step_sigma) <= 0.0:
            raise ValueError('max_step_sigma must be positive')
        reset_probability = float(self.reset_probability)
        if reset_probability < 0.0 or reset_probability >= 1.0:
            raise ValueError('reset_probability must be in [0, 1)')
        reset_initial_position_prior_weight = float(self.reset_initial_position_prior_weight)
        if reset_initial_position_prior_weight < 0.0 or reset_initial_position_prior_weight > 1.0:
            raise ValueError('reset_initial_position_prior_weight must be in [0, 1]')
        component_switch_probability = float(self.component_switch_probability)
        if component_switch_probability < 0.0 or component_switch_probability > 1.0:
            raise ValueError('component_switch_probability must be in [0, 1]')
        direction_mode = _coerce_direction_mode(self.direction_mode)
        terminal_goal_prior_sigma_cm = float(self.terminal_goal_prior_sigma_cm)
        if terminal_goal_prior_sigma_cm < 0.0:
            raise ValueError('terminal_goal_prior_sigma_cm must be non-negative')
        terminal_goal_prior_weight = float(self.terminal_goal_prior_weight)
        if terminal_goal_prior_weight < 0.0 or terminal_goal_prior_weight > 1.0:
            raise ValueError('terminal_goal_prior_weight must be in [0, 1]')
        initial_goal_prior_sigma_cm = float(self.initial_goal_prior_sigma_cm)
        if initial_goal_prior_sigma_cm < 0.0:
            raise ValueError('initial_goal_prior_sigma_cm must be non-negative')
        initial_goal_prior_weight = float(self.initial_goal_prior_weight)
        if initial_goal_prior_weight < 0.0 or initial_goal_prior_weight > 1.0:
            raise ValueError('initial_goal_prior_weight must be in [0, 1]')
        toward_direction_prior_weight = float(self.toward_direction_prior_weight)
        if toward_direction_prior_weight < 0.0 or toward_direction_prior_weight > 1.0:
            raise ValueError('toward_direction_prior_weight must be in [0, 1]')
        reverse_terminal_position_prior_weight = float(self.reverse_terminal_position_prior_weight)
        if (
            reverse_terminal_position_prior_weight < 0.0
            or reverse_terminal_position_prior_weight > 1.0
        ):
            raise ValueError('reverse_terminal_position_prior_weight must be in [0, 1]')

        goals = _coerce_candidate_goals(self.candidate_goals, centers)
        goal_prior = _coerce_goal_prior_weights(self.goal_prior_weights, goals.shape[0])
        initial_position_prior = _coerce_initial_position_prior_weights(
            self.initial_position_prior_weights,
            centers.shape[0],
        )
        initial_position_prior_direction_mode = _coerce_position_prior_direction_mode(
            self.initial_position_prior_direction_mode
        )
        reverse_terminal_position_prior = _coerce_reverse_terminal_position_prior_weights(
            self.reverse_terminal_position_prior_weights,
            centers.shape[0],
        )
        transition_durations = transition_durations_s(emissions)
        transition_sigmas_cm = np.asarray(
            [
                _per_bin_sigma(self.transition_sigma_cm_sqrt_s, duration)
                for duration in transition_durations
            ],
            dtype=float,
        )
        drift_steps_cm = float(self.drift_speed_cm_s) * transition_durations
        component_goals, component_directions, component_goal_indices = _goal_components(
            goals,
            direction_mode,
        )
        transitions = tuple(
            _goal_transition_sequence(
                centers,
                goal,
                direction=float(direction),
                drift_steps_cm=drift_steps_cm,
                sigmas_cm=transition_sigmas_cm,
                lateral_sigma_scale=lateral_sigma_scale,
                diffusion_mixture_weight=diffusion_mixture_weight,
                max_step_sigma=float(self.max_step_sigma),
            )
            for goal, direction in zip(component_goals, component_directions, strict=True)
        )
        component_prior = _component_goal_prior(
            goal_prior,
            component_goal_indices,
            goals.shape[0],
            component_directions=component_directions,
            toward_direction_prior_weight=toward_direction_prior_weight,
        )
        terminal_goal_factors = _terminal_goal_factors(
            centers,
            component_goals,
            component_directions,
            sigma_cm=terminal_goal_prior_sigma_cm,
            weight=terminal_goal_prior_weight,
        )
        reverse_terminal_position_factors = _reverse_terminal_position_factors(
            reverse_terminal_position_prior,
            component_directions,
            provided=self.reverse_terminal_position_prior_weights is not None,
            weight=reverse_terminal_position_prior_weight,
        )
        initial_goal_factors = _initial_goal_factors(
            centers,
            component_goals,
            component_directions,
            sigma_cm=initial_goal_prior_sigma_cm,
            weight=initial_goal_prior_weight,
        )
        initial_position_factors = _directional_initial_position_factors(
            initial_position_prior,
            component_directions,
            provided=self.initial_position_prior_weights is not None,
            mode=initial_position_prior_direction_mode,
        )
        recursion_initial_position_prior = initial_position_prior
        if (
            self.initial_position_prior_weights is not None
            and initial_position_prior_direction_mode != 'all'
        ):
            recursion_initial_position_prior = np.full(
                centers.shape[0],
                1.0 / centers.shape[0],
                dtype=float,
            )
        logp, trajectory, goal_posteriors = _forward_backward_goal_mixture(
            emissions.log_likelihood,
            transitions,
            component_prior,
            recursion_initial_position_prior,
            reset_probability=reset_probability,
            reset_initial_position_prior_weight=reset_initial_position_prior_weight,
            component_switch_probability=component_switch_probability,
            component_goal_indices=component_goal_indices,
            n_goals=goals.shape[0],
            initial_factors=initial_goal_factors * initial_position_factors,
            terminal_factors=terminal_goal_factors * reverse_terminal_position_factors,
        )
        terminal = trajectory[-1]
        terminal_goal_posterior = goal_posteriors[-1]
        best_goal_index = int(np.argmax(terminal_goal_posterior))
        diagnostics: dict[str, float | int | str] = {
            'state_space_mode': 'goal',
            'state_space_observation_model': 'sorted-spike-poisson',
            'state_space_time_bin_s': float(emissions.dt),
            'state_space_trajectory_posterior': 1,
            'state_space_trajectory_time_bins': int(emissions.n_time),
            'mean_trajectory_posterior_entropy': _mean_entropy(trajectory),
            'goal_state_space_candidate_goals': int(goals.shape[0]),
            'goal_state_space_goal_prior': (
                'uniform' if self.goal_prior_weights is None else 'provided'
            ),
            'goal_state_space_goal_prior_entropy': _entropy_from_probabilities(goal_prior),
            'goal_state_space_goal_prior_max_probability': float(np.max(goal_prior)),
            'goal_state_space_initial_position_prior': (
                'uniform' if self.initial_position_prior_weights is None else 'provided'
            ),
            'goal_state_space_initial_position_prior_entropy': _entropy_from_probabilities(
                initial_position_prior
            ),
            'goal_state_space_initial_position_prior_max_probability': float(
                np.max(initial_position_prior)
            ),
            'goal_state_space_initial_position_prior_direction_mode': (
                initial_position_prior_direction_mode
            ),
            'goal_state_space_initial_position_prior_max_factor': float(
                np.max(initial_position_factors)
            ),
            'goal_state_space_reverse_terminal_position_prior': (
                'disabled'
                if (
                    self.reverse_terminal_position_prior_weights is None
                    or reverse_terminal_position_prior_weight <= 0.0
                )
                else 'provided'
            ),
            'goal_state_space_reverse_terminal_position_prior_entropy': (
                _entropy_from_probabilities(reverse_terminal_position_prior)
            ),
            'goal_state_space_reverse_terminal_position_prior_max_probability': float(
                np.max(reverse_terminal_position_prior)
            ),
            'goal_state_space_reverse_terminal_position_prior_weight': float(
                reverse_terminal_position_prior_weight
            ),
            'goal_state_space_reverse_terminal_position_prior_max_factor': float(
                np.max(reverse_terminal_position_factors)
            ),
            'goal_state_space_transition_sigma_cm_sqrt_s': float(self.transition_sigma_cm_sqrt_s),
            'goal_state_space_lateral_sigma_scale': lateral_sigma_scale,
            'goal_state_space_diffusion_mixture_weight': diffusion_mixture_weight,
            'goal_state_space_transition_sigma_cm': (
                float(np.median(transition_sigmas_cm))
                if transition_sigmas_cm.size
                else 0.0
            ),
            'goal_state_space_transition_durations': ','.join(
                f'{duration:.12g}' for duration in transition_durations
            ),
            'goal_state_space_drift_speed_cm_s': float(self.drift_speed_cm_s),
            'goal_state_space_drift_step_cm': (
                float(np.median(drift_steps_cm)) if drift_steps_cm.size else 0.0
            ),
            'goal_state_space_max_step_sigma': float(self.max_step_sigma),
            'goal_state_space_reset_probability': float(reset_probability),
            'goal_state_space_reset_initial_position_prior_weight': float(
                reset_initial_position_prior_weight
            ),
            'goal_state_space_reset_position_prior_max_probability': float(
                np.max(
                    _reset_position_prior(
                        initial_position_prior,
                        reset_initial_position_prior_weight,
                    )
                )
            ),
            'goal_state_space_component_switch_probability': component_switch_probability,
            'goal_state_space_direction_mode': direction_mode,
            'goal_state_space_components': int(len(transitions)),
            'goal_state_space_toward_direction_prior_weight': toward_direction_prior_weight,
            'goal_state_space_component_prior_entropy': _entropy_from_probabilities(
                component_prior
            ),
            'goal_state_space_terminal_goal_prior_sigma_cm': terminal_goal_prior_sigma_cm,
            'goal_state_space_terminal_goal_prior_weight': terminal_goal_prior_weight,
            'goal_state_space_terminal_goal_prior': (
                'disabled'
                if terminal_goal_prior_sigma_cm <= 0.0 or terminal_goal_prior_weight <= 0.0
                else 'provided'
            ),
            'goal_state_space_terminal_goal_prior_max_factor': float(
                np.max(terminal_goal_factors)
            ),
            'goal_state_space_initial_goal_prior_sigma_cm': initial_goal_prior_sigma_cm,
            'goal_state_space_initial_goal_prior_weight': initial_goal_prior_weight,
            'goal_state_space_initial_goal_prior': (
                'disabled'
                if initial_goal_prior_sigma_cm <= 0.0 or initial_goal_prior_weight <= 0.0
                else 'provided'
            ),
            'goal_state_space_initial_goal_prior_max_factor': float(
                np.max(initial_goal_factors)
            ),
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
    transitions: tuple[csr_matrix | tuple[csr_matrix, ...], ...],
    goal_prior_weights: np.ndarray | None = None,
    initial_position_prior_weights: np.ndarray | None = None,
    reset_probability: float = 0.0,
    reset_initial_position_prior_weight: float = 0.0,
    component_switch_probability: float = 0.0,
    component_goal_indices: np.ndarray | None = None,
    n_goals: int | None = None,
    initial_factors: np.ndarray | None = None,
    terminal_factors: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    '''Run exact forward-backward recursion on the augmented goal-position state.'''

    if not transitions:
        raise ValueError('at least one goal transition matrix is required')
    n_time, n_bins = log_likelihood.shape
    n_components = len(transitions)
    goal_index_map = _coerce_component_goal_indices(
        component_goal_indices,
        n_components,
        n_goals,
    )
    n_output_goals = int(goal_index_map.max()) + 1
    reset_probability = float(reset_probability)
    if reset_probability < 0.0 or reset_probability >= 1.0:
        raise ValueError('reset_probability must be in [0, 1)')
    reset_prior_weight = float(reset_initial_position_prior_weight)
    if reset_prior_weight < 0.0 or reset_prior_weight > 1.0:
        raise ValueError('reset_initial_position_prior_weight must be in [0, 1]')
    switch_probability = float(component_switch_probability)
    if switch_probability < 0.0 or switch_probability > 1.0:
        raise ValueError('component_switch_probability must be in [0, 1]')
    goal_prior = _coerce_goal_prior_weights(goal_prior_weights, n_components)
    initial_position_prior = _coerce_initial_position_prior_weights(
        initial_position_prior_weights,
        n_bins,
    )
    reset_position_prior = _reset_position_prior(initial_position_prior, reset_prior_weight)
    initial = _coerce_component_factors(initial_factors, n_components, n_bins, 'initial_factors')
    terminal = _coerce_terminal_factors(terminal_factors, n_components, n_bins)
    scaled, offsets = _scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_components, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = goal_prior[:, None] * initial_position_prior[None, :] * initial * scaled[0][None, :]
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError('first emission row has no finite likelihood mass')
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        predicted = np.empty_like(alpha)
        for component_index in range(n_components):
            transition = _component_transition(transitions, component_index, time_index - 1)
            moved = np.asarray(transition @ alpha[component_index], dtype=float)
            if reset_probability > 0.0:
                reset_mass = float(alpha[component_index].sum())
                moved = (
                    (1.0 - reset_probability) * moved
                    + reset_probability * reset_mass * reset_position_prior
                )
            predicted[component_index] = moved
        if switch_probability > 0.0:
            position_mass = predicted.sum(axis=0)
            predicted = (
                (1.0 - switch_probability) * predicted
                + switch_probability * goal_prior[:, None] * position_mass[None, :]
            )
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f'emission row {time_index} has no finite predicted mass')
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    terminal_scale = float(np.sum(filtered[-1] * terminal))
    if terminal_scale <= 0.0:
        raise ValueError('terminal prior has no finite predicted mass')
    logp += float(np.log(terminal_scale))

    smoothed = np.zeros_like(filtered)
    beta = terminal / terminal_scale
    gamma = filtered[-1] * beta
    total = float(gamma.sum())
    smoothed[-1] = gamma / total if total > 0.0 else filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        values = scaled[time_index][None, :] * beta
        switched_values = None
        if switch_probability > 0.0:
            switched_values = np.sum(goal_prior[:, None] * values, axis=0)
        beta_prev = np.empty_like(beta)
        for component_index in range(n_components):
            transition = _component_transition(transitions, component_index, time_index - 1)
            future = values[component_index]
            if switched_values is not None:
                future = (
                    (1.0 - switch_probability) * future
                    + switch_probability * switched_values
                )
            moved = np.asarray(transition.T @ future, dtype=float)
            if reset_probability > 0.0:
                reset_value = float(np.dot(reset_position_prior, future))
                moved = (1.0 - reset_probability) * moved + reset_probability * reset_value
            beta_prev[component_index] = moved
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    position_posteriors = smoothed.sum(axis=1)
    component_posteriors = smoothed.sum(axis=2)
    goal_posteriors = np.zeros((n_time, n_output_goals), dtype=float)
    for component_index, goal_index in enumerate(goal_index_map):
        goal_posteriors[:, int(goal_index)] += component_posteriors[:, component_index]
    row_sums = goal_posteriors.sum(axis=1, keepdims=True)
    goal_posteriors = np.divide(
        goal_posteriors,
        row_sums,
        out=np.full_like(goal_posteriors, 1.0 / n_output_goals),
        where=row_sums > 0.0,
    )
    return float(logp), _as_log_probs(position_posteriors), goal_posteriors


def _component_transition(
    transitions: tuple[csr_matrix | tuple[csr_matrix, ...], ...],
    component_index: int,
    transition_index: int,
) -> csr_matrix:
    transition = transitions[component_index]
    if isinstance(transition, tuple):
        return transition[transition_index]
    return transition


def _reset_position_prior(
    initial_position_prior: np.ndarray,
    reset_initial_position_prior_weight: float,
) -> np.ndarray:
    prior = np.asarray(initial_position_prior, dtype=float)
    if prior.ndim != 1 or prior.size == 0:
        raise ValueError('initial_position_prior must be a non-empty vector')
    weight = float(reset_initial_position_prior_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError('reset_initial_position_prior_weight must be in [0, 1]')
    if weight <= 0.0:
        return np.full(prior.size, 1.0 / prior.size, dtype=float)
    uniform = np.full(prior.size, 1.0 / prior.size, dtype=float)
    blended = (1.0 - weight) * uniform + weight * prior
    total = float(blended.sum())
    if total <= 0.0 or not np.all(np.isfinite(blended)):
        raise ValueError('reset position prior must contain finite positive mass')
    return blended / total


def _goal_transition_sequence(
    bin_centers: np.ndarray,
    goal: np.ndarray,
    *,
    direction: float,
    drift_steps_cm: np.ndarray,
    sigmas_cm: np.ndarray,
    max_step_sigma: float,
    lateral_sigma_scale: float = 1.0,
    diffusion_mixture_weight: float = 0.0,
) -> tuple[csr_matrix, ...]:
    cache: dict[tuple[float, float, float, float], csr_matrix] = {}
    transitions: list[csr_matrix] = []
    for drift_step_cm, sigma_cm in zip(drift_steps_cm, sigmas_cm, strict=True):
        key = (
            float(drift_step_cm) * float(direction),
            float(sigma_cm),
            float(lateral_sigma_scale),
            float(diffusion_mixture_weight),
        )
        transition = cache.get(key)
        if transition is None:
            transition = _goal_transition_matrix(
                bin_centers,
                goal,
                drift_step_cm=key[0],
                sigma_cm=key[1],
                lateral_sigma_scale=key[2],
                diffusion_mixture_weight=key[3],
                max_step_sigma=max_step_sigma,
            )
            cache[key] = transition
        transitions.append(transition)
    return tuple(transitions)


def _goal_transition_matrix(
    bin_centers: np.ndarray,
    goal: np.ndarray,
    *,
    drift_step_cm: float,
    sigma_cm: float,
    max_step_sigma: float,
    lateral_sigma_scale: float = 1.0,
    diffusion_mixture_weight: float = 0.0,
) -> csr_matrix:
    '''Build a column-stochastic drift-diffusion transition toward one goal.'''

    if sigma_cm <= 0.0:
        raise ValueError('sigma_cm must be positive')
    lateral_scale = float(lateral_sigma_scale)
    if lateral_scale <= 0.0:
        raise ValueError('lateral_sigma_scale must be positive')
    diffusion_weight = float(diffusion_mixture_weight)
    if diffusion_weight < 0.0 or diffusion_weight > 1.0:
        raise ValueError('diffusion_mixture_weight must be in [0, 1]')
    if max_step_sigma <= 0.0:
        raise ValueError('max_step_sigma must be positive')
    centers = np.asarray(bin_centers, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (centers.shape[1],):
        raise ValueError('goal must have one coordinate per position dimension')

    n_bins = centers.shape[0]
    sigma2 = float(sigma_cm) * float(sigma_cm)
    lateral_sigma2 = (float(sigma_cm) * lateral_scale) ** 2
    isotropic = centers.shape[1] == 1 or np.isclose(lateral_scale, 1.0)
    radius2 = (sigma_cm * max_step_sigma) ** 2
    max_mahalanobis2 = float(max_step_sigma) * float(max_step_sigma)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(centers):
        predicted = _goal_drift_prediction(center, goal, drift_step_cm)
        delta = centers - predicted[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        if isotropic:
            transition_metric2 = dist2 / sigma2
            keep = dist2 <= radius2
        else:
            transition_metric2 = _goal_axis_mahalanobis2(
                delta,
                center,
                goal,
                sigma2=sigma2,
                lateral_sigma2=lateral_sigma2,
            )
            keep = transition_metric2 <= max_mahalanobis2
        if not np.any(keep):
            keep[int(np.argmin(transition_metric2))] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * transition_metric2[dst])
        weights /= float(weights.sum())
        rows.extend(int(idx) for idx in dst)
        cols.extend([src] * len(dst))
        data.extend(float((1.0 - diffusion_weight) * value) for value in weights)
        if diffusion_weight > 0.0:
            diffusion_dist2 = np.sum((centers - center[None, :]) ** 2, axis=1)
            diffusion_keep = diffusion_dist2 <= radius2
            if not np.any(diffusion_keep):
                diffusion_keep[int(np.argmin(diffusion_dist2))] = True
            diffusion_dst = np.flatnonzero(diffusion_keep)
            diffusion_weights = np.exp(-0.5 * diffusion_dist2[diffusion_dst] / sigma2)
            diffusion_weights /= float(diffusion_weights.sum())
            rows.extend(int(idx) for idx in diffusion_dst)
            cols.extend([src] * len(diffusion_dst))
            data.extend(float(diffusion_weight * value) for value in diffusion_weights)
    return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))


def _goal_axis_mahalanobis2(
    delta: np.ndarray,
    position: np.ndarray,
    goal: np.ndarray,
    *,
    sigma2: float,
    lateral_sigma2: float,
) -> np.ndarray:
    axis = np.asarray(goal, dtype=float) - np.asarray(position, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= np.finfo(float).eps:
        return np.sum(delta * delta, axis=1) / sigma2
    unit_axis = axis / axis_norm
    parallel = np.asarray(delta, dtype=float) @ unit_axis
    dist2 = np.sum(delta * delta, axis=1)
    lateral2 = np.maximum(0.0, dist2 - parallel * parallel)
    return (parallel * parallel) / sigma2 + lateral2 / lateral_sigma2


def _goal_drift_prediction(position: np.ndarray, goal: np.ndarray, drift_step_cm: float) -> np.ndarray:
    vector = goal - position
    distance = float(np.linalg.norm(vector))
    if drift_step_cm == 0.0 or distance <= np.finfo(float).eps:
        return np.asarray(position, dtype=float)
    step = np.sign(float(drift_step_cm)) * min(abs(float(drift_step_cm)), distance)
    return np.asarray(position, dtype=float) + (step / distance) * vector


def _goal_components(
    goals: np.ndarray,
    direction_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if direction_mode == 'toward':
        directions = np.ones(goals.shape[0], dtype=float)
        indices = np.arange(goals.shape[0], dtype=int)
        return goals, directions, indices
    if direction_mode == 'away':
        directions = -np.ones(goals.shape[0], dtype=float)
        indices = np.arange(goals.shape[0], dtype=int)
        return goals, directions, indices
    if direction_mode == 'bidirectional':
        component_goals = np.repeat(goals, 2, axis=0)
        directions = np.tile(np.array([1.0, -1.0], dtype=float), goals.shape[0])
        indices = np.repeat(np.arange(goals.shape[0], dtype=int), 2)
        return component_goals, directions, indices
    raise ValueError("direction_mode must be 'toward', 'away', or 'bidirectional'")


def _terminal_goal_factors(
    bin_centers: np.ndarray,
    component_goals: np.ndarray,
    component_directions: np.ndarray,
    *,
    sigma_cm: float,
    weight: float = 1.0,
) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    goals = np.asarray(component_goals, dtype=float)
    directions = np.asarray(component_directions, dtype=float)
    factor_weight = float(weight)
    if factor_weight < 0.0 or factor_weight > 1.0:
        raise ValueError('weight must be in [0, 1]')
    if sigma_cm <= 0.0 or factor_weight <= 0.0:
        return np.ones((goals.shape[0], centers.shape[0]), dtype=float)
    if goals.ndim != 2 or goals.shape[1] != centers.shape[1]:
        raise ValueError('component_goals must have shape (n_components, position_dim)')
    if directions.shape != (goals.shape[0],):
        raise ValueError('component_directions must have shape (n_components,)')
    factors = np.ones((goals.shape[0], centers.shape[0]), dtype=float)
    toward = directions > 0.0
    for component_index in np.flatnonzero(toward):
        delta = centers - goals[component_index][None, :]
        distances2 = np.sum(delta * delta, axis=1)
        weights = np.exp(-0.5 * distances2 / (float(sigma_cm) * float(sigma_cm)))
        total = float(weights.sum())
        if total <= 0.0 or not np.all(np.isfinite(weights)):
            continue
        mean_one = weights * (centers.shape[0] / total)
        factors[component_index] = (1.0 - factor_weight) + factor_weight * mean_one
    return factors


def _initial_goal_factors(
    bin_centers: np.ndarray,
    component_goals: np.ndarray,
    component_directions: np.ndarray,
    *,
    sigma_cm: float,
    weight: float = 1.0,
) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    goals = np.asarray(component_goals, dtype=float)
    directions = np.asarray(component_directions, dtype=float)
    factor_weight = float(weight)
    if factor_weight < 0.0 or factor_weight > 1.0:
        raise ValueError('weight must be in [0, 1]')
    if sigma_cm <= 0.0 or factor_weight <= 0.0:
        return np.ones((goals.shape[0], centers.shape[0]), dtype=float)
    if goals.ndim != 2 or goals.shape[1] != centers.shape[1]:
        raise ValueError('component_goals must have shape (n_components, position_dim)')
    if directions.shape != (goals.shape[0],):
        raise ValueError('component_directions must have shape (n_components,)')
    factors = np.ones((goals.shape[0], centers.shape[0]), dtype=float)
    away = directions < 0.0
    for component_index in np.flatnonzero(away):
        delta = centers - goals[component_index][None, :]
        distances2 = np.sum(delta * delta, axis=1)
        weights = np.exp(-0.5 * distances2 / (float(sigma_cm) * float(sigma_cm)))
        total = float(weights.sum())
        if total <= 0.0 or not np.all(np.isfinite(weights)):
            continue
        mean_one = weights * (centers.shape[0] / total)
        factors[component_index] = (1.0 - factor_weight) + factor_weight * mean_one
    return factors


def _reverse_terminal_position_factors(
    position_prior: np.ndarray,
    component_directions: np.ndarray,
    *,
    provided: bool,
    weight: float = 1.0,
) -> np.ndarray:
    directions = np.asarray(component_directions, dtype=float)
    factor_weight = float(weight)
    if factor_weight < 0.0 or factor_weight > 1.0:
        raise ValueError('weight must be in [0, 1]')
    if directions.ndim != 1:
        raise ValueError('component_directions must have shape (n_components,)')
    prior = np.asarray(position_prior, dtype=float)
    if prior.ndim != 1:
        raise ValueError('position_prior must have shape (n_bins,)')
    factors = np.ones((directions.shape[0], prior.size), dtype=float)
    if not provided or factor_weight <= 0.0:
        return factors
    mean_one = prior * float(prior.size)
    away = directions < 0.0
    for component_index in np.flatnonzero(away):
        factors[component_index] = (1.0 - factor_weight) + factor_weight * mean_one
    return factors


def _directional_initial_position_factors(
    position_prior: np.ndarray,
    component_directions: np.ndarray,
    *,
    provided: bool,
    mode: str,
) -> np.ndarray:
    directions = np.asarray(component_directions, dtype=float)
    prior = np.asarray(position_prior, dtype=float)
    if directions.ndim != 1:
        raise ValueError('component_directions must have shape (n_components,)')
    if prior.ndim != 1:
        raise ValueError('position_prior must have shape (n_bins,)')
    factors = np.ones((directions.shape[0], prior.size), dtype=float)
    if not provided or mode == 'all':
        return factors
    if mode == 'toward':
        selected = np.flatnonzero(directions > 0.0)
    elif mode == 'away':
        selected = np.flatnonzero(directions < 0.0)
    else:
        raise ValueError("mode must be 'all', 'toward', or 'away'")
    mean_one = prior * float(prior.size)
    for component_index in selected:
        factors[component_index] = mean_one
    return factors


def _coerce_component_factors(
    factors: np.ndarray | None,
    n_components: int,
    n_bins: int,
    name: str,
) -> np.ndarray:
    if factors is None:
        return np.ones((n_components, n_bins), dtype=float)
    array = np.asarray(factors, dtype=float)
    if array.shape != (n_components, n_bins):
        raise ValueError(f'{name} must have shape (n_components, n_bins)')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    if np.any(array < 0.0):
        raise ValueError(f'{name} must be non-negative')
    return array


def _coerce_terminal_factors(
    terminal_factors: np.ndarray | None,
    n_components: int,
    n_bins: int,
) -> np.ndarray:
    return _coerce_component_factors(
        terminal_factors,
        n_components,
        n_bins,
        'terminal_factors',
    )


def _component_goal_prior(
    goal_prior: np.ndarray,
    component_goal_indices: np.ndarray,
    n_goals: int,
    *,
    component_directions: np.ndarray | None = None,
    toward_direction_prior_weight: float = 0.5,
) -> np.ndarray:
    indices = np.asarray(component_goal_indices, dtype=int)
    if component_directions is None:
        counts = np.bincount(indices, minlength=n_goals).astype(float)
        if np.any(counts <= 0.0):
            raise ValueError('each goal must have at least one component')
        return goal_prior[indices] / counts[indices]

    directions = np.asarray(component_directions, dtype=float)
    if directions.shape != indices.shape:
        raise ValueError('component_directions must match component_goal_indices')
    toward_weight = float(toward_direction_prior_weight)
    if toward_weight < 0.0 or toward_weight > 1.0:
        raise ValueError('toward_direction_prior_weight must be in [0, 1]')

    prior = np.zeros(indices.shape[0], dtype=float)
    for goal_index in range(int(n_goals)):
        goal_components = np.flatnonzero(indices == goal_index)
        if goal_components.size == 0:
            raise ValueError('each goal must have at least one component')
        toward = goal_components[directions[goal_components] > 0.0]
        away = goal_components[directions[goal_components] < 0.0]
        if toward.size and away.size:
            prior[toward] = float(goal_prior[goal_index]) * toward_weight / toward.size
            prior[away] = float(goal_prior[goal_index]) * (1.0 - toward_weight) / away.size
        else:
            prior[goal_components] = float(goal_prior[goal_index]) / goal_components.size
    return prior


def _coerce_direction_mode(direction_mode: str) -> str:
    mode = str(direction_mode).strip().lower()
    if mode not in {'toward', 'away', 'bidirectional'}:
        raise ValueError("direction_mode must be 'toward', 'away', or 'bidirectional'")
    return mode


def _coerce_position_prior_direction_mode(direction_mode: str) -> str:
    mode = str(direction_mode).strip().lower()
    if mode not in {'all', 'toward', 'away'}:
        raise ValueError(
            "initial_position_prior_direction_mode must be 'all', 'toward', or 'away'"
        )
    return mode


def _coerce_component_goal_indices(
    component_goal_indices: np.ndarray | None,
    n_components: int,
    n_goals: int | None,
) -> np.ndarray:
    if component_goal_indices is None:
        return np.arange(n_components, dtype=int)
    indices = np.asarray(component_goal_indices, dtype=int)
    if indices.shape != (n_components,):
        raise ValueError('component_goal_indices must have shape (n_components,)')
    if indices.size == 0 or int(indices.min()) < 0:
        raise ValueError('component_goal_indices must be non-negative')
    if n_goals is not None and int(indices.max()) >= int(n_goals):
        raise ValueError('component_goal_indices exceed n_goals')
    if n_goals is not None and len(np.unique(indices)) != int(n_goals):
        raise ValueError('component_goal_indices must include every goal')
    return indices


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


def _coerce_goal_prior_weights(goal_prior_weights: np.ndarray | None, n_goals: int) -> np.ndarray:
    if n_goals <= 0:
        raise ValueError('n_goals must be positive')
    if goal_prior_weights is None:
        return np.full(n_goals, 1.0 / n_goals, dtype=float)
    weights = np.asarray(goal_prior_weights, dtype=float)
    if weights.shape != (n_goals,):
        raise ValueError('goal_prior_weights must have shape (n_goals,)')
    if not np.all(np.isfinite(weights)):
        raise ValueError('goal_prior_weights must be finite')
    if np.any(weights < 0.0):
        raise ValueError('goal_prior_weights must be non-negative')
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError('goal_prior_weights must contain positive mass')
    return weights / total


def _coerce_initial_position_prior_weights(
    initial_position_prior_weights: np.ndarray | None,
    n_bins: int,
) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError('n_bins must be positive')
    if initial_position_prior_weights is None:
        return np.full(n_bins, 1.0 / n_bins, dtype=float)
    weights = np.asarray(initial_position_prior_weights, dtype=float)
    if weights.shape != (n_bins,):
        raise ValueError('initial_position_prior_weights must have shape (n_bins,)')
    if not np.all(np.isfinite(weights)):
        raise ValueError('initial_position_prior_weights must be finite')
    if np.any(weights < 0.0):
        raise ValueError('initial_position_prior_weights must be non-negative')
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError('initial_position_prior_weights must contain positive mass')
    return weights / total


def _coerce_reverse_terminal_position_prior_weights(
    reverse_terminal_position_prior_weights: np.ndarray | None,
    n_bins: int,
) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError('n_bins must be positive')
    if reverse_terminal_position_prior_weights is None:
        return np.full(n_bins, 1.0 / n_bins, dtype=float)
    weights = np.asarray(reverse_terminal_position_prior_weights, dtype=float)
    if weights.shape != (n_bins,):
        raise ValueError('reverse_terminal_position_prior_weights must have shape (n_bins,)')
    if not np.all(np.isfinite(weights)):
        raise ValueError('reverse_terminal_position_prior_weights must be finite')
    if np.any(weights < 0.0):
        raise ValueError('reverse_terminal_position_prior_weights must be non-negative')
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError('reverse_terminal_position_prior_weights must contain positive mass')
    return weights / total


def _farthest_point_subset(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError('bin_centers must contain at least one position')
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


def _entropy_from_probabilities(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    positive = probs > 0.0
    return float(-np.sum(probs[positive] * np.log(probs[positive])))
