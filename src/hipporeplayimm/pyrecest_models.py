"""PyRecEst-backed replay models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .duration_dynamics import transition_durations_s
from .encoding import LogEmissionTensor
from .evidence_reporting import PYRECEST_PARTICLE_EVIDENCE_SUPPORT
from .models import EventScore, _posterior_diagnostics

PYRECEST_INSTALL_HINT = (
    "PyRecEst-backed replay models require the optional 'pyrecest' dependency. "
    "Install it from a checkout with `python -m pip install -e \".[pyrecest]\"` "
    "or from a package install with `python -m pip install hipporeplayimm[pyrecest]`."
)


def _missing_pyrecest_error() -> RuntimeError:
    return RuntimeError(PYRECEST_INSTALL_HINT)


def _is_missing_pyrecest_exception(exc: ModuleNotFoundError) -> bool:
    module_name = str(getattr(exc, "name", ""))
    return module_name == "pyrecest" or module_name.startswith("pyrecest.")


@dataclass(frozen=True)
class _PyRecEstReplayGridHelpers:
    build_grid_likelihood_lookup: object
    grid_log_likelihood_values: object
    grid_proposal_weights: object
    particle_position_log_posterior: object
    effective_sample_size_fraction: object


def _load_pyrecest_replay_grid_helpers() -> _PyRecEstReplayGridHelpers:
    """Load replay-grid helper functions from PyRecEst on demand.

    HippoReplayIMM keeps PyRecEst optional for non-PyRecEst models.  The
    PyRecEst-backed particle models nevertheless should not maintain private
    copies of PyRecEst helper logic; this lazy import preserves the existing
    optional-extra behavior while delegating reusable grid/proposal utilities to
    PyRecEst.
    """

    try:
        from pyrecest.filters.replay_grid_likelihood import (
            build_grid_likelihood_lookup,
            effective_sample_size_fraction,
            grid_log_likelihood_values,
            grid_proposal_weights,
            particle_position_log_posterior,
        )
    except ModuleNotFoundError as exc:
        if _is_missing_pyrecest_exception(exc):
            raise _missing_pyrecest_error() from exc
        raise
    return _PyRecEstReplayGridHelpers(
        build_grid_likelihood_lookup=build_grid_likelihood_lookup,
        grid_log_likelihood_values=grid_log_likelihood_values,
        grid_proposal_weights=grid_proposal_weights,
        particle_position_log_posterior=particle_position_log_posterior,
        effective_sample_size_fraction=effective_sample_size_fraction,
    )


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
    position_proposal_probability: float = 0.0
    position_proposal_ess_threshold: float | None = 0.5
    position_likelihood_interpolation: str = "linear"
    random_seed: int = 1
    name: str = "pyrecest-goal-particle"

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_particles, "n_particles")
        _validate_positive_float(self.initial_velocity_sigma_cm_s, "initial_velocity_sigma_cm_s")
        _validate_positive_float(self.process_noise_sigma_cm_s, "process_noise_sigma_cm_s")
        _validate_positive_float(self.position_jump_sigma_cm, "position_jump_sigma_cm")
        _validate_probability(self.jump_probability, "jump_probability")
        _validate_probability(self.goal_reset_probability, "goal_reset_probability")
        _validate_probability(self.position_proposal_probability, "position_proposal_probability")
        if self.position_proposal_ess_threshold is not None:
            _validate_probability(self.position_proposal_ess_threshold, "position_proposal_ess_threshold")
        if str(self.position_likelihood_interpolation).lower() not in {"nearest", "linear"}:
            raise ValueError("position_likelihood_interpolation must be 'nearest' or 'linear'")
        if hasattr(self, "mode_stickiness"):
            _validate_probability(getattr(self, "mode_stickiness"), "mode_stickiness")
        if hasattr(self, "jump_fraction"):
            _validate_probability(getattr(self, "jump_fraction"), "jump_fraction")
        for name in (
            "stationary_velocity_decay",
            "diffusion_velocity_decay",
            "momentum_velocity_decay",
            "jump_velocity_decay",
        ):
            if hasattr(self, name):
                _validate_nonnegative_float(getattr(self, name), name)

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        _validate_probability(
            self.position_proposal_probability,
            "position_proposal_probability",
        )
        if self.position_proposal_ess_threshold is not None:
            _validate_probability(
                self.position_proposal_ess_threshold,
                "position_proposal_ess_threshold",
            )

        seed = _event_seed(self.random_seed, emissions)
        np.random.seed(seed)

        grid_helpers = _load_pyrecest_replay_grid_helpers()
        bin_tree = cKDTree(bin_centers)
        likelihood_lookup = grid_helpers.build_grid_likelihood_lookup(
            bin_centers,
            self.position_likelihood_interpolation,
        )
        goals = _coerce_candidate_goals(self.candidate_goals, bin_centers)
        transition_durations = transition_durations_s(emissions)
        filter_dt = _representative_filter_dt(emissions, transition_durations)
        filter_ = self._build_filter(bin_centers, goals, filter_dt)

        logp = 0.0
        trajectory_log_posterior: list[np.ndarray] = []
        pre_update_ess_fractions: list[float] = []
        proposal_probabilities: list[float] = []
        for time_index in range(emissions.n_time):
            if time_index > 0:
                filter_.predict_replay(
                    dt=float(transition_durations[time_index - 1]),
                    use_semi_implicit_position_update=True,
                )
            proposal_probability, ess_fraction = _position_proposal_probability(
                filter_,
                self.position_proposal_probability,
                self.position_proposal_ess_threshold,
            )
            pre_update_ess_fractions.append(ess_fraction)
            proposal_probabilities.append(proposal_probability)
            logp += _update_filter_from_grid_likelihood(
                filter_,
                emissions.log_likelihood[time_index],
                bin_centers,
                bin_tree,
                likelihood_lookup,
                position_proposal_probability=proposal_probability,
            )
            trajectory_log_posterior.append(
                grid_helpers.particle_position_log_posterior(
                    filter_.position_particles,
                    np.asarray(filter_.filter_state.w, dtype=float),
                    bin_centers,
                    bin_tree,
                )
            )

        terminal_log_posterior = trajectory_log_posterior[-1]
        diagnostics = {
            "pyrecest_evidence_support": PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
            "pyrecest_particles": int(self.n_particles),
            "pyrecest_candidate_goals": int(goals.shape[0]),
            "pyrecest_time_bin_s": float(filter_dt),
            "pyrecest_transition_durations": _format_transition_durations(transition_durations),
            "pyrecest_position_likelihood_interpolation": likelihood_lookup.method,
            "pyrecest_position_proposal_probability": float(
                self.position_proposal_probability
            ),
            "pyrecest_position_proposal_ess_threshold": (
                "none"
                if self.position_proposal_ess_threshold is None
                else float(self.position_proposal_ess_threshold)
            ),
            "pyrecest_mean_pre_update_ess_fraction": float(
                np.mean(pre_update_ess_fractions)
            ),
            "pyrecest_mean_position_proposal_probability": float(
                np.mean(proposal_probabilities)
            ),
            "pyrecest_last_jump_fraction": float(filter_.last_jump_fraction),
            "pyrecest_last_goal_remap_fraction": float(filter_.last_goal_remap_fraction),
            "pyrecest_last_position_proposal_fraction": float(
                filter_.last_position_proposal_fraction
            ),
        }
        diagnostics.update(_goal_diagnostics(filter_, goals))
        diagnostics.update(_mode_diagnostics(filter_))
        diagnostics.update(_posterior_diagnostics(terminal_log_posterior, bin_centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal_log_posterior,
            trajectory_log_posterior=np.stack(trajectory_log_posterior, axis=0),
        )

    def _build_filter(
        self,
        bin_centers: np.ndarray,
        candidate_goals: np.ndarray,
        dt: float,
    ):
        try:
            from pyrecest.filters import GoalConditionedReplayParticleFilter
        except ModuleNotFoundError as exc:
            if _is_missing_pyrecest_exception(exc):
                raise _missing_pyrecest_error() from exc
            raise

        position_prior, velocity_prior = _initial_replay_priors(
            bin_centers,
            self.initial_velocity_sigma_cm_s,
        )
        process_noise, position_jump = self._build_noise_distributions(bin_centers.shape[1])
        return GoalConditionedReplayParticleFilter(
            n_particles=self.n_particles,
            position_dim=bin_centers.shape[1],
            dt=dt,
            alpha=self.alpha,
            beta=self.beta,
            initial_position_distribution=position_prior,
            initial_velocity_distribution=velocity_prior,
            candidate_goals=candidate_goals,
            process_noise=process_noise,
            position_jump_distribution=position_jump,
            jump_probability=self.jump_probability,
            goal_reset_probability=self.goal_reset_probability,
        )

    def _build_noise_distributions(self, position_dim: int):
        try:
            from pyrecest.distributions import GaussianDistribution
        except ModuleNotFoundError as exc:
            if _is_missing_pyrecest_exception(exc):
                raise _missing_pyrecest_error() from exc
            raise

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


@dataclass
class PyRecEstGoalParticleIMMModel(PyRecEstGoalParticleModel):
    """Goal-conditioned PyRecEst particle IMM replay scorer."""

    mode_stickiness: float = 0.95
    stationary_velocity_decay: float = 0.0
    diffusion_velocity_decay: float = 0.0
    momentum_velocity_decay: float = 0.95
    jump_fraction: float = 0.9
    jump_velocity_decay: float = 0.25
    name: str = "pyrecest-goal-particle-imm"

    def _build_filter(
        self,
        bin_centers: np.ndarray,
        candidate_goals: np.ndarray,
        dt: float,
    ):
        try:
            from pyrecest.filters import GoalConditionedReplayParticleIMMFilter
        except ModuleNotFoundError as exc:
            if _is_missing_pyrecest_exception(exc):
                raise _missing_pyrecest_error() from exc
            raise

        position_prior, velocity_prior = _initial_replay_priors(
            bin_centers,
            self.initial_velocity_sigma_cm_s,
        )
        process_noise, position_jump = self._build_noise_distributions(bin_centers.shape[1])
        return GoalConditionedReplayParticleIMMFilter(
            n_particles=self.n_particles,
            position_dim=bin_centers.shape[1],
            dt=dt,
            alpha=self.alpha,
            beta=self.beta,
            initial_position_distribution=position_prior,
            initial_velocity_distribution=velocity_prior,
            candidate_goals=candidate_goals,
            process_noise=process_noise,
            position_jump_distribution=position_jump,
            jump_probability=self.jump_probability,
            goal_reset_probability=self.goal_reset_probability,
            mode_stickiness=self.mode_stickiness,
            stationary_velocity_decay=self.stationary_velocity_decay,
            diffusion_velocity_decay=self.diffusion_velocity_decay,
            momentum_velocity_decay=self.momentum_velocity_decay,
            jump_fraction=self.jump_fraction,
            jump_velocity_decay=self.jump_velocity_decay,
        )


def _initial_replay_priors(
    bin_centers: np.ndarray,
    initial_velocity_sigma_cm_s: float,
):
    try:
        from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
    except ModuleNotFoundError as exc:
        if _is_missing_pyrecest_exception(exc):
            raise _missing_pyrecest_error() from exc
        raise

    bin_centers = np.asarray(bin_centers, dtype=float)
    position_dim = bin_centers.shape[1]
    position_prior = LinearDiracDistribution(bin_centers)
    velocity_prior = GaussianDistribution(
        np.zeros(position_dim, dtype=float),
        np.eye(position_dim) * float(initial_velocity_sigma_cm_s) ** 2,
    )
    return position_prior, velocity_prior


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
    if points.shape[0] == 0:
        raise ValueError("bin_centers must contain at least one position")
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


def _update_filter_from_grid_likelihood(
    filter_,
    log_likelihood: np.ndarray,
    bin_centers: np.ndarray,
    bin_tree: cKDTree,
    likelihood_lookup: object | None = None,
    *,
    position_proposal_probability: float = 0.0,
) -> float:
    grid_helpers = _load_pyrecest_replay_grid_helpers()
    if likelihood_lookup is None:
        likelihood_lookup = grid_helpers.build_grid_likelihood_lookup(bin_centers, "linear")

    def log_likelihood_at(positions: np.ndarray) -> np.ndarray:
        return grid_helpers.grid_log_likelihood_values(
            positions,
            log_likelihood,
            bin_tree,
            likelihood_lookup,
        )

    particle_log_likelihood = log_likelihood_at(filter_.position_particles)
    finite = np.isfinite(particle_log_likelihood)

    if position_proposal_probability > 0.0:
        finite_grid_values = np.asarray(log_likelihood, dtype=float)[
            np.isfinite(log_likelihood)
        ]
        if finite_grid_values.size == 0:
            raise ValueError("all grid log-likelihoods are non-finite")
        if np.any(finite):
            max_log = float(
                max(np.max(particle_log_likelihood[finite]), np.max(finite_grid_values))
            )
        else:
            max_log = float(np.max(finite_grid_values))
    else:
        if not np.any(finite):
            raise ValueError("all particle log-likelihoods are non-finite")
        max_log = float(np.max(particle_log_likelihood[finite]))

    def scaled_likelihood(positions: np.ndarray) -> np.ndarray:
        position_log_likelihood = log_likelihood_at(positions)
        return np.exp(np.clip(position_log_likelihood - max_log, -745.0, 0.0))

    if position_proposal_probability > 0.0:
        update_log = filter_.update_position_likelihood_with_proposal(
            scaled_likelihood,
            position_proposal=bin_centers,
            proposal_weights=grid_helpers.grid_proposal_weights(log_likelihood),
            proposal_probability=position_proposal_probability,
            return_log_marginal=True,
        )
    else:
        update_log = filter_.update_position_likelihood(
            scaled_likelihood,
            return_log_marginal=True,
        )
    return max_log + float(update_log)


def _position_proposal_probability(
    filter_,
    base_probability: float,
    ess_threshold: float | None,
) -> tuple[float, float]:
    grid_helpers = _load_pyrecest_replay_grid_helpers()
    ess_fraction = grid_helpers.effective_sample_size_fraction(
        np.asarray(filter_.filter_state.w, dtype=float)
    )
    if base_probability <= 0.0:
        return 0.0, ess_fraction
    if base_probability >= 1.0:
        return float(base_probability), ess_fraction
    if ess_threshold is None:
        return float(base_probability), ess_fraction
    if ess_fraction < float(ess_threshold):
        return float(base_probability), ess_fraction
    return 0.0, ess_fraction


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


def _mode_diagnostics(filter_) -> dict[str, float | str]:
    if not hasattr(filter_, "mode_probabilities"):
        return {}
    probabilities = np.asarray(filter_.mode_probabilities, dtype=float)
    names = tuple(getattr(filter_, "mode_names", ()))
    diagnostics: dict[str, float | str] = {
        f"pyrecest_mode_{name}_probability": float(probability)
        for name, probability in zip(names, probabilities, strict=False)
    }
    if hasattr(filter_, "most_likely_mode"):
        diagnostics["pyrecest_most_likely_mode"] = str(filter_.most_likely_mode())
    if hasattr(filter_, "last_mode_transition_fraction"):
        diagnostics["pyrecest_last_mode_transition_fraction"] = float(
            filter_.last_mode_transition_fraction
        )
    return diagnostics


def _representative_filter_dt(
    emissions: LogEmissionTensor,
    transition_durations: np.ndarray,
) -> float:
    """Return the positive scalar dt needed to initialize a PyRecEst filter.

    Prediction uses the per-step transition durations.  The scalar stored on the
    PyRecEst filter is only the fallback used by that library when a prediction
    call does not pass an explicit dt.
    """

    base_dt = getattr(emissions.dt, "base", emissions.dt)
    dt = float(base_dt)
    if not np.isfinite(dt) or dt <= 0.0:
        if transition_durations.size == 0:
            raise ValueError("emissions.dt must be finite and positive")
        dt = float(np.median(transition_durations))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("PyRecEst filter dt must be finite and positive")
    return dt


def _format_transition_durations(transition_durations: np.ndarray) -> str:
    return ",".join(f"{float(duration):.12g}" for duration in transition_durations)


def _validate_probability(probability: float, name: str) -> None:
    value = float(probability)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _validate_positive_int(value: int, name: str) -> None:
    if int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_positive_float(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative_float(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
