"""Make duration-aware IMM switching a continuous-time Markov semigroup.

The historical duration-aware implementation used ``exp(-dt / tau)`` as the
*final* same-mode probability. That is only the probability that no switch
occurred. It omits paths that leave a mode and return within the interval,
forces the diagonal toward zero for long bins, and makes ``P(t + s)`` differ
from ``P(t) @ P(s)``.

This runtime patch preserves the existing per-step no-switch diagnostics while
embedding each model's conditional switch-destination pattern in a continuous-
time Markov generator with mean dwell time ``tau``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
from scipy.linalg import expm

_PATCHED_FLAG = "_continuous_time_imm_transition_patch_applied"
_TRANSITION_WRAPPER_FLAG = "_continuous_time_imm_transition_wrapper"
_DIAGNOSTIC_WRAPPER_FLAG = "_continuous_time_imm_diagnostic_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _continuous_time_mode_transition_matrix(
    base_transition: np.ndarray,
    duration_s: float,
    mean_dwell_s: float,
) -> np.ndarray:
    """Embed conditional IMM switch destinations in continuous time."""

    transition = np.asarray(base_transition, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1] or not transition.size:
        raise ValueError("base_transition must be a nonempty square matrix")
    if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
        raise ValueError("base_transition must contain finite nonnegative probabilities")
    if not np.allclose(transition.sum(axis=1), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("base_transition rows must sum to 1")

    duration = float(duration_s)
    dwell = float(mean_dwell_s)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and nonnegative")
    if not np.isfinite(dwell) or dwell <= 0.0:
        raise ValueError("mean_dwell_s must be finite and positive")

    n_modes = transition.shape[0]
    if n_modes == 1 or duration == 0.0:
        return np.eye(n_modes, dtype=float)

    destinations = transition.copy()
    np.fill_diagonal(destinations, 0.0)
    switch_mass = destinations.sum(axis=1)
    for mode_index, mass in enumerate(switch_mass):
        if mass > 0.0:
            destinations[mode_index] /= mass
        else:
            destinations[mode_index].fill(1.0 / (n_modes - 1))
            destinations[mode_index, mode_index] = 0.0

    generator = destinations / dwell
    np.fill_diagonal(generator, -1.0 / dwell)
    result = np.asarray(expm(generator * duration), dtype=float)
    tolerance = 256.0 * np.finfo(float).eps
    if not np.all(np.isfinite(result)) or np.any(result < -tolerance):
        raise ValueError("continuous-time mode transition produced invalid probabilities")
    result = np.maximum(result, 0.0)
    result /= result.sum(axis=1, keepdims=True)
    return result


def _positional_or_keyword(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    position: int,
    name: str,
) -> Any:
    if len(args) > position:
        return args[position]
    try:
        return kwargs[name]
    except KeyError as exc:
        raise TypeError(f"missing required argument: {name}") from exc


def _wrap_mode_transition_sequence(
    helper: Callable[..., list[np.ndarray]],
    *,
    tau_position: int,
) -> Callable[..., list[np.ndarray]]:
    """Wrap a symmetric duration-aware transition-sequence helper."""

    if getattr(helper, _TRANSITION_WRAPPER_FLAG, False):
        return helper

    @wraps(helper)
    def mode_transition_matrices(*args, **kwargs):
        legacy = helper(*args, **kwargs)
        tau_s = float(
            _positional_or_keyword(
                args,
                kwargs,
                tau_position,
                "imm_switch_tau_s",
            )
        )
        durations = np.asarray(
            _positional_or_keyword(
                args,
                kwargs,
                tau_position + 1,
                "durations",
            ),
            dtype=float,
        )
        if tau_s == 0.0:
            return legacy
        if len(legacy) != int(durations.size):
            raise ValueError("mode transitions must contain one matrix per transition duration")
        return [
            _continuous_time_mode_transition_matrix(matrix, float(duration), tau_s)
            for matrix, duration in zip(legacy, durations, strict=True)
        ]

    setattr(mode_transition_matrices, _TRANSITION_WRAPPER_FLAG, True)
    setattr(mode_transition_matrices, _ORIGINAL_ATTR, helper)
    return mode_transition_matrices


def _wrap_trajectory_mode_transition_sequence(
    helper: Callable[..., list[np.ndarray]],
    base_helper: Callable[[Any, float], np.ndarray],
) -> Callable[..., list[np.ndarray]]:
    """Embed one duration-invariant trajectory-IMM routing pattern.

    ``trajectory_imm_momentum_switch_probability`` is a legacy absolute
    per-step probability defined together with the configured reference
    stickiness. Its ratio to the other off-diagonal probabilities defines the
    conditional destination distribution. Rebuilding that ratio from
    ``exp(-duration / tau)`` would make the generator duration-dependent and
    violate the semigroup property.
    """

    if getattr(helper, _TRANSITION_WRAPPER_FLAG, False):
        return helper

    @wraps(helper)
    def trajectory_mode_transition_matrices(*args, **kwargs):
        config = _positional_or_keyword(args, kwargs, 0, "config")
        stickiness = float(_positional_or_keyword(args, kwargs, 1, "stickiness"))
        durations = np.asarray(
            _positional_or_keyword(args, kwargs, 2, "durations"),
            dtype=float,
        )
        tau_s = float(getattr(config, "imm_switch_tau_s", 0.0))
        if tau_s == 0.0:
            return helper(*args, **kwargs)
        if not np.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("imm_switch_tau_s must be finite and positive")
        if durations.ndim != 1:
            raise ValueError("durations must be one-dimensional")
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError("transition durations must be finite and positive")

        reference_transition = base_helper(config, stickiness)
        return [
            _continuous_time_mode_transition_matrix(
                reference_transition,
                float(duration),
                tau_s,
            )
            for duration in durations
        ]

    setattr(trajectory_mode_transition_matrices, _TRANSITION_WRAPPER_FLAG, True)
    setattr(trajectory_mode_transition_matrices, _ORIGINAL_ATTR, helper)
    return trajectory_mode_transition_matrices


def _format_survival(module: Any, durations: np.ndarray, tau_s: float) -> str:
    values = np.exp(-np.asarray(durations, dtype=float) / float(tau_s))
    formatter = getattr(module, "_format_float_series", None)
    if callable(formatter):
        return str(formatter(values))
    return ",".join(f"{float(value):.12g}" for value in values)


def _wrap_displacement_diagnostics(helper: Callable[..., Any], module: Any) -> Callable[..., Any]:
    if getattr(helper, _DIAGNOSTIC_WRAPPER_FLAG, False):
        return helper

    @wraps(helper)
    def score_displacement_imm_exact(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory: bool = True,
    ):
        result = helper(
            emissions,
            bin_centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        tau_s = float(getattr(config, "imm_switch_tau_s", 0.0))
        if tau_s == 0.0:
            return result
        logp, trajectory, terminal, mode_post, displacement_post, diagnostics = result
        diagnostics = dict(diagnostics)
        key = "state_space_displacement_imm_mode_stickiness_per_step"
        diagnostics["state_space_displacement_imm_mode_self_transition_probability_per_step"] = diagnostics[key]
        diagnostics[key] = _format_survival(module, np.asarray(transition_durations_s), tau_s)
        return logp, trajectory, terminal, mode_post, displacement_post, diagnostics

    setattr(score_displacement_imm_exact, _DIAGNOSTIC_WRAPPER_FLAG, True)
    setattr(score_displacement_imm_exact, _ORIGINAL_ATTR, helper)
    return score_displacement_imm_exact


def _wrap_trajectory_diagnostics(helper: Callable[..., Any], module: Any) -> Callable[..., Any]:
    if getattr(helper, _DIAGNOSTIC_WRAPPER_FLAG, False):
        return helper

    @wraps(helper)
    def score_trajectory_imm_exact_sparse(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory: bool = True,
    ):
        result = helper(
            emissions,
            bin_centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        tau_s = float(getattr(config, "imm_switch_tau_s", 0.0))
        if tau_s == 0.0:
            return result
        logp, trajectory, terminal, mode_post, diagnostics = result
        diagnostics = dict(diagnostics)
        key = "state_space_trajectory_imm_mode_stickiness_per_step"
        diagnostics["state_space_trajectory_imm_mode_self_transition_probability_per_step"] = diagnostics[key]
        diagnostics[key] = _format_survival(module, np.asarray(transition_durations_s), tau_s)
        return logp, trajectory, terminal, mode_post, diagnostics

    setattr(score_trajectory_imm_exact_sparse, _DIAGNOSTIC_WRAPPER_FLAG, True)
    setattr(score_trajectory_imm_exact_sparse, _ORIGINAL_ATTR, helper)
    return score_trajectory_imm_exact_sparse


def _synchronize_aliases(previous: Any, active: Any) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        for name, value in list(vars(module).items()):
            if value is previous:
                setattr(module, name, active)


def apply_continuous_time_imm_transition_patch() -> None:
    """Install semigroup-consistent duration-aware IMM transitions."""

    from . import duration_occupancy
    from . import state_space_displacement_imm as displacement_imm
    from . import state_space_trajectory_imm as trajectory_imm

    if getattr(duration_occupancy, _PATCHED_FLAG, False):
        return

    previous_duration = duration_occupancy._mode_transition_matrices
    active_duration = _wrap_mode_transition_sequence(previous_duration, tau_position=3)
    duration_occupancy._mode_transition_matrices = active_duration
    _synchronize_aliases(previous_duration, active_duration)

    previous_displacement = displacement_imm._mode_transition_matrices
    active_displacement = _wrap_mode_transition_sequence(previous_displacement, tau_position=2)
    displacement_imm._mode_transition_matrices = active_displacement
    _synchronize_aliases(previous_displacement, active_displacement)

    previous_trajectory = trajectory_imm._trajectory_imm_mode_transition_matrices
    active_trajectory = _wrap_trajectory_mode_transition_sequence(
        previous_trajectory,
        trajectory_imm._trajectory_imm_mode_transition_matrix,
    )
    trajectory_imm._trajectory_imm_mode_transition_matrices = active_trajectory
    _synchronize_aliases(previous_trajectory, active_trajectory)

    previous_displacement_score = displacement_imm._score_displacement_imm_exact
    active_displacement_score = _wrap_displacement_diagnostics(
        previous_displacement_score,
        displacement_imm,
    )
    displacement_imm._score_displacement_imm_exact = active_displacement_score
    _synchronize_aliases(previous_displacement_score, active_displacement_score)

    previous_trajectory_score = trajectory_imm._score_trajectory_imm_exact_sparse
    active_trajectory_score = _wrap_trajectory_diagnostics(
        previous_trajectory_score,
        trajectory_imm,
    )
    trajectory_imm._score_trajectory_imm_exact_sparse = active_trajectory_score
    _synchronize_aliases(previous_trajectory_score, active_trajectory_score)

    setattr(duration_occupancy, _PATCHED_FLAG, True)


__all__ = [
    "_continuous_time_mode_transition_matrix",
    "apply_continuous_time_imm_transition_patch",
]
