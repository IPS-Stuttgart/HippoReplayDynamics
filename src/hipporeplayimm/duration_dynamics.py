"""Duration-aware replay dynamics helpers.

The replay emission builders use each time bin's own observation duration for
Poisson likelihoods. State-space dynamics, however, should use the elapsed time
between emission-bin centres. This module centralizes that transition-duration
logic so models can consume it natively instead of monkey-patching model
scorers at import time.
"""

from __future__ import annotations

import numpy as np


def transition_durations_s(emissions) -> np.ndarray:
    """Return one positive transition duration per adjacent emission pair."""

    n_transitions = max(int(emissions.n_time) - 1, 0)
    explicit = getattr(emissions, "transition_durations", None)
    if explicit is not None:
        return _check_transition_durations(explicit, n_transitions, "transition_durations")

    times = np.asarray(getattr(emissions, "times", []), dtype=float)
    if times.shape == (int(emissions.n_time),) and n_transitions:
        durations = np.diff(times)
        if np.all(np.isfinite(durations)) and np.all(durations > 0.0):
            return _check_transition_durations(durations, n_transitions, "times")

    return np.full(n_transitions, float(emissions.dt), dtype=float)


def attach_duration_metadata(emissions):
    """Attach validated transition durations to an emission tensor."""

    emissions.transition_durations = transition_durations_s(emissions)
    return emissions


def apply_duration_dynamics_patch() -> None:
    """Install backwards-compatible emission-metadata wrappers only.

    State-space duration dynamics are implemented directly in the model
    recursions. The small builder wrapper remains for modules that imported
    ``build_emissions`` before package initialization synchronized aliases.
    """

    import hipporeplayimm.encoding as encoding
    import hipporeplayimm.kd_reference as kd_reference

    _wrap_builder(encoding, "build_emissions")
    _wrap_builder(kd_reference, "build_kd_emissions")


def _wrap_builder(module, name: str) -> None:
    builder = getattr(module, name)
    if getattr(builder, "_duration_metadata_wrapped", False):
        return

    def wrapped(*args, _builder=builder, **kwargs):
        return attach_duration_metadata(_builder(*args, **kwargs))

    wrapped._duration_metadata_wrapped = True
    setattr(module, name, wrapped)


def _check_transition_durations(values, n_transitions: int, name: str) -> np.ndarray:
    durations = np.asarray(values, dtype=float)
    if durations.shape != (n_transitions,):
        raise ValueError(f"{name} must have shape {(n_transitions,)}, got {durations.shape}")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError(f"{name} must contain finite positive durations")
    return durations


def _ps(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    return max(
        float(sigma_cm_sqrt_s) * np.sqrt(max(float(dt_s), np.finfo(float).tiny)),
        np.finfo(float).eps,
    )


def _pss(sigma_cm_sqrt_s: float, durations_s: np.ndarray, fallback_dt_s: float) -> np.ndarray:
    durations = np.asarray(durations_s, dtype=float)
    if durations.size == 0:
        return np.empty(0, dtype=float)
    return np.asarray([_ps(sigma_cm_sqrt_s, duration) for duration in durations], dtype=float)


def _rep(sigma_cm_sqrt_s: float, durations_s: np.ndarray, fallback_dt_s: float) -> float:
    durations = np.asarray(durations_s, dtype=float)
    dt = float(np.median(durations)) if durations.size else float(fallback_dt_s)
    return _ps(sigma_cm_sqrt_s, dt)


def _decays(base_decay: float, durations_s: np.ndarray, reference_dt_s: float) -> np.ndarray:
    durations = np.asarray(durations_s, dtype=float)
    if durations.size == 0:
        return np.empty(0, dtype=float)
    reference = max(float(reference_dt_s), np.finfo(float).tiny)
    base = max(float(base_decay), np.finfo(float).tiny)
    return np.asarray([base ** (float(duration) / reference) for duration in durations], dtype=float)


def _scales(durations_s: np.ndarray) -> np.ndarray:
    durations = np.asarray(durations_s, dtype=float)
    scales = np.ones_like(durations, dtype=float)
    if durations.size > 1:
        scales[1:] = durations[1:] / durations[:-1]
    return scales
