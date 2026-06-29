"""Runtime patch for copied emission time coordinates."""

from __future__ import annotations

import numpy as np

from .encoding import LogEmissionTensor

_PATCH_MARKER = "_time_order_patch_wrapped"


def apply_reverse_emission_time_patch() -> None:
    """Keep copied time coordinates increasing while reversing observation rows."""

    from . import result_improvement_extensions as improved
    from . import reverse_models

    improved_copy = getattr(improved, "copy_emissions_with_log_likelihood", None)
    reverse_copy = getattr(reverse_models, "reverse_emissions", None)
    improved_patched = getattr(improved, "_time_order_patch_applied", False) and getattr(improved_copy, _PATCH_MARKER, False)
    reverse_models_patched = getattr(reverse_models, "_time_order_patch_applied", False) and getattr(reverse_copy, _PATCH_MARKER, False)
    if improved_patched and reverse_models_patched:
        return

    def copy_emissions_with_log_likelihood(
        emissions: LogEmissionTensor,
        log_likelihood: np.ndarray,
        *,
        reverse_time: bool = False,
    ) -> LogEmissionTensor:
        likelihood = np.asarray(log_likelihood, dtype=float)
        counts = np.asarray(emissions.spike_counts)
        if reverse_time:
            likelihood = likelihood[::-1].copy()
            counts = counts[::-1].copy()
        return LogEmissionTensor(
            log_likelihood=likelihood.copy(),
            spike_counts=counts.copy(),
            times=_time_vector(
                emissions,
                reverse_time=reverse_time,
            ),
            dt=emissions.dt,
            cell_ids=np.asarray(emissions.cell_ids).copy(),
            n_spikes=int(emissions.n_spikes),
            bin_durations=_duration_vector(
                getattr(emissions, "bin_durations", None),
                reverse_time=reverse_time,
                expected_length=emissions.n_time,
                name="bin_durations",
            ),
            transition_durations=_transition_duration_vector(
                emissions,
                reverse_time=reverse_time,
            ),
            metadata=dict(getattr(emissions, "metadata", {}) or {}),
        )

    def reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
        return copy_emissions_with_log_likelihood(
            emissions,
            np.asarray(emissions.log_likelihood, dtype=float),
            reverse_time=True,
        )

    setattr(copy_emissions_with_log_likelihood, _PATCH_MARKER, True)
    setattr(reverse_emissions, _PATCH_MARKER, True)
    improved.copy_emissions_with_log_likelihood = copy_emissions_with_log_likelihood
    reverse_models.reverse_emissions = reverse_emissions
    improved._time_order_patch_applied = True
    reverse_models._time_order_patch_applied = True


def _time_vector(
    emissions: LogEmissionTensor,
    *,
    reverse_time: bool,
) -> np.ndarray:
    times = np.asarray(emissions.times, dtype=float)
    if not reverse_time or times.shape == (0,):
        return times.copy()
    if times.shape != (emissions.n_time,):
        return times.copy()
    return (float(times[-1]) - times[::-1] + float(times[0])).copy()


def _duration_vector(
    values: np.ndarray | None,
    *,
    reverse_time: bool,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise ValueError(f"{name} must contain {expected_length} values; got shape {array.shape}")
    return array[::-1].copy() if reverse_time else array.copy()


def _transition_duration_vector(
    emissions: LogEmissionTensor,
    *,
    reverse_time: bool,
) -> np.ndarray | None:
    values = getattr(emissions, "transition_durations", None)
    if values is None:
        values = _transition_durations_from_times(emissions)
    return _duration_vector(
        values,
        reverse_time=reverse_time,
        expected_length=max(emissions.n_time - 1, 0),
        name="transition_durations",
    )


def _transition_durations_from_times(emissions: LogEmissionTensor) -> np.ndarray | None:
    if emissions.n_time <= 1:
        return np.empty(0, dtype=float)
    times = np.asarray(emissions.times, dtype=float)
    if times.shape == (0,):
        return np.full(emissions.n_time - 1, float(emissions.dt), dtype=float)
    if times.shape != (emissions.n_time,):
        return None
    durations = np.diff(times)
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("times must be strictly increasing when transition_durations is missing")
    return durations
