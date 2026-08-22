"""Runtime patches for copied emission time coordinates and epoch intervals."""

from __future__ import annotations

import sys

import numpy as np

from .encoding import LogEmissionTensor

_PATCH_MARKER = "_time_order_patch_wrapped"
_DURATION_TIMESTAMP_PATCH_MARKER = "_duration_timestamp_validation_patch_wrapped"


def apply_reverse_emission_time_patch() -> None:
    """Keep copied time coordinates increasing while reversing observation rows."""

    from .data_interval_validation import apply_data_interval_validation_patch

    apply_data_interval_validation_patch()
    _apply_duration_timestamp_validation_patch()

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
        from . import encoding

        return encoding.LogEmissionTensor(
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


def _apply_duration_timestamp_validation_patch() -> None:
    """Reject malformed present timestamps instead of silently using scalar ``dt``."""

    from . import duration_dynamics

    resolver = duration_dynamics.transition_durations_s
    if getattr(resolver, _DURATION_TIMESTAMP_PATCH_MARKER, False):
        original = getattr(resolver, "__hipporeplayimm_original__", None)
        if original is not None:
            _synchronize_duration_resolver_aliases(original, resolver)
        return

    original = resolver

    def transition_durations_s(emissions) -> np.ndarray:
        if getattr(emissions, "transition_durations", None) is not None:
            return original(emissions)
        if duration_dynamics._dur_from_dt(emissions.dt) is not None:
            return original(emissions)

        raw_times = np.asarray(getattr(emissions, "times", []))
        if raw_times.shape == (0,):
            return original(emissions)
        if raw_times.shape != (int(emissions.n_time),):
            raise ValueError("times must contain one value per emission row")
        times = _timestamp_array_preserving_extended_precision(raw_times)
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite")
        return _strictly_increasing_transition_durations(times)

    setattr(transition_durations_s, _DURATION_TIMESTAMP_PATCH_MARKER, True)
    setattr(transition_durations_s, "__hipporeplayimm_original__", original)
    duration_dynamics.transition_durations_s = transition_durations_s
    _synchronize_duration_resolver_aliases(original, transition_durations_s)


def _timestamp_array_preserving_extended_precision(values: object) -> np.ndarray:
    """Keep wider floating-point timestamps until adjacent differences are formed."""

    raw = np.asarray(values)
    if raw.dtype.kind == "f" and np.finfo(raw.dtype).nmant > np.finfo(float).nmant:
        return raw
    return np.asarray(values, dtype=float)


def _strictly_increasing_transition_durations(times: np.ndarray) -> np.ndarray:
    """Return representable positive differences without first narrowing timestamps."""

    if times.size <= 1:
        return np.empty(0, dtype=float)
    if np.any(times[1:] <= times[:-1]):
        raise ValueError(
            "times must be strictly increasing when transition_durations is missing"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        native_durations = np.diff(times)
    if not np.all(np.isfinite(native_durations)):
        raise ValueError("timestamp differences exceed floating-point range")
    durations = np.asarray(native_durations, dtype=float)
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("timestamp differences exceed floating-point range")
    return durations


def _synchronize_duration_resolver_aliases(original, replacement) -> None:
    """Update package modules that imported the duration resolver by value."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name != "hipporeplayimm" and not module_name.startswith("hipporeplayimm."):
            continue
        if getattr(module, "transition_durations_s", None) is original:
            module.transition_durations_s = replacement


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

    transition_durations = _transition_durations_for_time_vector(emissions)
    if transition_durations is None:
        with np.errstate(over="ignore", invalid="ignore"):
            output = float(times[-1]) - times[::-1] + float(times[0])
        return _validated_reversed_times(output)

    reversed_durations = transition_durations[::-1]
    output = np.empty_like(times, dtype=float)
    output[0] = float(times[0])
    if reversed_durations.size:
        with np.errstate(over="ignore", invalid="ignore"):
            output[1:] = output[0] + np.cumsum(reversed_durations, dtype=float)
    return _validated_reversed_times(output)


def _validated_reversed_times(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(output)):
        raise ValueError("reversed times exceed floating-point range")
    if output.size > 1 and np.any(output[1:] <= output[:-1]):
        raise ValueError("reversed times must remain strictly increasing")
    return output.copy()


def _transition_durations_for_time_vector(emissions: LogEmissionTensor) -> np.ndarray | None:
    values = getattr(emissions, "transition_durations", None)
    if values is None:
        return _transition_durations_from_times(emissions)
    expected_length = max(emissions.n_time - 1, 0)
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise ValueError(f"transition_durations must contain {expected_length} values; got shape {array.shape}")
    if array.size and (not np.all(np.isfinite(array)) or np.any(array <= 0.0)):
        raise ValueError("transition_durations must contain finite positive values")
    return array


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
    raw_times = np.asarray(emissions.times)
    if raw_times.shape == (0,):
        return np.full(emissions.n_time - 1, float(emissions.dt), dtype=float)
    if raw_times.shape != (emissions.n_time,):
        return None
    times = _timestamp_array_preserving_extended_precision(raw_times)
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    return _strictly_increasing_transition_durations(times)
