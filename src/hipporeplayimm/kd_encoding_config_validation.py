"""Validate KD reference encoding configuration before fitting.

The KD reference encoder uses fixed grid dimensions and occupancy floors directly
inside NumPy shape, edge, smoothing, and rate calculations. Invalid parameters
can otherwise reach divide-by-zero, empty-grid, or Gaussian-filter failures that
are harder to diagnose than a configuration error.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_kd_encoding_config_validation_patch_applied"


def _coerce_float_scalar(config: Any, name: str) -> float:
    value = getattr(config, name)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a scalar float")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar float") from exc
    if array.shape != ():
        raise TypeError(f"{name} must be a scalar float")
    try:
        value = array.item()
    except (AttributeError, IndexError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a scalar float")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar float") from exc


def _validate_positive_float(config: Any, name: str) -> None:
    value = _coerce_float_scalar(config, name)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative_float(config: Any, name: str) -> None:
    value = _coerce_float_scalar(config, name)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_positive_integer(config: Any, name: str) -> None:
    value = getattr(config, name)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    arr = np.asarray(value)
    if arr.ndim != 0 or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be a positive integer")
    if int(arr) <= 0:
        raise ValueError(f"{name} must be positive")


def validate_kd_encoding_config(config: Any) -> None:
    """Reject KD encoder parameters that cannot produce a valid fixed grid."""

    _validate_positive_float(config, "bin_size_cm")
    _validate_positive_integer(config, "n_bins_x")
    _validate_positive_integer(config, "n_bins_y")
    _validate_nonnegative_float(config, "smoothing_sigma_cm")
    _validate_nonnegative_float(config, "min_speed_cm_s")
    _validate_positive_float(config, "min_occupancy_s")
    _validate_positive_float(config, "rate_floor_hz")
    _validate_nonnegative_float(config, "min_peak_rate_hz")


def _session_with_excitatory_fallback(session: Any, config: Any) -> Any:
    """Let the KD encoder match the standard encoder's no-label fallback.

    The reference encoder requests ``session.excitatory_spikes()`` whenever
    ``use_excitatory`` is true.  For datasets without excitatory labels that
    method returns no spikes, so the KD encoder silently fits an empty cell set
    even when spikes are available.  The main encoder falls back to all spikes in
    this case; mirror that behavior by treating all observed cells as excitatory
    only when the label list is absent.
    """

    if not bool(getattr(config, "use_excitatory", True)):
        return session
    excitatory_neurons = np.asarray(getattr(session, "excitatory_neurons", np.array([])))
    if excitatory_neurons.size:
        return session
    spikes = np.asarray(getattr(session, "spikes", np.empty((0, 2))))
    if spikes.size == 0:
        return session
    cell_ids = np.asarray(getattr(session, "cell_ids"), dtype=int)
    if cell_ids.size == 0:
        return session
    return replace(session, excitatory_neurons=cell_ids)


def apply_kd_encoding_config_validation_patch() -> None:
    """Install KD encoding-config validation on the reference encoder."""

    from . import kd_reference

    if getattr(kd_reference, _PATCHED_FLAG, False):
        return

    original_fit = kd_reference.fit_kd_place_field_encoding

    @wraps(original_fit)
    def fit_kd_place_field_encoding(session, config=None):
        config = kd_reference.KDEncodingConfig() if config is None else config
        validate_kd_encoding_config(config)
        return original_fit(_session_with_excitatory_fallback(session, config), config)

    kd_reference.fit_kd_place_field_encoding = fit_kd_place_field_encoding
    setattr(kd_reference, _PATCHED_FLAG, True)


__all__ = ["apply_kd_encoding_config_validation_patch", "validate_kd_encoding_config"]