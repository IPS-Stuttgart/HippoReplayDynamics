"""Validate ``LogEmissionTensor`` count summaries and cell identifiers."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from .encoding import LogEmissionTensor

_PATCH_FLAG = "_n_spikes_validation_applied"
_POST_INIT_WRAPPER_MARKER = "_n_spikes_validation_post_init_wrapper"


def _log_emission_n_spikes_patch_current() -> bool:
    return bool(getattr(LogEmissionTensor.__post_init__, _POST_INIT_WRAPPER_MARKER, False))


def apply_log_emission_n_spikes_validation_patch() -> None:
    """Install idempotent ``LogEmissionTensor`` post-construction guards."""

    if _log_emission_n_spikes_patch_current():
        setattr(LogEmissionTensor, _PATCH_FLAG, True)
        return

    original_post_init = LogEmissionTensor.__post_init__

    @wraps(original_post_init)
    def _validated_post_init(self: LogEmissionTensor) -> None:
        original_post_init(self)
        _validate_log_likelihood(self)
        _validate_n_spikes(self)
        _validate_cell_ids(self)

    setattr(_validated_post_init, _POST_INIT_WRAPPER_MARKER, True)
    LogEmissionTensor.__post_init__ = _validated_post_init  # type: ignore[method-assign]
    setattr(LogEmissionTensor, _PATCH_FLAG, True)


def _validate_log_likelihood(emissions: LogEmissionTensor) -> None:
    values = np.asarray(emissions.log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must be a two-dimensional array")
    if values.shape[1] == 0:
        raise ValueError("log_likelihood must include at least one spatial bin")
    if np.any(np.isnan(values)):
        raise ValueError("log_likelihood must not contain NaN values")


def _contains_boolean_values(values: Any) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _validate_n_spikes(emissions: LogEmissionTensor) -> None:
    if _contains_boolean_values(emissions.spike_counts):
        raise ValueError("spike_counts must be numeric counts, not boolean values")
    spike_counts = np.asarray(emissions.spike_counts, dtype=float)
    rounded_counts = np.rint(spike_counts)
    if not np.all(np.isclose(spike_counts, rounded_counts, rtol=0.0, atol=0.0)):
        raise ValueError("spike_counts must be integer-valued")
    total_spikes = float(rounded_counts.sum())
    if _contains_boolean_values(emissions.n_spikes):
        raise ValueError("n_spikes must be a numeric count, not boolean")
    try:
        n_spikes = float(emissions.n_spikes)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_spikes must be numeric") from exc
    if not np.isfinite(n_spikes) or n_spikes < 0.0:
        raise ValueError("n_spikes must be finite and nonnegative")
    rounded = float(np.rint(n_spikes))
    if not np.isclose(n_spikes, rounded, rtol=0.0, atol=0.0):
        raise ValueError("n_spikes must be integer-valued")
    if not np.isclose(rounded, total_spikes, rtol=0.0, atol=0.0):
        raise ValueError("n_spikes must equal the total spike_counts sum")
    emissions.spike_counts = rounded_counts.astype(int, copy=False)
    emissions.n_spikes = int(rounded)


def _validate_cell_ids(emissions: LogEmissionTensor) -> None:
    if _contains_boolean_values(emissions.cell_ids):
        raise ValueError("cell_ids must be numeric integer identifiers, not boolean values")
    try:
        cell_ids = np.asarray(emissions.cell_ids, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell_ids must contain finite integer identifiers") from exc
    if cell_ids.ndim != 1:
        raise ValueError("cell_ids must be one-dimensional")
    if cell_ids.size == 0:
        emissions.cell_ids = np.empty(0, dtype=int)
        return
    if not np.all(np.isfinite(cell_ids)):
        raise ValueError("cell_ids must contain finite integer identifiers")
    rounded = np.rint(cell_ids)
    if not np.all(np.isclose(cell_ids, rounded, rtol=0.0, atol=0.0)):
        raise ValueError("cell_ids must be integer-valued")
    integer_info = np.iinfo(np.dtype(int))
    if not np.all((rounded >= integer_info.min) & (rounded <= integer_info.max)):
        raise ValueError("cell_ids must fit into integer identifier range")
    canonical = rounded.astype(int)
    if np.unique(canonical).shape[0] != canonical.shape[0]:
        raise ValueError("cell_ids must be unique")
    emissions.cell_ids = canonical


__all__ = ["apply_log_emission_n_spikes_validation_patch"]
