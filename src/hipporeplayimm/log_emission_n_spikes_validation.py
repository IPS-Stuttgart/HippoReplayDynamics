"""Validate ``LogEmissionTensor`` summary fields after base construction.

The base tensor permits ``-inf`` log-likelihood entries to mark impossible states,
but ``NaN`` entries poison posterior normalization and evidence calculations.  This
patch also keeps the stored ``n_spikes`` summary consistent with the validated
``spike_counts`` tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .encoding import LogEmissionTensor

_PATCH_FLAG = "_n_spikes_validation_applied"


def apply_log_emission_n_spikes_validation_patch() -> None:
    """Install idempotent ``LogEmissionTensor`` post-construction guards."""

    if getattr(LogEmissionTensor, _PATCH_FLAG, False):
        return

    original_post_init = LogEmissionTensor.__post_init__

    def _validated_post_init(self: LogEmissionTensor) -> None:
        original_post_init(self)
        _validate_log_likelihood(self)
        _validate_n_spikes(self)

    LogEmissionTensor.__post_init__ = _validated_post_init  # type: ignore[method-assign]
    setattr(LogEmissionTensor, _PATCH_FLAG, True)


def _validate_log_likelihood(emissions: LogEmissionTensor) -> None:
    """Reject NaN likelihood entries while preserving ``-inf`` impossible states."""

    values = np.asarray(emissions.log_likelihood, dtype=float)
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
    """Reject summary spike counts that disagree with ``spike_counts``."""

    if _contains_boolean_values(emissions.spike_counts):
        raise ValueError("spike_counts must be numeric counts, not boolean values")

    spike_counts = np.asarray(emissions.spike_counts, dtype=float)
    rounded_counts = np.rint(spike_counts)
    if not np.all(np.isclose(spike_counts, rounded_counts, rtol=0.0, atol=0.0)):
        raise ValueError("spike_counts must be integer-valued")

    total_spikes = float(rounded_counts.sum())
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

    emissions.n_spikes = int(rounded)
