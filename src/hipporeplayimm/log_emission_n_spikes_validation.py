"""Validate ``LogEmissionTensor.n_spikes`` against its count matrix."""

from __future__ import annotations

import numpy as np

from .encoding import LogEmissionTensor

_PATCH_FLAG = "_n_spikes_validation_applied"


def apply_log_emission_n_spikes_validation_patch() -> None:
    """Install an idempotent ``n_spikes`` consistency guard."""

    if getattr(LogEmissionTensor, _PATCH_FLAG, False):
        return

    original_post_init = LogEmissionTensor.__post_init__

    def _validated_post_init(self: LogEmissionTensor) -> None:
        original_post_init(self)
        _validate_n_spikes(self)

    LogEmissionTensor.__post_init__ = _validated_post_init  # type: ignore[method-assign]
    setattr(LogEmissionTensor, _PATCH_FLAG, True)


def _validate_n_spikes(emissions: LogEmissionTensor) -> None:
    """Reject summary spike counts that disagree with ``spike_counts``."""

    total_spikes = float(np.asarray(emissions.spike_counts, dtype=float).sum())
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
