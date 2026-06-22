from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def apply_simulation_recovery_count_validation_patch() -> None:
    from . import simulation_recovery

    if getattr(simulation_recovery, "_count_validation_patch_applied", False):
        return

    original = simulation_recovery.emissions_from_counts

    @wraps(original)
    def emissions_from_counts_with_validated_counts(
        encoding: Any,
        counts: Any,
        *,
        dt: float,
        spike_rate_scale: float = 1.0,
        likelihood_temperature: float = 1.0,
        negative_binomial_overdispersion: float = 0.0,
    ):
        validated_counts = _validated_count_matrix(
            counts,
            n_cells=int(getattr(encoding, "n_cells")),
        )
        return original(
            encoding,
            validated_counts,
            dt=dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )

    simulation_recovery.emissions_from_counts = emissions_from_counts_with_validated_counts
    simulation_recovery._count_validation_patch_applied = True


def _validated_count_matrix(counts: Any, *, n_cells: int) -> np.ndarray:
    try:
        values = np.asarray(counts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc

    if values.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    if values.shape[1] != int(n_cells):
        raise ValueError("counts columns must match encoding.n_cells")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    rounded = np.rint(values)
    if not np.all(np.isclose(values, rounded, rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    return np.asarray(rounded, dtype=int)
