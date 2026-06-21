"""Input checks for synthetic-recovery emission arrays."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def apply_synthetic_count_input_patch() -> None:
    """Validate count arrays before the legacy helper converts their dtype."""

    import hipporeplayimm.simulation_recovery as recovery

    if getattr(recovery, "_synthetic_count_input_patch_applied", False):
        return

    original = recovery.emissions_from_counts

    @wraps(original)
    def emissions_from_counts_checked(encoding: Any, counts: Any, *args: Any, **kwargs: Any) -> Any:
        return original(encoding, _checked_count_array(counts), *args, **kwargs)

    recovery.emissions_from_counts = emissions_from_counts_checked
    recovery._synthetic_count_input_patch_applied = True


def _checked_count_array(counts: Any) -> np.ndarray:
    raw = np.asarray(counts)
    if raw.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must contain numeric values") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("counts must contain finite nonnegative values")
    if not np.all(np.isclose(numeric, np.rint(numeric), rtol=0.0, atol=0.0)):
        raise ValueError("counts must contain integer-valued counts")
    return numeric.astype(int)
