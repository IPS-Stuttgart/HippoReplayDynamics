"""Validate clusterless mark-group identifiers before integer coercion.

Clusterless grouping keys are identifiers, not boolean flags.  They are often
stored as floating-point MATLAB values, so integral floats remain supported, but
malformed boolean, fractional, non-finite, or out-of-range values must be rejected
before NumPy integer casts can alias them to unrelated groups or force a silent
fallback to the global mark likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCHED_FLAG = "_clusterless_mark_group_validation_patch_applied"


def _contains_boolean_ids(values: Any) -> bool:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        raw = np.asarray(values)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _coerce_integral_group_ids(values: Any, name: str) -> np.ndarray:
    """Return integer group IDs without lossy bool/fraction/range coercion."""

    raw = np.asarray(values, dtype=object)
    if _contains_boolean_ids(raw):
        raise ValueError(f"{name} must not contain boolean identifiers")
    numeric = np.asarray(raw, dtype=float)
    if numeric.size == 0:
        return np.asarray(numeric, dtype=int)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite integer identifiers")
    rounded = np.rint(numeric)
    if not np.all(np.isclose(numeric, rounded, rtol=0.0, atol=1e-9)):
        raise ValueError(f"{name} must be integer-valued")
    integer_info = np.iinfo(np.dtype(int))
    if not np.all((rounded >= integer_info.min) & (rounded <= integer_info.max)):
        raise ValueError(f"{name} must fit into integer identifier range")
    return rounded.astype(int)


def apply_clusterless_mark_group_validation_patch() -> None:
    """Install strict validation for clusterless mark group IDs."""

    from . import clusterless

    if getattr(clusterless, _PATCHED_FLAG, False):
        return

    def mark_group_ids_for_config(session, config):
        marks = session.spike_marks
        if marks is None:
            raise ValueError("Session does not contain spike marks.")
        group_by = clusterless._normalize_mark_group_by(config.mark_group_by)
        if group_by == "none":
            return None
        if group_by == "cell":
            return None if marks.cell_ids is None else _coerce_integral_group_ids(marks.cell_ids, "clusterless cell group IDs")
        if group_by == "tetrode":
            if marks.group_ids is None:
                raise ValueError("clusterless mark grouping by tetrode requires spike-mark group IDs from Tetrode_Cell_IDs")
            return _coerce_integral_group_ids(marks.group_ids, "clusterless tetrode group IDs")
        if marks.group_ids is not None:
            return _coerce_integral_group_ids(marks.group_ids, "clusterless mark group IDs")
        return None

    def coerce_group_indices(self, group_ids, n_marks: int):
        if group_ids is None or self.group_ids is None:
            return None
        raw_group_ids = np.asarray(group_ids, dtype=object)
        if raw_group_ids.ndim == 0:
            raw_group_ids = np.full(int(n_marks), raw_group_ids.item(), dtype=object if raw_group_ids.dtype == object else raw_group_ids.dtype)
        raw_group_ids = raw_group_ids.reshape(-1)
        if raw_group_ids.shape[0] != int(n_marks):
            raise ValueError(f"Expected {n_marks} mark group IDs, got {raw_group_ids.shape[0]}")
        coerced = _coerce_integral_group_ids(raw_group_ids, "mark group IDs")
        encoding_group_ids = _coerce_integral_group_ids(self.group_ids, "encoding mark group IDs")
        sorted_order = np.argsort(encoding_group_ids)
        sorted_groups = encoding_group_ids[sorted_order]
        positions = np.searchsorted(sorted_groups, coerced)
        in_bounds = positions < sorted_groups.shape[0]
        matches = np.zeros(int(n_marks), dtype=bool)
        matches[in_bounds] = sorted_groups[positions[in_bounds]] == coerced[in_bounds]
        group_indices = np.full(int(n_marks), -1, dtype=int)
        group_indices[matches] = sorted_order[positions[matches]]
        return group_indices

    clusterless._mark_group_ids_for_config = mark_group_ids_for_config
    clusterless.ClusterlessMarkEncoding._coerce_group_indices = coerce_group_indices
    setattr(clusterless, _PATCHED_FLAG, True)


__all__ = ["apply_clusterless_mark_group_validation_patch"]
