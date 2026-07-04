"""Validate clusterless mark-group identifiers before integer coercion.

Clusterless grouping keys are identifiers, not boolean flags.  They are often
stored as floating-point MATLAB values, so integral floats remain supported, but
malformed boolean, fractional, non-finite, or out-of-range values must be rejected
before NumPy integer casts can alias them to unrelated groups or force a silent
fallback to the global mark likelihood.  Row-count validation is part of the same
contract: group/cell identifiers must stay aligned with the mark rows before they
are used as boolean masks during clusterless encoding.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

_PATCHED_FLAG = "_clusterless_mark_group_validation_patch_applied"
_WRAPPER_MARKER = "_clusterless_mark_group_validation_wrapper"


def _mark_group_guard(wrapper):
    setattr(wrapper, _WRAPPER_MARKER, True)
    return wrapper


def _is_current_group_guard(value: object) -> bool:
    return bool(getattr(value, _WRAPPER_MARKER, False))


def _wrappers_are_current(clusterless) -> bool:
    try:
        return (
            _is_current_group_guard(clusterless._mark_group_ids_for_config)
            and _is_current_group_guard(clusterless.ClusterlessMarkEncoding._coerce_group_indices)
        )
    except AttributeError:
        return False


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


def _coerce_integral_group_ids(
    values: Any,
    name: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Return integer group IDs without lossy bool/fraction/range coercion."""

    raw = np.asarray(values, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    else:
        raw = raw.reshape(-1)
    if expected_size is not None and raw.shape[0] != int(expected_size):
        raise ValueError(
            f"{name} must contain one value per spike mark row; "
            f"expected {int(expected_size)}, got {raw.shape[0]}"
        )
    if _contains_boolean_ids(raw):
        raise ValueError(f"{name} must not contain boolean identifiers")
    integer_info = np.iinfo(np.dtype(int))
    coerced = [
        _coerce_integral_group_id(value, name, integer_info)
        for value in raw
    ]
    return np.asarray(coerced, dtype=int)


def _coerce_integral_group_id(value: Any, name: str, integer_info: np.iinfo) -> int:
    """Coerce one group identifier without sending integer inputs through float."""

    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=object)
        if arr.ndim != 0:
            raise ValueError(f"{name} must be one-dimensional")
        value = arr.item()

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean identifiers")

    if isinstance(value, (int, np.integer)):
        identifier = int(value)
    elif isinstance(value, Decimal):
        identifier = _coerce_decimal_group_id(value, name)
    elif isinstance(value, (str, bytes)):
        identifier = _coerce_text_group_id(value, name)
    elif isinstance(value, (float, np.floating)):
        identifier = _coerce_float_group_id(float(value), name)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite integer identifiers") from exc
        identifier = _coerce_float_group_id(numeric, name)

    if identifier < int(integer_info.min) or identifier > int(integer_info.max):
        raise ValueError(f"{name} must fit into integer identifier range")
    return identifier


def _coerce_float_group_id(value: float, name: str) -> int:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite integer identifiers")
    if not value.is_integer():
        raise ValueError(f"{name} must be integer-valued")
    return int(value)


def _coerce_decimal_group_id(value: Decimal, name: str) -> int:
    if not value.is_finite():
        raise ValueError(f"{name} must be finite integer identifiers")
    integer = value.to_integral_value()
    if value != integer:
        raise ValueError(f"{name} must be integer-valued")
    return int(integer)


def _coerce_text_group_id(value: str | bytes, name: str) -> int:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must be finite integer identifiers") from exc
    else:
        text = value
    text = text.strip()
    if not text:
        raise ValueError(f"{name} must be finite integer identifiers")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return _coerce_decimal_group_id(Decimal(text), name)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be finite integer identifiers") from exc


def apply_clusterless_mark_group_validation_patch() -> None:
    """Install strict validation for clusterless mark group IDs."""

    from . import clusterless

    if getattr(clusterless, _PATCHED_FLAG, False) and _wrappers_are_current(clusterless):
        return

    @_mark_group_guard
    def mark_group_ids_for_config(session, config):
        marks = session.spike_marks
        if marks is None:
            raise ValueError("Session does not contain spike marks.")
        expected_size = int(marks.n_spikes)
        group_by = clusterless._normalize_mark_group_by(config.mark_group_by)
        if group_by == "none":
            return None
        if group_by == "cell":
            return None if marks.cell_ids is None else _coerce_integral_group_ids(
                marks.cell_ids,
                "clusterless cell group IDs",
                expected_size=expected_size,
            )
        if group_by == "tetrode":
            if marks.group_ids is None:
                raise ValueError("clusterless mark grouping by tetrode requires spike-mark group IDs from Tetrode_Cell_IDs")
            return _coerce_integral_group_ids(
                marks.group_ids,
                "clusterless tetrode group IDs",
                expected_size=expected_size,
            )
        if marks.group_ids is not None:
            return _coerce_integral_group_ids(
                marks.group_ids,
                "clusterless mark group IDs",
                expected_size=expected_size,
            )
        return None

    @_mark_group_guard
    def coerce_group_indices(self, group_ids, n_marks: int):
        if group_ids is None or self.group_ids is None:
            return None
        raw_group_ids = np.asarray(group_ids, dtype=object)
        if raw_group_ids.ndim == 0:
            raw_group_ids = np.full(int(n_marks), raw_group_ids.item(), dtype=object if raw_group_ids.dtype == object else raw_group_ids.dtype)
        raw_group_ids = raw_group_ids.reshape(-1)
        coerced = _coerce_integral_group_ids(
            raw_group_ids,
            "mark group IDs",
            expected_size=int(n_marks),
        )
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
