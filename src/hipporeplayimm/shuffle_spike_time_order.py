"""Runtime patches for shuffle-control ordering and scope keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import operator

import numpy as np

_PATCHED_FLAG = "_shuffle_spike_time_order_patch_applied"
_SCOPE_KEY_PATCHED_FLAG = "_shuffle_scope_numeric_key_patch_applied"
_GRID_SHAPE_PATCHED_FLAG = "_shuffle_grid_shape_validation_patch_applied"


def apply_shuffle_spike_time_order_patch() -> None:
    """Install sorted spike-time shuffling and dtype-stable shuffle scope keys."""

    from . import result_improvements as ri
    from . import shuffle_controls

    if not (getattr(ri, _PATCHED_FLAG, False) and getattr(ri, "shuffle_spike_times_session", None) is _shuffle_spike_times_session_sorted):
        ri.shuffle_spike_times_session = _shuffle_spike_times_session_sorted
        setattr(ri, _PATCHED_FLAG, True)

    if not getattr(shuffle_controls, _SCOPE_KEY_PATCHED_FLAG, False):
        original_scope_label = shuffle_controls._scope_label

        def scope_label(value: object) -> str:
            if isinstance(value, Mapping):
                return _mapping_scope_label(value, scope_label)
            nonfinite_numeric = _nonfinite_numeric_scope_label(value)
            if nonfinite_numeric is not None:
                return repr(("scalar", nonfinite_numeric))
            numeric = _numeric_scope_label(value)
            if numeric is not None:
                return repr(("numeric", numeric))
            return original_scope_label(value)

        shuffle_controls._scope_label = scope_label
        setattr(shuffle_controls, _SCOPE_KEY_PATCHED_FLAG, True)

    if not getattr(shuffle_controls, _GRID_SHAPE_PATCHED_FLAG, False):
        original_validate_grid_shape = shuffle_controls._validate_grid_shape

        def validate_grid_shape(grid_shape: object) -> tuple[int, int]:
            return _validated_grid_shape(grid_shape, original_validate_grid_shape)

        shuffle_controls._validate_grid_shape = validate_grid_shape
        setattr(shuffle_controls, _GRID_SHAPE_PATCHED_FLAG, True)


def _mapping_scope_label(value: Mapping[object, object], scope_label) -> str:
    items = sorted(
        ((scope_label(key), scope_label(item)) for key, item in value.items()),
        key=repr,
    )
    return repr(("mapping", items))


def _validated_grid_shape(grid_shape: object, original_validate_grid_shape) -> tuple[int, int]:
    try:
        values = tuple(grid_shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("grid_shape must contain exactly two integer dimensions") from exc
    if len(values) != 2:
        raise ValueError("grid_shape must contain exactly two integer dimensions")
    return tuple(_positive_integer_grid_dimension(value) for value in values)  # type: ignore[return-value]


def _positive_integer_grid_dimension(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("grid_shape dimensions must be positive integers")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_shape dimensions must be positive integers") from exc
    if array.ndim != 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("grid_shape dimensions must be positive integers")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("grid_shape dimensions must be positive integers") from exc
    if not np.isfinite(numeric):
        raise ValueError("grid_shape dimensions must be finite positive integers")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
        raise ValueError("grid_shape dimensions must be positive integers")
    if integer <= 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    return integer


def _shuffle_spike_times_session_sorted(session, random_seed: int = 1):
    from . import result_improvements as ri

    rng = np.random.default_rng(_nonnegative_integer_seed(random_seed))
    spikes = np.asarray(session.spikes, dtype=float).copy()
    if spikes.size == 0:
        return session
    if spikes.ndim != 2 or spikes.shape[1] < 1:
        raise ValueError("session.spikes must be a two-dimensional array with a time column")
    spikes[:, 0] = rng.permutation(spikes[:, 0])
    order = np.argsort(spikes[:, 0], kind="mergesort")
    spikes = spikes[order]
    marks = session.spike_marks
    if marks is not None:
        mark_times = np.asarray(marks.times, dtype=float).copy()
        if mark_times.shape[0] == order.shape[0]:
            mark_times = spikes[:, 0].copy()
        elif mark_times.size:
            mark_times = rng.permutation(mark_times)
        marks = ri._replace_spike_mark_rows(marks, times=mark_times, order=order)
    return replace(session, spikes=spikes, spike_marks=marks)


def _numeric_scope_label(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not isinstance(value, (float, np.floating)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return str(int(numeric)) if numeric.is_integer() else format(numeric, ".17g")


def _nonfinite_numeric_scope_label(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_, int, np.integer)) or not isinstance(value, (float, np.floating)):
        return None
    numeric = float(value)
    if np.isfinite(numeric):
        return None
    return str(value).strip()


def _nonnegative_integer_seed(value: object) -> int:
    try:
        array = np.asarray(value)
    except ValueError as exc:
        raise ValueError("random_seed must be an integer scalar") from exc
    if array.ndim != 0:
        raise ValueError("random_seed must be an integer scalar")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("random_seed must be an integer, not boolean")
    if isinstance(scalar, (str, bytes, np.str_, np.bytes_)):
        raise ValueError("random_seed must be an integer, not string")
    try:
        integer = operator.index(scalar)
    except TypeError:
        try:
            integer = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("random_seed must be a finite integer") from exc
        try:
            is_exact = bool(scalar == integer)
        except (TypeError, ValueError):
            is_exact = False
        if not is_exact:
            raise ValueError("random_seed must be an integer")
    if integer < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    return int(integer)
