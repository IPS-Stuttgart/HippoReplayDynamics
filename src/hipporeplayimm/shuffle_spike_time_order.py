"""Runtime patches for shuffle-control ordering and scope keys."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

_PATCHED_FLAG = "_shuffle_spike_time_order_patch_applied"
_SCOPE_KEY_PATCHED_FLAG = "_shuffle_scope_numeric_key_patch_applied"


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
            numeric = _numeric_scope_label(value)
            if numeric is not None:
                return repr(("numeric", numeric))
            return original_scope_label(value)

        shuffle_controls._scope_label = scope_label
        setattr(shuffle_controls, _SCOPE_KEY_PATCHED_FLAG, True)


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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return str(int(numeric)) if numeric.is_integer() else format(numeric, ".17g")


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
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("random_seed must be an integer") from exc
    if not np.isfinite(numeric):
        raise ValueError("random_seed must be a finite integer")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
        raise ValueError("random_seed must be an integer")
    if integer < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    return integer
