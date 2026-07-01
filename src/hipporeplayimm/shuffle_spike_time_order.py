"""Keep shuffled spike-time null controls in decoder-safe time order."""

from __future__ import annotations

from dataclasses import replace

import numpy as np


_PATCHED_FLAG = "_shuffle_spike_time_order_patch_applied"


def apply_shuffle_spike_time_order_patch() -> None:
    """Install a sorted spike-time shuffling helper.

    ``shuffle_spike_times_session`` intentionally randomizes spike timestamps for
    null controls, but downstream replay decoders expect spike rows to be sorted
    by time.  Keep clusterless mark rows aligned with the reordered spike rows.
    """

    from . import result_improvements as ri

    if (
        getattr(ri, _PATCHED_FLAG, False)
        and getattr(ri, "shuffle_spike_times_session", None) is _shuffle_spike_times_session_sorted
    ):
        return
    ri.shuffle_spike_times_session = _shuffle_spike_times_session_sorted
    setattr(ri, _PATCHED_FLAG, True)


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
