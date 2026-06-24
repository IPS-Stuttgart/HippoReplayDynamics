"""Keep shuffled spike-time null controls in decoder-safe time order."""

from __future__ import annotations

from dataclasses import replace

import numpy as np


def apply_shuffle_spike_time_order_patch() -> None:
    """Install a sorted spike-time shuffling helper.

    ``shuffle_spike_times_session`` intentionally randomizes spike timestamps for
    null controls, but downstream replay decoders expect spike rows to be sorted
    by time.  Keep clusterless mark rows aligned with the reordered spike rows.
    """

    from . import result_improvements as ri

    if getattr(ri, "_shuffle_spike_time_order_patch_applied", False):
        return
    ri.shuffle_spike_times_session = _shuffle_spike_times_session_sorted
    ri._shuffle_spike_time_order_patch_applied = True


def _shuffle_spike_times_session_sorted(session, random_seed: int = 1):
    from . import result_improvements as ri

    rng = np.random.default_rng(random_seed)
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
