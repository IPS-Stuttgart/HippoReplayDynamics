"""Validate clusterless spike-mark identifiers before held-out subsetting.

Clusterless held-out scoring filters spike marks by the train/test cell split.
The legacy helper used ``dtype=int`` casts for requested cell IDs and saved
spike-mark cell/group IDs.  That can silently turn malformed identifiers such as
``1.9`` or ``True`` into valid integer IDs before the clusterless encoding layer
gets a chance to validate them, aliasing marks into the wrong train/test subset
or tetrode group.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import SpikeMarkData
from .data_cell_id_validation import _coerce_integral_ids

_PATCHED_FLAG = "_benchmark_mark_cell_id_validation_patch_applied"


def apply_benchmark_mark_cell_id_validation_patch() -> None:
    """Install strict ID validation for clusterless held-out mark subsets."""

    from . import benchmarks

    if getattr(benchmarks, _PATCHED_FLAG, False):
        return

    original_subset = benchmarks._session_with_mark_cell_subset

    def session_with_validated_mark_cell_subset(session, cell_ids, *, role: str):
        marks = session.spike_marks
        if marks is None or marks.n_features == 0 or marks.cell_ids is None:
            return original_subset(session, cell_ids, role=role)

        selected = _coerce_integral_ids(cell_ids, f"{role} cell IDs").reshape(-1)
        if selected.size == 0:
            raise ValueError(f"No {role} cell IDs were selected for clusterless scoring.")

        mark_cell_ids = _coerce_integral_ids(marks.cell_ids, "spike-mark cell IDs").reshape(-1)
        if mark_cell_ids.shape[0] != int(marks.n_spikes):
            raise ValueError("spike-mark cell IDs must have one entry per mark")

        mark_group_ids = None
        if marks.group_ids is not None:
            mark_group_ids = _coerce_integral_ids(marks.group_ids, "spike-mark group IDs").reshape(-1)
            if mark_group_ids.shape[0] != mark_cell_ids.shape[0]:
                raise ValueError("spike-mark group IDs must have one entry per mark")

        keep = np.isin(mark_cell_ids, selected)
        if not np.any(keep):
            selected_text = benchmarks._format_cell_ids(selected)
            raise ValueError(f"No {role} spike marks found for selected cell IDs: {selected_text}")

        filtered_marks = SpikeMarkData(
            times=np.asarray(marks.times, dtype=float)[keep].copy(),
            marks=np.asarray(marks.marks, dtype=float)[keep].copy(),
            source_file=marks.source_file,
            source_variable=marks.source_variable,
            feature_names=marks.feature_names,
            cell_ids=mark_cell_ids[keep].copy(),
            group_ids=None if mark_group_ids is None else mark_group_ids[keep].copy(),
        )
        return replace(session, spike_marks=filtered_marks)

    benchmarks._session_with_mark_cell_subset = session_with_validated_mark_cell_subset
    try:
        from . import ground_truth
    except Exception:
        ground_truth = None
    if ground_truth is not None:
        ground_truth._session_with_mark_cell_subset = session_with_validated_mark_cell_subset
    setattr(benchmarks, _PATCHED_FLAG, True)


__all__ = ["apply_benchmark_mark_cell_id_validation_patch"]
