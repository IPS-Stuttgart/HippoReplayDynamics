"""Prevent sparse position samples from certifying continuous well dwell."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_ground_truth_sparse_gap_dwell_guard_wrapped"
_MAX_CONTIGUOUS_SAMPLE_GAP_MULTIPLIER = 5.0


def apply_ground_truth_sparse_gap_guard_patch(gt: Any) -> None:
    """Require sampled continuity for post-ripple well-dwell labels."""

    base_visit = gt.first_post_ripple_well_visit
    if getattr(base_visit, _PATCHED_FLAG, False):
        return

    @wraps(base_visit)
    def first_post_ripple_well_visit_without_sparse_gap_dwell(
        position: np.ndarray,
        wells: pd.DataFrame,
        ripple_peak: float,
        *,
        visit_radius_cm: float,
        min_dwell_s: float,
        future_horizon_s: float,
    ) -> dict[str, float | int] | None:
        cleaned = gt._clean_position(position)
        future = cleaned[
            (cleaned[:, 0] >= ripple_peak)
            & (cleaned[:, 0] <= ripple_peak + future_horizon_s)
        ]
        if future.size == 0 or wells.empty:
            return None

        robust_gap_s = _max_contiguous_sample_gap_s(cleaned[:, 0])
        dwell_evidence_gap_s = max(float(min_dwell_s), np.finfo(float).eps)
        max_sample_gap_s = min(robust_gap_s, dwell_evidence_gap_s)
        candidates: list[dict[str, float | int]] = []
        for well in wells.itertuples(index=False):
            center = np.array([float(well.well_x), float(well.well_y)])
            distances = np.sqrt(np.sum((future[:, 1:3] - center[None, :]) ** 2, axis=1))
            in_radius = distances <= visit_radius_cm
            runs = _split_true_runs_at_sample_gaps(
                gt._true_runs(in_radius),
                future[:, 0],
                max_sample_gap_s,
            )
            for start_idx, end_idx in runs:
                dwell_s = float(future[end_idx - 1, 0] - future[start_idx, 0])
                if dwell_s >= min_dwell_s:
                    candidates.append(
                        {
                            "well_id": int(well.well_id),
                            "well_x": float(well.well_x),
                            "well_y": float(well.well_y),
                            "arrival_time": float(future[start_idx, 0]),
                            "dwell_s": dwell_s,
                        }
                    )
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item["arrival_time"]))
        return candidates[0]

    setattr(first_post_ripple_well_visit_without_sparse_gap_dwell, _PATCHED_FLAG, True)
    gt.first_post_ripple_well_visit = first_post_ripple_well_visit_without_sparse_gap_dwell


def _max_contiguous_sample_gap_s(times: np.ndarray) -> float:
    """Return a robust upper bound for samples considered temporally adjacent."""

    values = np.asarray(times, dtype=float).reshape(-1)
    if values.size < 2:
        return float("inf")
    differences = np.diff(values)
    valid = differences[np.isfinite(differences) & (differences > 0.0)]
    if valid.size == 0:
        return float("inf")
    nominal_interval_s = float(np.median(valid))
    return max(
        _MAX_CONTIGUOUS_SAMPLE_GAP_MULTIPLIER * nominal_interval_s,
        np.finfo(float).eps,
    )


def _split_true_runs_at_sample_gaps(
    runs: list[tuple[int, int]],
    times: np.ndarray,
    max_sample_gap_s: float,
) -> list[tuple[int, int]]:
    """Split contiguous mask runs at missing, reversed, or oversized time steps."""

    values = np.asarray(times, dtype=float).reshape(-1)
    split_runs: list[tuple[int, int]] = []
    for start, end in runs:
        run_times = values[start:end]
        if run_times.size <= 1:
            split_runs.append((start, end))
            continue
        differences = np.diff(run_times)
        breaks = np.flatnonzero(
            ~np.isfinite(differences)
            | (differences <= 0.0)
            | (differences > max_sample_gap_s)
        ) + 1
        segment_start = int(start)
        for offset in breaks:
            segment_end = int(start + offset)
            if segment_end > segment_start:
                split_runs.append((segment_start, segment_end))
            segment_start = segment_end
        if end > segment_start:
            split_runs.append((segment_start, int(end)))
    return split_runs


__all__ = ["apply_ground_truth_sparse_gap_guard_patch"]
