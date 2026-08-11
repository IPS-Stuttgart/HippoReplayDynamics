from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm import ground_truth
from hipporeplayimm.ground_truth_window_scope import _max_contiguous_sample_gap_s


def _single_well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [1],
            "well_x": [0.0],
            "well_y": [0.0],
            "n_estimates": [1],
        }
    )


def test_short_trace_cadence_uses_lower_middle_interval() -> None:
    times = np.array([0.0, 0.1, 3.0], dtype=float)

    assert _max_contiguous_sample_gap_s(times) == 0.5


def test_short_trace_dropout_is_not_counted_as_ground_truth_dwell() -> None:
    hipporeplayimm.apply_runtime_patches()
    times = np.array([0.0, 0.1, 3.0], dtype=float)
    position = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)]
    )

    visit = ground_truth.first_post_ripple_well_visit(
        position,
        _single_well(),
        ripple_peak=0.0,
        visit_radius_cm=5.0,
        min_dwell_s=0.5,
        future_horizon_s=4.0,
    )

    assert visit is None
