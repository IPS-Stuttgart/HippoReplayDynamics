from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm import ground_truth


def _single_well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [1],
            "well_x": [0.0],
            "well_y": [0.0],
            "n_estimates": [1],
        }
    )


def test_first_post_ripple_visit_does_not_count_recording_gap_as_dwell() -> None:
    hipporeplayimm.apply_runtime_patches()
    times = np.array([0.0, 0.1, 0.2, 1.0, 3.0, 3.1, 3.2], dtype=float)
    x = np.array([50.0, 50.0, 50.0, 0.0, 0.0, 50.0, 50.0], dtype=float)
    position = np.column_stack([times, x, np.zeros_like(times), np.zeros_like(times)])

    visit = ground_truth.first_post_ripple_well_visit(
        position,
        _single_well(),
        ripple_peak=0.0,
        visit_radius_cm=5.0,
        min_dwell_s=0.5,
        future_horizon_s=4.0,
    )

    assert visit is None


def test_first_post_ripple_visit_keeps_continuously_sampled_dwell() -> None:
    hipporeplayimm.apply_runtime_patches()
    times = np.array(
        [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.1],
        dtype=float,
    )
    x = np.array(
        [50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0],
        dtype=float,
    )
    position = np.column_stack([times, x, np.zeros_like(times), np.zeros_like(times)])

    visit = ground_truth.first_post_ripple_well_visit(
        position,
        _single_well(),
        ripple_peak=0.0,
        visit_radius_cm=5.0,
        min_dwell_s=0.4,
        future_horizon_s=3.0,
    )

    assert visit is not None
    assert visit["well_id"] == 1
    assert visit["arrival_time"] == 1.0
    assert visit["dwell_s"] == 0.5
