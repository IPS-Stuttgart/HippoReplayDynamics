from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from scripts.audit_tanni2022_virtual_movement import (
    empirical_upper_p,
    independently_shift_cell_maps,
    ordered_margin,
    source_event_groups,
)


def test_ordered_margin_compares_best_ordered_with_best_nonordered() -> None:
    scores = {
        "stationary": SimpleNamespace(log_likelihood=4.0),
        "diffusion": SimpleNamespace(log_likelihood=9.0),
        "fragmented": SimpleNamespace(log_likelihood=6.0),
        "first-order-imm": SimpleNamespace(log_likelihood=8.0),
    }

    assert ordered_margin(scores) == 3.0


def test_independent_map_shift_preserves_each_cells_rate_distribution() -> None:
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0, 2.0, 3.0]),
        bin_centers=np.zeros((6, 2)),
        rates_hz=np.arange(12, dtype=float).reshape(2, 6),
        occupancy_s=np.ones(6),
        cell_ids=np.array([1, 2]),
        config=EncodingConfig(),
    )

    shifted = independently_shift_cell_maps(encoding, np.random.default_rng(4))

    for cell_index in range(encoding.n_cells):
        assert sorted(shifted.rates_hz[cell_index]) == sorted(encoding.rates_hz[cell_index])
        assert not np.array_equal(shifted.rates_hz[cell_index], encoding.rates_hz[cell_index])


def test_source_event_groups_merge_overlapping_windows_only_within_session() -> None:
    events = pd.DataFrame(
        {
            "animal": ["A", "A", "A", "B"],
            "session": ["S", "S", "S", "S"],
            "window_start_time_s": [1.0, 1.1, 2.0, 1.1],
            "window_end_time_s": [1.2, 1.3, 2.2, 1.3],
        }
    )

    groups = source_event_groups(events, overlap_gap_s=0.0)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] != groups.iloc[0]
    assert groups.iloc[3] not in {groups.iloc[0], groups.iloc[2]}


def test_empirical_upper_p_uses_plus_one_correction() -> None:
    assert empirical_upper_p(5.0, np.array([1.0, 2.0, 3.0])) == 0.25
