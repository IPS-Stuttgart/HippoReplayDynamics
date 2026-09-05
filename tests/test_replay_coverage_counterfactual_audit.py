import numpy as np
import pytest

from scripts.audit_replay_coverage_counterfactual import direct_support_rows


def test_pooled_channel_cannot_create_sorted_cell_support():
    full = np.zeros((20, 4), int)
    full[:, 0], full[:, 2] = 1, 100
    subsets = {1.: np.arange(4), .5: np.array([0, 1])}
    pooled = np.column_stack([full[:, :2], full[:, 2:].sum(axis=1)])
    spec = {"dataset": "test", "animal": "a", "session": "s", "source_event_index": 1,
            "replicate": 0, "truth_kind": "stationary", "gradient": 0}
    counts = {(3., "native", 1.): full, (3., "native", .5): full[:, :2],
              (3., "pooled_removed", .5): pooled}
    trials = [{"spec": spec, "counts": counts}]
    rows = direct_support_rows(trials, subsets, 3.)
    assert len(rows) == 24
    assert rows[rows.cell_fraction.eq(1.)].supported_frame_fraction.eq(1.).all()
    assert rows[rows.cell_fraction.eq(.5)].supported_frame_fraction.eq(0.).all()
    assert rows[rows.regime.eq("pooled_removed")].observation_channels.eq(3).all()
    counts[3., "pooled_removed", .5] = pooled.copy()
    counts[3., "pooled_removed", .5][0, -1] += 1
    with pytest.raises(AssertionError, match="deterministic parent sum"):
        direct_support_rows(trials, subsets, 3.)
