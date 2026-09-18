"""Coverage changes must not silently change targets or aggregation units."""

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.coverage_dose_forecast import attach_contrasts, event_tables, retention_gate, subsets, summaries
from hipporeplayimm.independent_rejected_forecast import BASELINES


def test_nested_and_legacy_half_reproducible():
    train, half = np.arange(24), np.arange(0, 24, 2)
    for repeat in range(10):
        a = subsets(train, half, ("PF", "Rat1"), 0, repeat)
        b = subsets(train, half, ("PF", "Rat1"), 0, repeat)
        assert set(a[0.25]) < set(a[0.5]) < set(a[0.75]) < set(a[1])
        assert [len(a[f]) for f in [1, 0.75, 0.5, 0.25]] == [24, 18, 12, 6]
        for f in a:
            np.testing.assert_array_equal(a[f], b[f])
        if repeat == 0:
            np.testing.assert_array_equal(a[0.5], half)
    assert not np.array_equal(subsets(train, half, ("PF",), 0, 1)[0.5], half)
    with pytest.raises(ValueError):
        subsets(train, [99], ("PF",), 0, 0)


def rows():
    out = []
    for event in range(2):
        for split in range(2):
            full = {
                "dataset": "PF",
                "animal": "Rat1",
                "session": "s1",
                "event_id": event,
                "split": split,
                "repeat": 0,
                "fraction": 1.0,
                "arm": "fixed_codebook",
                "status": "scored",
                "n_full_bins": 6,
                "n_heldout_target_spikes": 2,
                "n_train_cells": 24,
                "n_train_spikes": 50,
                "geometric_pass": True,
                "valid_frames": 12,
                "score_dynamic": -10.0,
            }
            full.update({"score_" + b: -14.0 for b in BASELINES})
            out.append(full)
            for repeat, value in enumerate([2.0, 4.0, 20.0] if split == 0 else [8.0]):
                row = full | {"repeat": repeat, "fraction": 0.5, "geometric_pass": False, "n_train_cells": 12, "score_dynamic": -14.0 + value}
                out.append(row)
    return pd.DataFrame(out)


def test_paired_differences_and_two_level_resampling():
    x = attach_contrasts(rows())
    half = x[x.fraction.eq(0.5)]
    np.testing.assert_allclose(half.paired_change_dynamic_minus_matched_own, half.dynamic_minus_matched_own - 2)
    events = event_tables(rows())
    lost = events[events.group.eq("lost")]
    # First split median is 2; second split is 4, so event median is 3.
    assert lost.dynamic_minus_matched_own.eq(3).all()
    assert lost.paired_change_dynamic_minus_matched_own.eq(1).all()
    assert lost.pass_change.eq(-1).all()
    summary = summaries(events)[2]
    assert retention_gate(summary, "PF", "fixed_codebook", "lost", 1)
    assert not retention_gate(summary, "PF", "fixed_codebook", "lost", 4)


def test_failures_and_undefined_targets_are_explicit():
    x = rows()
    with pytest.raises(ValueError):
        attach_contrasts(pd.concat([x, x.iloc[:1]]))
    with pytest.raises(ValueError):
        attach_contrasts(x[x.fraction.ne(1)])
    y = x.copy()
    y.loc[y.fraction.lt(1), "n_heldout_target_spikes"] = 3
    with pytest.raises(ValueError):
        attach_contrasts(y)
    x["n_heldout_target_spikes"] = 0
    e = event_tables(x)
    assert e.dynamic_minus_matched_own.isna().all()
    assert not retention_gate(summaries(e)[2], "PF", "fixed_codebook", "lost", 1)


def test_gained_and_short_events_not_hidden():
    x = rows()
    x.loc[x.event_id.eq(0) & x.fraction.eq(1), "geometric_pass"] = False
    x.loc[x.event_id.eq(0) & x.fraction.eq(0.5), "geometric_pass"] = True
    x.loc[x.event_id.eq(1) & x.fraction.eq(0.5), "valid_frames"] = 3
    e = event_tables(x)
    assert set(e[e.group.eq("gained")].event_id) == {0}
    assert set(e[e.group.eq("lost")].event_id) == {1}
    assert e[e.group.eq("lost_supported")].empty


def test_full_reference_not_replicated_per_repeat_or_arm():
    x = rows()
    reduced = x[x.fraction.eq(0.5) & x.repeat.eq(0)].assign(arm="restricted_calibration")
    out = attach_contrasts(pd.concat([x, reduced], ignore_index=True))
    assert len(out[out.fraction.eq(1)]) == 4
    assert out[out.arm.eq("restricted_calibration")].full_dynamic_minus_matched_own.eq(2).all()
