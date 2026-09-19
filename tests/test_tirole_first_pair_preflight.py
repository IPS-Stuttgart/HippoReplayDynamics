from dataclasses import replace

import numpy as np
import pytest

from hipporeplayimm.tirole_two_track import TrackSession, fit_maps, nearest_samples
from scripts.audit_tirole_first_pair_preflight import first_pair_view, summarize_view


def four_epochs():
    t = np.arange(1000, 3000, 0.04)
    x = (t * 20) % 200
    intervals = [(1400, 1450), (1500, 1550), (2500, 2550), (2600, 2650)]
    positions = np.array([np.where((t >= a) & (t < b), x, np.nan) for a, b in intervals])
    s = np.arange(1000, 3000, 0.2)
    return TrackSession(
        "S",
        t,
        np.where(np.isfinite(positions).any(axis=0), 20, 0),
        ~np.isfinite(positions).any(axis=0),
        positions,
        np.full(4, 200.0),
        np.arange(1, 31),
        np.arange(1001, 1031),
        s,
        np.arange(len(s)) % 30,
        nearest_samples(t, s),
    )


def test_first_pair_keeps_original_clock_and_removes_later_exposure():
    original = four_epochs()
    view, intervals, cutoff = first_pair_view(original)
    assert view.n_tracks == 2
    assert original.n_tracks == 4
    assert view.times[0] == original.times[0]
    assert view.times[-1] < cutoff == intervals[2][0]
    assert view.spike_times.max() <= view.times[-1]
    np.testing.assert_equal(view.positions, original.positions[:2, original.times < cutoff])
    np.testing.assert_equal(view.spike_samples, nearest_samples(view.times, view.spike_times))
    np.testing.assert_equal(view.original_ids, original.original_ids)
    view.positions[0, 0] = 0
    assert np.isnan(original.positions[0, 0])


def test_later_spikes_do_not_change_first_pair_maps():
    s = four_epochs()
    view, _, cutoff = first_pair_view(s)
    expected = fit_maps(view)
    extra = np.repeat(s.times[s.times >= cutoff][::5], 20)
    spikes = np.r_[s.spike_times, extra]
    modified = replace(s, spike_times=spikes, spike_units=np.r_[s.spike_units, np.zeros(len(extra), int)], spike_samples=nearest_samples(s.times, spikes))
    after = fit_maps(first_pair_view(modified)[0])
    for key in expected:
        np.testing.assert_equal(after[key], expected[key], err_msg=key)


@pytest.mark.parametrize("kind", ["two", "empty", "nonchronological", "overlap"])
def test_bad_epoch_layout_fails(kind):
    s = four_epochs()
    if kind == "two":
        s = replace(s, positions=s.positions[:2], lengths_cm=s.lengths_cm[:2])
    elif kind == "empty":
        s.positions[3] = np.nan
    elif kind == "nonchronological":
        s.positions = s.positions[[1, 0, 2, 3]]
    else:
        s.positions[1, np.isfinite(s.positions[0])] = 1
    with pytest.raises(ValueError):
        first_pair_view(s)


def test_unchanged_run_gate_never_promotes_primary_cohort():
    view, intervals, cutoff = first_pair_view(four_epochs())
    maps = {"common_units": np.arange(25), "union_units": np.arange(30), "valid_bins": np.ones((2, 20), bool)}
    cv = [{"status": "complete", "n_scored_windows": 100, "context_accuracy": 0.9, "median_position_error_cm": 10, "valid_bin_fraction": 1} for _ in range(10)]
    row = summarize_view(view, maps, intervals, cutoff, cv)
    assert row["RUN_gate_passed"]
    assert not row["primary_cohort_promoted"]
    assert not row["source_reexposure_flag_available"]
    assert row["replay_events_scored"] == 0
    cv[1]["context_accuracy"] = 0.79
    row = summarize_view(view, maps, intervals, cutoff, cv)
    assert not row["RUN_gate_passed"]
    assert row["exclusion_reason"] == "RUN_decoder_not_calibrated"
    assert "incomplete_RUN_crossvalidation" in summarize_view(view, maps, intervals, cutoff, [])["exclusion_reason"]
