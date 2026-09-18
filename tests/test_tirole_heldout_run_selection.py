from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.tirole_two_track import TrackSession, blocked_training_mask, decode_counts, fit_maps, nearest_samples
from hipporeplayimm.two_track_content import event_bin_counts, field_shift_posteriors
from scripts.calibrate_tirole_heldout_run_selection import exposure_rates, select_windows
from scripts.report_tirole_heldout_run_selection import block_weights, check_cube, metrics, window_statistics


def session_fixture():
    times = 3.17 + np.arange(50000) * 0.01
    track2 = (np.arange(len(times)) // 5000) % 2 == 1
    x = (np.arange(len(times)) * 0.2) % 100
    pos = np.array([np.where(~track2, x, np.nan), np.where(track2, x, np.nan)])
    st = times[::5]
    su = np.arange(len(st)) % 20
    return TrackSession(
        "fixture", times, np.full(len(times), 20.0), np.zeros(len(times), bool), pos, np.array([100.0, 100.0]), np.arange(20), np.arange(20), st, su, nearest_samples(times, st)
    )


def test_selection_is_balanced_behavior_only_and_heldout():
    s = session_fixture()
    a, availability = select_windows(s)
    assert len(a) == 200 and availability.selected_windows.eq(20).all()
    empty = replace(s, spike_times=np.array([]), spike_units=np.array([], int), spike_samples=np.array([], int))
    pd.testing.assert_frame_equal(a, select_windows(empty)[0])
    for row in a.itertuples():
        take = (s.times >= row.start_s) & (s.times < row.end_s)
        train = blocked_training_mask(s.times, s.times[0], row.fold)
        assert take.any() and not train[take].any()
        assert np.isfinite(s.positions[row.truth_track - 1, take]).all()
        assert row.time_block % 5 == row.fold
    pd.testing.assert_frame_equal(a, select_windows(s)[0])


def test_missing_behavior_not_replaced_by_spike_selection():
    s = session_fixture()
    s.speed[:25000] = 0
    a, available = select_windows(s)
    assert (available.selected_windows <= available.available_windows).all()
    assert a.start_s.ge(s.times[25000]).all()
    s.speed[:] = 0
    with pytest.raises(ValueError, match="no behavior"):
        select_windows(s)


def test_heldout_spikes_cannot_change_training_maps_or_QC():
    s = session_fixture()
    train = blocked_training_mask(s.times, s.times[0], 2)
    before = fit_maps(s, train)
    test_times = s.times[~train][::2]
    st = np.r_[s.spike_times, test_times]
    su = np.r_[s.spike_units, np.zeros(len(test_times), int)]
    order = np.argsort(st)
    modified = replace(s, spike_times=st[order], spike_units=su[order], spike_samples=nearest_samples(s.times, st[order]))
    after = fit_maps(modified, train)
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])
    assert len(event_bin_counts(s, s.times[0], s.times[0] + 1, 0.1)) == 10


def test_exposure_conversion_preserves_actual_poisson_and_conditional_posteriors():
    rng = np.random.default_rng(12)
    rates = rng.uniform(0, 5, (2, 9, 15))
    rates[:, 0, :4] = 0
    counts = rng.poisson(0.5, (10, 9))
    valid = rng.random((2, 15)) > 0.2
    np.testing.assert_allclose(decode_counts(counts, rates, 0.1, valid), decode_counts(counts, exposure_rates(rates), 0.02, valid), atol=1e-14)
    shifts = rng.integers(0, 15, (6, 2, 9))
    for conditional in [False, True]:
        np.testing.assert_allclose(
            field_shift_posteriors(counts, rates, valid, shifts, bin_s=0.1, conditional_count=conditional),
            field_shift_posteriors(counts, exposure_rates(rates), valid, shifts, conditional_count=conditional),
            atol=1e-14,
        )


def score_fixture():
    rows = []
    content = []
    windows = []
    for eid, track in enumerate([1, 2, 1, 2]):
        meta = {"window_id": eid, "session": "fixture", "animal": "RAT", "truth_track": track, "fold": 0, "time_block": eid * 5}
        windows.append(meta)
        for split in range(5):
            content.append({**meta, "split": split, "evaluation_true_probability": 0.8, "evaluation_true_z": 2.0, "evaluation_correct": 1.0})
            for order in ["original", "whole_bin_shuffled"]:
                for likelihood in ["poisson", "conditional_count"]:
                    for repeat in range(-1, 5):
                        accepted = order == "original" and (repeat == -1 or track == 2)
                        rows.append(
                            {
                                **meta,
                                "split": split,
                                "order": order,
                                "likelihood": likelihood,
                                "repeat": repeat,
                                "fraction": 1.0 if repeat == -1 else 0.5,
                                "sequence_accepted": accepted,
                                "sequence_eligible": True,
                                "inferred_track": track,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(content), pd.DataFrame(windows)


def test_window_aggregation_recovers_known_behavioral_selection_shift():
    s, b, w = score_fixture()
    check_cube(s, b, w)
    stats = window_statistics(s, b)
    table = stats[stats.order.eq("original") & stats.likelihood.eq("poisson")]
    result = metrics(table, np.ones(4))
    assert result["full_true_track2_fraction"][0] == 0.5
    assert result["half_true_track2_fraction"][0] == 1
    assert result["half_minus_full_true_track2_fraction"][0] == 0.5
    assert result["lost_evaluation_true_z"][0] == 2
    assert result["lost_support_mass"][0] == 0
    assert result["lost_sequence_mass"][0] == 0.5
    assert np.isnan(result["gained_true_track2_fraction"][0])
    blank = stats[stats.order.eq("whole_bin_shuffled") & stats.likelihood.eq("poisson")]
    assert np.isnan(metrics(blank, np.ones(4))["half_minus_full_true_track2_fraction"][0])


@pytest.mark.parametrize("broken", ["row", "readout", "identity", "truth", "duplicate"])
def test_missing_or_misidentified_scores_fail(broken):
    s, b, w = score_fixture()
    if broken == "row":
        s = s.iloc[1:]
    if broken == "readout":
        b = b.iloc[1:]
    if broken == "identity":
        s.loc[0, "fraction"] = 0.5
    if broken == "truth":
        b.loc[0, "truth_track"] = 2
    if broken == "duplicate":
        s = pd.concat([s.iloc[:-1], s.iloc[:1]])
    with pytest.raises(ValueError):
        check_cube(s, b, w)


def test_block_bootstrap_preserves_balanced_strata_and_block_pairing():
    rows = []
    for fold in range(5):
        for track in [1, 2]:
            for block, n in [(fold, 1), (fold + 5, 3)]:
                for _ in range(n):
                    rows.append({"window_id": len(rows), "truth_track": track, "fold": fold, "time_block": block + track * 100})
    w = pd.DataFrame(rows)
    weights, strata, ok = block_weights(w, 123, 100)
    assert ok and len(strata) == 10
    for _, g in w.groupby(["truth_track", "fold"]):
        np.testing.assert_allclose(weights[:, g.index].sum(axis=1), len(g))
        for _, b in g.groupby("time_block"):
            np.testing.assert_allclose(weights[:, b.index], np.repeat(weights[:, b.index[:1]], len(b), axis=1))
    np.testing.assert_array_equal(weights, block_weights(w, 123, 100)[0])
    assert not block_weights(w[w.fold.eq(0)], 123, 100)[2]
