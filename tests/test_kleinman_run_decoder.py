import numpy as np
import pytest

from scripts.validate_kleinman_run_decoder import (
    align_behavior,
    decode,
    fit_maps,
    integer_indices,
    interval_counts,
    interval_data,
    make_traversals,
    session_pass,
    split_units,
    window_behavior,
)


def behavior():
    return {"position": np.arange(21.0), "velocity": np.column_stack([np.arange(1, 21), np.ones(20)]),
                "reward_ends": [5, 15], "left_visit": [[2, 4]], "right_visit": [[18, 20]],
                "epoch_change": [[10, 11]]}


def test_source_alignment_distinguishes_visit_and_epoch_indices():
    t, x, _, _, visits, epochs = align_behavior(behavior())
    np.testing.assert_array_equal(t, x)
    assert visits[0]["start_s"] == 1
    assert visits[0]["entry_position_cm"] == 1
    assert epochs == [(1, 10), (11, 20)]


def test_nonmonotonic_clock_not_silently_sorted():
    info = behavior()
    info["velocity"][10, 0] = 1
    with pytest.raises(ValueError, match="nonmonotonic"):
        align_behavior(info)


def test_first_position_has_no_timestamp_and_fractional_indices_rejected():
    info = behavior()
    info["left_visit"] = [[1, 4]]
    with pytest.raises(ValueError, match="indices"):
        align_behavior(info)
    with pytest.raises(ValueError):
        integer_indices([2.5], 1, 4, "test")


def test_epoch_gaps_and_same_end_pairs_are_excluded():
    visits = [{"side": i % 2, "start_s": i * 10, "end_s": i * 10 + 2} for i in range(13)]
    runs = make_traversals(visits, [(0, 50), (55, 200)])
    assert not any(r["start_s"] < 55 < r["end_s"] for r in runs)
    assert len(runs) == 11
    for group in {r["lap_group"] for r in runs}:
        assert len({r["fold"] for r in runs if r["lap_group"] == group}) == 1
    assert {r["fold"] for r in runs} == set(range(5))


def test_composite_unit_keys_and_half_open_counting():
    keys, trains, excluded = split_units([[1, 1, 2], [0, 1, 1], [1, 1, 1], [2, 0, 1]])
    assert keys.tolist() == [[1, 1], [2, 1]]
    assert excluded == 1
    assert interval_counts(trains, [0, 1, 2]).tolist() == [[1, 0], [0 + 1, 1]]


def test_invalid_clock_intervals_and_local_reverse_steps_are_not_training():
    t = np.array([0, .02, .04, .3, .32])
    x = np.array([0, 1, .5, 3, 4])
    runs = [{"start_s": 0, "end_s": .32, "fold": 0, "direction": 1}]
    _, valid, trainable, *_ = interval_data(t, x, np.full(5, 30), runs, np.arange(0, 6, 2))
    assert valid.tolist() == [True, True, False, True]
    assert trainable.tolist() == [True, False, False, True]


def test_window_time_weighting_and_tracking_gap_exclusion():
    t = np.array([0, .1, .3, .4])
    x = 10 * t
    speed = np.full(4, 10)
    result = window_behavior(t, x, speed, .05, .35, np.ones(3, bool))
    assert result == pytest.approx((2, 10))
    assert window_behavior(t, x, speed, .05, .35, np.array([True, False, True])) is None


def synthetic_map():
    rng = np.random.default_rng(4)
    b = np.tile(np.repeat(np.arange(30), 40), 2)
    d = np.repeat(np.arange(2), 30 * 40)
    rates = np.zeros((2, 30, 24)) + .05
    for direction in range(2):
        for unit in range(12):
            rates[direction, :, unit + direction * 12] += 40 * np.exp(-((np.arange(30) - unit * 29 / 11) / 3) ** 2 / 2)
    counts = rng.poisson(.1 * rates[d, b])
    return rng, b, d, rates, counts


def test_known_directional_fields_recover_independent_observations():
    rng, b, d, truth, counts = synthetic_map()
    rates, support, units, _ = fit_maps(counts, np.full(len(b), .1), d, b, np.ones(len(b), bool), 30)
    observed = rng.poisson(.25 * truth.reshape(60, 24))[:, units]
    lp = decode(observed, rates, support, .25)
    centers = np.tile(np.arange(30) * 2 + 1, 2)
    mean = np.exp(lp) @ centers
    assert np.mean(abs(mean - centers)) < 8
    assert np.mean(lp.argmax(axis=1) // 30 == np.repeat([0, 1], 30)) > .9
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1)


def test_heldout_spikes_cannot_select_units_or_change_maps():
    _, b, d, _, counts = synthetic_map()
    mask = np.arange(len(b)) % 4 != 0
    first = fit_maps(counts, np.full(len(b), .1), d, b, mask, 30)
    changed = counts.copy()
    changed[~mask] += 100000
    second = fit_maps(changed, np.full(len(b), .1), d, b, mask, 30)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_never_smooth_across_directions_or_decode_unsupported_states():
    counts = np.ones((20, 6)) * 2
    rates, support, _, occupancy = fit_maps(counts, np.ones(20), np.zeros(20, int),
                                           np.arange(20) % 10, np.ones(20, bool), 10)
    assert occupancy[1].sum() == 0
    p = np.exp(decode(np.ones((1, 6)), rates, support, .25))
    assert p[:, 10:].sum() == 0


def test_gates_not_vacuous_and_fold_failures_visible():
    assert session_pass(20, .8, 40, 8, 5)
    assert not session_pass(20, .8, 0, 8, 5)
    assert not session_pass(20, .8, 40, 8, 4)
    assert not session_pass(20, .8, 40, 4, 5)
    assert not session_pass(np.nan, .8, 40, 8, 5)
    assert not session_pass(36, .8, 40, 8, 5)
    assert not session_pass(20, .59, 40, 8, 5)
