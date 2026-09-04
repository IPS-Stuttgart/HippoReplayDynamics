"""Paired recording loss, window alignment, and non-bridging speed tests."""

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.replay_coverage import decode_independent
from scripts.analyze_replay_coverage_subsampling import (
    aggregate,
    decode_compact,
    decoding_support,
    event_windows,
    observed_metrics,
    pair_to_full,
    population_subsets,
)


def test_partial_edge_bin_center_excluded_only_with_known_walls():
    grid = np.array([[4., 4.], [12., 4.], [20., 4.]])
    occupied = np.ones(3, dtype=bool)
    np.testing.assert_array_equal(decoding_support(occupied, grid, [[0, 0], [18, 8]]), [True, True, False])
    np.testing.assert_array_equal(decoding_support(occupied, grid, np.full((2, 2), np.nan)), occupied)
    with pytest.raises(ValueError):
        decoding_support(occupied, grid, [[0, np.nan], [18, 8]])


def test_nested_recording_subsets_deterministic_and_full_not_replicated():
    ids = np.arange(100)
    first = list(population_subsets(ids, [.25, .5, .75], 3, 7, "rat/session"))
    second = list(population_subsets(ids, [.75, .25, .5], 3, 7, "rat/session"))
    assert len(first) == 10
    assert sum(fraction == 1 for fraction, _, _ in first) == 1
    for (fraction, rep, a), (other_fraction, other_rep, b) in zip(first, second, strict=True):
        assert (fraction, rep) == (other_fraction, other_rep)
        np.testing.assert_array_equal(a, b)
        assert len(a) == int(fraction * 100)
    for rep in range(3):
        subsets = {f: set(c) for f, r, c in first if r == rep}
        assert subsets[.25] < subsets[.5] < subsets[.75]
    alternate = list(population_subsets(ids, [.5], 1, 7, "other/session"))
    assert not np.array_equal(alternate[1][2], first[2][2])


def test_windows_ignore_partial_tail_without_joining_different_events():
    counts = np.arange(7)[:, None]
    windows = event_windows(counts, np.r_[np.full(6, .005), .002])
    np.testing.assert_array_equal(windows[:, 0], [6, 10, 14])
    assert event_windows(counts[:3], np.full(3, .005)).shape == (0, 1)
    with pytest.raises(ValueError):
        event_windows(counts, [.005, .002, .005, .005, .005, .005, .005])
    with pytest.raises(ValueError):
        event_windows(counts, [.005] * 6 + [.010])


def test_compact_chunked_decoder_exactly_matches_reference():
    rng = np.random.default_rng(7)
    counts = rng.poisson(2, (21, 5))
    rates = rng.uniform(.02, 5, (5, 11))
    grid = rng.uniform(0, 40, (11, 2))
    for likelihood in ["poisson", "conditional_multinomial"]:
        actual = decode_compact(counts, rates, grid, likelihood, chunk_size=4)
        expected = decode_independent(counts, rates, grid, .020, likelihood=likelihood)
        for key, values in actual.items():
            np.testing.assert_allclose(values, expected[key], atol=1e-12)


def test_speed_uses_nonoverlap_and_support_does_not_bridge_missing_frame():
    path = np.column_stack((np.arange(16) * 5.0, np.zeros(16)))
    counts = np.tile([2, 1], (16, 1))
    result = observed_metrics(path, counts, np.ones(16), np.ones(16), True)
    assert result["continuity_pass"]
    assert result["nonoverlapping_valid_steps"] == 3
    assert result["nonoverlapping_event_median_speed_cm_s"] == 1000
    counts[4] = 0
    result = observed_metrics(path, counts, np.ones(16), np.ones(16), True)
    assert result["nonoverlapping_valid_steps"] == 1
    assert result["nonoverlapping_event_median_speed_cm_s"] == 1000
    assert result["continuous_start"] == 5


def test_empty_and_unsupported_events_remain_finite_failures_not_zero_speed():
    result = observed_metrics(np.empty((0, 2)), np.empty((0, 3)), np.empty(0), np.empty(0), True)
    assert not result["continuity_pass"]
    assert result["nonoverlapping_valid_steps"] == 0
    assert np.isnan(result["nonoverlapping_event_median_speed_cm_s"])
    result = observed_metrics(np.zeros((16, 2)), np.zeros((16, 3)), np.ones(16), np.ones(16), True)
    assert not result["continuity_pass"] and result["supported_frame_fraction"] == 0


def test_selected_core_speed_requires_adjacent_independent_windows():
    path = np.column_stack((np.arange(12) * 5.0, np.zeros(12)))
    counts = np.tile([2, 1], (12, 1))
    result = observed_metrics(path, counts, np.ones(12), np.ones(12), True)
    assert result["selected_core_nonoverlapping_steps"] == 2
    assert result["selected_core_median_speed_cm_s"] == 1000
    counts[0] = 0
    counts[8] = 0
    result = observed_metrics(path, counts, np.ones(12), np.ones(12), True)
    assert result["selected_core_nonoverlapping_steps"] == 0
    assert np.isnan(result["selected_core_median_speed_cm_s"])


def score_fixture():
    rows = []
    for animal in ["A", "B"]:
        for event in range(3):
            for fraction, rep in [(1., 0), (.5, 0), (.5, 1)]:
                rows.append({
                    "dataset": "test", "animal": animal, "session": "S1", "event_index": event,
                    "likelihood": "poisson", "estimator": "map", "bin_filter": "unfiltered",
                    "cell_fraction": fraction, "population_replicate": rep, "retained_cells": int(10 * fraction),
                    "continuity_pass": event > 0 if fraction == 1 else event == 0,
                    "nonoverlapping_event_median_speed_cm_s": 1000., "median_posterior_rms_cm": 10.,
                    "large_jump_fraction": .2, "selected_core_median_speed_cm_s": 1000.,
                })
    return pd.DataFrame(rows)


def test_full_reference_pairing_tracks_losses_and_gains():
    frame = score_fixture()
    paired = pair_to_full(frame)
    assert len(paired) == len(frame)
    assert paired.lost_full_continuity.sum() == 8
    assert paired.gained_continuity.sum() == 4
    population, _, animals, summary = aggregate(frame, 200, 3)
    assert len(population) == 6
    selected = summary[summary.metric.eq("continuity_pass_difference") & summary.cell_fraction.eq(.5)].iloc[0]
    assert selected.animals == 2
    assert selected.equal_animal_mean == pytest.approx(-1 / 3)
    assert len(animals) == 4
    with pytest.raises(ValueError, match="missing"):
        pair_to_full(frame[~(frame.cell_fraction.eq(1) & frame.animal.eq("A") & frame.event_index.eq(0))])
    with pytest.raises(ValueError, match="duplicate"):
        pair_to_full(pd.concat([frame, frame[frame.cell_fraction.eq(1)]]))
