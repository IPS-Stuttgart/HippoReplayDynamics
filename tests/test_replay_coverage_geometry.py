"""Exact pairing, time/grid semantics and one-draw recovery tests."""

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.replay_coverage_geometry import (
    CORE,
    arena_bounds,
    centered_grid,
    criterion,
    decoder_settings,
    field_population,
    generate_observations,
    make_paths,
    measurement_metrics,
    resolution_truth,
    window_counts,
)
from scripts.simulate_replay_coverage_geometry import (
    build_batch,
    observation_manifest,
    score_batch,
    summarize_batch,
)


def args():
    return Namespace(seed=32, paths=2, duration_ms=200, peak_hz=20., floor_hz=.02,
                     count_rate_hz=300., baseline_only=True)


def test_arena_area_aspect_and_nested_centered_grids():
    bounds = arena_bounds(8.75, 1.4)
    np.testing.assert_allclose(bounds, [[-175, -125], [175, 125]])
    fine, coarse = centered_grid(bounds, 4), centered_grid(bounds, 8)
    assert set(map(tuple, coarse)).issubset(set(map(tuple, fine)))
    assert np.all(fine >= bounds[0]) and np.all(fine <= bounds[1])
    np.testing.assert_array_equal(centered_grid(CORE, 8), centered_grid(CORE, 8))


def test_ratinabox_rates_match_gaussian_sigma_cm_and_units():
    pytest.importorskip("ratinabox")
    evaluate, physical, _ = field_population(3.7, 1.4, 20, np.array([[0, 0], [.1, -.2]]))
    points = np.array([[0, 0], [20, 0], [40, 0]])
    expected = .02 + 19.98 * np.exp(-((points[:, None] - physical)**2).sum(axis=2) / (2 * 20**2))
    np.testing.assert_allclose(evaluate(points), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("window,stride", [(10, 5), (20, 5), (20, 10), (40, 10)])
def test_windows_equal_direct_sums(window, stride):
    fine = np.random.default_rng(2).poisson(.3, (200, 3))
    expected = np.stack([fine[k:k + window].sum(axis=0) for k in range(0, 201-window, stride)])
    np.testing.assert_array_equal(window_counts(fine, window, stride), expected)


def straight_path():
    edges = np.column_stack([np.arange(201) - 100., np.zeros(201)])
    return {"edges_cm": edges, "midpoints_cm": (edges[:-1] + edges[1:]) / 2,
            "speed_cm_s": np.full(200, 1000.), "covariate": (edges[:-1, 0] + .5) / 70,
            "fine_s": .001, "n_per_base": 5}


@pytest.mark.parametrize("window,stride", [(10, 5), (20, 5), (20, 10), (40, 10)])
def test_nonoverlapping_speed_correct_at_each_resolution(window, stride):
    truth = resolution_truth(straight_path(), window, stride)
    counts = np.full((len(truth["center_cm"]), 2), 2)
    values = measurement_metrics(truth["window_mean_cm"], counts, truth, window, stride,
                                "literal_20cm_10frames", False, np.zeros(len(counts)), np.zeros(len(counts)))
    assert values["all_median_speed_cm_s"] == pytest.approx(1000)
    np.testing.assert_allclose(truth["chord_speed"], 1000)
    np.testing.assert_allclose(truth["arclength_speed"], 1000)
    assert values["median_position_error_cm"] == pytest.approx(0)


def test_missing_support_never_bridged_for_speed():
    truth = resolution_truth(straight_path(), 20, 5)
    counts = np.full((len(truth["center_cm"]), 2), 2)
    counts[2] = 0
    values = measurement_metrics(truth["center_cm"], counts, truth, 20, 5, "literal_20cm_10frames",
                                True, np.zeros(len(counts)), np.zeros(len(counts)))
    assert values["all_steps"] == len(truth["indices"]) - 2


def test_time_scaled_and_literal_rules_are_not_silently_interchanged():
    assert criterion(5, "literal_20cm_10frames") == criterion(5, "time_scaled_4000cm_s_45ms")
    assert criterion(10, "time_scaled_4000cm_s_45ms") == {"max_jump_cm": 40, "min_frames": 6, "min_displacement_cm": 40}
    assert criterion(10, "literal_20cm_10frames")["min_frames"] == 10
    assert len(decoder_settings(128, 30, 1.4)) == 18
    assert len(decoder_settings(64, 30, 1.4)) == 1


def test_exact_native_subsetting_common_totals_and_shuffle():
    paths = make_paths(5, 0, 2, 200)
    def evaluate(points):
        return 2 + np.exp(-((points[:, 0, None] - np.arange(128)) / 30)**2)
    first = generate_observations(paths, evaluate, 5, 0, 0)
    other = generate_observations(paths, lambda p: evaluate(p)[:, ::-1], 5, 0, 1)
    for trial, obs, different in zip(paths, first, other, strict=True):
        np.testing.assert_array_equal(obs["native_poisson", 64], obs["native_poisson", 128][:, :64])
        totals = obs["common_count_schedule", 128].sum(axis=1)
        np.testing.assert_array_equal(totals, obs["common_count_schedule", 64].sum(axis=1))
        np.testing.assert_array_equal(totals, different["common_count_schedule", 128].sum(axis=1))
        if trial["truth_kind"] == "shuffled_continuous":
            parent = first[trial["path_id"] * 6 + 1]
            for key, arr in obs.items():
                expected = parent[key].reshape(40, 5, -1)[trial["permutation"]].reshape(arr.shape)
                np.testing.assert_array_equal(arr, expected)
                np.testing.assert_array_equal(arr.sum(axis=0), parent[key].sum(axis=0))


def test_field_changes_preserve_paths_but_not_rates_and_common_counts():
    pytest.importorskip("ratinabox")
    a = build_batch(args(), 0, 0)
    b = build_batch(args(), 0, 10)
    for left, right in zip(a[0], b[0], strict=True):
        np.testing.assert_array_equal(left["path"]["midpoints_cm"], right["path"]["midpoints_cm"])
    am = observation_manifest(a[0], a[1], 0, 0)
    bm = observation_manifest(b[0], b[1], 0, 10)
    np.testing.assert_array_equal(am.path_sha256, bm.path_sha256)
    np.testing.assert_array_equal(am[am.observation.eq("common_count_schedule")].total_schedule_sha256,
                                  bm[bm.observation.eq("common_count_schedule")].total_schedule_sha256)


def test_scoring_denominators_and_gradient_abstention():
    pytest.importorskip("ratinabox")
    config = args()
    paths, obs, evaluate, _, bounds = build_batch(config, 0, 0)
    events, geometry = score_batch(config, 0, 0, paths, obs, evaluate, bounds)
    assert len(events) == config.paths * 6 * 2 * 2 * 2 * 2 * 2
    assert len(geometry) == 4
    summary, slopes = summarize_batch(events)
    assert summary.events.eq(2).all()
    assert slopes.normalized_slope.isna().all()
    assert not slopes.status.eq("available").any()
    assert not summary[summary.truth_kind.ne("continuous")].eligible_recovery_fraction.notna().any()
    assert pd.api.types.is_bool_dtype(events.continuity_pass)


def test_bad_windows_fail():
    with pytest.raises(ValueError):
        window_counts(np.ones((10, 2)), 20, 5)
    with pytest.raises(ValueError):
        window_counts(np.ones((100, 2)), 20, 7)
