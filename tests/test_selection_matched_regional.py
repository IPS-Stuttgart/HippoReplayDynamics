import numpy as np
import pytest

from hipporeplayimm.selection_matched_regional import (
    calibration_hist,
    draw_candidate,
    draw_panel,
    endpoint_interval,
    evaluate,
    latent_intensities,
    threshold,
)
from scripts.measure_edge_support_content import occupied_graph


def fixture():
    grid = np.array([[0., 0.], [8., 0.], [16., 0.]])
    rates = np.array([[20., 2., .1], [.1, 2., 20.], [2., 2., 2.]])
    region = np.array([True, False, False])
    graph = occupied_graph(grid)
    times = np.arange(.001, .101, .001)
    return grid, rates, region, graph, times


def test_preserves_timestamps_and_endpoint_total():
    grid, rates, region, graph, times = fixture()
    saved = times.copy()
    counts, active, identity = draw_candidate(times, 0., .102, 0, "moving", grid, rates, graph, region, np.random.default_rng(1))
    np.testing.assert_array_equal(times, saved)
    a, b = endpoint_interval(0., .102)
    assert counts.sum() == ((times >= a) & (times < b)).sum()
    assert len(identity) == len(times) and active == len(np.unique(identity))
    assert counts.shape == (3,)


def test_late_jump_is_only_last_five_ms():
    grid, rates, region, graph, _ = fixture()
    times = np.array([.08, .09, .096, .099])
    intensity = latent_intensities(times, 0., .1, 0, "late_jump", grid, rates, graph, region, np.random.default_rng(1))
    np.testing.assert_array_equal(intensity[:2], np.tile(rates[:, 2], (2, 1)))
    np.testing.assert_array_equal(intensity[2:], np.tile(rates[:, 0], (2, 1)))


def test_global_gain_is_exactly_unidentifiable_given_counts():
    grid, rates, region, graph, times = fixture()
    templates = [{"times": times, "start": 0., "end": .1}]*10
    a = draw_panel(templates, .3, "mix", grid, rates, graph, region, np.random.default_rng(1))
    b = draw_panel(templates, .3, "mix", grid, rates, graph, region, np.random.default_rng(1), "global_gain_x2")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["labels"].sum() == 3


def test_calibration_uses_only_retained_events():
    y = np.array([[0, 1, 2], [0, 2, 1]])
    z = np.array([[0, 0, 1], [0, 1, 1]])
    accepted = np.array([[True, False, True], [True, True, False]])
    np.testing.assert_array_equal(calibration_hist(y, z, accepted), [[2, 0, 0], [0, 0, 2]])


def test_known_mixture_recovers_prevalence_and_no_empty_pass():
    cal = np.array([[900, 100, 0], [0, 100, 900]])
    calls = np.repeat([0, 1, 2], [630, 100, 270])
    truth = np.r_[np.zeros(700, int), np.ones(300, int)]
    fit = evaluate(calls, truth, np.ones(1000, bool), cal)
    assert abs(fit["prevalence"]-.3) < .002 and fit["fit_tv"] < .002
    empty = evaluate(calls, truth, np.zeros(1000, bool), cal)
    assert empty["events"] == 0 and empty["status"] == "missing_class" and np.isnan(empty["prevalence"])


def test_threshold_requires_independent_bank_and_uses_upper_order_statistic():
    assert threshold(np.arange(20)) == 19
    with pytest.raises(ValueError):
        threshold(np.arange(5))


def test_long_clock_endpoint_matches_complete_base_bin_tolerance():
    start, end = 37748.29399700123, 37748.41399700082
    a, b = endpoint_interval(start, end)
    np.testing.assert_allclose([a, b], [start+.100, start+.120], atol=1e-11, rtol=0)
