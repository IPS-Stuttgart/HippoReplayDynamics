"""Ground-truth and observation invariants, not desired biological outcomes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.replay_coverage_recovery import (
    decode_batches,
    fine_population_totals,
    fold_box,
    gradient_from_moments,
    map_interpolator,
    paired_speed_moments,
    sample_observations,
    simulate_path,
    true_state_indices,
    truth_windows,
)
from scripts.simulate_replay_coverage_recovery import (
    build_trials,
    cluster_summary,
    score_trials,
    select_sources,
    simulation_domain,
    summarize_events,
    technical_gates,
)


def fixture_cache(n_base=40):
    edges = np.arange(0, 208, 8.)
    x, y = np.meshgrid((edges[:-1] + edges[1:]) / 2, (edges[:-1] + edges[1:]) / 2, indexing="ij")
    grid = np.column_stack([x.ravel(), y.ravel()])
    rng = np.random.default_rng(6)
    centers = rng.uniform(0, 200, (12, 2))
    rates = .1 + 30 * np.exp(-np.sum((centers[:, None] - grid)**2, axis=2) / (2 * 25**2))
    counts = rng.poisson(.7, (n_base, 12))
    return {"unit_qc_mask": np.ones(12, bool), "cell_ids": np.arange(12), "rates_hz": rates,
            "arena_bounds_cm": np.array([[0., 0.], [200., 200.]]),
            "x_edges_cm": edges, "y_edges_cm": edges, "valid_spatial_bins": np.ones(len(grid), bool),
            "bin_centers_cm": grid, "candidate_base_counts": counts,
            "candidate_base_durations_s": np.full(n_base, .005),
            "candidate_offsets": np.array([0, n_base]), "candidate_event_indices": np.array([8])}


def arguments():
    return SimpleNamespace(seed=20260907, events_per_session=1, speed_cm_s=1000., fine_s=.001, rate_scale=3.)


def test_box_multiple_reflections():
    positions, signs = fold_box(np.array([[25., -5.], [5., 5.]]), [[0., 0.], [10., 10.]])
    np.testing.assert_allclose(positions, [[5, 5], [5, 5]])
    np.testing.assert_allclose(signs, [[1, -1], [1, 1]])


@pytest.mark.parametrize("gradient", [-.5, 0., .5])
def test_truth_speed_field_and_integrator_convergence(gradient):
    bounds = np.array([[0, 0], [200, 350]])
    coarse = simulate_path(100, bounds, "continuous", gradient, 1000., 3)
    fine = simulate_path(100, bounds, "continuous", gradient, 1000., 3, fine_s=.0005)
    assert np.all(coarse["edges_cm"] >= bounds[0]) and np.all(coarse["edges_cm"] <= bounds[1])
    np.testing.assert_allclose(coarse["speed_cm_s"], 1000 * (1 + gradient * coarse["covariate"]), atol=.02)
    np.testing.assert_allclose(coarse["edges_cm"], fine["edges_cm"][::2], atol=.05)


def test_reflections_separate_arclength_from_chord():
    path = simulate_path(100, [[0, 0], [20, 20]], "continuous", 0, 1000., 3)
    truth = truth_windows(path)
    np.testing.assert_allclose(truth["speed_cm_s"], 1000.)
    assert np.median(truth["window_mean_chord_speed_cm_s"]) < 800


@pytest.mark.parametrize("kwargs", [{"fine_s": 0}, {"fine_s": np.nan}, {"n_base": 4.5}, {"gradient": 1}, {"bounds": [[0, 0], [0, 1]]}])
def test_invalid_truth_inputs(kwargs):
    args = {"n_base": 40, "bounds": [[0, 0], [200, 200]], "kind": "continuous", "gradient": 0, "speed_cm_s": 1000, "seed": 4}
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_path(**args)


def test_stationary_and_discontinuous_are_not_redrawn():
    stationary = simulate_path(40, [[0, 0], [200, 200]], "stationary", 0, 1000, 5)
    assert np.unique(stationary["midpoints_cm"], axis=0).shape[0] == 1
    assert np.all(stationary["speed_cm_s"] == 0)
    jumps = simulate_path(40, [[0, 0], [200, 200]], "discontinuous", 0, 1000, 5)
    assert np.unique(jumps["midpoints_cm"], axis=0).shape[0] == 40
    assert np.isnan(jumps["speed_cm_s"]).all()


def test_bilinear_surrogate_and_unvisited_truth():
    edges = np.array([0., 8., 16.])
    evaluate = map_interpolator(np.array([[1., 2., 3., 4.]]), edges, edges)
    np.testing.assert_allclose(evaluate([[8., 8.], [0., 0.]]), [[2.5], [1.]])
    actual = true_state_indices(np.array([[4., 4.], [12., 4.], [-1., 4.]]), edges, edges, np.array([True, True, False, True]))
    np.testing.assert_array_equal(actual, [0, -1, -1])


def test_observation_invariants_determinism_and_count_isolation():
    path = simulate_path(40, [[0, 0], [200, 200]], "continuous", 0, 1000, 3)
    rates = np.tile([1., 10., 30., 100.], (200, 1))
    source = np.arange(40) % 7
    subsets = {1.: np.arange(4), .5: np.array([0, 2]), .25: np.array([2])}
    first, _ = sample_observations(rates, path, source, subsets, 3., 9)
    second, _ = sample_observations(rates, path, source, subsets, 3., 9)
    for (regime, fraction), counts in first.items():
        np.testing.assert_array_equal(counts, second[regime, fraction])
        if regime == "fixed_count":
            np.testing.assert_array_equal(counts.sum(axis=1), source)
        else:
            np.testing.assert_array_equal(counts, first[regime, 1.][:, subsets[fraction]])
    with pytest.raises(ValueError):
        sample_observations(rates, path, source, subsets, np.nan, 9)
    with pytest.raises(ValueError):
        fine_population_totals([np.nan], 5, 9)


def test_batch_decoder_empty_and_missing_truth():
    output = decode_batches(np.empty((0, 1)), np.ones((1, 2)), np.array([[0, 0], [1, 0]]), np.empty(0), "poisson")
    assert output["map"].shape == (0, 2)
    with pytest.raises(ValueError):
        decode_batches(np.ones((1, 1)), np.ones((1, 2)), np.array([[0, 0], [1, 0]]), np.empty(0), "poisson")


def test_speed_support_never_bridges_missing_windows():
    xy = np.column_stack([np.arange(17) * 5., np.zeros(17)])
    counts = np.full((17, 2), 3)
    counts[4] = 0
    truth = {"speed_cm_s": np.full(4, 1000.), "true_covariate": np.linspace(-.8, .8, 4),
             "window_mean_chord_speed_cm_s": np.full(4, 1000.)}
    result = paired_speed_moments(xy, counts, truth, np.array([[0, 0], [200, 200]]), True)
    assert result["all_steps"] == 2
    assert result["all_median_speed_error_vs_arclength_cm_s"] == 0


def test_gradient_uses_equal_event_weights_and_abstains():
    prefix = "all__true_coordinate__decoded"
    frame = pd.DataFrame([{f"{prefix}__steps": n, f"{prefix}__mean_x": x,
                           f"{prefix}__mean_y": 1000 * (1 + .5 * x),
                           f"{prefix}__mean_xx": x*x, f"{prefix}__mean_xy": x*1000*(1+.5*x)}
                          for x, n in zip(np.linspace(-.8, .8, 8), [1, 3, 5, 7, 20, 300, 10, 100], strict=True)])
    assert gradient_from_moments(frame, prefix, 1000)["normalized_slope"] == pytest.approx(.5)
    assert gradient_from_moments(frame.iloc[:2], prefix, 1000)["status"] != "available"
    same = frame.copy()
    same[f"{prefix}__mean_x"] = 0
    same[f"{prefix}__mean_xx"] = 0
    assert np.isnan(gradient_from_moments(same, prefix, 1000)["normalized_slope"])


def test_source_selection_ignores_input_order_and_unknown_pf_walls():
    ids = np.array([8, 2, 14, 35, 17])
    first = ids[select_sources(ids, 3, 5, "session")]
    reverse = ids[::-1]
    second = reverse[select_sources(reverse, 3, 5, "session")]
    assert set(first) == set(second)
    cache = fixture_cache()
    cache["arena_bounds_cm"][:] = np.nan
    _, source = simulation_domain(cache)
    assert source == "encoding_grid_extent_not_verified_walls"


@pytest.mark.parametrize("n_base", [3, 40])
def test_vertical_fixture_denominators_and_missing_rows(n_base):
    args = arguments()
    record = SimpleNamespace(dataset="synthetic", animal="rat", session="session")
    trials, specs, observations, subsets, ids, rates, grid, bounds = build_trials(fixture_cache(n_base), record, args)
    assert len(trials) == 5 and len(ids) == 12
    frame = score_trials(trials, subsets, rates, grid, bounds, args)
    assert len(frame) == 360
    session, gradients = summarize_events(frame, 1000.)
    assert len(session) == 360 and len(gradients) == 216 * 12
    _, summary = cluster_summary(gradients, 3, 100)
    assert summary.normalized_slope.isna().all()  # Only one source trial per condition.
    gates = technical_gates(frame, pd.DataFrame(specs), pd.DataFrame(observations), 1, True)
    assert bool(gates.loc[gates.gate.eq("overall_technical"), "passed"].iloc[0]) == (n_base >= 4)
    if n_base >= 4:
        bad = technical_gates(frame.iloc[:-1], pd.DataFrame(specs), pd.DataFrame(observations), 1, True)
        assert not bad.loc[bad.gate.eq("all_trial_conditions_present"), "passed"].iloc[0]
        for spec in specs:
            assert spec["source_full_bin_spikes"] > 0
            assert spec["path_sha256"]
