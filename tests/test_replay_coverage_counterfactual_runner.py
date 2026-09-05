from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.simulate_replay_coverage_counterfactual import (
    aggregate_decomposition,
    analysis_sets,
    build_counterfactual_trials,
    event_decomposition,
    plot_results,
    score_batch,
)


def fixture():
    edges = np.array([0., 80., 160.])
    grid = np.array([[40., 40.], [40., 120.], [120., 40.], [120., 120.]])
    cache = {"unit_qc_mask": np.ones(4, bool), "cell_ids": np.arange(4),
             "rates_hz": np.tile(np.array([10., 20., 30., 40.])[:, None], (1, 4)),
             "arena_bounds_cm": np.array([[0., 0.], [160., 160.]]),
             "x_edges_cm": edges, "y_edges_cm": edges, "valid_spatial_bins": np.ones(4, bool),
             "bin_centers_cm": grid, "candidate_base_counts": np.ones((40, 4), int),
             "candidate_base_durations_s": np.full(40, .005), "candidate_offsets": np.array([0, 40]),
             "candidate_event_indices": np.array([8])}
    profiles = pd.DataFrame({"source_event_index": [8], "n_base_bins": [40]})
    record = SimpleNamespace(dataset="synthetic", animal="a", session="s")
    args = SimpleNamespace(seed=9, rate_scales=[1., 3.], speed_cm_s=1000.)
    return cache, profiles, record, args


def test_frozen_profile_vertical_slice_and_shuffle():
    cache, profiles, record, args = fixture()
    trials, specs, observations, subsets, _, rates, grid, bounds = build_counterfactual_trials(cache, profiles, record, args, 0)
    assert len(trials) == 6 and len(observations) == 84
    original = next(t for t in trials if t["spec"]["truth_kind"] == "continuous" and t["spec"]["gradient"] == 0)
    shuffled = next(t for t in trials if t["spec"]["truth_kind"] == "shuffled_continuous")
    for key, values in original["counts"].items():
        np.testing.assert_array_equal(values.sum(axis=0), shuffled["counts"][key].sum(axis=0))
        assert sorted(map(tuple, values)) == sorted(map(tuple, shuffled["counts"][key]))
    for scale in args.rate_scales:
        frame = score_batch(trials, subsets, rates, grid, bounds, args, scale)
        assert len(frame) == 336 and frame.status.eq("scored").all()
        assert not frame.continuity_pass.any()  # Spatially uninformative constant maps.
        assert not ((frame.regime == "count_restored") & (frame.cell_fraction == 1)).any()
        compare = frame[frame.regime.isin(["native", "pooled_removed"]) & frame.cell_fraction.eq(.5)].pivot(
            index=["truth_kind", "gradient", "likelihood", "estimator", "bin_filter"], columns="regime", values="supported_frame_fraction")
        np.testing.assert_array_equal(compare.native, compare.pooled_removed)
    assert all(s["n_base_bins"] == 40 for s in specs)


def test_information_partition_is_exact_and_rejects_missing_comparator():
    common = {"dataset": "d", "animal": "a", "session": "s", "source_event_index": 1, "replicate": 0,
              "truth_kind": "continuous", "gradient": 0, "truth_geometric_eligible": True,
              "rate_scale": 3., "likelihood": "poisson", "estimator": "map", "bin_filter": "unfiltered"}
    rows = [{**common, "regime": family, "cell_fraction": fraction, "continuity_pass": passed}
            for family, fraction, passed in [("native", 1., True), ("native", .5, False), ("native", .25, False),
                                             ("pooled_removed", .5, True), ("pooled_removed", .25, False),
                                             ("count_restored", .5, True), ("count_restored", .25, False)]]
    frame = pd.DataFrame(rows)
    partition = event_decomposition(frame)
    assert len(partition) == 2
    np.testing.assert_array_equal(partition.native_minus_full, partition.pooled_minus_full + partition.native_minus_pooled)
    with pytest.raises(ValueError):
        event_decomposition(frame.iloc[:-1])
    empty = event_decomposition(frame.assign(truth_geometric_eligible=False))
    _, _, summary = aggregate_decomposition(empty, 1, 100)
    assert summary.empty and "effect" in summary


def test_dose_series_does_not_mix_in_extra_primary_replicates():
    data = pd.DataFrame({"rate_scale": [3., 10., 3.], "replicate": [0, 0, 1], "value": [1, 2, 999]})
    split = analysis_sets(data, 3., 1)
    assert split[split.analysis_set.eq("paired_dose")].value.tolist() == [1, 2]
    assert split[split.analysis_set.eq("primary_replication")].value.tolist() == [1, 999]


def test_smoke_figure_handles_unavailable_gradients(tmp_path):
    common = {"analysis_set": "paired_dose", "dataset": "synthetic", "rate_scale": 3., "cell_fraction": .5,
              "likelihood": "conditional_multinomial", "estimator": "map", "bin_filter": "unfiltered"}
    decomposition = pd.DataFrame([{**common, "contrast": name, "effect": 0., "ci95_low": np.nan, "ci95_high": np.nan}
                                  for name in ["native_minus_full", "pooled_minus_full"]])
    gradients = pd.DataFrame([{**common, "regime": "native", "cell_fraction": 1., "likelihood": "poisson",
                               "estimator": "posterior_mean", "selection": "all", "coordinate": "true_coordinate",
                               "readout": readout, "response": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
                              for readout in ["decoded", "true_chord"]])
    plot_results(decomposition, gradients, tmp_path / "smoke.png")
    assert (tmp_path / "smoke.png").stat().st_size > 1000
