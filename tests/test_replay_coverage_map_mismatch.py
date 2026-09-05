"""Independent-half invariants, paired scoring and non-vacuous gates."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_replay_coverage_run_validation import circular_run  # noqa: F401

from hipporeplayimm.replay_coverage_data import CoverageInputConfig
from hipporeplayimm.replay_coverage_geometry import resolution_truth, window_counts
from hipporeplayimm.replay_coverage_map_mismatch import (
    decode_with_coverage,
    fit_map_pair,
    mismatch_metrics,
    mismatch_observations,
)
from hipporeplayimm.replay_coverage_recovery import simulate_path, true_state_indices
from scripts.audit_replay_coverage_map_mismatch import sampled_decode_check
from scripts.simulate_replay_coverage_map_mismatch import IDENTITY, score_trials, summarize_batch, technical_gates


def test_training_map_and_selection_ignore_generator_spikes(circular_run):
    original, qc, meta = fit_map_pair(circular_run, 0, CoverageInputConfig())
    changed = circular_run.spikes.copy()
    keep = changed[:, 0] > meta["split_time_s"] + 1
    changed[keep, 1] = 13 - changed[keep, 1]
    other, other_qc, _ = fit_map_pair(replace(circular_run, spikes=changed), 0, CoverageInputConfig())
    for key in ["rates_hz", "grid_cm", "cell_ids", "valid_spatial_bins", "training_intervals"]:
        np.testing.assert_array_equal(original[key], other[key])
    pd.testing.assert_frame_equal(qc, other_qc)
    assert not np.array_equal(original["generator_rates_hz"], other["generator_rates_hz"])
    assert not meta["generator_half_used_for_decoder_QC"]
    assert not meta["generator_half_used_for_decoder_support"]
    assert meta["both_position_extents_used_for_synthetic_domain"]


def test_map_directions_are_disjoint_and_share_decode_states(circular_run):
    for direction in [0, 1]:
        model, _, meta = fit_map_pair(circular_run, direction, CoverageInputConfig())
        assert model["rates_hz"].shape == model["generator_at_training_states_hz"].shape
        assert model["rates_hz"].shape[1] == len(model["grid_cm"])
        assert model["rates_hz"].shape[0] == len(model["cell_ids"])
        assert all(max(a, c) > min(b, d) for a, b in model["training_intervals"] for c, d in model["generator_intervals"])
        assert meta["training_spikes_sha256"] != meta["generator_spikes_sha256"]
    with pytest.raises(ValueError):
        fit_map_pair(circular_run, 2, CoverageInputConfig())
    with pytest.raises(ValueError, match="insufficient training-only"):
        fit_map_pair(circular_run, 0, CoverageInputConfig(min_running_spikes=1000000))


def test_shared_gain_is_explicit_blockwise_stress_with_finite_counts():
    rates = np.full((20000, 8), 20.)
    first, gains = mismatch_observations(rates, 913, rate_scale=2.)
    second, repeated = mismatch_observations(rates, 913, rate_scale=2.)
    np.testing.assert_array_equal(gains, repeated)
    assert np.all(gains.reshape(-1, 20) == gains.reshape(-1, 20)[:, :1])
    assert .85 < gains.mean() < 1.15
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].shape == rates.shape
        assert np.issubdtype(first[key].dtype, np.integer)
        assert .8 < first[key].sum() / (rates.sum() * .002) < 1.2
    with pytest.raises(ValueError):
        mismatch_observations(-rates, 1)


def test_observation_family_and_oracle_map_scores_agree_on_same_map():
    counts = np.array([[5, 0], [0, 5], [0, 0]])
    rates = np.array([[10., .1], [.1, 10.]])
    grid = np.array([[0., 0.], [10., 0.]])
    for likelihood in ["poisson", "conditional_multinomial"]:
        decoded = decode_with_coverage(counts, rates, grid, [0, 1, -1], likelihood)
        np.testing.assert_array_equal(decoded["map"][:2], grid)
        assert decoded["hpd95"].tolist() == [True, True, False]
        assert np.isfinite(decoded["posterior_mean"]).all()


def test_metrics_do_not_bridge_excluded_bins_and_use_actual_domain():
    domain = np.array([[100., 200.], [200., 300.]])
    path = simulate_path(40, domain, "continuous", 0., 1000., 4)
    truth = resolution_truth(path, 20, 5)
    n = len(truth["center_cm"])
    points = truth["window_mean_cm"]
    counts = np.full((n, 2), 2)
    counts[2] = 0
    out = mismatch_metrics(points, counts, truth, domain, True, np.ones(n), np.ones(n), np.ones(n, bool))
    assert out["all_steps"] == len(truth["indices"]) - 2
    assert -1 <= out["all__decoded_coordinate__decoded__mean_x"] <= 1
    short = resolution_truth(simulate_path(4, domain, "stationary", 0., 1000., 4), 20, 5)
    result = mismatch_metrics(short["center_cm"], np.zeros((1, 2)), short, domain, False, np.ones(1), np.ones(1), np.ones(1, bool))
    assert result["all_steps"] == 0 and not result["continuity_pass"]
    assert np.isnan(result["all_median_speed_cm_s"])


def test_paired_trial_scoring_is_complete_without_needing_a_positive_result(circular_run):
    model, _, _ = fit_map_pair(circular_run, 0, CoverageInputConfig())
    model["generator_at_training_states_hz"] = model["rates_hz"].copy()
    n_cells = len(model["cell_ids"])
    subsets = {1.: np.arange(n_cells), .5: np.arange(n_cells//2)}
    path = simulate_path(40, model["domain_cm"], "continuous", 0., 1000., 9)
    truth = resolution_truth(path, 20, 5)
    counts = np.random.default_rng(6).poisson(2., (200, n_cells))
    spec = {"dataset": "fixture", "animal": "R1", "session": "S1", "direction": 0, "source_event_index": 1,
        "truth_kind": "continuous", "gradient": 0., "truth_training_supported_fraction": 1., "truth_generator_supported_fraction": 1.}
    trial = {"spec": spec, "truth": truth,
        "fine_counts": {name: counts for name in ["poisson", "shared_gain"]},
        "states": true_state_indices(truth["center_cm"], model["x_edges_cm"], model["y_edges_cm"], model["valid_spatial_bins"]),
        "counts": {(name, fraction): window_counts(counts[:, cells], 20, 5) for name in ["poisson", "shared_gain"] for fraction, cells in subsets.items()}}
    frame = score_trials([trial], model, subsets, 3.)
    assert len(frame) == 64
    left = frame[frame.decoder_map.eq("generator_known")].drop(columns="decoder_map").reset_index(drop=True)
    right = frame[frame.decoder_map.eq("independent_RUN_half")].drop(columns="decoder_map").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert sampled_decode_check(frame, trial, model, subsets, 3.) == 64
    corrupted = frame.copy()
    corrupted.loc[0, "median_position_error_cm"] += 1
    with pytest.raises(AssertionError):
        sampled_decode_check(corrupted, trial, model, subsets, 3.)
    summary, gradients = summarize_batch(frame)
    assert len(summary) == 64 and len(gradients) == 64*12
    assert gradients.normalized_slope.isna().all()


def test_empty_or_missing_directions_cannot_pass():
    sessions = pd.DataFrame({"dataset": ["fixture"]})
    assert not technical_gates(pd.DataFrame(), sessions, True).passed.all()
    rows = [{**dict(zip(IDENTITY, ["fixture", "R1", "S1", d], strict=True)), "status": "scored", "source_profiles": 2, "metric_rows": 768} for d in [0, 1]]
    gates = technical_gates(pd.DataFrame(rows), sessions, True)
    assert gates.passed.all()
    rows[1]["status"] = "encoding_unavailable"
    gates = technical_gates(pd.DataFrame(rows), sessions, True).set_index("gate").passed
    assert gates["overall_technical"] and not gates["all_planned_encoding_directions_available"]
    rows[1]["status"] = "failed"
    assert not technical_gates(pd.DataFrame(rows), sessions, True).set_index("gate").passed["overall_technical"]
    rows[1] = rows[0]
    assert not technical_gates(pd.DataFrame(rows), sessions, True).set_index("gate").passed["planned_directions_reported"]
