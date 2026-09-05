"""Test training-only selection, behavioral truth, and nonvacuous validation."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage_data import CoverageInputConfig, index_spike_times, prepare_position_support
from hipporeplayimm.replay_coverage_validation import (
    count_windows,
    fit_training_fold,
    heldout_centers,
    posterior_coverage,
    subtract_block,
    training_session,
    truth_state_indices,
    window_truth,
)
from scripts.audit_replay_coverage_run_outputs import direct_window_support
from scripts.validate_replay_coverage_run_decoder import (
    score_fold,
    session_from_cache,
    summarize_predictions,
    validation_gates,
)


@pytest.fixture(scope="module")
def circular_run():
    rng = np.random.default_rng(425)
    t = np.arange(0, 80, .01)
    xy = 50 + 40 * np.column_stack((np.sin(.7 * t), np.cos(.7 * t)))
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    fields = 50 + 40 * np.column_stack((np.sin(angle), np.cos(angle)))
    rates = .02 + 25 * np.exp(-np.sum((xy[:, None] - fields[None]) ** 2, axis=2) / (2 * 9**2))
    counts = rng.poisson(rates * .01)
    frames, cells = np.nonzero(counts)
    frames = np.repeat(frames, counts[frames, cells])
    cell_ids = np.repeat(cells, counts[np.nonzero(counts)]) + 1
    spikes = np.column_stack((t[frames] + rng.uniform(0, .01, len(frames)), cell_ids))
    empty = np.empty((0, 2))
    session = ReplaySession(
        rat="R1", name="RUN1", path=Path("RUN1"), position=np.column_stack((t, xy)),
        spikes=spikes, tetrode_cell_ids=empty, excitatory_neurons=np.arange(1, 13),
        inhibitory_neurons=np.empty(0, dtype=int), ripple_events=np.empty((0, 6)),
        run_times=np.array([[0, t[-1]]]), sleep_box_immobile_times=empty,
        sleep_times=empty, rem_times=empty, well_sequence=None,
        metadata={"source_dataset": "tanni2022", "arena_bounds_cm": [[0, 0], [100, 100]]},
    )
    return prepare_position_support(session, .1, [[0, 0], [100, 100]])[0]


def test_guarded_interval_subtraction_preserves_disconnected_support():
    np.testing.assert_allclose(subtract_block([[0, 10], [12, 20], [30, 40]], 5, 15), [[0, 5], [15, 20], [30, 40]])
    assert subtract_block([[0, 10]], -1, 11).shape == (0, 2)
    with pytest.raises(ValueError):
        subtract_block([[0, 10]], 10, 2)


def test_heldout_spikes_positions_and_new_cell_cannot_change_training(circular_run):
    full = circular_run
    config = CoverageInputConfig()
    model, qc, meta = fit_training_fold(full, 24, 40, 1, config)
    position = full.position.copy()
    position[(position[:, 0] >= 23) & (position[:, 0] <= 41), 1:3] = [5000, -5000]
    spikes = full.spikes.copy()
    spikes[(spikes[:, 0] >= 23) & (spikes[:, 0] <= 41), 1] = 9999
    spikes = np.vstack([spikes, np.column_stack((np.linspace(24, 40, 2000), np.full(2000, 5555)))])
    after, later_qc, later_meta = fit_training_fold(replace(full, position=position, spikes=spikes), 24, 40, 1, config)
    for key in model:
        np.testing.assert_array_equal(model[key], after[key])
    pd.testing.assert_frame_equal(qc, later_qc)
    assert meta == later_meta
    assert len(model["cell_ids"]) >= 5
    assert 9999 not in model["cell_ids"] and 5555 not in model["cell_ids"]
    assert not meta["full_RUN_unit_QC_used"] and not meta["heldout_positions_used_for_grid"]
    train = training_session(full, 24, 40, 1, .1)
    assert not ((train.position[:, 0] >= 23) & (train.position[:, 0] < 41)).any()
    assert not ((train.spikes[:, 0] >= 23) & (train.spikes[:, 0] < 41)).any()
    assert np.all((train.run_times[:, 1] < 23) | (train.run_times[:, 0] > 41))


def test_train_failure_does_not_relax_qc(circular_run):
    with pytest.raises(ValueError, match="insufficient training-only"):
        fit_training_fold(circular_run, 0, 16, 1, CoverageInputConfig(min_running_spikes=100000))


def test_window_selection_is_behavior_only_and_does_not_cross_gaps(circular_run):
    full = replace(circular_run, run_times=np.array([[1., 9.], [12., 20.]]))
    centers, eligible = heldout_centers(full, 3, 18, .25, .25, 20, 56)
    other, later = heldout_centers(replace(full, spikes=np.empty((0, 2))), 3, 18, .25, .25, 20, 56)
    np.testing.assert_array_equal(centers, other)
    assert eligible == later and eligible > len(centers) == 20
    assert np.all((centers - .125 >= 3) & (centers + .125 <= 18))
    assert np.all((centers + .125 <= 9) | (centers - .125 >= 12))
    with pytest.raises(ValueError):
        heldout_centers(full, 3, 18, .25, .02, 20, 56)


def test_half_open_windows_count_each_sorted_cell():
    spikes = np.array([[.5, 1], [1., 1], [1.5, 1], [2.5, 2]])
    counts = count_windows(index_spike_times(spikes, [1, 2]), [1, 2], np.array([1., 2.]), 1.)
    np.testing.assert_array_equal(counts, [[2, 0], [1, 0]])
    count, active = direct_window_support(spikes, [1, 2], [1., 2.], 1.)
    np.testing.assert_array_equal(count, counts.sum(axis=1))
    np.testing.assert_array_equal(active, np.count_nonzero(counts, axis=1))


def test_independent_recount_handles_silence_other_units_and_exact_boundaries():
    spikes = np.array([[1.25, 1], [1.0, 2], [.875, 1], [1.125, 1], [.875, 99], [2., 2]])
    count, active = direct_window_support(spikes, [1, 2], [1., 1.25, 3.], .25)
    np.testing.assert_array_equal(count, [2, 2, 0])
    np.testing.assert_array_equal(active, [2, 1, 0])
    with pytest.raises(ValueError):
        direct_window_support(spikes, [1, 2], [1., 1.1], .25)


def test_truth_integrates_tracking_and_distinguishes_chord_from_path_length():
    truth = window_truth(np.array([[0., 0., 0.], [1., 2., 0.], [2., 2., 2.]]), np.array([1.]), 2.)
    np.testing.assert_allclose(truth["center_xy"], [[2., 0.]])
    np.testing.assert_allclose(truth["window_mean_xy"], [[1.5, .5]])
    np.testing.assert_allclose(truth["mean_behavior_speed_cm_s"], [2.])
    line = window_truth(np.array([[0., 0., 0.], [1., 2., 0.], [2., 4., 0.]]), np.array([.5, 1.5]), .5)
    np.testing.assert_allclose(line["center_xy"], line["window_mean_xy"])
    with pytest.raises(ValueError):
        window_truth(np.array([[0., 0., 0.], [1., 2., 0.]]), np.array([.1]), .5)


def test_hpd_membership_ties_and_unsupported_truth():
    posterior = np.array([[.6, .3, .1], [.6, .3, .1], [.6, .3, .1], [.5, .5, 0], [.6, .3, .1]])
    result = posterior_coverage(posterior, [0, 1, 2, 1, -1])
    np.testing.assert_array_equal(result["hpd50"], [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(result["hpd80"], [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(result["hpd95"], [1, 1, 1, 1, 0])
    for indices in [[3], [-2], [.5]]:
        with pytest.raises(ValueError):
            posterior_coverage(np.array([[.6, .3, .1]]), indices)
    with pytest.raises(ValueError):
        posterior_coverage(np.array([[.6, .6, .1]]), [0])


def test_truth_outside_training_spatial_support_is_not_clipped():
    model = {"x_edges_cm": np.array([0, 10, 20]), "y_edges_cm": np.array([0, 10, 20]), "valid_spatial_bins": np.array([1, 0, 1, 1], dtype=bool)}
    np.testing.assert_array_equal(truth_state_indices(model, [[5, 5], [5, 15], [15, 15], [25, 25]]), [0, -1, 2, -1])


def test_cache_loader_does_not_require_full_run_maps_or_qc(tmp_path, circular_run):
    path = tmp_path / "raw_only.npz"
    np.savez(path, position=circular_run.position, spikes=circular_run.spikes,
             supported_run_intervals=circular_run.run_times, arena_bounds_cm=np.array([[0, 0], [100, 100]]))
    record = SimpleNamespace(artifact_path=path, animal="R1", session="RUN1", dataset="pfeiffer_foster")
    units = pd.DataFrame({"cell_id": [1, 2, 3], "source_cell_type_allowed": [True, False, True]})
    session = session_from_cache(record, units)
    np.testing.assert_array_equal(session.excitatory_neurons, [1, 3])
    np.testing.assert_array_equal(session.spikes, circular_run.spikes)


def test_synthetic_run_decodes_better_than_wrong_cell_map(circular_run):
    model, _, _ = fit_training_fold(circular_run, 24, 40, 1, CoverageInputConfig())
    centers, _ = heldout_centers(circular_run, 24, 40, .25, .25, 60, 125)
    frame, permutations = score_fold(circular_run, model, centers, [.25, .02], 5, 451)
    long = frame[frame.likelihood.eq("poisson") & frame.window_s.eq(.25)]
    assert long.posterior_mean_error_cm.median() < 20
    assert long.paired_improvement_over_cell_ID_null_cm.median() > 10
    assert len(frame) == len(centers) * 4 and len(permutations) == 5
    assert np.isfinite(frame.posterior_rms_cm).all()
    for key, value in {"dataset": "fixture", "animal": "R1", "session": "RUN1", "fold": 0, "n_training_units": len(model["cell_ids"])}.items():
        frame[key] = value
    summary = summarize_predictions(frame)
    assert len(summary) == 8
    assert summary.measured_windows.max() == len(centers)
    folds = pd.DataFrame({"status": ["scored"], "n_test_windows": [len(centers)]})
    assert dict(validation_gates(folds, frame, 1, True))["overall_technical"]
    duplicate = pd.concat([frame.iloc[:-1], frame.iloc[[0]]])
    assert not dict(validation_gates(folds, duplicate, 1, True))["all_window_likelihood_rows_present"]
    frame["spike_support_pass"] = False
    filtered = summarize_predictions(frame).query("test_bin_filter == 'at_least_2cells_3spikes'")
    assert filtered.measured_windows.eq(0).all()
    assert filtered.median_posterior_mean_error_cm.isna().all()
    assert not filtered.beats_cell_ID_null_descriptively.any()


def test_empty_validation_cannot_pass():
    gates = dict(validation_gates(pd.DataFrame(), pd.DataFrame(), 0, True))
    assert not gates["overall_technical"]
    assert not gates["all_window_likelihood_rows_present"]
