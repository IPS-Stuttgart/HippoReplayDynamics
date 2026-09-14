from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage_data import CoverageInputConfig, fit_coverage_population
from scripts.prepare_edge_support_content_inputs import freeze_candidates, training_session
from scripts.measure_edge_support_content import (
    conditional_draw, known_path, occupied_graph, read_event, select_windows, window_sums,
)


def synthetic_run():
    rng = np.random.default_rng(30)
    t = np.arange(0, 200, .02)
    xy = np.column_stack([60+60*np.sin(2*np.pi*t/10), 60+60*np.cos(2*np.pi*t/13)])
    centers = rng.uniform(15, 105, (20, 2))
    rates = .05 + 12*np.exp(-np.sum((xy[:, None, :]-centers[None, :, :])**2, axis=2)/(2*20**2))
    counts = rng.poisson(.02*rates)
    spikes = np.concatenate([np.column_stack([np.repeat(t+.01, counts[:, i]),
                                             np.full(counts[:, i].sum(), i+1)]) for i in range(20)])
    empty = np.empty((0, 2))
    return ReplaySession(rat="r", name="s", path=Path("synthetic"), position=np.column_stack([t, xy]),
        spikes=spikes, tetrode_cell_ids=np.column_stack([np.ones(20), np.arange(1, 21)]),
        excitatory_neurons=np.arange(1, 21), inhibitory_neurons=np.empty(0, int), ripple_events=np.empty((0, 6)),
        run_times=np.array([[0., 200.]]), sleep_box_immobile_times=empty, sleep_times=empty, rem_times=empty,
        well_sequence=None, metadata={"source_dataset": "synthetic"})


def test_first_half_rates_and_qc_ignore_heldout_spikes():
    session = synthetic_run()
    train, mid, bounds = training_session(session)
    assert mid == 100 and bounds == (0, 200)
    assert train.run_times[-1, 1] < 100
    before, units, _ = fit_coverage_population(train, CoverageInputConfig())
    changed = session.spikes.copy()
    changed[changed[:, 0] >= 100, 0] = 150
    changed = np.vstack([changed, np.column_stack([np.full(1000, 100.), np.ones(1000)])])
    after, altered, _ = fit_coverage_population(training_session(replace(session, spikes=changed))[0], CoverageInputConfig())
    np.testing.assert_array_equal(before["rates_hz"], after["rates_hz"])
    np.testing.assert_array_equal(before["unit_qc_mask"], after["unit_qc_mask"])
    pd.testing.assert_frame_equal(units, altered)
    assert before["unit_qc_mask"].sum() >= 10


def test_frozen_sampling_ignores_outcomes_and_input_row_order():
    candidates = pd.DataFrame(dict(event_index=np.arange(500), start_s=np.arange(500.),
                                    end_s=np.arange(500.)+.1, spikes=np.arange(500)))
    first = freeze_candidates(candidates, "data", "session", 12)
    assert len(first) == 200
    altered = candidates.sample(frac=1, random_state=2)
    altered["spikes"] = 999
    second = freeze_candidates(altered, "data", "session", 12)
    np.testing.assert_array_equal(first.event_index, second.event_index)
    with pytest.raises(ValueError, match="duplicate"):
        freeze_candidates(pd.concat([candidates, candidates]), "d", "s", 1)


def test_window_sum_and_a_only_timing_does_not_see_b():
    base = np.array([[1, 0], [0, 2], [0, 0], [1, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(window_sums(base), [[2, 3], [1, 3], [1, 1]])
    a = np.array([[2, 1], [1, 2], [0, 1], [0, 0]])
    b = np.zeros_like(a)
    first = select_windows(a, b)
    second = select_windows(a, np.full_like(b, 5))
    assert first["a_supported_edge"] == second["a_supported_edge"] == 1
    assert first["joint_supported_edge"] == -1
    assert second["joint_supported_edge"] == 1
    assert first["pooled_two_spike_edge"] == 1
    assert second["pooled_two_spike_edge"] == 3
    assert first["raw_endpoint"] == 3


def test_no_valid_window_abstains_without_fallback():
    zero = np.zeros((8, 5), int)
    result = select_windows(zero, zero)
    assert result["raw_endpoint"] == 7
    assert all(result[k] == -1 for k in result if k != "raw_endpoint")


def test_known_path_stays_on_occupied_graph_and_exact_total_draw():
    grid = np.column_stack([np.arange(0., 88, 8), np.zeros(11)])
    rates = np.arange(1, 34).reshape(3, 11)
    graph = occupied_graph(grid)
    truth, rate = known_path(grid, rates, graph, 40, "sim_moving", 20)
    assert truth.shape == (40, 2) and rate.shape == (40, 3)
    assert ((truth[:, 0] >= 0) & (truth[:, 0] <= 80)).all()
    assert np.all(truth[:, 1] == 0)
    assert np.max(np.linalg.norm(np.diff(truth, axis=0), axis=1)) <= 5.0001
    totals = np.arange(40) % 8
    sample = conditional_draw(totals, rate, 55)
    np.testing.assert_array_equal(sample.sum(axis=1), totals)
    np.testing.assert_array_equal(sample, conditional_draw(totals, rate, 55))
    np.testing.assert_array_equal(conditional_draw(totals, rate, 55, True).sum(axis=1), totals)
    stationary, _ = known_path(grid, rates, graph, 40, "sim_stationary", 20)
    np.testing.assert_array_equal(stationary, np.tile(stationary[0], (40, 1)))
    jumped, _ = known_path(grid, rates, graph, 40, "sim_late_jump", 20)
    np.testing.assert_allclose(jumped[:-4], truth[:-4])
    np.testing.assert_array_equal(jumped[-4:], np.tile(jumped[-1], (4, 1)))


def test_earlier_readout_cannot_be_called_terminal_recovery():
    grid = np.column_stack([np.arange(0., 88, 8), np.zeros(11)])
    left = .01+100*np.exp(-(grid[:, 0]/8)**2/2)
    rates = np.tile(left, (6, 1))
    base = np.zeros((12, 6), int)
    base[:4, [0, 1, 3, 4]] = 1
    truth = np.zeros((12, 2))
    truth[-4:, 0] = 80
    rows = pd.DataFrame(read_event(base, rates, grid, (np.arange(3), np.arange(3, 6)), 10., 1, truth)).set_index("policy")
    trimmed = rows.loc["a_supported_edge"]
    assert trimmed.status == "available" and trimmed.shift_earlier_ms > 0
    assert trimmed.truth_shift_cm == pytest.approx(80)
    assert trimmed.a_selected_truth_error_cm < trimmed.a_original_truth_error_cm
    assert trimmed.a_original_truth_error_cm > trimmed.raw_a_original_truth_error_cm
    assert rows.loc["raw_endpoint", "shift_earlier_ms"] == 0
