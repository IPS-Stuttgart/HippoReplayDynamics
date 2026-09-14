from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage_data import CoverageInputConfig, fit_coverage_population
from scripts.prepare_edge_support_content_inputs import freeze_candidates, training_session


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
