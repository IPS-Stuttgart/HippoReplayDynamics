import numpy as np
import pandas as pd
import pytest

from scripts.audit_encoding_uncertainty_content import METHODS, SOURCES
from scripts.report_encoding_uncertainty_content import (
    METRICS, aggregate, gates, paired_sessions,
)


def frame():
    rows = []
    for method in METHODS:
        for j in range(12):
            r = dict(dataset='hc11', animal='rat', session='run1', source='run_q4', split=0,
                method=method, event_index=j, original_start_s=float(j), original_end_s=j+.02,
                n_cells_per_group=5, a_spikes=j, b_spikes=j+1, a_active=min(j, 3), b_active=2,
                a_entropy_control_available=True, b_entropy_control_available=True, a_encoding_sensitivity_tv=j/5)
            r.update({m: float(j+1) for m in METRICS if not m.startswith('mean_')})
            rows.append(r)
    return pd.DataFrame(rows)


def summary_fixture():
    rows = []
    for dataset in ('pfeiffer_foster', 'hc11'):
        for method in ('gamma_exposure', 'gamma_training_drift'):
            for source in SOURCES:
                for comparator in ('poisson', 'poisson_entropy_matched'):
                    for metric in METRICS:
                        rows.append(dict(dataset=dataset, method=method, source=source, split=0,
                            comparator=comparator, metric=metric, delta=-.1, relative_delta=-.2,
                            delta_upper95=.5, animals=4, animals_improved=4, sessions=8, events=1600,
                            control_available=True, finite_animals=4))
    return pd.DataFrame(rows)


def test_paired_summaries_keep_fixed_events_and_diagnostics():
    s, d = paired_sessions(frame())
    assert s.delta.eq(0).all()
    assert s.events.eq(12).all()
    assert d.events.eq(12).all()
    assert np.allclose(d.spearman_rho, 1)


@pytest.mark.parametrize('kind', ['missing_event', 'missing_model', 'shift_time', 'nan_metric', 'duplicate', 'empty'])
def test_incomplete_or_changed_inputs_fail(kind):
    f = frame()
    if kind == 'missing_event': f = f.iloc[1:]
    if kind == 'missing_model': f = f.loc[f.method.ne('gamma_exposure')]
    if kind == 'shift_time': f.loc[0, 'original_start_s'] += .005
    if kind == 'nan_metric': f.loc[0, 'b_error'] = np.nan
    if kind == 'duplicate': f = pd.concat([f, f.iloc[:1]])
    if kind == 'empty': f = f.iloc[:0]
    with pytest.raises((ValueError, AssertionError)):
        paired_sessions(f)


def test_animals_not_events_or_sessions_set_inference_weights():
    s, _ = paired_sessions(frame())
    s = s.iloc[:1]
    copies = []
    for animal, session, before, after, events in (('a', 'a1', 10, 8, 1000), ('a', 'a2', 10, 8, 10), ('b', 'b1', 20, 20, 5)):
        p = s.copy()
        p['animal'], p['session'], p['before'], p['after'], p['delta'], p['events'] = animal, session, before, after, after-before, events
        copies.append(p)
    animal, result = aggregate(pd.concat(copies))
    assert len(animal) == 2
    assert result.delta.iloc[0] == -1
    assert result.before.iloc[0] == 15
    assert result.after.iloc[0] == 14


def test_frozen_gate_pass_profile_and_nonvacuous_failures():
    s = summary_fixture()
    assert gates(s, True).passed.all()
    assert not gates(s.iloc[:0], True).loc[lambda x: x.gate.eq('promising_remedy_screen'), 'passed'].any()
    assert not gates(s, False).loc[lambda x: x.gate.eq('promising_remedy_screen'), 'passed'].any()
    s.loc[s.source.eq('real'), 'events'] = 0
    assert not gates(s, True).loc[lambda x: x.gate.eq('promising_remedy_screen'), 'passed'].any()


def test_entropy_and_truth_controls_block_false_agreement_success():
    s = summary_fixture()
    s.loc[s.comparator.eq('poisson_entropy_matched') & s.source.eq('real'), 'delta'] = .01
    s.loc[s.source.eq('sim_late_jump') & s.metric.eq('b_error'), 'delta_upper95'] = 2.1
    s.loc[s.source.eq('sim_moving') & s.metric.eq('mean_brier'), 'delta'] = .001
    g = gates(s, True)
    for name in ('beats_entropy_control_separation', 'beats_entropy_control_regional_tv',
                 'each_known_population_error_upper95_at_most_2cm', 'each_simulated_mean_population_brier_not_worse', 'promising_remedy_screen'):
        assert not g.loc[g.gate.eq(name), 'passed'].any()


def test_other_splits_cannot_rescue_primary_and_missing_sources_fail():
    good = summary_fixture()
    bad = good.copy()
    good['split'] = 1
    bad.loc[bad.source.eq('real') & bad.metric.eq('regional_tv'), 'relative_delta'] = -.01
    g = gates(pd.concat([bad, good]), True)
    assert not g.loc[g.gate.eq('promising_remedy_screen'), 'passed'].any()
    g = gates(summary_fixture().loc[lambda x: x.source.ne('run_q4')], True)
    assert not g.loc[g.gate.eq('promising_remedy_screen'), 'passed'].any()
