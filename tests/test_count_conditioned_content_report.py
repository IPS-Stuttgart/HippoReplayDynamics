import numpy as np
import pandas as pd

from scripts.audit_count_conditioned_content import SOURCES, METHODS
from scripts.measure_count_conditioned_content import readout
from scripts.report_count_conditioned_content import gates, support_strata
from scripts.report_encoding_uncertainty_content import METRICS, paired_sessions


def summary_fixture():
    rows = []
    for dataset in ('pfeiffer_foster', 'hc11'):
        for source in SOURCES:
            for comparator in ('poisson', 'poisson_entropy_matched'):
                for metric in METRICS:
                    rows.append(dict(dataset=dataset, source=source, method='conditional_multinomial',
                        split=0, comparator=comparator, metric=metric, delta=-.1, relative_delta=-.2,
                        delta_upper95=.5, animals=4, animals_improved=4, sessions=8, events=1600,
                        control_available=True, finite_animals=4))
    return pd.DataFrame(rows)


def test_frozen_gate_pass_and_empty_failure():
    s = summary_fixture()
    assert gates(s, True).passed.all()
    assert not gates(s, False).loc[lambda x: x.gate.eq('promising_remedy_screen'), 'passed'].any()
    assert not gates(s.iloc[:0], True).loc[lambda x: x.gate.eq('promising_remedy_screen'), 'passed'].any()


def test_unconditional_controls_cannot_be_omitted_or_ignored():
    s = summary_fixture()
    missing = gates(s.loc[~s.source.str.contains('poisson')], True)
    assert not missing.loc[missing.gate.eq('promising_remedy_screen'), 'passed'].any()
    s.loc[s.source.eq('sim_poisson_stationary') & s.metric.eq('b_error'), 'delta_upper95'] = 2.1
    s.loc[s.source.eq('sim_poisson_stationary') & s.metric.eq('mean_brier'), 'delta'] = .01
    g = gates(s, True)
    assert not g.loc[g.gate.eq('promising_remedy_screen'), 'passed'].any()


def test_entropy_matching_blocks_agreement_by_flattening():
    s = summary_fixture()
    s.loc[s.comparator.eq('poisson_entropy_matched') & s.source.eq('real'), 'delta'] = 0.
    g = gates(s, True)
    assert not g.loc[g.gate.isin(['beats_entropy_control_separation', 'beats_entropy_control_regional_tv', 'promising_remedy_screen']), 'passed'].any()


def test_other_splits_and_silent_populations_cannot_hide_primary_failure():
    s = summary_fixture()
    sensitivity = s.copy()
    sensitivity['split'] = 1
    s.loc[s.source.eq('real'), 'events'] = 0
    s.loc[s.source.eq('real'), 'relative_delta'] = .1
    g = gates(pd.concat([s, sensitivity]), True)
    assert not g.loc[g.gate.eq('promising_remedy_screen'), 'passed'].any()


def test_shared_pairing_and_empty_activity_strata():
    rng = np.random.default_rng(315)
    counts, rates = np.zeros((12, 8), int), rng.uniform(.1, 5, (8, 25))
    grid = np.array([[x, y] for x in range(5) for y in range(5)], float)*8
    frames, _ = readout(counts, np.full((12, 2), np.nan), rates, grid, [np.arange(4), np.arange(4, 8)])
    for f in frames:
        f['event_index'] = np.arange(12)
        f['dataset'], f['animal'], f['session'], f['source'], f['split'] = 'hc11', 'rat', 'session1', 'real', 0
        f['original_start_s'], f['original_end_s'], f['n_cells_per_group'] = 0., .02, 4
    all_rows = pd.concat(frames, ignore_index=True)
    session, diagnostic = paired_sessions(all_rows, methods=METHODS, diagnostic_column='a_total_count_reliance_tv')
    assert not session.empty
    assert diagnostic.spearman_rho.isna().all()
    strata = support_strata(all_rows).set_index('activity_stratum')
    assert strata.loc['both_zero', 'events'] == 12
    assert strata.loc['both_zero', 'separation_after'] == 0.
    assert strata.loc['both_supported_sensitivity', 'events'] == 0
    assert np.isnan(strata.loc['both_supported_sensitivity', 'mean_error_delta'])
