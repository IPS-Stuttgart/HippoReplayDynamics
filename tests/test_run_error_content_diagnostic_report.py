import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts._provenance import file_sha256
from scripts.report_run_error_content_diagnostic import (
    CASES, POLICIES, build_gates, forecast_tables, load_verified, location_reference,
    main, selected_indices, selection_tables, summarize_selection, validate_frame,
)


def fixture(dataset='tanni2022', animal='R0', session='S0', n=8):
    rows = []
    for split in range(3):
        for source, draw in sorted(CASES):
            for i in range(n):
                rows.append(dict(dataset=dataset, animal=animal, session=session, split=split,
                    source=source, draw=draw, event_index=i, a_spikes=n-i, a_active=n-i, b_spikes=i,
                    b_active=i, a_entropy=.1 + .05*i, b_entropy=.1 + .05*i,
                    a_width_cm=10+i, b_width_cm=10+i, endpoint_separation_cm=1+i, regional_tv=.02*(1+i),
                    a_truth_error_cm=np.nan if source == 'real' else 1+i,
                    b_truth_error_cm=np.nan if source == 'real' else 1+i,
                    true_tile=-1 if source == 'real' else i % 3, grid_diagonal_cm=100,
                    prediction_mean=.2, prediction_full=np.log1p((1+i)/100), prediction_spikes_entropy=.1 + .01*i))
    return pd.DataFrame(rows)


def five_animals():
    return pd.concat([fixture(animal=f'R{i}', session=f'S{i}') for i in range(5)], ignore_index=True)


def results():
    f = five_animals()
    metrics, _, _ = selection_tables(f)
    _, _, summary = summarize_selection(metrics, n_bootstrap=10)
    _, _, forecast = forecast_tables(f)
    return summary, forecast


def test_selection_has_no_B_or_truth_access_and_deterministic_ties():
    frame = fixture().query("source == 'real' and split == 0").copy()
    frame.prediction_full = .3
    changed = frame.sample(frac=1, random_state=14).copy()
    for name in ['b_spikes', 'b_entropy', 'b_truth_error_cm', 'true_tile', 'endpoint_separation_cm']:
        changed[name] = np.arange(len(frame)) + 100
    np.testing.assert_array_equal(frame.loc[selected_indices(frame, 'diagnostic_full'), 'event_index'], np.arange(4))
    np.testing.assert_array_equal(changed.loc[selected_indices(changed, 'diagnostic_full'), 'event_index'], np.arange(4))
    odd = frame.iloc[:5]
    assert len(selected_indices(odd, 'diagnostic_full')) == 3


def test_known_location_reference_exposes_selection_of_easy_tiles():
    data = pd.DataFrame({'true_tile': [0, 0, 1, 1], 'mean_truth_error_cm': [1., 1., 11., 11.]})
    chosen = data.iloc[:2]
    assert location_reference(data, chosen, 'mean_truth_error_cm') == 1
    assert chosen.mean_truth_error_cm.mean() - data.mean_truth_error_cm.mean() == -5


def test_missing_truth_cases_or_splits_fail_closed():
    frame = fixture()
    validate_frame(frame)
    for bad in [frame.iloc[:0], frame.loc[frame.split.ne(2)], frame.loc[~(frame.source.eq('sim_matched') & frame.draw.eq(1))]]:
        with pytest.raises(ValueError): validate_frame(bad)
    broken = frame.copy()
    broken.loc[broken.source.eq('run_test'), 'b_truth_error_cm'] = np.nan
    with pytest.raises(ValueError, match='missing known truth'): validate_frame(broken)
    with pytest.raises(ValueError, match='duplicate'): validate_frame(pd.concat([frame, frame.iloc[:1]]))


def test_unequal_event_and_draw_counts_cannot_dominate_animals():
    first = fixture(animal='A', session='large', n=8)
    second = fixture(animal='B', session='small', n=4)
    second['endpoint_separation_cm'] = 100.
    frame = pd.concat([first, second], ignore_index=True)
    metric, _, selected = selection_tables(frame)
    _, animals, summary = summarize_selection(metric, n_bootstrap=10)
    result = summary.query("source == 'real' and split == 0 and policy == 'diagnostic_full' and metric == 'endpoint_separation_cm'").iloc[0]
    assert result.baseline == pytest.approx((4.5 + 100) / 2)
    assert result.animals == 2
    assert len(animals.query("source == 'sim_matched' and split == 0 and policy == 'diagnostic_full' and metric == 'endpoint_separation_cm'")) == 2
    assert set(selected.policy) == set(POLICIES)


def test_all_frozen_gates_pass_on_known_success_case():
    summary, forecast = results()
    gates = build_gates(summary, forecast, technical_pass=True)
    assert gates.status.eq('pass').all(), gates.to_string(index=False)


@pytest.mark.parametrize('metric,source,column,value,expected', [
    ('mean_entropy', 'real', 'change', .05, 'no_entropy_broadening'),
    ('mean_truth_error_cm', 'sim_drift', 'change', .1, 'sim_drift_truth_not_worse'),
    ('mean_truth_error_cm', 'run_test', 'location_matched_change', 1., 'run_test_location_matched_not_worse'),
    ('regional_tv', 'real', 'animals_improved', 3, 'regional_tv_reduced'),
])
def test_accuracy_entropy_and_animal_failures_cannot_hide(metric, source, column, value, expected):
    summary, forecast = results()
    mask = summary.metric.eq(metric) & summary.source.eq(source) & summary.split.eq(0) & summary.policy.eq('diagnostic_full')
    summary.loc[mask, column] = value
    gates = build_gates(summary, forecast, technical_pass=True).set_index('gate')
    assert gates.loc[expected, 'status'] == 'fail'
    assert gates.loc['overall_external_screen', 'status'] == 'fail'


def test_missing_summary_animal_or_baseline_never_passes():
    summary, forecast = results()
    forecast.loc[forecast.model.eq('full'), 'animals'] = 4
    with pytest.raises(ValueError, match='forecast cohort'):
        build_gates(summary, forecast, technical_pass=True)


def create_root(tmp_path):
    audit_entries = []
    for short, dataset, n_animals, n_sessions in [('pf', 'pfeiffer_foster', 4, 2), ('tanni', 'tanni2022', 5, 5)]:
        folder = tmp_path / short
        folder.mkdir()
        results = []
        for a in range(n_animals):
            for s in range(n_sessions):
                animal, session = f'R{a}', f'S{a}_{s}'
                frame = fixture(dataset, animal, session)
                path = folder / (session + '.csv.gz')
                frame.to_csv(path, index=False)
                results.append(dict(dataset=dataset, animal=animal, session=session, path=str(path), rows=len(frame), sha256=file_sha256(path)))
                audit_entries.append(dict(dataset=dataset, animal=animal, session=session, status='pass'))
        (folder / 'manifest.json').write_text(json.dumps(dict(status='complete', inputs_unchanged=True, results=results)))
    (tmp_path / 'frozen').mkdir()
    (tmp_path / 'frozen/frozen_model.json').write_text('{}')
    (tmp_path / 'audit').mkdir()
    inputs = {x: str(tmp_path / x / 'manifest.json') for x in ('pf', 'tanni')}
    (tmp_path / 'audit/reconstruction.json').write_text(json.dumps(dict(status='pass', applied=audit_entries,
        input_file_paths=inputs, input_file_sha256={k: file_sha256(v) for k, v in inputs.items()})))


def test_cli_report_outputs_and_hash_refusal(tmp_path, monkeypatch):
    create_root(tmp_path)
    output = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['report', '--root', str(tmp_path), '--output-dir', str(output)])
    main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['external_screen_passed']
    assert manifest['objective_achieved'] is False
    assert manifest['non_rescoring']
    assert (output / 'external_validation.png').stat().st_size > 1000
    assert 'not pristine confirmation' in (output / 'report.md').read_text()
    source = json.loads((tmp_path / 'tanni/manifest.json').read_text())['results'][0]['path']
    Path(source).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='applied rows changed'): load_verified(tmp_path)
