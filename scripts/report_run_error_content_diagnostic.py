#!/usr/bin/env python3
"""Non-rescoring validation of the frozen PF-trained RUN-error diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

KEY = ['dataset', 'animal', 'session', 'source', 'split', 'draw', 'event_index']
GROUP = KEY[:-1]
POLICIES = {
    'diagnostic_full': ('prediction_full', True),
    'spikes_entropy': ('prediction_spikes_entropy', True),
    'lowest_A_entropy': ('a_entropy', True),
    'highest_A_spikes': ('a_spikes', False),
}
MODELS = ('mean', 'spikes_entropy', 'full')
CASES = {('real', -1), ('run_test', -1), ('sim_matched', 0), ('sim_matched', 1), ('sim_drift', 0), ('sim_drift', 1)}
METRICS = ('endpoint_separation_cm', 'regional_tv', 'a_entropy', 'b_entropy', 'mean_entropy',
           'a_width_cm', 'b_width_cm', 'a_spikes', 'b_spikes', 'a_active', 'b_active')
TRUTH_METRICS = ('a_truth_error_cm', 'b_truth_error_cm', 'mean_truth_error_cm')


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_frame(frame):
    required = set(KEY + list(METRICS[:4]) + ['a_width_cm', 'b_width_cm', 'a_spikes', 'b_spikes',
        'a_active', 'b_active', 'a_truth_error_cm', 'b_truth_error_cm', 'true_tile', 'grid_diagonal_cm'] +
        ['prediction_' + m for m in MODELS])
    require(required.issubset(frame.columns), 'missing required columns')
    require(not frame.empty and not frame.duplicated(KEY).any(), 'empty/duplicate readouts')
    require(frame[KEY].notna().all().all(), 'missing event identity')
    for _, session in frame.groupby(['dataset', 'animal', 'session']):
        require(set(session.split) == {0, 1, 2}, 'missing frozen split')
        reference_ids = None
        for _, split in session.groupby('split'):
            require(set(zip(split.source, split.draw, strict=True)) == CASES, 'missing or extra source/draw')
            ids = set(split.loc[split.source.eq('real'), 'event_index'])
            if reference_ids is None: reference_ids = ids
            require(ids == reference_ids, 'split-specific candidate selection')
            for source, draw in CASES - {('run_test', -1)}:
                require(set(split.loc[split.source.eq(source) & split.draw.eq(draw), 'event_index']) == ids,
                        'simulation/event cohort mismatch')
    numeric = [x for x in required if x not in KEY + ['true_tile', 'a_truth_error_cm', 'b_truth_error_cm']]
    require(np.isfinite(frame[numeric].to_numpy(float)).all(), 'nonfinite metric/prediction')
    require((frame.grid_diagonal_cm > 0).all(), 'invalid arena size')
    require((frame[['prediction_' + m for m in MODELS]] >= 0).all().all(), 'negative predicted error')
    require(((frame[['a_entropy', 'b_entropy', 'regional_tv']] >= -1e-12) &
             (frame[['a_entropy', 'b_entropy', 'regional_tv']] <= 1 + 1e-12)).all().all(), 'invalid entropy/TV')
    real = frame.source.eq('real')
    require(frame.loc[real, ['a_truth_error_cm', 'b_truth_error_cm']].isna().all().all(), 'replay truth invented')
    require(frame.loc[real, 'true_tile'].eq(-1).all(), 'replay truth tile invented')
    known = frame.loc[~real]
    require(np.isfinite(known[list(TRUTH_METRICS[:2])].to_numpy()).all(), 'missing known truth')
    require((known[list(TRUTH_METRICS[:2])] >= 0).all().all(), 'negative true error')
    require(known.true_tile.isin(range(9)).all(), 'missing known true tile')


def selected_indices(frame, policy):
    require(policy in POLICIES and len(frame) > 0, 'unknown policy/empty selection')
    score, ascending = POLICIES[policy]
    require(np.isfinite(frame[score]).all(), 'nonfinite selection score')
    return frame.sort_values([score, 'event_index'], ascending=[ascending, True], kind='stable').index[:(len(frame) + 1) // 2]


def location_reference(frame, selected, metric):
    require(len(selected) > 0, 'empty retained set')
    means = frame.groupby('true_tile')[metric].mean()
    proportions = selected.true_tile.value_counts(normalize=True)
    require(proportions.index.isin(means.index).all(), 'reference missing selected tile')
    return float(np.sum(means.loc[proportions.index] * proportions))


def selection_tables(frame):
    data = frame.copy()
    data['mean_entropy'] = (data.a_entropy + data.b_entropy) / 2
    data['mean_truth_error_cm'] = (data.a_truth_error_cm + data.b_truth_error_cm) / 2
    metrics, retention, identities = [], [], []
    for values, local in data.groupby(GROUP, sort=True):
        identity = dict(zip(GROUP, values, strict=True))
        known = identity['source'] != 'real'
        names = METRICS + TRUTH_METRICS if known else METRICS
        for policy in POLICIES:
            chosen = local.loc[selected_indices(local, policy)]
            common = dict(**identity, policy=policy, n_available=len(local), n_selected=len(chosen),
                          actual_retention=len(chosen) / len(local))
            for metric in names:
                metrics.append(dict(**common, metric=metric, baseline=float(local[metric].mean()),
                    selected=float(chosen[metric].mean()),
                    location_reference=location_reference(local, chosen, metric) if metric in TRUTH_METRICS else np.nan))
            selections = chosen[KEY].copy()
            score = POLICIES[policy][0]
            selections['policy'] = policy
            selections['selection_rank'] = np.arange(1, len(chosen) + 1)
            selections['selection_score_name'] = score
            selections['selection_score_value'] = chosen[score].to_numpy()
            identities.append(selections)
            if known:
                for tile in range(9):
                    available = int(local.true_tile.eq(tile).sum())
                    kept = int(chosen.true_tile.eq(tile).sum())
                    retention.append(dict(**common, true_tile=tile, tile_available=available, tile_selected=kept,
                        tile_retention=kept / available if available else np.nan,
                        selected_location_share=kept / len(chosen)))
    return pd.DataFrame(metrics), pd.DataFrame(retention), pd.concat(identities, ignore_index=True)


def summarize_selection(metrics, seed=20260914, n_bootstrap=2000):
    keys = ['dataset', 'source', 'split', 'policy', 'metric']
    columns = ['baseline', 'selected', 'location_reference']
    # Repeated simulated draws and uneven event/session counts never create extra animals.
    sessions = metrics.groupby(keys + ['animal', 'session'], as_index=False)[columns].mean()
    animals = sessions.groupby(keys + ['animal'], as_index=False)[columns].mean()
    animals['reduction'] = animals.baseline - animals.selected
    animals['relative_reduction'] = animals.reduction / animals.baseline.replace(0, np.nan)
    animals['location_matched_change'] = animals.selected - animals.location_reference
    rows = []
    rng = np.random.default_rng(seed)
    for values, local in animals.groupby(keys, sort=True):
        identity = dict(zip(keys, values, strict=True))
        draws = rng.integers(len(local), size=(n_bootstrap, len(local)))
        reductions = local.reduction.to_numpy()[draws].mean(axis=1)
        means = local[columns].mean()
        baseline, selected = float(means.baseline), float(means.selected)
        rows.append(dict(**identity, animals=len(local), baseline=baseline, selected=selected,
            change=selected - baseline, relative_reduction=(baseline - selected) / baseline if baseline > 0 else np.nan,
            reduction_ci_low=float(np.quantile(reductions, .025)), reduction_ci_high=float(np.quantile(reductions, .975)),
            animals_improved=int(local.reduction.gt(0).sum()), location_reference=float(means.location_reference),
            location_matched_change=float(selected - means.location_reference)))
    return sessions, animals, pd.DataFrame(rows)


def weighted_corr(x, y, w):
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
    require(np.isfinite(np.c_[x, y, w]).all() and np.all(w > 0), 'invalid correlation observations')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    w = w / w.sum()
    xc, yc = x - w @ x, y - w @ y
    denominator = np.sqrt((w @ (xc * xc)) * (w @ (yc * yc)))
    return float(w @ (xc * yc) / denominator) if denominator > 0 else np.nan


def forecast_tables(frame):
    native = frame.loc[frame.source.eq('run_test')].copy()
    native['target'] = np.log1p(native.b_truth_error_cm / native.grid_diagonal_cm)
    session_rows, animal_rows = [], []
    for values, local in native.groupby(['dataset', 'animal', 'session', 'split']):
        identity = dict(zip(['dataset', 'animal', 'session', 'split'], values, strict=True))
        for model in MODELS:
            prediction = local['prediction_' + model]
            session_rows.append(dict(**identity, model=model, rows=len(local), log_mse=float(np.mean((prediction - local.target) ** 2))))
    session = pd.DataFrame(session_rows)
    for values, local in native.groupby(['dataset', 'animal', 'split']):
        identity = dict(zip(['dataset', 'animal', 'split'], values, strict=True))
        weights = 1 / local.groupby('session').session.transform('size').to_numpy()
        for model in MODELS:
            p = local['prediction_' + model]
            animal_rows.append(dict(**identity, model=model, sessions=local.session.nunique(), rows=len(local),
                log_mse=float(np.average((p - local.target) ** 2, weights=weights)),
                pearson_risk_B_error_cm=weighted_corr(p, local.b_truth_error_cm, weights),
                pearson_risk_log_normalized_error=weighted_corr(p, local.target, weights)))
    animals = pd.DataFrame(animal_rows)
    summary = animals.groupby(['dataset', 'split', 'model'], as_index=False).agg(
        log_mse=('log_mse', 'mean'), animals=('animal', 'nunique'),
        animals_positive_error_correlation=('pearson_risk_B_error_cm', lambda x: int(x.gt(0).sum())),
        animals_finite_error_correlation=('pearson_risk_B_error_cm', lambda x: int(np.isfinite(x).sum())))
    return session, animals, summary


def build_gates(summary, forecast, technical_pass):
    rows = []
    def gate(name, passed, detail):
        rows.append(dict(gate=name, status='pass' if bool(passed) else 'fail', detail=detail))

    gate('complete_external_cohort_and_audit', technical_pass, '25 Tanni sessions / 5 animals; all frozen splits and cases; verified numerical audit')
    def metric(name, source='real'):
        local = summary.loc[summary.dataset.eq('tanni2022') & summary.split.eq(0) &
                            summary.policy.eq('diagnostic_full') & summary.metric.eq(name) & summary.source.eq(source)]
        require(len(local) == 1 and int(local.iloc[0].animals) == 5, 'missing external summary cohort')
        return local.iloc[0]
    models = forecast.loc[forecast.dataset.eq('tanni2022') & forecast.split.eq(0)].set_index('model')
    require(set(models.index) == set(MODELS) and models.animals.eq(5).all(), 'missing forecast cohort')
    full = models.loc['full']
    for baseline in ('mean', 'spikes_entropy'):
        error = float(models.loc[baseline, 'log_mse'])
        reduction = (error - full.log_mse) / error if error > 0 else np.nan
        gate('forecast_improves_vs_' + baseline, np.isfinite(reduction) and reduction >= .05,
             f'relative log-MSE reduction={reduction:.6g}; required >=0.05')
    gate('forecast_error_correlation_positive', full.animals_finite_error_correlation == 5 and full.animals_positive_error_correlation >= 4,
         f'positive physical-error correlations={int(full.animals_positive_error_correlation)}/5; finite={int(full.animals_finite_error_correlation)}/5')
    for name in ('regional_tv', 'endpoint_separation_cm'):
        m = metric(name)
        gate(name + '_reduced', m.relative_reduction >= .1 and m.animals_improved >= 4 and m.reduction_ci_low > 0,
             f'reduction={m.relative_reduction:.6g}; positive animals={m.animals_improved}/5; reduction CI=[{m.reduction_ci_low:.6g},{m.reduction_ci_high:.6g}]')
    entropies = [metric(x) for x in ('mean_entropy', 'a_entropy', 'b_entropy')]
    gate('no_entropy_broadening', entropies[0].change <= 1e-12 and all(x.change <= .01 + 1e-12 for x in entropies[1:]),
         'changes mean/A/B=' + ','.join(f'{x.change:.6g}' for x in entropies))
    for source in ('run_test', 'sim_matched', 'sim_drift'):
        errors = [metric(x, source) for x in ('mean_truth_error_cm', 'a_truth_error_cm', 'b_truth_error_cm')]
        gate(source + '_truth_not_worse', errors[0].change <= 1e-12 and all(x.change <= 2 + 1e-12 for x in errors[1:]),
             'error changes mean/A/B cm=' + ','.join(f'{x.change:.6g}' for x in errors))
        gate(source + '_location_matched_not_worse', np.isfinite(errors[0].location_matched_change) and errors[0].location_matched_change <= 1e-12,
             f'mean A/B error change vs retained-location-matched reference={errors[0].location_matched_change:.6g} cm')
    gate('overall_external_screen', all(r['status'] == 'pass' for r in rows),
         'All criteria required; this screen alone cannot certify the original matched-population remedy')
    return pd.DataFrame(rows)


def load_verified(root):
    audit_path = root / 'audit/reconstruction.json'
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'pass', 'numerical audit incomplete')
    for key, value in audit['input_file_paths'].items():
        require(file_sha256(value) == audit['input_file_sha256'][key], 'audit input changed: ' + key)
    frames = []
    for short, dataset, expected_sessions, expected_animals in [('pf', 'pfeiffer_foster', 8, 4), ('tanni', 'tanni2022', 25, 5)]:
        manifest = json.loads((root / short / 'manifest.json').read_text())
        require(manifest['status'] == 'complete' and manifest['inputs_unchanged'], 'application incomplete')
        results = manifest['results']
        require(len(results) == expected_sessions and len({r['animal'] for r in results}) == expected_animals, 'incomplete animal/session cohort')
        require(len({(r['animal'], r['session']) for r in results}) == expected_sessions, 'duplicate manifest session')
        checked = [r for r in audit['applied'] if r['dataset'] == dataset and r['status'] == 'pass']
        require({(r['animal'], r['session']) for r in checked} == {(r['animal'], r['session']) for r in results}, 'unaudited session')
        for entry in results:
            require(file_sha256(entry['path']) == entry['sha256'], 'applied rows changed')
            data = pd.read_csv(entry['path'])
            require(len(data) == entry['rows'] and data.dataset.eq(dataset).all() and data.animal.eq(entry['animal']).all()
                    and data.session.eq(entry['session']).all(), 'readout identity mismatch')
            validate_frame(data)
            frames.append(data)
    return pd.concat(frames, ignore_index=True)


def figure(summary, forecast, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = summary.loc[summary.dataset.eq('tanni2022') & summary.split.eq(0) & summary.policy.eq('diagnostic_full')]
    f = forecast.loc[forecast.dataset.eq('tanni2022') & forecast.split.eq(0)].set_index('model')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), layout='constrained')
    axes[0, 0].bar(['Constant', 'Spikes +\nentropy', 'Full risk'], f.loc[list(MODELS), 'log_mse'], color=['#737373', '#4c956c', '#b55252'])
    axes[0, 0].set(title='Known RUN error forecasting', ylabel='Mean squared log-normalized error')
    for axis, name, title, label in [(axes[0, 1], 'endpoint_separation_cm', 'Real endpoint disagreement', 'A/B mean separation (cm)'),
                                      (axes[0, 2], 'regional_tv', 'Real regional disagreement', 'Nine-tile posterior TV'),
                                      (axes[1, 0], 'mean_entropy', 'Real posterior concentration', 'Mean A/B normalized entropy')]:
        row = s.loc[s.source.eq('real') & s.metric.eq(name)].iloc[0]
        axis.bar(['Random\nexpectation', 'Selected\n50%'], [row.baseline, row.selected], color=['#737373', '#b55252'])
        axis.set(title=title, ylabel=label)
    known = s.loc[s.metric.eq('mean_truth_error_cm')].set_index('source').loc[['run_test', 'sim_matched', 'sim_drift']]
    x = np.arange(3)
    axes[1, 1].bar(x - .2, known.baseline, .4, label='Random expectation', color='#737373')
    axes[1, 1].bar(x + .2, known.selected, .4, label='Selected 50%', color='#b55252')
    axes[1, 1].set(xticks=x, xticklabels=['RUN', 'Matched\nsimulation', 'Drift\nsimulation'], title='Known location accuracy', ylabel='Mean A/B error (cm)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].bar(x, known.location_matched_change, color='#b55252')
    axes[1, 2].axhline(0, color='black', lw=.8)
    axes[1, 2].set(xticks=x, xticklabels=['RUN', 'Matched\nsimulation', 'Drift\nsimulation'],
                   title='After matching retained locations', ylabel='Selected minus location-matched error (cm)')
    for axis in axes.flat:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen PF-trained diagnostic applied to Tanni | primary split | equal-animal summaries', fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    data = load_verified(args.root)
    metrics, retention, selected = selection_tables(data)
    sessions, animals, summary = summarize_selection(metrics)
    forecast_session, forecast_animal, forecast = forecast_tables(data)
    gates = build_gates(summary, forecast, technical_pass=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tables = dict(selection_by_draw=metrics, selection_by_session=sessions, selection_by_animal=animals,
        selection_summary=summary, true_location_retention=retention, forecast_by_session=forecast_session,
        forecast_by_animal=forecast_animal, forecast_summary=forecast, gate_summary=gates)
    for name, frame in tables.items(): frame.to_csv(args.output_dir / (name + '.csv'), index=False)
    selected.to_csv(args.output_dir / 'selected_event_ids.csv.gz', index=False)
    figure(summary, forecast, args.output_dir / 'external_validation.png')
    passed = gates.iloc[-1].status == 'pass'
    verdict = 'external_screen_pass_original_contrast_still_required' if passed else 'external_screen_failed_no_validated_remedy'
    primary = summary.loc[summary.dataset.eq('tanni2022') & summary.split.eq(0) & summary.policy.eq('diagnostic_full')]
    lines = ['# RUN-error-trained diagnostic: external validation', '', 'Verdict: **' + verdict + '**', '',
        'PF third-quarter RUN trained the frozen predictor. Tanni is external to training but has been used in previous failed development experiments; it is not pristine confirmation.', '',
        'All 25 Tanni sessions / 5 animals retained. Selection: lowest predicted risk 50% (rounded up) within session, source, split and draw. Split0 primary; other splits are sensitivity only.', '',
        'No events rescored. Every reported source and all known errors passed an independent reconstruction. No B replay features enter selection.', '',
        'Baseline is the exact expected mean under random fixed-size retention, computed over all eligible rows; it is not an actual random 50% draw. Events average within session, draws within source, sessions within animal, then equal animals.', '',
        'Forecast error is MSE for log(1+B position error/arena diagonal). The correlation gate uses session-balanced Pearson correlation of predicted risk with physical B error in cm, separately by animal. Log-normalized error correlation is also reported.', '',
        '| Gate | Status | Detail |', '|---|---|---|']
    lines += [f'| {r.gate} | {r.status} | {r.detail} |' for r in gates.itertuples()]
    lines += ['', '## Primary retained-content metrics', '', '| Source | Metric | Random expectation | Selected 50% | Change | Location-matched change |', '|---|---|---:|---:|---:|---:|']
    keep = primary.loc[primary.metric.isin(['endpoint_separation_cm', 'regional_tv', 'mean_entropy', 'mean_truth_error_cm'])]
    lines += [f'| {r.source} | {r.metric} | {r.baseline:.6g} | {r.selected:.6g} | {r.change:.6g} | {r.location_matched_change:.6g} |' for r in keep.itertuples()]
    lines += ['', '## Scope', '',
        'Disjoint population agreement is not replay ground truth. Known RUN and conditional simulations constrain accuracy but do not reproduce all replay noise correlations or state changes. True-tile reweighting is validation only; it does not select events or estimate real replay truth.', '',
        'The five-animal cluster intervals are descriptive. The full-RUN unit/grid eligibility remains a conditioning limitation. No predictor, retention, split or animal was chosen based on external outcomes.', '',
        'Even a pass here would not establish a correction of regional replay prevalence. Transfer back to the original targeted matched-population contrast remains required before declaring the user objective achieved.', '']
    (args.output_dir / 'report.md').write_text('\n'.join(lines))
    inputs = dict(reporter=Path(__file__), audit=args.root / 'audit/reconstruction.json',
        model=args.root / 'frozen/frozen_model.json', pf=args.root / 'pf/manifest.json', tanni=args.root / 'tanni/manifest.json')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status='complete', verdict=verdict, external_screen_passed=passed, objective_achieved=False,
        non_rescoring=True, primary_split=0, external_pristine_confirmation=False,
        output_sha256={path.name: file_sha256(path) for path in args.output_dir.iterdir() if path.is_file()})
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='complete', verdict=verdict)), flush=True)


if __name__ == '__main__':
    main()
