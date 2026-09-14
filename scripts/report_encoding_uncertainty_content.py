#!/usr/bin/env python3
"""Non-rescoring, audit-gated report for the frozen uncertainty experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_encoding_uncertainty_content import METHODS, SOURCES

PRIMARY = 'gamma_training_drift'
GROUP = ['dataset', 'source', 'split', 'method', 'comparator', 'metric']
KEYS = ['dataset', 'animal', 'session', 'source', 'split', 'event_index']
METRICS = ('separation_cm', 'regional_tv', 'a_error', 'b_error', 'mean_error',
           'a_brier', 'b_brier', 'mean_brier', 'a_nll', 'b_nll', 'mean_nll',
           'a_entropy', 'b_entropy', 'a_width', 'b_width')


def paired_sessions(frame):
    if frame.empty or frame.duplicated(KEYS+['method']).any():
        raise ValueError('empty or duplicated readouts')
    frame = frame.copy()
    for name in ('error', 'brier', 'nll'):
        frame['mean_'+name] = (frame['a_'+name]+frame['b_'+name])/2
    rows, diagnostics = [], []
    for metadata, g in frame.groupby(['dataset', 'animal', 'session', 'source', 'split']):
        info = dict(zip(['dataset', 'animal', 'session', 'source', 'split'], metadata, strict=True))
        if set(g.method) != set(METHODS):
            raise ValueError('missing method')
        bank = {method: p.set_index('event_index').sort_index() for method, p in g.groupby('method')}
        base = bank['poisson']
        if len(base) == 0:
            raise ValueError('no events')
        for method, p in bank.items():
            if not p.index.equals(base.index):
                raise ValueError('method event sets differ')
            for col in ('original_start_s', 'original_end_s', 'n_cells_per_group', 'a_spikes', 'b_spikes', 'a_active', 'b_active'):
                np.testing.assert_array_equal(p[col], base[col], err_msg='changed time/population: '+col)
            if not np.allclose(p.original_end_s-p.original_start_s, .02, atol=1e-10, rtol=0):
                raise ValueError('changed duration')
            for metric in METRICS:
                values = p[metric]
                known_only = metric.endswith(('error', 'brier', 'nll'))
                if info['source'] == 'real' and known_only:
                    if not values.isna().all():
                        raise ValueError('real candidates have no position truth')
                elif not np.isfinite(values).all():
                    raise ValueError('missing finite metric: '+metric)
            for comparator in ('poisson', 'poisson_entropy_matched'):
                if method == comparator:
                    continue
                for metric in METRICS:
                    a, b = p[metric], bank[comparator][metric]
                    rows.append(dict(info, method=method, comparator=comparator, metric=metric,
                        events=len(p), finite_events=int(np.isfinite(a-b).sum()),
                        before=b.mean(), after=a.mean(), delta=(a-b).mean(),
                        control_available=bool(p.a_entropy_control_available.all() and p.b_entropy_control_available.all())))
        targets = ('separation_cm', 'regional_tv') if info['source'] == 'real' else ('b_error', 'b_brier')
        x = base.a_encoding_sensitivity_tv
        for target in targets:
            y = base[target]
            finite = np.isfinite(x) & np.isfinite(y)
            defined = finite.sum() >= 10 and x[finite].nunique() > 1 and y[finite].nunique() > 1
            rho = float(spearmanr(x[finite], y[finite]).statistic) if defined else np.nan
            diagnostics.append(dict(info, target=target, events=int(finite.sum()), spearman_rho=rho,
                diagnostic_scope='A_only_training_quarter_sensitivity; association_not_validated_certificate'))
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def aggregate(session):
    if session.empty:
        raise ValueError('no session results')
    animal = session.groupby(GROUP+['animal'], as_index=False).agg(
        before=('before', 'mean'), after=('after', 'mean'), delta=('delta', 'mean'),
        sessions=('session', 'nunique'), events=('events', 'sum'), finite_events=('finite_events', 'sum'),
        control_available=('control_available', 'all'))
    results = []
    rng = np.random.default_rng(20260914)
    for metadata, g in animal.groupby(GROUP, sort=True):
        d = g.delta.dropna().to_numpy()
        bounds = [np.nan]*3
        if len(d):
            draws = rng.choice(d, (10000, len(d)), replace=True).mean(axis=1)
            bounds = np.quantile(draws, [.025, .975, .95])
        before, after = g.before.mean(), g.after.mean()
        results.append(dict(zip(GROUP, metadata, strict=True), before=before, after=after, delta=g.delta.mean(),
            relative_delta=after/before-1 if before > 0 else np.nan,
            delta_ci_low=bounds[0], delta_ci_high=bounds[1], delta_upper95=bounds[2],
            animals=len(g), finite_animals=len(d), animals_improved=int((d < 0).sum()),
            sessions=int(g.sessions.sum()), events=int(g.events.sum()), finite_events=int(g.finite_events.sum()),
            control_available=bool(g.control_available.all()), ci_scope='descriptive_four_rat_bootstrap'))
    return animal, pd.DataFrame(results)


def gates(summary, technical_passed):
    rows = []
    for dataset in ('pfeiffer_foster', 'hc11'):
        for method in ('gamma_training_drift', 'gamma_exposure'):
            g = summary.loc[summary.dataset.eq(dataset) & summary.method.eq(method) & summary.split.eq(0)]

            def value(source, metric, field='delta', comparator='poisson'):
                part = g.loc[g.source.eq(source) & g.metric.eq(metric) & g.comparator.eq(comparator)]
                return float(part.iloc[0][field]) if len(part) == 1 else np.nan

            complete = len(g) == len(SOURCES)*2*len(METRICS) and g.animals.eq(4).all() and g.sessions.eq(8).all()
            known = g.loc[g.source.ne('real') & g.comparator.eq('poisson')]
            error = known.loc[known.metric.isin(['a_error', 'b_error'])]
            brier = known.loc[known.source.ne('run_q4') & known.metric.eq('mean_brier')]
            checks = dict(independent_audit_and_cohort=bool(technical_passed and complete),
                all_1600_original_real_events_preserved=bool(value('real', 'regional_tv', 'events') == 1600),
                entropy_controls_complete=bool(complete and g.control_available.all()),
                real_separation_reduced_at_least_10_percent=bool(value('real', 'separation_cm', 'relative_delta') <= -.10),
                real_regional_tv_reduced_at_least_10_percent=bool(value('real', 'regional_tv', 'relative_delta') <= -.10),
                both_agreement_directions_positive_in_three_rats=bool(value('real', 'separation_cm', 'animals_improved') >= 3 and value('real', 'regional_tv', 'animals_improved') >= 3),
                beats_entropy_control_separation=bool(value('real', 'separation_cm', comparator='poisson_entropy_matched') < 0),
                beats_entropy_control_regional_tv=bool(value('real', 'regional_tv', comparator='poisson_entropy_matched') < 0),
                run_mean_population_error_not_worse=bool(value('run_q4', 'mean_error') <= 1e-10),
                run_mean_population_brier_not_worse=bool(value('run_q4', 'mean_brier') <= 1e-10),
                each_known_population_error_upper95_at_most_2cm=bool(len(error) == 10 and error.finite_animals.eq(4).all() and (error.delta_upper95 <= 2).all()),
                each_simulated_mean_population_brier_not_worse=bool(len(brier) == 4 and brier.finite_animals.eq(4).all() and (brier.delta <= 1e-10).all()))
            checks['promising_remedy_screen'] = all(checks.values())
            for name, passed in checks.items():
                rows.append(dict(dataset=dataset, method=method, gate=name, passed=passed,
                    role='primary_external_screen' if dataset == 'hc11' and method == PRIMARY else 'sensitivity_cannot_rescue_primary'))
    return pd.DataFrame(rows)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    s = summary.loc[summary.split.eq(0) & summary.comparator.eq('poisson')]
    methods = ['gamma_exposure', PRIMARY, 'poisson_entropy_matched']
    colors = ['#4477aa', '#228866', '#aa5577']
    labels = ['Sampling uncertainty', 'Sampling + training drift', 'Entropy-matched Poisson']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for i, dataset in enumerate(('pfeiffer_foster', 'hc11')):
        local = s.loc[s.dataset.eq(dataset)]
        for j, metric, title in ((0, 'separation_cm', 'Population separation change (cm)'),
                                 (1, 'regional_tv', 'Regional disagreement change (TV)')):
            table = local.loc[local.source.eq('real') & local.metric.eq(metric)].set_index('method').loc[methods]
            ax = axes[i, j]
            ax.bar(range(3), table.delta, color=colors)
            ax.vlines(range(3), table.delta_ci_low, table.delta_ci_high, color='black')
            ax.axhline(0, color='black', lw=.8)
            ax.set(xticks=range(3), xticklabels=labels, title=dataset+'\n'+title)
            ax.tick_params(axis='x', labelrotation=25, labelsize=8)
        ax = axes[i, 2]
        sources = ['run_q4', 'sim_stationary', 'sim_moving', 'sim_moving_gain', 'sim_late_jump']
        for method, color, label in zip(methods, colors, labels, strict=True):
            table = local.loc[local.method.eq(method) & local.metric.eq('mean_error')].set_index('source').loc[sources]
            ax.plot(range(5), table.delta, marker='o', color=color, label=label)
        ax.axhline(0, color='black', lw=.8)
        ax.set(xticks=range(5), xticklabels=['RUN', 'Stationary', 'Moving', 'Gain shift', 'Late jump'], title='Known-position mean A/B error change (cm)')
        ax.tick_params(axis='x', labelrotation=25, labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed-time rate uncertainty: change from Poisson at the SAME 20-ms endpoint\nLower is better; equal-rat averages and descriptive four-rat intervals')
    fig.savefig(output/'encoding_uncertainty_content.png', dpi=160)
    fig.savefig(output/'encoding_uncertainty_content.pdf')
    plt.close(fig)


def write_report(summary, gate, diagnostic, output):
    s = summary.loc[summary.split.eq(0) & summary.comparator.eq('poisson')]
    lines = ['# Fixed-time encoding-uncertainty benchmark', '',
        'Non-rescoring report. Split0 is primary, with events averaged within session, sessions within rat, and rats equally weighted.',
        'Each dataset has eight sessions/four rats and 1600 real candidates. This reused methodological cohort is not pristine confirmation.', '',
        '## Real candidates', '',
        '| Dataset | Method | Separation before / after (cm) | Change | TV before / after | Change |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for (dataset, method), g in s.loc[s.source.eq('real')].groupby(['dataset', 'method']):
        a, b = g.set_index('metric').loc[['separation_cm', 'regional_tv']].itertuples()
        lines.append(f'| {dataset} | {method} | {a.before:.3f} / {a.after:.3f} | {a.relative_delta:+.2%} | {b.before:.4f} / {b.after:.4f} | {b.relative_delta:+.2%} |')
    lines += ['', '## Frozen screen', '']
    for (dataset, method), g in gate.groupby(['dataset', 'method']):
        failed = g.loc[~g.passed & g.gate.ne('promising_remedy_screen'), 'gate'].tolist()
        lines.append(f'- {dataset}, {method}: {"FAIL" if failed else "PASS SCREEN ONLY"}; failed: {", ".join(failed) or "none"}.')
    lines += ['', '## Primary method: known-position accuracy', '',
        '| Dataset | Source | Mean A/B error change (cm) | Mean Brier change | A error upper95 (cm) | B error upper95 (cm) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for (dataset, source), g in s.loc[s.source.ne('real') & s.method.eq(PRIMARY)].groupby(['dataset', 'source']):
        t = g.set_index('metric')
        lines.append(f'| {dataset} | {source} | {t.loc["mean_error", "delta"]:+.4f} | {t.loc["mean_brier", "delta"]:+.5f} | {t.loc["a_error", "delta_upper95"]:+.3f} | {t.loc["b_error", "delta_upper95"]:+.3f} |')
    lines += ['', '## A-only training-map sensitivity', '',
        'Unadjusted within-session Spearman associations, then sessions averaged within rat. No fitted selector; no destination certification.', '']
    for (dataset, source, target), g in diagnostic.loc[diagnostic.split.eq(0)].groupby(['dataset', 'source', 'target']):
        rat = g.groupby('animal').spearman_rho.mean()
        lines.append(f'- {dataset}, {source}, {target}: mean within-rat/session rho {rat.mean():+.3f}; positive {int((rat > 0).sum())}/{int(rat.notna().sum())} rats.')
    lines += ['', '## Scope and interpretation', '',
        '- No replay latent ground truth: cross-population agreement alone cannot establish accuracy or correct destination.',
        '- No temporal model, no shifted endpoint, no candidate abstention, and no changed cells or map mean. Only the fixed-time count likelihood differs.',
        '- Gamma-Poisson integration is established; the rate variance is a moment approximation, not an exact posterior over smoothed fields or a new biological mechanism.',
        '- Entropy-matched Poisson is a mandatory control against improvement by posterior flattening. Proper regional Brier/NLL and true-position errors are also reported.',
        '- POST high-MUA hc11 events are not automatically immobile, NREM, ripple-positive or replay. They differ from PF awake candidates.',
        '- Simulations preserve observed population count totals, not real noise correlations; occupied-grid paths are not verified maze topology.',
        '- Four-rat bootstrap intervals are descriptive. The 2-cm noninferiority tolerance is frozen, not proof of equal accuracy.',
        '- A pass here would require disjoint-event confirmation and a direct original matched-Home/accepted-endpoint test. Sensitivity methods/splits cannot rescue a primary failure.',
        '- Correlations of a diagnostic are not evidence that using it reduces instability. No validated remedy is implied by technical completion.']
    (output/'encoding_uncertainty_content_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-dir', type=Path, action='append', required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    inputs = {f'sessions{i}': d/'measurement_sessions.csv' for i, d in enumerate(args.measurement_dir)}
    inputs.update(audit=args.audit_dir/'independent_audit.json', audit_sessions=args.audit_dir/'independent_audit_sessions.csv', reporter=Path(__file__))
    audit = json.loads(inputs['audit'].read_text())
    audited = pd.read_csv(inputs['audit_sessions'])
    if audit['status'] != 'passed' or not audit['inputs_unchanged'] or len(audited) != 16 or not audited.status.eq('passed').all():
        raise ValueError('complete independent audit required')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    sessions = pd.concat([pd.read_csv(d/'measurement_sessions.csv') for d in args.measurement_dir], ignore_index=True)
    complete = (set(sessions.dataset) == {'pfeiffer_foster', 'hc11'} and not sessions.duplicated(['dataset', 'session']).any()
        and all(len(g) == 8 and g.animal.nunique() == 4 and g.selected_candidates.eq(200).all() and g.status.eq('complete').all()
                for _, g in sessions.groupby('dataset')))
    if not complete:
        raise ValueError('incomplete frozen cohort')
    frames = []
    for row in sessions.itertuples(index=False):
        validated = audited.loc[audited.dataset.eq(row.dataset) & audited.session.eq(row.session)]
        path = Path(row.artifact_dir)/'event_readouts.csv.gz'
        if len(validated) != 1 or file_sha256(path) != validated.iloc[0].readout_sha256:
            raise ValueError('unaudited or changed readouts')
        local = pd.read_csv(path, float_precision='round_trip')
        if len(local) != validated.iloc[0].readout_rows or set(local.source) != set(SOURCES) or set(local.split) != {0, 1, 2}:
            raise ValueError('incomplete recording')
        frames.append(local)
    frame = pd.concat(frames, ignore_index=True)
    by_session, diagnostic = paired_sessions(frame)
    animal, summary = aggregate(by_session)
    gate = gates(summary, True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (('encoding_uncertainty_by_session', by_session), ('encoding_uncertainty_by_animal', animal),
                        ('encoding_uncertainty_summary', summary), ('encoding_uncertainty_gate_summary', gate),
                        ('encoding_uncertainty_diagnostic_by_session', diagnostic), ('measurement_sessions', sessions)):
        table.to_csv(args.output_dir/f'{name}.csv', index=False)
    write_report(summary, gate, diagnostic, args.output_dir)
    plot(summary, args.output_dir)
    manifest.update(status='complete', non_rescoring=True, independent_audit_passed=True,
        inputs_unchanged=all(file_sha256(v) == manifest['input_file_sha256'][k] for k, v in inputs.items()),
        outputs_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not manifest['inputs_unchanged']:
        raise ValueError('report inputs changed')


if __name__ == '__main__':
    main()
