#!/usr/bin/env python3
"""Non-rescoring count-conditioning report: agreement is not accuracy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_count_conditioned_content import METHODS, SOURCES
from scripts.report_encoding_uncertainty_content import METRICS, aggregate, paired_sessions

PRIMARY = 'conditional_multinomial'


def gates(summary, technical):
    rows = []
    for dataset in ('pfeiffer_foster', 'hc11'):
        s = summary.loc[summary.dataset.eq(dataset) & summary.method.eq(PRIMARY) & summary.split.eq(0)]

        def number(source, metric, field='delta', comparator='poisson'):
            r = s.loc[s.source.eq(source) & s.metric.eq(metric) & s.comparator.eq(comparator)]
            return float(r.iloc[0][field]) if len(r) == 1 else np.nan

        complete = len(s) == 2*len(SOURCES)*len(METRICS) and s.animals.eq(4).all() and s.sessions.eq(8).all()
        known = s.loc[s.source.ne('real') & s.comparator.eq('poisson')]
        errors = known.loc[known.metric.isin(['a_error', 'b_error'])]
        brier = known.loc[known.source.ne('run_q4') & known.metric.eq('mean_brier')]
        checks = dict(independent_audit_and_cohort=bool(technical and complete),
            all_1600_fixed_real_endpoints=bool(number('real', 'regional_tv', 'events') == 1600),
            all_entropy_controls_available=bool(complete and s.control_available.all()),
            real_separation_reduced_at_least_10_percent=bool(number('real', 'separation_cm', 'relative_delta') <= -.1),
            real_regional_tv_reduced_at_least_10_percent=bool(number('real', 'regional_tv', 'relative_delta') <= -.1),
            both_directions_positive_three_rats=bool(number('real', 'separation_cm', 'animals_improved') >= 3 and number('real', 'regional_tv', 'animals_improved') >= 3),
            beats_entropy_control_separation=bool(number('real', 'separation_cm', comparator='poisson_entropy_matched') < 0),
            beats_entropy_control_regional_tv=bool(number('real', 'regional_tv', comparator='poisson_entropy_matched') < 0),
            run_mean_population_error_not_worse=bool(number('run_q4', 'mean_error') <= 1e-10),
            run_mean_population_brier_not_worse=bool(number('run_q4', 'mean_brier') <= 1e-10),
            all_known_population_error_upper95_within_2cm=bool(len(errors) == 14 and errors.finite_animals.eq(4).all() and (errors.delta_upper95 <= 2).all()),
            all_simulated_mean_brier_not_worse=bool(len(brier) == 6 and brier.finite_animals.eq(4).all() and (brier.delta <= 1e-10).all()))
        checks['promising_remedy_screen'] = all(checks.values())
        for name, passed in checks.items():
            rows.append(dict(dataset=dataset, gate=name, passed=passed,
                role='primary_external_screen' if dataset == 'hc11' else 'development_comparison'))
    return pd.DataFrame(rows)


def support_strata(frame):
    rows = []
    for keys, g in frame.loc[frame.split.eq(0)].groupby(['dataset', 'animal', 'session', 'source']):
        meta = dict(zip(['dataset', 'animal', 'session', 'source'], keys, strict=True))
        bank = {name: part.set_index('event_index').sort_index() for name, part in g.groupby('method')}
        base, new = bank['poisson'], bank[PRIMARY]
        masks = dict(both_zero=base.a_spikes.eq(0) & base.b_spikes.eq(0),
                     one_zero=base.a_spikes.eq(0) ^ base.b_spikes.eq(0),
                     both_nonzero=base.a_spikes.gt(0) & base.b_spikes.gt(0),
                     both_supported_sensitivity=(base.a_spikes >= 3) & (base.b_spikes >= 3) & (base.a_active >= 2) & (base.b_active >= 2))
        for name, mask in masks.items():
            selected, original = new.loc[mask], base.loc[mask]
            rows.append(dict(meta, activity_stratum=name, events=int(mask.sum()), total_events=len(base),
                fraction=float(mask.mean()), separation_before=original.separation_cm.mean(),
                separation_after=selected.separation_cm.mean(),
                separation_delta=(selected.separation_cm-original.separation_cm).mean(),
                regional_tv_delta=(selected.regional_tv-original.regional_tv).mean(),
                mean_error_delta=((selected.a_error+selected.b_error-original.a_error-original.b_error)/2).mean(),
                scope='descriptive_fixed_activity_stratum_not_primary_selection_or_validation'))
    return pd.DataFrame(rows)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    methods, colors = [PRIMARY, 'poisson_entropy_matched'], ['#008866', '#aa5577']
    labels = ['Count-conditioned', 'Entropy-matched Poisson']
    s = summary.loc[summary.split.eq(0) & summary.comparator.eq('poisson')]
    for i, dataset in enumerate(('pfeiffer_foster', 'hc11')):
        local = s.loc[s.dataset.eq(dataset)]
        for j, metric, title in ((0, 'separation_cm', 'Population separation change (cm)'),
                                 (1, 'regional_tv', 'Regional disagreement change (TV)')):
            data = local.loc[local.source.eq('real') & local.metric.eq(metric)].set_index('method').loc[methods]
            axes[i, j].bar(range(2), data.delta, color=colors)
            axes[i, j].vlines(range(2), data.delta_ci_low, data.delta_ci_high, color='black')
            axes[i, j].axhline(0, color='black', lw=.8)
            axes[i, j].set(xticks=range(2), xticklabels=labels, title=dataset+'\n'+title)
            axes[i, j].tick_params(axis='x', labelsize=9)
        sources = ['run_q4', 'sim_stationary', 'sim_moving', 'sim_moving_gain', 'sim_late_jump', 'sim_poisson_stationary', 'sim_poisson_stationary_gain20']
        for name, color, label in zip(methods, colors, labels, strict=True):
            x = local.loc[local.method.eq(name) & local.metric.eq('mean_error')].set_index('source').loc[sources]
            axes[i, 2].plot(range(7), x.delta, marker='o', color=color, label=label)
        axes[i, 2].axhline(0, color='black', lw=.8)
        axes[i, 2].set(xticks=range(7), xticklabels=['RUN', 'Static', 'Moving', 'Cell gains', 'Late jump', 'Poisson', 'Poisson gain20'],
            title='Known-position mean A/B error change (cm)')
        axes[i, 2].tick_params(axis='x', labelrotation=30, labelsize=8)
        axes[i, 2].legend(fontsize=8)
    fig.suptitle('Count-conditioned likelihood at the SAME 20-ms endpoint\nLower is better; entropy and matched-Poisson controls prevent false agreement claims')
    fig.savefig(output/'count_conditioned_content.png', dpi=160)
    fig.savefig(output/'count_conditioned_content.pdf')
    plt.close(fig)


def write_report(summary, gate, diagnostic, strata, output):
    s = summary.loc[summary.split.eq(0) & summary.comparator.eq('poisson')]
    lines = ['# Count-conditioned content remedy screen', '',
        'Non-rescoring report; original endpoints and populations remain unchanged. Split0 primary. Each dataset: eight sessions, four rats, 1600 real candidates.',
        'Events average within session, sessions within rat, then rats equally. Bootstrap intervals are descriptive (four rats).', '',
        '## Real candidate agreement', '',
        '| Dataset | Method | Separation before / after cm | Change | TV before / after | Change |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for (dataset, method), g in s.loc[s.source.eq('real')].groupby(['dataset', 'method']):
        t = g.set_index('metric')
        a, b = t.loc['separation_cm'], t.loc['regional_tv']
        lines.append(f'| {dataset} | {method} | {a.before:.3f} / {a.after:.3f} | {a.relative_delta:+.2%} | {b.before:.4f} / {b.after:.4f} | {b.relative_delta:+.2%} |')
    lines += ['', '## Frozen gates', '']
    for dataset, g in gate.groupby('dataset'):
        failed = g.loc[~g.passed & g.gate.ne('promising_remedy_screen'), 'gate'].tolist()
        lines.append(f'- {dataset}: {"FAIL" if failed else "PASS SCREEN ONLY"}; failed: {", ".join(failed) or "none"}.')
    lines += ['', '## Accuracy and proper scores', '',
        '| Dataset | Source | Mean A/B error change cm | Mean Brier change | A error upper95 | B error upper95 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for (dataset, source), g in s.loc[s.method.eq(PRIMARY) & s.source.ne('real')].groupby(['dataset', 'source']):
        t = g.set_index('metric')
        lines.append(f'| {dataset} | {source} | {t.loc["mean_error", "delta"]:+.3f} | {t.loc["mean_brier", "delta"]:+.5f} | {t.loc["a_error", "delta_upper95"]:+.3f} | {t.loc["b_error", "delta_upper95"]:+.3f} |')
    lines += ['', '## Count-information diagnostic', '',
        'A-only regionalTV between the original and count-conditioned posterior, evaluated against independently observed B error or agreement. Unadjusted correlation alone is not a validated diagnostic.', '']
    for (dataset, source, target), g in diagnostic.loc[diagnostic.split.eq(0)].groupby(['dataset', 'source', 'target']):
        rat = g.groupby('animal').spearman_rho.mean()
        lines.append(f'- {dataset}, {source}, {target}: mean within-session/rat rho {rat.mean():+.3f}; positive {int((rat > 0).sum())}/{int(rat.notna().sum())} rats.')
    lines += ['', '## Silent populations', '',
        'Conditioning on zero observed spikes yields a uniform posterior. Lower disagreement here is not observed spatial information.', '']
    for (dataset, name), g in strata.loc[strata.source.eq('real')].groupby(['dataset', 'activity_stratum']):
        fraction = g.groupby('animal').fraction.mean().mean()
        lines.append(f'- {dataset}, {name}: {int(g.events.sum())} endpoints; equal-rat fraction {fraction:.1%}.')
    lines += ['', '## Boundaries', '',
        '- Flat spatial prior; no temporal model, path smoothing, window shift, cell resampling or decoder-based event deletion.',
        '- Count conditioning is an established likelihood. It is invariant to common gain but discards potentially informative location-dependent total firing rates.',
        '- Existing simulations fix observed population totals. The added Poisson/no-gain and gain20 sources deliberately test unconditional observations too.',
        '- Simulations and independent RUN provide known positions; real candidate endpoints do not. Agreement can improve while true error worsens.',
        '- hc11 POST MUA is not verified NREM, ripple-positive, immobile or replay. Geometry of occupied-grid simulation paths is not verified maze topology.',
        '- The hc11 cohort has informed previous failed methods. Passing this screen requires disjoint-event confirmation and direct original matched-Home/accepted-endpoint evaluation before success.',
        '- Fixed activity strata are descriptive, not a new selection rule that can rescue the primary result.',
        '- No thresholds, map gain, support rules or partitions were tuned after these outcomes.']
    (output/'count_conditioned_content_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-dir', type=Path, action='append', required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    inputs = {f'sessions{i}': d/'measurement_sessions.csv' for i, d in enumerate(args.measurement_dir)}
    inputs.update(audit=args.audit_dir/'independent_audit.json', audit_sessions=args.audit_dir/'independent_audit_sessions.csv',
        reporter=Path(__file__), aggregation=ROOT/'scripts/report_encoding_uncertainty_content.py')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    audit = json.loads(inputs['audit'].read_text())
    audited = pd.read_csv(inputs['audit_sessions'])
    if audit['status'] != 'passed' or not audit['inputs_unchanged'] or len(audited) != 16 or not audited.status.eq('passed').all():
        raise ValueError('full independent audit required')
    sessions = pd.concat([pd.read_csv(d/'measurement_sessions.csv') for d in args.measurement_dir], ignore_index=True)
    complete = (set(sessions.dataset) == {'pfeiffer_foster', 'hc11'} and not sessions.duplicated(['dataset', 'session']).any()
        and all(len(g) == 8 and g.animal.nunique() == 4 and g.selected_candidates.eq(200).all() and g.status.eq('complete').all()
                for _, g in sessions.groupby('dataset')))
    if not complete: raise ValueError('incomplete cohort')
    frames = []
    for row in sessions.itertuples(index=False):
        record = audited.loc[audited.dataset.eq(row.dataset) & audited.session.eq(row.session)]
        path = Path(row.artifact_dir)/'event_readouts.csv.gz'
        if len(record) != 1 or file_sha256(path) != record.iloc[0].readout_sha256: raise ValueError('unaudited output')
        f = pd.read_csv(path, float_precision='round_trip')
        if len(f) != record.iloc[0].readout_rows or set(f.source) != set(SOURCES) or set(f.split) != {0, 1, 2}:
            raise ValueError('incomplete sources/splits')
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    session, diagnostic = paired_sessions(frame, methods=METHODS, diagnostic_column='a_total_count_reliance_tv')
    animal, summary = aggregate(session)
    gate, strata = gates(summary, True), support_strata(frame)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (('count_conditioned_by_session', session), ('count_conditioned_by_animal', animal),
                        ('count_conditioned_summary', summary), ('count_conditioned_gate_summary', gate),
                        ('count_conditioned_diagnostic_by_session', diagnostic), ('count_conditioned_support_strata', strata),
                        ('measurement_sessions', sessions)):
        table.to_csv(args.output_dir/f'{name}.csv', index=False)
    write_report(summary, gate, diagnostic, strata, args.output_dir)
    plot(summary, args.output_dir)
    manifest.update(status='complete', non_rescoring=True, independent_audit_passed=True,
        inputs_unchanged=all(file_sha256(v) == manifest['input_file_sha256'][k] for k, v in inputs.items()),
        outputs_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not manifest['inputs_unchanged']: raise ValueError('report inputs changed')


if __name__ == '__main__':
    main()
