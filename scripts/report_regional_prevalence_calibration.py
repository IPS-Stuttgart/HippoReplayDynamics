#!/usr/bin/env python3
"""Non-rescoring report of frozen regional prevalence calibration gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

METRICS = ('raw_high', 'raw_low', 'corrected_high', 'corrected_low', 'raw_discrepancy',
           'corrected_discrepancy', 'raw_mean_absolute_error', 'corrected_mean_absolute_error')


def rat_mean(frame, column):
    if frame.empty or not np.isfinite(frame[column]).all():
        return np.nan
    return float(frame.groupby(['animal', 'session'])[column].mean().groupby('animal').mean().mean())


def screen(frame, responses):
    primary = frame.loc[frame.encoding.eq('early_run') & frame.calibration.eq('run_q3')]
    native = primary.loc[primary.source.eq('run_q4') & primary.panel.str.startswith('prevalence_')]
    candidate = primary.loc[primary.source.eq('all_fixed_candidates')]
    accepted = primary.loc[primary.source.eq('full_accepted_segment')]
    fits = responses.loc[responses.encoding.eq('early_run') & responses.calibration.eq('run_q3')]
    error_raw, error_corrected = [rat_mean(native, name + '_mean_absolute_error') for name in ('raw', 'corrected')]
    native_by_rat = native.groupby(['animal', 'session'])[['raw_mean_absolute_error', 'corrected_mean_absolute_error']].mean().groupby('animal').mean()
    candidate_raw, candidate_corrected = [rat_mean(candidate, name + '_discrepancy') for name in ('raw', 'corrected')]
    accepted_raw, accepted_corrected = [rat_mean(accepted, name + '_discrepancy') for name in ('raw', 'corrected')]
    gates = [
        ('all_primary_calibrations_available', len(fits) == 8 and fits.status.eq('available').all()),
        ('native_validation_coverage', len(native) == 20 and native.session.nunique() == 4 and native.animal.nunique() == 3),
        ('native_validation_estimates_compatible', len(native) > 0 and native.high_status.eq('available').all() and native.low_status.eq('available').all()),
        ('native_error_reduction_at_least_20_percent', np.isfinite(error_corrected) and error_raw > 0 and error_corrected <= .8 * error_raw),
        ('native_mean_absolute_error_at_most_5_pp', np.isfinite(error_corrected) and error_corrected <= .05),
        ('native_error_not_worse_in_each_rat', len(native_by_rat) == 3 and np.isfinite(native.corrected_mean_absolute_error).all() and
         (native_by_rat.corrected_mean_absolute_error <= native_by_rat.raw_mean_absolute_error).all()),
        ('candidate_discrepancy_reduction_at_least_20_percent', len(candidate) == 4 and np.isfinite(candidate_corrected) and
         candidate_raw > 0 and candidate_corrected <= .8 * candidate_raw),
        ('real_candidate_estimates_compatible', len(candidate) == 4 and candidate.high_status.eq('available').all() and candidate.low_status.eq('available').all()),
        ('accepted_discrepancy_not_worse', len(accepted) == 4 and np.isfinite(accepted_corrected) and accepted_corrected <= accepted_raw),
        ('real_accepted_estimates_compatible', len(accepted) == 4 and accepted.high_status.eq('available').all() and accepted.low_status.eq('available').all()),
    ]
    gates.append(('pf_development_screen', all(bool(passed) for _, passed in gates)))
    gates.extend([('independent_validation_complete', False), ('validated_remedy', False)])
    metrics = dict(native_error_raw=error_raw, native_error_corrected=error_corrected,
                   candidate_discrepancy_raw=candidate_raw, candidate_discrepancy_corrected=candidate_corrected,
                   accepted_discrepancy_raw=accepted_raw, accepted_discrepancy_corrected=accepted_corrected)
    return pd.DataFrame([dict(gate=name, passed=bool(value)) for name, value in gates]), metrics


def summarize(frame):
    keys = ['encoding', 'calibration', 'source', 'panel']
    rows = []
    for values, group in frame.groupby(keys):
        row = dict(zip(keys, values, strict=True))
        row.update({col: rat_mean(group, col) for col in METRICS})
        row.update(sessions=group.session.nunique(), animals=group.animal.nunique(),
                   unavailable_pairs=int((~np.isfinite(group.corrected_high) | ~np.isfinite(group.corrected_low)).sum()),
                   incompatible_pairs=int((group.high_status.eq('incompatible_out_of_range') | group.low_status.eq('incompatible_out_of_range')).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def figure(frame, output):
    primary = frame.loc[frame.encoding.eq('early_run') & frame.calibration.eq('run_q3')]
    candidates = primary.loc[primary.source.eq('all_fixed_candidates')].sort_values('session')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = ('#2166ac', '#b2182b')
    for ax, prefix, title in zip(axes[0], ('raw', 'corrected'), ('Original endpoint mass', 'Native-calibrated prevalence (unclipped)'), strict=True):
        x = np.arange(len(candidates))
        for j, side in enumerate(('high', 'low')):
            values = 100 * candidates[prefix + '_' + side]
            ax.bar(x + (j - .5) * .35, values, .35, color=colors[j], label=side + ' Home coverage')
            for xpos, value in zip(x + (j - .5) * .35, values, strict=True):
                if not np.isfinite(value):
                    ax.text(xpos, .04, 'NA', transform=ax.get_xaxis_transform(), color=colors[j], ha='center', fontsize=8)
        ax.set_xticks(x, candidates.session, rotation=20)
        ax.set_xlim(-.6, len(candidates) - .4)
        ax.set_ylabel('Percent'); ax.set_title(title); ax.axhline(0, color='black', lw=.6); ax.legend(fontsize=8)
    native = primary.loc[primary.source.eq('run_q4') & primary.panel.str.startswith('prevalence_')]
    ax = axes[1, 0]
    for prefix, color in zip(('raw', 'corrected'), colors, strict=True):
        for side, marker in (('high', 'o'), ('low', 'x')):
            ax.scatter(100 * native.true_prevalence, 100 * native[prefix + '_' + side], s=18, marker=marker, alpha=.65, color=color,
                       label=prefix + ' ' + side)
    ax.plot([0, 100], [0, 100], color='black', ls=':'); ax.set_xlabel('Known regional proportion (%)')
    ax.set_ylabel('Estimated proportion (%)'); ax.set_title('Held-out RUN recovery (missing fits omitted)'); ax.legend(fontsize=8, ncol=2)
    ax = axes[1, 1]
    sources = ('run_q4', 'test_poisson_gain1', 'test_poisson_gain4', 'test_conditional', 'test_conditional_map_drift', 'test_conditional_shared_assembly')
    x = np.arange(len(sources))
    for j, prefix in enumerate(('raw', 'corrected')):
        values = [100 * rat_mean(primary.loc[primary.source.eq(s) & primary.panel.str.startswith('prevalence_')], prefix + '_mean_absolute_error') for s in sources]
        ax.bar(x + (j - .5) * .35, values, .35, color=colors[j], label=prefix)
        for xpos, value in zip(x + (j - .5) * .35, values, strict=True):
            if not np.isfinite(value):
                ax.text(xpos, .04, 'NA', transform=ax.get_xaxis_transform(), color=colors[j], ha='center', fontsize=8)
    ax.set_xticks(x, ('RUN', 'Poisson', 'Gain 4', 'Fixed totals', 'Map drift', 'Assembly'), rotation=30)
    ax.set_xlim(-.6, len(sources) - .4)
    ax.set_ylabel('Mean absolute prevalence error (pp)'); ax.set_title('Transfer checks; native RUN calibration'); ax.legend(fontsize=8)
    fig.suptitle('Original matched-population content: calibration must improve truth recovery, not only agreement', fontsize=12)
    fig.savefig(output / 'regional_prevalence_calibration.png', dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.audit_dir / 'reconstruction.json').read_text())
    sessions = pd.read_csv(args.result_dir / 'sessions.csv')
    if audit['status'] != 'pass' or len(audit['results']) != 4 or set(x['session'] for x in audit['results']) != set(sessions.session):
        raise ValueError('complete independent audit required')
    frames, responses, inputs = [], [], dict(script=Path(__file__), audit=args.audit_dir / 'reconstruction.json', source=args.result_dir / 'manifest.json')
    for row in sessions.itertuples():
        folder = Path(row.artifact_dir)
        for name, value in json.loads((folder / 'outputs.json').read_text()).items():
            if file_sha256(folder / name) != value: raise ValueError('source output changed')
        frames.append(pd.read_csv(folder / 'prevalence.csv'))
        responses.append(pd.read_csv(folder / 'calibration.csv'))
        for name in ('prevalence.csv', 'calibration.csv'): inputs[row.session + '_' + name] = folder / name
    frame, responses = pd.concat(frames, ignore_index=True), pd.concat(responses, ignore_index=True)
    gates, metrics = screen(frame, responses)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / 'per_session_prevalence.csv', index=False)
    responses.to_csv(args.output_dir / 'calibration_responses.csv', index=False)
    summary.to_csv(args.output_dir / 'equal_rat_summary.csv', index=False)
    gates.to_csv(args.output_dir / 'gate_summary.csv', index=False)
    frame.groupby(['encoding', 'calibration', 'source', 'panel', 'animal'])[list(METRICS)].mean().reset_index().to_csv(args.output_dir / 'by_rat_summary.csv', index=False)
    figure(frame, args.output_dir)
    passed = bool(gates.set_index('gate').loc['pf_development_screen', 'passed'])
    text = ['# Regional prevalence calibration', '',
            'Non-rescoring report. Original targeted matched populations; all four sessions / three rats retained.',
            'Primary: early-RUN maps and native third-quarter RUN calibration, fourth-quarter RUN truth validation.',
            'Full-RUN map sensitivity is NOT held-out decoder validation. Original full-RUN unit eligibility remains a limitation.', '',
            '## Frozen screen', '', f'- PF development screen: {"pass" if passed else "fail"}.',
            '- Independent dataset validation: not run; validated remedy: NOT established.', '']
    text += ['Native error below averages the five frozen prevalence stress panels (5-75%),',
             'not the natural RUN occupancy distribution. Natural-prevalence rows remain in the CSVs.',
             'A missing aggregate is retained as unavailable whenever any required population is missing.', '']
    for key, value in metrics.items(): text.append(f'- {key}: {100 * value:.4f} percentage points.')
    text += ['', '## Interpretation limits', '',
             '- Adjusted prevalence is an aggregate estimator, not a new event posterior or trajectory label.',
             '- Estimates remain unbounded; unavailable/incompatible pairs are not discarded or clipped.',
             '- Known-prevalence panels share source observations. They are not extra independent animals or trials.',
             '- Transfer requires stable class-conditional responses, including the within-region spatial distribution.',
             '- RUN, synthetic matched maps, gain, drift and shared-assembly stresses are reported separately.',
             '- A narrower high/low discrepancy alone cannot establish the true biological regional proportion.',
             '- The original producer and failed audit are retained; v2 fixes one duplicate native boundary spike before outcome inspection.', '',
             '## Gates', '']
    text.extend(f'- {r.gate}: {"pass" if r.passed else "fail"}' for r in gates.itertuples())
    (args.output_dir / 'report.md').write_text('\n'.join(text) + '\n')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status='complete', independent_validation=False, pf_screen_pass=passed,
                    output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(pf_screen_pass=passed, **metrics)), flush=True)


if __name__ == '__main__':
    main()
