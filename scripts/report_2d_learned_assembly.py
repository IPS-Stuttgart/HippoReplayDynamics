#!/usr/bin/env python3
"""Non-rescoring report of learned-assembly and spatial predictive comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from _provenance import build_script_provenance, file_sha256
from audit_2d_learned_assembly import load_fit

DATASETS = {'pfeiffer_foster': ('Pfeiffer-Foster', 4), 'tanni2022': ('Tanni', 5)}
ENDPOINTS = ('spatial_imm_minus_learned_hmm', 'learned_hmm_minus_nonspatial_global',
             'learned_hmm_minus_same_emissions_iid', 'learned_hmm_order_advantage',
             'spatial_minus_assembly_order_advantage')


def decisions(summary, fits):
    rows = []
    for dataset, (_, count) in DATASETS.items():
        part = summary[summary.dataset.eq(dataset)].set_index('contrast')
        required = {f'k{k}__{axis}' for k in (20, 50, 100) for axis in ENDPOINTS}
        if part.index.duplicated().any() or not required.issubset(part.index) or not part.animals.eq(count).all():
            raise ValueError('missing full-cohort endpoint or animal coverage')
        fit = fits[fits.dataset.eq(dataset) & fits.n_states.eq(50)]
        converged = not fit.empty and bool(fit.fit_converged.all())

        def positive(axis, source=part, n_animals=count):
            row = source.loc[f'k50__{axis}']
            return bool(row['mean'] > 0 and row.ci_low > 0 and row.positive_animals == n_animals)

        valid = converged and positive('learned_hmm_minus_nonspatial_global')
        spatial = valid and positive('spatial_imm_minus_learned_hmm')
        contrast = part.loc['k50__spatial_imm_minus_learned_hmm']
        assembly = valid and contrast.ci_high < 0 and contrast.positive_animals == 0
        verdict = 'comparator_adequacy_not_established'
        if valid:
            verdict = 'spatial_advantage_over_tested_assembly' if spatial else 'learned_assembly_advantage' if assembly else 'no_rat_uniform_separation'
        rows.append({
            'dataset': dataset, 'primary_states': 50, 'primary_fits_converged': converged,
            'learned_comparator_beats_global': valid,
            'learned_temporal_structure_supported': valid and positive('learned_hmm_minus_same_emissions_iid') and positive('learned_hmm_order_advantage'),
            'spatial_advantage_over_tested_assembly_supported': spatial,
            'verdict': verdict, 'independent_confirmation': False,
            'new_neural_mechanism_established': False,
        })
    return pd.DataFrame(rows)


def markdown(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def run(args):
    source, audit, out = (Path(p).resolve() for p in (args.input_dir, args.audit_dir, args.output_dir))
    mp, ap = source / 'learned_assembly_manifest.json', audit / 'learned_assembly_reconstruction.json'
    manifest, verified = [json.loads(p.read_text()) for p in (mp, ap)]
    if manifest['status'] != 'complete' or manifest['scope'] != 'all_9225' or verified['status'] != 'pass' or verified['input_file_sha256']['scoring_manifest'] != file_sha256(mp):
        raise ValueError('complete linked full-cohort reconstruction required')
    for name, expected in manifest['output_sha256'].items():
        if file_sha256(source / name) != expected:
            raise ValueError(f'changed input: {name}')
    if out.exists() and any(out.iterdir()):
        raise ValueError('refusing overwrite')
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(source / 'learned_assembly_summary.csv')
    animals = pd.read_csv(source / 'learned_assembly_by_animal.csv')
    fits = pd.read_csv(source / 'learned_assembly_fit_summary.csv')
    decision = decisions(summary, fits)
    primary = summary[summary.contrast.isin([f'k50__{e}' for e in ENDPOINTS])].copy()
    diagnostic = []
    for result in manifest['completed']:
        fit = load_fit(source / f"{result['stem']}_fit.npz")
        n_cells, states = fit.probabilities.shape
        diagnostic.append({k: result[k] for k in ('dataset', 'animal', 'session', 'fold', 'n_states', 'fit_converged')} | {
            'n_cells': n_cells, 'calibration_events': fit.n_calibration_events,
            'calibration_bins': fit.n_calibration_bins,
            'calibration_bins_per_state': fit.n_calibration_bins / states,
            'effective_occupied_states': float(np.exp(-np.sum(fit.occupancy * np.log(fit.occupancy)))),
            'occupancy_weighted_self_transition': float(fit.occupancy @ np.diag(fit.transition)),
            'free_parameters': states - 1 + states * (states - 1) + states * (n_cells - 1),
            'parameter_count_exceeds_count_entries': states - 1 + states * (states - 1) + states * (n_cells - 1) > fit.n_calibration_bins * n_cells,
        })
    diagnostic = pd.DataFrame(diagnostic)
    for name, frame in (('primary', primary), ('capacity', summary), ('decisions', decision), ('fit_diagnostics', diagnostic)):
        frame.to_csv(out / f'learned_assembly_{name}.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = ('#167b7b', '#bb4a40')
    for column, (dataset, (label, _)) in enumerate(DATASETS.items()):
        for row, axis in enumerate(('learned_hmm_minus_nonspatial_global', 'spatial_imm_minus_learned_hmm')):
            ax = axes[row, column]
            for j, states in enumerate((20, 50, 100)):
                key = f'k{states}__{axis}'
                stat = summary[summary.dataset.eq(dataset) & summary.contrast.eq(key)].iloc[0]
                rats = animals[animals.dataset.eq(dataset) & animals.contrast.eq(key)].sort_values('animal')
                ax.scatter(j + np.linspace(-0.12, 0.12, len(rats)), rats.delta, color='0.6', s=22, zorder=2)
                ax.vlines(j, stat.ci_low, stat.ci_high, color=colors[row], lw=2)
                ax.scatter(j, stat['mean'], color=colors[row], s=50, zorder=3)
            ax.axhline(0, color='black', lw=0.7)
            ax.set_xticks(range(3), ['20', '50 (primary)', '100'])
            ax.set_xlabel('Learned assembly states')
            ax.set_ylabel('Paired held-out score difference (nats)')
            ax.set_title(label + ': ' + ('learned HMM - global' if row == 0 else 'spatial IMM - learned HMM'), fontsize=11)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Does a spatial map add prediction beyond learned neural assemblies?', fontsize=13)
    fig.savefig(out / 'learned_assembly_comparison.png', dpi=180)
    plt.close(fig)
    readable = pd.DataFrame({
        'dataset': primary.dataset, 'contrast': primary.contrast.str.replace('k50__', '', regex=False),
        'mean_nats': primary['mean'].map(lambda x: f'{x:+.3f}'),
        '95%_CI': [f'[{a:+.3f}, {b:+.3f}]' for a, b in zip(primary.ci_low, primary.ci_high, strict=True)],
        'positive_animals': primary.positive_animals.astype(str) + '/' + primary.animals.astype(str),
    })
    text = '\n\n'.join([
        '# Learned Assembly Predictive Comparison',
        ('Non-rescoring report. All 9,225 frozen MUA candidates, 33 sessions and nine animals. '
        'Primary K50; K20/K100 are capacity sensitivities, not outcome-selected alternatives.'),
        markdown(readable), markdown(decision[['dataset', 'verdict', 'learned_temporal_structure_supported']]),
        ('Inference uses only target training cells. Both models can use held-out-cell spikes in other calibration events, '
        'never the scored event fold or its one-second guard. The spatial predictor additionally uses RUN maps. '
        'Scores condition on held-out spike totals and sum proper bin-marginal log probabilities; they are not joint log evidence.'),
        ('Each event contributes its median paired difference across five cell splits. Means weight sessions equally within animals '
        'and animals equally within each dataset; 95% intervals use the frozen hierarchical animal/session/event bootstrap. '
        'Gray figure points are animal means. These reused-event, four/five-animal intervals do not absorb all analysis-selection uncertainty.'),
        (f'Fits converged: {int(fits.fit_converged.sum())}/{len(fits)}. '
        f'Parameter-count warnings (descriptive, not proof of failure): {int(diagnostic.parameter_count_exceeds_count_entries.sum())}. '
        'All capacity and fit diagnostics remain in the CSVs. Convergence is not evidence of generalization; global and independent-state comparisons test that separately.'),
        ('A position-free model can learn spatial relationships implicitly. Its success does not show that neural activity lacks spatial content; '
        'a spatial win does not establish a unique neural mechanism. A confidence interval crossing zero does not prove equivalence. '
        'Fully learned burst-event HMMs and order-shuffle controls already have direct precedent in '
        '[Maboudi et al. (2018)](https://elifesciences.org/articles/34467). This is a count-conditioned skeptical comparator, not a new replay detector.'),
        ('No replay prevalence, constant-speed, or high-importance novelty claim follows from this comparison. '
        'The observation-model form, calibration budget, model capacity and time-step convention still differ from exact published Poisson-HMM implementations.'),
        (f"Reconstruction: {verified['scores_checked']} independently recomputed scores; "
        f"{verified['rows_validated']} validated score rows; max error {verified['max_score_error']:.3g}. "
        'All calibration objectives and original plus two shuffled predictions per event/split/capacity were reconstructed. '
        'Additional null predictions receive coverage, finite-score and invariance checks, not exhaustive independent re-inference.'),
        f"Scorer commit: `{manifest['code_commit']}`. Manifest SHA256: `{file_sha256(mp)}`.",
    ])
    (out / 'learned_assembly_report.md').write_text(text + '\n')
    provenance = build_script_provenance(input_paths={'scoring_manifest': mp, 'audit': ap, 'reporter': Path(__file__)})
    provenance['output_sha256'] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}
    (out / 'learned_assembly_report_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(readable.to_string(index=False), flush=True)
    print(decision.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    run(parser.parse_args())
