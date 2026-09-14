#!/usr/bin/env python3
"""Reconstruct primary report aggregates and describe frozen feature transfer."""
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


def independent_metrics(data, metric):
    source_rows = []
    for (animal, session, source, draw), local in data.groupby(['animal', 'session', 'source', 'draw']):
        local = local.reset_index(drop=True)
        order = np.lexsort((local.event_index.to_numpy(), local.prediction_full.to_numpy()))
        kept = order[:int(np.ceil(len(local) / 2))]
        values = local[metric].to_numpy()
        if source == 'real' and 'truth' in metric: continue
        if not np.isfinite(values).all(): raise ValueError('missing metric')
        location_reference = np.nan
        if 'truth' in metric:
            tiles = local.true_tile.to_numpy()
            location_reference = 0.
            for tile in np.unique(tiles[kept]):
                location_reference += np.mean(values[tiles == tile]) * np.mean(tiles[kept] == tile)
        source_rows.append(dict(animal=animal, session=session, source=source, draw=draw,
            baseline=float(np.mean(values)), selected=float(np.mean(values[kept])), location_reference=location_reference))
    rows = pd.DataFrame(source_rows)
    columns = ['baseline', 'selected', 'location_reference']
    by_session = rows.groupby(['animal', 'session', 'source'])[columns].mean()
    by_animal = by_session.groupby(['animal', 'source'])[columns].mean()
    return by_animal.groupby('source')[columns].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report_manifest = json.loads((args.report_dir / 'manifest.json').read_text())
    for name, sha in report_manifest['output_sha256'].items():
        if file_sha256(args.report_dir / name) != sha: raise ValueError('report changed')
    model = json.loads((args.root / 'frozen/frozen_model.json').read_text())
    training = pd.read_csv(args.root / 'frozen/training_run.csv.gz')
    application = json.loads((args.root / 'tanni/manifest.json').read_text())
    chunks = []
    for entry in application['results']:
        if file_sha256(entry['path']) != entry['sha256']: raise ValueError('application changed')
        frame = pd.read_csv(entry['path'])
        chunks.append(frame.loc[frame.split.eq(0)])
    data = pd.concat(chunks, ignore_index=True)
    data['mean_entropy'] = (data.a_entropy + data.b_entropy) / 2
    data['mean_truth_error_cm'] = (data.a_truth_error_cm + data.b_truth_error_cm) / 2
    summary = pd.read_csv(args.report_dir / 'selection_summary.csv')
    summary = summary.loc[summary.dataset.eq('tanni2022') & summary.split.eq(0) & summary.policy.eq('diagnostic_full')]
    checks = []
    for metric in sorted(summary.metric.unique()):
        reference = independent_metrics(data, metric)
        for source, values in reference.iterrows():
            reported = summary.loc[summary.source.eq(source) & summary.metric.eq(metric)].iloc[0]
            for column in ('baseline', 'selected', 'location_reference'):
                np.testing.assert_allclose(reported[column], values[column], atol=1e-9, equal_nan=True)
            checks.append(dict(metric=metric, source=source, status='pass'))
    native = data.loc[data.source.eq('run_test')].copy()
    target = np.log1p(native.b_truth_error_cm / native.grid_diagonal_cm)
    forecast = pd.read_csv(args.report_dir / 'forecast_summary.csv')
    for name in ('mean', 'spikes_entropy', 'full'):
        native['squared_error'] = (native['prediction_' + name] - target) ** 2
        reference = native.groupby(['animal', 'session']).squared_error.mean().groupby('animal').mean().mean()
        reported = forecast.loc[forecast.dataset.eq('tanni2022') & forecast.split.eq(0) & forecast.model.eq(name)].iloc[0]
        np.testing.assert_allclose(reference, reported.log_mse, atol=1e-12)
    state = model['states']['full']
    transfers = []
    for i, feature in enumerate(state['columns']):
        low, high = float(training[feature].min()), float(training[feature].max())
        contribution = (native[feature] - state['means'][i]) / state['scales'][i] * state['coefficients'][i]
        rows = native[['animal', 'session']].copy()
        rows['out_of_training_range'] = (native[feature] < low) | (native[feature] > high)
        rows['contribution'] = contribution
        equal = rows.groupby(['animal', 'session'])[['out_of_training_range', 'contribution']].mean().groupby('animal').mean().mean()
        transfers.append(dict(feature=feature, training_min=low, training_max=high,
            external_min=float(native[feature].min()), external_max=float(native[feature].max()),
            out_of_training_range_fraction=float(equal.out_of_training_range),
            mean_unclipped_prediction_contribution=float(equal.contribution),
            min_unclipped_prediction_contribution=float(contribution.min()),
            max_unclipped_prediction_contribution=float(contribution.max()),
            training_scale=state['scales'][i], standardized_coefficient=state['coefficients'][i]))
    session_rows = []
    for (animal, session), local in native.groupby(['animal', 'session']):
        error = np.log1p(local.b_truth_error_cm / local.grid_diagonal_cm)
        row = dict(animal=animal, session=session, rows=len(local), grid_diagonal_cm=float(local.grid_diagonal_cm.iloc[0]),
            predicted_risk_mean=float(local.prediction_full.mean()), target_mean=float(error.mean()))
        for name in ('a_truth_error_cm', 'b_truth_error_cm'):
            row['mean_' + name] = float(local[name].mean())
        row['within_session_risk_error_corr'] = float(np.corrcoef(local.prediction_full, local.b_truth_error_cm)[0, 1]) if local.prediction_full.nunique() > 1 else np.nan
        session_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(checks).to_csv(args.output_dir / 'aggregate_checks.csv', index=False)
    pd.DataFrame(transfers).to_csv(args.output_dir / 'feature_transfer_exploratory.csv', index=False)
    pd.DataFrame(session_rows).to_csv(args.output_dir / 'native_session_prediction_exploratory.csv', index=False)
    provenance = build_script_provenance(input_paths=dict(audit=Path(__file__), report=args.report_dir / 'manifest.json',
        model=args.root / 'frozen/frozen_model.json', tanni=args.root / 'tanni/manifest.json'), cwd=ROOT)
    provenance.update(status='pass', checked_primary_metric_groups=len(checks),
        exploratory_feature_transfer_not_a_new_predictor=True, no_refit=True,
        primary_real_events=int(data.source.eq('real').sum()), primary_native_windows=len(native))
    (args.output_dir / 'audit.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(status='pass', checked_primary_metric_groups=len(checks))), flush=True)


if __name__ == '__main__':
    main()
