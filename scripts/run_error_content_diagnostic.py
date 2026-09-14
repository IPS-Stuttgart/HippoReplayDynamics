#!/usr/bin/env python3
"""Freeze a RUN-error diagnostic on PF, then apply it without B replay inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge

from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_population_content_stability import decode, tile_ids
from scripts.validate_population_content_stability import KEY, load_readouts, weights

MODELS = ('mean', 'spikes_entropy', 'full')
BASE = ('log_spikes', 'log_active', 'a_entropy')
FULL = (*BASE, 'active_fraction', 'relative_width', 'a_peak', 'log_cells', 'log_a_coverage',
        'log_arena_size', 'relative_support_distance', 'log_b_coverage_at_a',
        'log_a_code_gradient', 'log_b_code_gradient_at_a')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def code_gradient(rates, grid):
    edges = cKDTree(grid).query_pairs(8.01, output_type='ndarray')
    values, degree = np.zeros(len(grid)), np.zeros(len(grid), int)
    if not len(edges): return values
    a, b = edges.T
    distance2 = np.sum((grid[a] - grid[b]) ** 2, axis=1)
    if np.any(distance2 <= 0): raise ValueError('duplicate supported grid locations')
    contrast = np.sum((np.sqrt(rates[:, a]) - np.sqrt(rates[:, b])) ** 2, axis=0) / distance2
    for endpoint in (a, b):
        np.add.at(values, endpoint, contrast)
        np.add.at(degree, endpoint, 1)
    return np.divide(values, degree, out=np.zeros_like(values), where=degree > 0)


def predictor_features(frame, rates_a, rates_b, grid):
    """B encoding is allowed; no B replay measurement is read here."""
    diagonal = float(np.linalg.norm(np.ptp(grid, axis=0)))
    if diagonal <= 0: raise ValueError('degenerate grid')
    distance, nearest = cKDTree(grid).query(frame[['a_x_cm', 'a_y_cm']].to_numpy())
    a_sum, b_sum = rates_a.sum(axis=0), rates_b.sum(axis=0)
    grad_a, grad_b = code_gradient(rates_a, grid), code_gradient(rates_b, grid)
    return pd.DataFrame(dict(log_spikes=np.log1p(frame.a_spikes), log_active=np.log1p(frame.a_active),
        a_entropy=frame.a_entropy, active_fraction=frame.a_active / len(rates_a),
        relative_width=frame.a_width_cm / diagonal, a_peak=frame.a_peak,
        log_cells=np.full(len(frame), np.log1p(len(rates_a))),
        log_a_coverage=np.log1p(a_sum[nearest] / a_sum.mean()),
        log_arena_size=np.full(len(frame), np.log(diagonal / 8.)), relative_support_distance=distance / diagonal,
        log_b_coverage_at_a=np.log1p(b_sum[nearest] / b_sum.mean()),
        log_a_code_gradient=np.log1p(.02 * diagonal ** 2 * grad_a[nearest]),
        log_b_code_gradient_at_a=np.log1p(.02 * diagonal ** 2 * grad_b[nearest])), index=frame.index)


def fit(frame, model):
    if frame.empty or not frame.dataset.eq('pfeiffer_foster').all() or not frame.source.eq('run_calibration').all() or not frame.split.eq(0).all():
        raise ValueError('training is PF primary-split third-quarter RUN only')
    if frame.duplicated(KEY).any(): raise ValueError('duplicate training observations')
    y = np.log1p(frame.b_truth_error_cm.to_numpy() / frame.grid_diagonal_cm.to_numpy())
    if not np.isfinite(y).all() or np.any(y < 0): raise ValueError('known RUN truth required')
    w = weights(frame)
    columns = () if model == 'mean' else BASE if model == 'spikes_entropy' else FULL if model == 'full' else None
    if columns is None: raise ValueError('unknown model')
    if not columns:
        return dict(model=model, columns=[], intercept=float(w @ y))
    x = frame[list(columns)].to_numpy(float)
    medians = np.nanmedian(x, axis=0)
    medians = np.nan_to_num(medians)
    missing = ~np.isfinite(x)
    x = np.column_stack([np.where(missing, medians, x), missing.astype(float)])
    mean = w @ x
    scale = np.sqrt(w @ ((x - mean) ** 2))
    scale[scale < 1e-10] = 1
    estimator = Ridge(alpha=10).fit((x - mean) / scale, y, sample_weight=w * len(frame))
    return dict(model=model, columns=list(columns), medians=medians.tolist(), means=mean.tolist(),
                scales=scale.tolist(), coefficients=estimator.coef_.tolist(), intercept=float(estimator.intercept_))


def predict(frame, state):
    if not state['columns']: return np.full(len(frame), state['intercept'])
    x = frame[state['columns']].to_numpy(float)
    missing = ~np.isfinite(x)
    x = np.column_stack([np.where(missing, state['medians'], x), missing.astype(float)])
    return np.maximum(0, ((x - state['means']) / state['scales']) @ state['coefficients'] + state['intercept'])


def verified_sessions(root, expected_dataset):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'complete': raise ValueError('source incomplete')
    results = manifest['results']
    if any(r['status'] != 'complete' or r['dataset'] != expected_dataset for r in results):
        raise ValueError('unexpected source cohort')
    for row in results:
        folder = Path(row['artifact_dir'])
        if file_sha256(folder / 'event_readouts.csv.gz') != row['readouts_sha256']:
            raise ValueError('readouts changed')
    return results


def session_parts(row):
    folder = Path(row['artifact_dir'])
    arrays = load(folder / 'audit_arrays.npz')
    freeze = json.loads((folder / 'frozen_populations.json').read_text())
    if file_sha256(freeze['cache_path']) != freeze['cache_sha256']:
        raise ValueError('source cache changed')
    for part in freeze['parts']:
        for side in ('a', 'b'):
            np.testing.assert_array_equal(arrays['cell_ids'][part[side]], part[side + '_ids'])
        if set(part['a']).intersection(part['b']) or len(part['a']) != len(part['b']):
            raise ValueError('invalid disjoint population pair')
    return arrays, freeze


def training_rows(row, arrays, part):
    a, b = part['a'], part['b']
    counts, rates, grid, windows = [arrays[k] for k in ('calibration_counts', 'rates_hz', 'grid_cm', 'calibration_windows')]
    decoded_a = decode(counts[:, a], rates[a], grid)
    decoded_b = decode(counts[:, b], rates[b], grid)
    frame = pd.DataFrame(dict(a_spikes=counts[:, a].sum(axis=1), a_active=np.count_nonzero(counts[:, a], axis=1),
        a_entropy=decoded_a['entropy'], a_width_cm=decoded_a['width'], a_peak=decoded_a['peak'],
        a_x_cm=decoded_a['mean'][:, 0], a_y_cm=decoded_a['mean'][:, 1],
        b_truth_error_cm=np.linalg.norm(decoded_b['mean'] - windows[:, 2:], axis=1),
        grid_diagonal_cm=np.linalg.norm(np.ptp(grid, axis=0)),
        event_index=np.arange(len(windows)), start_s=windows[:, 0], end_s=windows[:, 1],
        dataset=row['dataset'], animal=row['animal'], session=row['session'], source='run_calibration', split=0, draw=-1))
    features = predictor_features(frame, rates[a], rates[b], grid)
    for key in features: frame[key] = features[key]
    return frame


def freeze_model(args):
    rows = verified_sessions(args.input_dir, 'pfeiffer_foster')
    if len(rows) != 8 or len({r['animal'] for r in rows}) != 4: raise ValueError('all PF sessions required')
    inputs = dict(source=args.input_dir / 'manifest.json', script=Path(__file__),
        protocol=ROOT / 'docs/run_error_content_diagnostic_protocol.md',
        decoder=ROOT / 'scripts/measure_population_content_stability.py',
        weights=ROOT / 'scripts/validate_population_content_stability.py')
    chunks = []
    for row in rows:
        arrays, frozen = session_parts(row)
        primary = next(p for p in frozen['parts'] if p['split'] == 0)
        chunks.append(training_rows(row, arrays, primary))
        folder = Path(row['artifact_dir'])
        for name in ('audit_arrays.npz', 'frozen_populations.json'): inputs[row['session'] + '/' + name] = folder / name
    train = pd.concat(chunks, ignore_index=True)
    oof, states = [], {}
    for animal in sorted(train.animal.unique()):
        local = train.loc[train.animal.eq(animal)].copy()
        states[animal] = {model: fit(train.loc[train.animal.ne(animal)], model) for model in MODELS}
        for model in MODELS: local['prediction_' + model] = predict(local, states[animal][model])
        oof.append(local)
    all_states = {model: fit(train, model) for model in MODELS}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train.to_csv(args.output_dir / 'training_run.csv.gz', index=False)
    pd.concat(oof, ignore_index=True).to_csv(args.output_dir / 'pf_leave_one_rat_out.csv.gz', index=False)
    model = build_script_provenance(input_paths=inputs, cwd=ROOT)
    model.update(status='frozen', states=all_states, leave_one_rat_out=states, training_rows=len(train),
        training_dataset='pfeiffer_foster', training_source='run_calibration', training_animals=sorted(train.animal.unique()),
        target='log1p_unused_B_RUN_error_over_arena_diagonal', retention=.5,
        training_sha256=file_sha256(args.output_dir / 'training_run.csv.gz'), b_replay_observations_are_predictors=False)
    (args.output_dir / 'frozen_model.json').write_text(json.dumps(model, indent=2) + '\n')
    print(json.dumps(dict(status='frozen', training_rows=len(train), training_animals=model['training_animals'])), flush=True)


def add_truth_tiles(frame, arrays, part, row, frozen):
    grid = arrays['grid_cm']
    tree = cKDTree(grid)
    tiles = tile_ids(grid)
    source = frame.source.iloc[0]
    if source == 'real':
        return np.full(len(frame), -1)
    if source == 'run_test':
        truth = arrays['test_windows'][:, 2:]
    else:
        import hashlib
        values = (frozen['seed'], row['dataset'], row['animal'], row['session'], part['split'], source, int(frame.draw.iloc[0]))
        digest = hashlib.sha256('|'.join(map(str, values)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        truth = grid[rng.integers(len(grid), size=len(arrays['endpoint_counts']))]
    lookup = {int(e): j for j, e in enumerate(np.arange(len(truth)) if source == 'run_test' else arrays['event_indices'])}
    indices = [lookup[int(e)] for e in frame.event_index]
    return tiles[tree.query(truth[indices])[1]]


def apply_model(args):
    frozen_model = json.loads(args.frozen_model.read_text())
    if frozen_model['status'] != 'frozen': raise ValueError('freeze coefficients first')
    for key, path in frozen_model['input_file_paths'].items():
        if file_sha256(path) != frozen_model['input_file_sha256'][key]: raise ValueError('frozen training input changed')
    rows = verified_sessions(args.input_dir, args.dataset)
    expected = (8, 4) if args.dataset == 'pfeiffer_foster' else (25, 5)
    if (len(rows), len({r['animal'] for r in rows})) != expected: raise ValueError('full cohort required')
    data = load_readouts(args.input_dir)
    inputs = dict(model=args.frozen_model, source=args.input_dir / 'manifest.json', script=Path(__file__),
                  protocol=ROOT / 'docs/run_error_content_diagnostic_protocol.md')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status='running', dataset=args.dataset, replay_rescored=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    outputs = []
    for row in rows:
        arrays, frozen = session_parts(row)
        rates, grid = arrays['rates_hz'], arrays['grid_cm']
        local = data.loc[data.session.eq(row['session']) & data.animal.eq(row['animal'])].copy()
        states = frozen_model['leave_one_rat_out'][row['animal']] if args.dataset == 'pfeiffer_foster' else frozen_model['states']
        pieces = []
        for part in frozen['parts']:
            subset = local.loc[local.split.eq(part['split'])].copy()
            features = predictor_features(subset, rates[part['a']], rates[part['b']], grid)
            for key in features: subset[key] = features[key]
            for model in MODELS: subset['prediction_' + model] = predict(subset, states[model])
            for (_, _), case in subset.groupby(['source', 'draw']):
                case = case.copy()
                case['true_tile'] = add_truth_tiles(case, arrays, part, row, frozen)
                pieces.append(case)
        result = pd.concat(pieces, ignore_index=True)
        output = args.output_dir / (row['animal'] + '__' + row['session'].replace('/', '_') + '.csv.gz')
        result.to_csv(output, index=False)
        entry = dict(dataset=row['dataset'], animal=row['animal'], session=row['session'],
                     path=str(output), sha256=file_sha256(output), rows=len(result), source_artifact=row['artifact_dir'])
        outputs.append(entry)
        pd.DataFrame(outputs).to_csv(args.output_dir / 'sessions.csv', index=False)
        print(json.dumps(entry), flush=True)
    unchanged = all(file_sha256(v) == manifest['input_file_sha256'][k] for k, v in inputs.items())
    manifest.update(status='complete' if unchanged else 'failed', inputs_unchanged=unchanged, results=outputs)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not unchanged: raise ValueError('inputs changed during application')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('freeze')
    train.add_argument('--input-dir', type=Path, required=True)
    train.add_argument('--output-dir', type=Path, required=True)
    apply = sub.add_parser('apply')
    apply.add_argument('--input-dir', type=Path, required=True)
    apply.add_argument('--output-dir', type=Path, required=True)
    apply.add_argument('--frozen-model', type=Path, required=True)
    apply.add_argument('--dataset', choices=('pfeiffer_foster', 'tanni2022'), required=True)
    args = p.parse_args()
    freeze_model(args) if args.command == 'freeze' else apply_model(args)


if __name__ == '__main__':
    main()
