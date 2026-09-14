#!/usr/bin/env python3
"""Independent audit of RUN-risk fitting, features, preserved outcomes and truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_regional_prevalence_calibration import raw_counts_by_assignment

KEY = ['dataset', 'animal', 'session', 'source', 'split', 'draw', 'event_index']
FEATURES = ('log_spikes', 'log_active', 'a_entropy', 'active_fraction', 'relative_width', 'a_peak',
            'log_cells', 'log_a_coverage', 'log_arena_size', 'relative_support_distance',
            'log_b_coverage_at_a', 'log_a_code_gradient', 'log_b_code_gradient_at_a')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def posterior(counts, rates):
    ll = np.einsum('ec,cp->ep', counts, np.log(rates)) - .02 * rates.sum(axis=0)
    p = np.exp(ll - np.max(ll, axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def spatial_tiles(grid):
    scaled = (grid - grid.min(axis=0)) / np.maximum(np.ptp(grid, axis=0), 1)
    indices = np.clip(np.floor(3 * scaled).astype(int), 0, 2)
    return indices[:, 0] * 3 + indices[:, 1]


def reconstructed_cases(arrays, part, frozen):
    a, b = part['a'], part['b']
    c, grid = arrays['endpoint_counts'], arrays['grid_cm']
    yield 'real', -1, c[:, a], c[:, b], None, arrays['event_indices']
    native = arrays['test_counts']
    yield 'run_test', -1, native[:, a], native[:, b], arrays['test_windows'][:, 2:], np.arange(len(native))
    for source in ('sim_matched', 'sim_drift'):
        for draw in range(2):
            values = (frozen['seed'], frozen['dataset'], frozen['animal'], frozen['session'], part['split'], source, draw)
            seed = int.from_bytes(hashlib.sha256('|'.join(map(str, values)).encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            locations = rng.integers(len(grid), size=len(c))
            rates = arrays['rates_hz'] if source == 'sim_matched' else arrays['drift_hz'] * rng.lognormal(0, .4, (len(arrays['cell_ids']), 1))
            draws = []
            for ids in (a, b):
                probabilities = rates[ids][:, locations].T
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                totals = c[:, ids].sum(axis=1)
                generated = np.array([rng.multinomial(int(n), p) for n, p in zip(totals, probabilities, strict=True)])
                np.testing.assert_array_equal(generated.sum(axis=1), totals)
                draws.append(generated)
            yield source, draw, *draws, grid[locations], arrays['event_indices']


def check_decoding(frame, ca, cb, ra, rb, grid, truth, event_ids):
    if frame.event_index.duplicated().any(): raise ValueError('duplicate event readout')
    frame = frame.set_index('event_index').loc[event_ids]
    if len(frame) != len(ca): raise ValueError('incomplete event readouts')
    tile = spatial_tiles(grid)
    decoded = []
    for side, counts, rates in (('a', ca, ra), ('b', cb, rb)):
        p = posterior(counts, rates)
        mean = p @ grid
        regional = np.column_stack([p[:, tile == i].sum(axis=1) for i in range(9)])
        values = dict(spikes=counts.sum(axis=1), active=np.count_nonzero(counts, axis=1),
            x_cm=mean[:, 0], y_cm=mean[:, 1],
            entropy=-np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1) / np.log(len(grid)),
            width_cm=np.sqrt(np.maximum(p @ (grid ** 2).sum(axis=1) - (mean ** 2).sum(axis=1), 0)),
            truth_error_cm=np.linalg.norm(mean - truth, axis=1) if truth is not None else np.full(len(frame), np.nan))
        for key, value in values.items():
            np.testing.assert_allclose(frame[side + '_' + key], value, atol=1e-6, rtol=1e-8, equal_nan=True)
        for i in range(9): np.testing.assert_allclose(frame[f'{side}_region_{i}'], regional[:, i], atol=1e-8)
        if side == 'a': np.testing.assert_allclose(frame.a_peak, p.max(axis=1), atol=1e-8)
        decoded.append((mean, regional))
    np.testing.assert_allclose(frame.endpoint_separation_cm, np.linalg.norm(decoded[0][0] - decoded[1][0], axis=1), atol=1e-6)
    np.testing.assert_allclose(frame.regional_tv, .5 * np.abs(decoded[0][1] - decoded[1][1]).sum(axis=1), atol=1e-8)
    expected_tiles = tile[cKDTree(grid).query(truth)[1]] if truth is not None else np.full(len(frame), -1)
    np.testing.assert_array_equal(frame.true_tile, expected_tiles)


def gradients(rates, grid):
    neighbors = cKDTree(grid).query_ball_point(grid, 8.01)
    result = np.zeros(len(grid))
    root_rates = np.sqrt(rates)
    for i, near in enumerate(neighbors):
        others = [j for j in near if j != i]
        if not others: continue
        contrasts = np.sum((root_rates[:, others] - root_rates[:, i, None]) ** 2, axis=0)
        distances = np.sum((grid[others] - grid[i]) ** 2, axis=1)
        result[i] = np.mean(contrasts / distances)
    return result


def features(frame, a_rates, b_rates, grid):
    diag = np.sqrt(np.sum(np.ptp(grid, axis=0) ** 2))
    dist, bin_ids = cKDTree(grid).query(np.c_[frame.a_x_cm, frame.a_y_cm])
    a_sum, b_sum = a_rates.sum(axis=0), b_rates.sum(axis=0)
    values = [np.log1p(frame.a_spikes), np.log1p(frame.a_active), frame.a_entropy,
              frame.a_active / len(a_rates), frame.a_width_cm / diag, frame.a_peak,
              np.full(len(frame), np.log1p(len(a_rates))), np.log1p(a_sum[bin_ids] / np.mean(a_sum)),
              np.full(len(frame), np.log(diag / 8)), dist / diag, np.log1p(b_sum[bin_ids] / np.mean(b_sum)),
              np.log1p(.02 * diag ** 2 * gradients(a_rates, grid)[bin_ids]),
              np.log1p(.02 * diag ** 2 * gradients(b_rates, grid)[bin_ids])]
    return np.column_stack(values)


def verify_fit(frame, state):
    counts = frame.groupby(['animal', 'session']).size().to_dict()
    sessions = frame.groupby('animal').session.nunique().to_dict()
    w = np.array([1 / (frame.animal.nunique() * sessions[r.animal] * counts[(r.animal, r.session)]) for r in frame.itertuples()])
    y = np.log1p(frame.b_truth_error_cm / frame.grid_diagonal_cm).to_numpy()
    intercept = float(w @ y)
    np.testing.assert_allclose(intercept, state['intercept'], atol=1e-10)
    if not state['columns']: return
    raw = frame[state['columns']].to_numpy(float)
    medians = np.nan_to_num(np.nanmedian(raw, axis=0))
    missing = ~np.isfinite(raw)
    raw = np.c_[np.where(missing, medians, raw), missing.astype(float)]
    mean = w @ raw
    scale = np.sqrt(w @ ((raw - mean) ** 2)); scale[scale < 1e-10] = 1
    x = (raw - mean) / scale
    weighted = w * len(frame)
    beta = np.linalg.solve(x.T @ (weighted[:, None] * x) + 10 * np.eye(x.shape[1]), x.T @ (weighted * (y - intercept)))
    for key, expected in [('medians', medians), ('means', mean), ('scales', scale), ('coefficients', beta)]:
        np.testing.assert_allclose(state[key], expected, rtol=1e-7, atol=1e-9)


def predictions(frame, state):
    if not state['columns']: return np.repeat(state['intercept'], len(frame))
    x = frame[state['columns']].to_numpy(float)
    missing = ~np.isfinite(x)
    x = np.c_[np.where(missing, state['medians'], x), missing.astype(float)]
    return np.clip(((x - state['means']) / state['scales']) @ state['coefficients'] + state['intercept'], 0, None)


def check_source(folder):
    frozen = json.loads((folder / 'frozen_populations.json').read_text())
    if file_sha256(frozen['cache_path']) != frozen['cache_sha256']: raise ValueError('changed source cache')
    raw = load(frozen['cache_path'])
    arrays = load(folder / 'audit_arrays.npz')
    mask = raw['unit_qc_mask'].astype(bool)
    support = raw['valid_spatial_bins'] & (raw['occupancy_first_half_s'] >= .05)
    np.testing.assert_array_equal(raw['cell_ids'][mask], arrays['cell_ids'])
    np.testing.assert_array_equal(raw['bin_centers_cm'][support], arrays['grid_cm'])
    np.testing.assert_array_equal(np.maximum(raw['rates_first_half_hz'][mask][:, support], 1e-4), arrays['rates_hz'])
    np.testing.assert_array_equal(np.maximum(raw['rates_second_half_hz'][mask][:, support], 1e-4), arrays['drift_hz'])
    endpoints = raw_counts_by_assignment(raw['spikes'], arrays['endpoint_start_s'], arrays['endpoint_end_s'], arrays['cell_ids'])
    np.testing.assert_array_equal(endpoints, arrays['endpoint_counts'])
    for name in ('calibration', 'test'):
        w = arrays[name + '_windows']
        reconstructed = raw_counts_by_assignment(raw['spikes'], w[:, 0], w[:, 1], arrays['cell_ids'])
        np.testing.assert_array_equal(reconstructed, arrays[name + '_counts'])
        truth = np.c_[np.interp(w[:, 0] + .01, raw['position'][:, 0], raw['position'][:, 1]),
                      np.interp(w[:, 0] + .01, raw['position'][:, 0], raw['position'][:, 2])]
        np.testing.assert_allclose(truth, w[:, 2:], atol=1e-8, rtol=0)
        quarter = np.linspace(raw['supported_run_intervals'][:, 0].min(), raw['supported_run_intervals'][:, 1].max(), 5)
        a, b = (quarter[2], quarter[3]) if name == 'calibration' else (quarter[3], quarter[4])
        if not np.all((w[:, 0] >= a) & (w[:, 1] <= b)): raise ValueError('RUN split leakage')
    return arrays, frozen


def audit_training(model, root, source_root):
    path = root / 'training_run.csv.gz'
    if file_sha256(path) != model['training_sha256']: raise ValueError('changed training rows')
    train = pd.read_csv(path)
    if not train.source.eq('run_calibration').all() or not train.dataset.eq('pfeiffer_foster').all() or not train.split.eq(0).all():
        raise ValueError('training leakage')
    if train.duplicated(KEY).any(): raise ValueError('duplicate training rows')
    source = json.loads((source_root / 'manifest.json').read_text())
    checked = []
    for row in source['results']:
        arrays, frozen = check_source(Path(row['artifact_dir']))
        part = next(p for p in frozen['parts'] if p['split'] == 0)
        local = train.loc[train.session.eq(row['session']) & train.animal.eq(row['animal'])].sort_values('event_index')
        c, rates, grid = arrays['calibration_counts'], arrays['rates_hz'], arrays['grid_cm']
        pa = posterior(c[:, part['a']], rates[part['a']]); pb = posterior(c[:, part['b']], rates[part['b']])
        amean, bmean = pa @ grid, pb @ grid
        expected = dict(a_spikes=c[:, part['a']].sum(axis=1), a_active=np.count_nonzero(c[:, part['a']], axis=1),
            a_x_cm=amean[:, 0], a_y_cm=amean[:, 1], a_peak=pa.max(axis=1),
            a_entropy=-np.sum(pa * np.log(np.maximum(pa, 1e-300)), axis=1) / np.log(len(grid)),
            a_width_cm=np.sqrt(np.maximum(pa @ (grid ** 2).sum(axis=1) - (amean ** 2).sum(axis=1), 0)),
            b_truth_error_cm=np.linalg.norm(bmean - arrays['calibration_windows'][:, 2:], axis=1))
        for key, value in expected.items(): np.testing.assert_allclose(local[key], value, atol=1e-7, rtol=1e-8)
        np.testing.assert_allclose(local[list(FEATURES)], features(local, rates[part['a']], rates[part['b']], grid), atol=1e-9)
        checked.append(dict(session=row['session'], training_rows=len(local), status='pass'))
    for state in model['states'].values(): verify_fit(train, state)
    for animal, states in model['leave_one_rat_out'].items():
        for state in states.values(): verify_fit(train.loc[train.animal.ne(animal)], state)
    return checked


def audit_applied(entry, model, dataset):
    path = Path(entry['path'])
    if file_sha256(path) != entry['sha256']: raise ValueError('changed applied predictions')
    result = pd.read_csv(path).sort_values(KEY).reset_index(drop=True)
    folder = Path(entry['source_artifact'])
    original = pd.read_csv(folder / 'event_readouts.csv.gz').sort_values(KEY).reset_index(drop=True)
    for key in original:
        if pd.api.types.is_numeric_dtype(original[key]):
            np.testing.assert_allclose(result[key], original[key], atol=1e-10, rtol=1e-9, equal_nan=True)
        else:
            np.testing.assert_array_equal(result[key], original[key])
    arrays, frozen = check_source(folder)
    rates, grid = arrays['rates_hz'], arrays['grid_cm']
    states = model['leave_one_rat_out'][entry['animal']] if dataset == 'pfeiffer_foster' else model['states']
    checked_rows = 0
    for part in frozen['parts']:
        local = result.loc[result.split.eq(part['split'])]
        a, b = part['a'], part['b']
        if set(a).intersection(b): raise ValueError('B population is not disjoint')
        np.testing.assert_allclose(local[list(FEATURES)], features(local, rates[a], rates[b], grid), atol=1e-9, rtol=1e-9)
        for source, draw, ca, cb, truth, ids in reconstructed_cases(arrays, part, frozen):
            case = local.loc[local.source.eq(source) & local.draw.eq(draw)]
            if len(case) != len(ids): raise ValueError('incomplete source/draw')
            check_decoding(case, ca, cb, rates[a], rates[b], grid, truth, ids)
            checked_rows += len(case)
    if checked_rows != len(result): raise ValueError('unrecognized or missing case')
    for name, state in states.items():
        np.testing.assert_allclose(result['prediction_' + name], predictions(result, state), atol=1e-10, rtol=1e-8)
        altered = result.copy()
        for key in ('b_spikes', 'b_active', 'b_entropy', 'b_x_cm', 'b_y_cm', 'b_truth_error_cm', 'endpoint_separation_cm'):
            altered[key] = 987654.
        np.testing.assert_array_equal(predictions(result, state), predictions(altered, state))
    return dict(dataset=dataset, animal=entry['animal'], session=entry['session'], status='pass', prediction_rows=len(result), reconstructed_rows=checked_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    path = args.root / 'frozen/frozen_model.json'
    model = json.loads(path.read_text())
    for key, value in model['input_file_paths'].items():
        if file_sha256(value) != model['input_file_sha256'][key]: raise ValueError('frozen model input changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    training = audit_training(model, args.root / 'frozen', Path(model['input_file_paths']['source']).parent)
    rows = []
    for short, dataset, expected in [('pf', 'pfeiffer_foster', 8), ('tanni', 'tanni2022', 25)]:
        applied = json.loads((args.root / short / 'manifest.json').read_text())
        if applied['status'] != 'complete' or len(applied['results']) != expected: raise ValueError('application incomplete')
        if applied['input_file_sha256']['model'] != file_sha256(path): raise ValueError('model mismatch')
        for key, value in applied['input_file_paths'].items():
            if file_sha256(value) != applied['input_file_sha256'][key]: raise ValueError('application input changed')
        for entry in applied['results']:
            row = audit_applied(entry, model, dataset)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output_dir / 'progress.csv', index=False)
            print(json.dumps(row), flush=True)
    manifest = build_script_provenance(input_paths=dict(model=path, audit=Path(__file__),
        pf=args.root / 'pf/manifest.json', tanni=args.root / 'tanni/manifest.json',
        raw_count_helper=ROOT / 'scripts/audit_regional_prevalence_calibration.py'), cwd=ROOT)
    manifest.update(status='pass', training=training, applied=rows,
        scope='all coefficients/features/predictions; raw RUN and endpoint counts; all real/RUN/simulated posterior moments, regional readouts and known errors; all truth tiles')
    (args.output_dir / 'reconstruction.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
