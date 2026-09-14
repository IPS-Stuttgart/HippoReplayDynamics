#!/usr/bin/env python3
"""PACC calibration of the original matched-population Home-content contrast."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd
from scipy.special import softmax

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import seed

DT = .02
PREVALENCES = (.05, .15, .30, .50, .75)
SESSIONS = ('Rat1/Open1', 'Rat1/Open2', 'Rat2/Open1', 'Rat4/Open2')


def regional_scores(counts, rates, near, groups):
    """Unchanged independent Poisson posteriors, returned as regional mass."""
    counts, rates, near = np.asarray(counts), np.asarray(rates), np.asarray(near, bool)
    if counts.ndim != 2 or rates.shape[0] != counts.shape[1]:
        raise ValueError('cell dimension mismatch')
    if not np.isfinite(rates).all() or np.any(rates <= 0) or np.any(counts < 0):
        raise ValueError('invalid rate/count')
    if not near.any() or near.all():
        raise ValueError('both spatial classes are required')
    out = []
    for indices in groups:
        mu = rates[np.asarray(indices, int)]
        ll = counts[:, indices] @ np.log(mu) - DT * mu.sum(axis=0)
        out.append(softmax(ll, axis=1)[:, near].sum(axis=1))
    return np.column_stack(out)


def fit_response(scores, labels, parent_ids=None):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('invalid calibration scores/labels')
    means, counts, blocks = [], [], []
    for value in (False, True):
        ix = labels == value
        counts.append(int(ix.sum()))
        means.append(float(scores[ix].mean()) if ix.any() else np.nan)
        blocks.append(int(np.unique(np.asarray(parent_ids)[ix]).size) if parent_ids is not None else counts[-1])
    gap = means[1] - means[0]
    status = ('insufficient_class_support' if min(counts) < 100 or min(blocks) < 10 else
              'weak_response_gap' if not np.isfinite(gap) or gap < .05 else 'available')
    return dict(mu0=means[0], mu1=means[1], response_gap=gap,
                n0=counts[0], n1=counts[1], blocks0=blocks[0], blocks1=blocks[1], status=status)


def adjust(mean_score, response):
    if response['status'] != 'available':
        return np.nan, response['status']
    value = (float(mean_score) - response['mu0']) / response['response_gap']
    return value, 'available' if 0 <= value <= 1 else 'incompatible_out_of_range'


def native_bank(cache, windows, cell_ids):
    edges = windows[:, 0, None] + np.arange(13)[None, :] * DT
    starts, ends = edges[:, :-1].ravel(), edges[:, 1:].ravel()
    position, spikes = cache['position'], cache['spikes']
    xy = np.column_stack([np.interp(starts + DT / 2, position[:, 0], position[:, d]) for d in (1, 2)])
    counts = np.empty((len(starts), len(cell_ids)), dtype=np.int64)
    for j, cell in enumerate(cell_ids):
        times = np.sort(spikes[spikes[:, 1] == cell, 0])
        counts[:, j] = np.searchsorted(times, ends) - np.searchsorted(times, starts)
    return dict(counts=counts, truth_cm=xy, starts_s=starts, ends_s=ends,
                parent_ids=np.repeat(np.arange(len(windows)), 12))


def synthetic_bank(session, name, rates, late, grid, near, totals, n_per_class=2000):
    rng = np.random.default_rng(seed(20260914, 'regional_prevalence', session, name))
    positions = np.concatenate([rng.choice(np.flatnonzero(near == y), n_per_class) for y in (False, True)])
    if 'poisson' in name:
        gain = 4. if name.endswith('gain4') else 1.
        counts = rng.poisson(DT * gain * rates[:, positions].T)
        drawn_totals = counts.sum(axis=1)
    else:
        drawn_totals = rng.choice(totals, len(positions))
        generator = late if 'map_drift' in name else rates
        probabilities = generator[:, positions].T.copy()
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if 'shared_assembly' in name:
            assembly = rates[:, near].mean(axis=1)
            probabilities = .2 * probabilities + .8 * (assembly / assembly.sum())[None, :]
        counts = np.array([rng.multinomial(int(n), p) for n, p in zip(drawn_totals, probabilities, strict=True)])
    return dict(counts=counts, truth_cm=grid[positions], position_ids=positions,
                drawn_totals=drawn_totals, labels=near[positions])


def checked(path, expected, inputs):
    path = Path(path)
    actual = file_sha256(path)
    if actual is None or (expected is not None and actual != expected):
        raise ValueError(f'changed/missing input: {path}')
    inputs[str(path)] = actual
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def session_data(rec, args, home, original):
    folder = args.matched_dir / rec['session'].replace('/', '_')
    inputs = {}
    freeze_path = checked(folder / 'freeze_checkpoint.json', None, inputs)
    freeze = json.loads(freeze_path.read_text())
    checked(folder / 'selected_before_confirmation.json', freeze['selection_sha256'], inputs)
    validation = load_npz(checked(folder / 'run_validation.npz', freeze['run_validation_sha256'], inputs))
    arrays = load_npz(checked(rec['input_arrays_path'], rec['input_arrays_sha256'], inputs))
    cache = load_npz(checked(rec['source']['source_cache_path'], rec['source']['source_cache_sha256'], inputs))
    pairs = [x for x in freeze['selected'] if x['family'] == 'targeted' and x['confirmation_pass']]
    if len(pairs) != 1:
        raise ValueError('original confirmed targeted pair missing')
    groups = [np.asarray(pairs[0][side]['indices'], int) for side in ('high', 'low')]
    ids, grid = arrays['cell_ids'], arrays['grid_cm']
    np.testing.assert_array_equal(ids, validation['cell_ids'])
    np.testing.assert_array_equal(grid, validation['grid_cm'])
    mask = cache['unit_qc_mask'].astype(bool)
    np.testing.assert_array_equal(cache['cell_ids'][mask], ids)
    rates = dict(early_run=validation['early_rates'], full_run=arrays['rates_hz'][:, arrays['support']])
    late = cache['rates_second_half_hz'][mask][:, arrays['support']]
    home_xy = np.asarray([home.home_x_cm, home.home_y_cm])
    near = np.linalg.norm(grid - home_xy, axis=1) <= 20
    lookup = {str(uid): j for j, uid in enumerate(arrays['window_uids'])}
    banks, matches = {}, []
    for cohort in ('all_fixed_candidates', 'full_accepted_segment'):
        ref = original.loc[original.session.eq(rec['session']) & original.family.eq('targeted') &
                           original.encoding.eq('early_run') & original.cohort.eq(cohort)].copy()
        if ref.empty or ref.window_uid.duplicated().any():
            raise ValueError('missing or duplicate original endpoints')
        indexes = []
        for row in ref.itertuples():
            j = lookup[str(row.window_uid)]
            start, end = arrays['frame_offsets'][2 * j:2 * j + 2]
            frame = int(row.fixed_endpoint_frame)
            if frame < 0 or start + frame >= end:
                raise ValueError('endpoint outside frozen window')
            indexes.append(start + frame)
        counts = arrays['frame_counts'][indexes]
        banks[cohort] = dict(counts=counts, event_ids=ref.window_uid.to_numpy(str),
                             frame_indices=np.array(indexes), truth_cm=np.full((len(ref), 2), np.nan))
        for encoding, mu in rates.items():
            p = regional_scores(counts, mu, near, groups)
            expected = original.loc[original.session.eq(rec['session']) & original.family.eq('targeted') &
                original.encoding.eq(encoding) & original.cohort.eq(cohort)].set_index('window_uid').loc[ref.window_uid]
            np.testing.assert_allclose(p, expected[['home_mass_high', 'home_mass_low']], rtol=1e-9, atol=1e-9)
            matches.append(dict(encoding=encoding, cohort=cohort, endpoints=len(ref),
                                max_error=float(np.max(np.abs(p - expected[['home_mass_high', 'home_mass_low']].to_numpy())))))
    for source, key in (('run_q3', 'match_windows'), ('run_q4', 'confirm_windows')):
        banks[source] = native_bank(cache, validation[key], ids)
        banks[source]['labels'] = np.linalg.norm(banks[source]['truth_cm'] - home_xy, axis=1) <= 20
    totals = banks['all_fixed_candidates']['counts'].sum(axis=1)
    for source in ('cal_poisson_gain1', 'cal_conditional', 'test_poisson_gain1', 'test_poisson_gain4',
                   'test_conditional', 'test_conditional_map_drift', 'test_conditional_shared_assembly'):
        banks[source] = synthetic_bank(rec['session'], source, rates['early_run'], late, grid, near, totals)
    meta = dict(inputs=inputs, groups={s: x.tolist() for s, x in zip(('high', 'low'), groups, strict=True)},
                baseline_reconstruction=matches, source_record={k: rec[k] for k in ('dataset', 'animal', 'session')})
    return banks, rates, grid, near, groups, ids, late, home_xy, meta


def measure_session(rec, args, home, original):
    banks, rates, grid, near, groups, ids, late, home_xy, meta = session_data(rec, args, home, original)
    output = args.output_dir / rec['session'].replace('/', '_')
    output.mkdir(exist_ok=False)
    (output / 'frozen_inputs.json').write_text(json.dumps(meta, indent=2) + '\n')
    np.savez_compressed(output / 'encoding.npz', **rates, late_rates=late, grid_cm=grid, near=near,
                        high_indices=groups[0], low_indices=groups[1], cell_ids=ids, home_xy=home_xy)
    scores = {}
    for source, bank in banks.items():
        score_encodings = rates if source in ('all_fixed_candidates', 'full_accepted_segment', 'run_q3') else {'early_run': rates['early_run']}
        score_bank = {name: regional_scores(bank['counts'], mu, near, groups) for name, mu in score_encodings.items()}
        np.savez_compressed(output / f'{source}.npz', **bank, **{f'{k}_scores': v for k, v in score_bank.items()})
        scores[source] = score_bank
    response_rows, rows = [], []
    for encoding in rates:
        calibration_sources = ('run_q3', 'cal_poisson_gain1', 'cal_conditional') if encoding == 'early_run' else ('run_q3',)
        for cal in calibration_sources:
            responses = [fit_response(scores[cal][encoding][:, j], banks[cal]['labels'], banks[cal].get('parent_ids')) for j in range(2)]
            for side, fit in zip(('high', 'low'), responses, strict=True):
                response_rows.append(dict(calibration=cal, encoding=encoding, side=side, **fit))
            for source, bank in banks.items():
                if source.startswith('cal_') or source == 'run_q3' or encoding not in scores[source]:
                    continue
                p = scores[source][encoding]
                panels = [('natural', float(bank['labels'].mean()))] if 'labels' in bank else [('unknown', np.nan)]
                if 'labels' in bank:
                    panels += [(f'prevalence_{pi}', pi) for pi in PREVALENCES]
                for panel, pi in panels:
                    if panel.startswith('prevalence_'):
                        mean = (1 - pi) * p[~bank['labels']].mean(axis=0) + pi * p[bank['labels']].mean(axis=0)
                    else:
                        mean = p.mean(axis=0)
                    corrected = [adjust(mean[j], responses[j]) for j in range(2)]
                    row = dict(encoding=encoding, calibration=cal, source=source, panel=panel, true_prevalence=pi,
                               n_observations=len(p), raw_high=float(mean[0]), raw_low=float(mean[1]),
                               corrected_high=corrected[0][0], corrected_low=corrected[1][0],
                               high_status=corrected[0][1], low_status=corrected[1][1])
                    row.update(raw_discrepancy=abs(row['raw_high'] - row['raw_low']),
                        corrected_discrepancy=abs(row['corrected_high'] - row['corrected_low']),
                        raw_mean_absolute_error=float(np.mean(np.abs(mean - pi))),
                        corrected_mean_absolute_error=float(np.mean(np.abs(np.array([x[0] for x in corrected]) - pi))))
                    rows.append(row)
    for name, values in (('calibration.csv', response_rows), ('prevalence.csv', rows)):
        frame = pd.DataFrame(values)
        for key in ('animal', 'session'): frame[key] = rec[key]
        frame.to_csv(output / name, index=False)
    unchanged = all(file_sha256(path) == h for path, h in meta['inputs'].items())
    if not unchanged: raise ValueError('session source changed during run')
    (output / 'outputs.json').write_text(json.dumps({p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}, indent=2) + '\n')
    return dict(animal=rec['animal'], session=rec['session'], status='complete', artifact_dir=str(output),
                baseline_checks=sum(x['endpoints'] for x in meta['baseline_reconstruction']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    original_manifest = json.loads((args.matched_dir / 'manifest.json').read_text())
    files = original_manifest['input_file_paths']
    inputs = dict(matched=args.matched_dir / 'manifest.json', benchmark=Path(files['benchmark']),
                  home=Path(files['home']), endpoints=args.matched_dir / 'matched_event_content.csv.gz',
                  population_status=args.matched_dir / 'population_match_status.csv', script=Path(__file__),
                  protocol=ROOT / 'docs/regional_prevalence_calibration_protocol.md',
                  seed_helper=ROOT / 'scripts/audit_edge_support_content.py')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status='running', method='probabilistic_adjusted_classify_and_count', independent_validation=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status = pd.read_csv(inputs['population_status'])
    confirmed = set(status.loc[status.family.eq('targeted') & status.confirmation_pass, 'session'])
    if confirmed != set(SESSIONS): raise ValueError('frozen matched cohort changed')
    records = [r for r in json.loads(inputs['benchmark'].read_text())['results'] if r['dataset'] == 'pfeiffer_foster' and r['session'] in SESSIONS]
    if len(records) != 4: raise ValueError('wrong record coverage')
    home = pd.read_csv(inputs['home']).set_index('session')
    original = pd.read_csv(inputs['endpoints'])
    results = []
    for rec in records:
        try:
            result = measure_session(rec, args, home.loc[rec['session']], original)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            result = dict(animal=rec['animal'], session=rec['session'], status='failed', reason=str(exc))
        results.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(results).to_csv(args.output_dir / 'sessions.csv', index=False)
    unchanged = all(file_sha256(v) == manifest['input_file_sha256'][k] for k, v in inputs.items())
    good = len(results) == 4 and all(x['status'] == 'complete' for x in results) and unchanged
    manifest.update(status='complete' if good else 'failed', inputs_unchanged=unchanged, results=results)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not good: raise ValueError('prevalence calibration producer failed')


if __name__ == '__main__':
    main()
