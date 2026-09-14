#!/usr/bin/env python3
"""Independently reconstruct regional calibration and its source observations."""
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

from scripts._provenance import build_script_provenance, file_sha256


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def independently_decode(counts, rates, mask, populations):
    results = []
    for pop in populations:
        log_rates = np.log(rates[pop])
        prior_silence = -.02 * np.sum(rates[pop], axis=0)
        values = []
        for start in range(0, len(counts), 256):
            log_p = np.einsum('ec,cb->eb', counts[start:start + 256, pop], log_rates) + prior_silence
            likelihood = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
            values.extend(likelihood[:, mask].sum(axis=1) / likelihood.sum(axis=1))
        results.append(values)
    return np.asarray(results).T


def reconstruct_response(scores, labels, parents=None):
    n0, n1 = int((~labels).sum()), int(labels.sum())
    b0 = int(np.unique(parents[~labels]).size) if parents is not None else n0
    b1 = int(np.unique(parents[labels]).size) if parents is not None else n1
    mu0 = float(scores[~labels].mean()) if n0 else np.nan
    mu1 = float(scores[labels].mean()) if n1 else np.nan
    gap = mu1 - mu0
    status = 'available'
    if min(n0, n1) < 100 or min(b0, b1) < 10:
        status = 'insufficient_class_support'
    elif not np.isfinite(gap) or gap < .05:
        status = 'weak_response_gap'
    return dict(mu0=mu0, mu1=mu1, response_gap=gap, n0=n0, n1=n1, blocks0=b0, blocks1=b1, status=status)


def raw_counts_by_assignment(spikes, starts, ends, ids):
    """Assign raw spikes to windows once, rather than per-cell search differences."""
    order = np.argsort(starts)
    a, b = starts[order], ends[order]
    if np.any(a[1:] < b[:-1] - 1e-9):
        raise ValueError('native windows overlap')
    cell = np.searchsorted(ids, spikes[:, 1].astype(int))
    window = np.searchsorted(a, spikes[:, 0], side='right') - 1
    valid = (window >= 0) & (cell < len(ids))
    ix = np.flatnonzero(valid)
    ix = ix[(spikes[ix, 0] < b[window[ix]]) & (ids[cell[ix]] == spikes[ix, 1])]
    result = np.bincount(window[ix] * len(ids) + cell[ix], minlength=len(a) * len(ids)).reshape(len(a), len(ids))
    return result[np.argsort(order)]


def audit_session(row, record, matched):
    folder = Path(row.artifact_dir)
    hashes = json.loads((folder / 'outputs.json').read_text())
    if any(file_sha256(folder / name) != value for name, value in hashes.items()):
        raise ValueError('producer output changed')
    meta = json.loads((folder / 'frozen_inputs.json').read_text())
    if any(file_sha256(path) != value for path, value in meta['inputs'].items()):
        raise ValueError('source changed')
    e = load(folder / 'encoding.npz')
    raw = load(record['source']['source_cache_path'])
    original = load(record['input_arrays_path'])
    validation = load(matched / row.session.replace('/', '_') / 'run_validation.npz')
    freeze = json.loads((matched / row.session.replace('/', '_') / 'freeze_checkpoint.json').read_text())
    pair = next(x for x in freeze['selected'] if x['family'] == 'targeted' and x['confirmation_pass'])
    for side in ('high', 'low'):
        np.testing.assert_array_equal(e[f'{side}_indices'], pair[side]['indices'])
    np.testing.assert_array_equal(e['cell_ids'], original['cell_ids'])
    np.testing.assert_array_equal(e['early_run'], validation['early_rates'])
    np.testing.assert_array_equal(e['full_run'], original['rates_hz'][:, original['support']])
    np.testing.assert_array_equal(e['late_rates'], raw['rates_second_half_hz'][raw['unit_qc_mask'].astype(bool)][:, original['support']])
    np.testing.assert_array_equal(e['grid_cm'], original['grid_cm'])
    original_manifest = json.loads((matched / 'manifest.json').read_text())
    home = pd.read_csv(original_manifest['input_file_paths']['home']).set_index('session').loc[row.session]
    np.testing.assert_array_equal(e['home_xy'], home[['home_x_cm', 'home_y_cm']].to_numpy(float))
    np.testing.assert_array_equal(e['near'], np.linalg.norm(e['grid_cm'] - e['home_xy'], axis=1) <= 20)
    banks = {p.stem: load(p) for p in folder.glob('*.npz') if p.stem != 'encoding'}
    score_count, spike_count = 0, 0
    score_banks = {}
    for name, bank in banks.items():
        count = bank['counts']
        if count.shape[1] != len(e['cell_ids']) or not np.isfinite(count).all() or np.any(count < 0) or np.any(count != count.astype(int)):
            raise ValueError('invalid counts')
        if name in ('all_fixed_candidates', 'full_accepted_segment'):
            np.testing.assert_array_equal(count, original['frame_counts'][bank['frame_indices']])
            old = pd.read_csv(matched / row.session.replace('/', '_') / 'replay_content.csv.gz')
            old = old.loc[old.family.eq('targeted') & old.encoding.eq('early_run') & old.cohort.eq(name)]
            np.testing.assert_array_equal(bank['event_ids'], old.window_uid.to_numpy(str))
            lookup = {str(uid): j for j, uid in enumerate(original['window_uids'])}
            expected_indices = [int(original['frame_offsets'][2 * lookup[str(r.window_uid)]] + r.fixed_endpoint_frame) for r in old.itertuples()]
            np.testing.assert_array_equal(bank['frame_indices'], expected_indices)
        elif name.startswith('run_'):
            windows = validation['match_windows' if name == 'run_q3' else 'confirm_windows']
            starts = np.array([w[0] + j * .02 for w in windows for j in range(12)])
            ends = np.array([w[0] + j * .02 for w in windows for j in range(1, 13)])
            np.testing.assert_array_equal(bank['starts_s'], starts)
            np.testing.assert_array_equal(bank['ends_s'], ends)
            np.testing.assert_array_equal(bank['parent_ids'], np.repeat(np.arange(len(windows)), 12))
            np.testing.assert_array_equal(count, raw_counts_by_assignment(raw['spikes'], starts, ends, e['cell_ids']))
            xy = np.column_stack([np.interp(starts + .01, raw['position'][:, 0], raw['position'][:, d]) for d in (1, 2)])
            np.testing.assert_allclose(bank['truth_cm'], xy, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(bank['labels'], np.linalg.norm(xy - e['home_xy'], axis=1) <= 20)
        else:
            digest = hashlib.sha256(f'20260914|regional_prevalence|{row.session}|{name}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
            position_ids = np.r_[rng.choice(np.flatnonzero(~e['near']), 2000), rng.choice(np.flatnonzero(e['near']), 2000)]
            np.testing.assert_array_equal(position_ids, bank['position_ids'])
            np.testing.assert_array_equal(bank['truth_cm'], e['grid_cm'][position_ids])
            np.testing.assert_array_equal(bank['labels'], e['near'][position_ids])
            if 'poisson' in name:
                gain = 4. if name.endswith('gain4') else 1.
                expected = rng.poisson(.02 * gain * e['early_run'][:, position_ids].T)
            else:
                n = rng.choice(banks['all_fixed_candidates']['counts'].sum(axis=1), len(position_ids))
                mu = e['late_rates'] if 'map_drift' in name else e['early_run']
                probs = mu[:, position_ids].T.copy()
                probs /= probs.sum(axis=1, keepdims=True)
                if 'shared_assembly' in name:
                    assembly = e['early_run'][:, e['near']].mean(axis=1)
                    probs = .2 * probs + .8 * (assembly / assembly.sum())[None, :]
                expected = np.array([rng.multinomial(int(total), p) for total, p in zip(n, probs, strict=True)])
            np.testing.assert_array_equal(count, expected)
            np.testing.assert_array_equal(bank['drawn_totals'], count.sum(axis=1))
        spike_count += int(count.sum())
        score_banks[name] = {}
        for encoding in ('early_run', 'full_run'):
            key = encoding + '_scores'
            if key not in bank: continue
            rebuilt = independently_decode(count, e[encoding], e['near'], [e['high_indices'], e['low_indices']])
            np.testing.assert_allclose(bank[key], rebuilt, rtol=1e-9, atol=1e-10)
            score_banks[name][encoding] = rebuilt
            score_count += rebuilt.size
    cal = pd.read_csv(folder / 'calibration.csv')
    responses = {}
    for r in cal.itertuples():
        j = ('high', 'low').index(r.side)
        bank = banks[r.calibration]
        result = reconstruct_response(score_banks[r.calibration][r.encoding][:, j], bank['labels'], bank.get('parent_ids'))
        for key, value in result.items():
            if key == 'status':
                assert value == getattr(r, key)
            else:
                np.testing.assert_allclose(value, getattr(r, key), atol=1e-10, rtol=1e-9, equal_nan=True)
        responses[(r.encoding, r.calibration, r.side)] = result
    estimates = pd.read_csv(folder / 'prevalence.csv')
    for r in estimates.itertuples():
        p = score_banks[r.source][r.encoding]
        pi = r.true_prevalence
        if r.panel.startswith('prevalence_'):
            y = banks[r.source]['labels']
            means = pi * p[y].mean(axis=0) + (1 - pi) * p[~y].mean(axis=0)
        else:
            means = p.mean(axis=0)
            if 'labels' in banks[r.source]:
                np.testing.assert_allclose(pi, banks[r.source]['labels'].mean())
        corrected = []
        for j, side in enumerate(('high', 'low')):
            fit = responses[(r.encoding, r.calibration, side)]
            np.testing.assert_allclose(means[j], getattr(r, 'raw_' + side), atol=1e-10)
            value = (means[j] - fit['mu0']) / fit['response_gap'] if fit['status'] == 'available' else np.nan
            status = ('available' if 0 <= value <= 1 else 'incompatible_out_of_range') if fit['status'] == 'available' else fit['status']
            assert status == getattr(r, side + '_status')
            np.testing.assert_allclose(value, getattr(r, 'corrected_' + side), atol=1e-9, equal_nan=True)
            corrected.append(value)
        for key, value in dict(raw_discrepancy=abs(means[0] - means[1]), corrected_discrepancy=abs(corrected[0] - corrected[1]),
                              raw_mean_absolute_error=np.mean(np.abs(means - pi)),
                              corrected_mean_absolute_error=np.mean(np.abs(np.array(corrected) - pi))).items():
            np.testing.assert_allclose(value, getattr(r, key), atol=1e-9, equal_nan=True)
    return dict(session=row.session, status='pass', reconstructed_scores=score_count,
                reconstructed_spikes=spike_count, responses=len(cal), prevalence_rows=len(estimates))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.result_dir / 'manifest.json').read_text())
    if source['status'] != 'complete': raise ValueError('producer not complete')
    for key, path in source['input_file_paths'].items():
        if file_sha256(path) != source['input_file_sha256'][key]: raise ValueError('producer input changed')
    sessions = pd.read_csv(args.result_dir / 'sessions.csv')
    if len(sessions) != 4 or not sessions.status.eq('complete').all(): raise ValueError('incomplete coverage')
    records = json.loads(Path(source['input_file_paths']['benchmark']).read_text())['results']
    matched = Path(source['input_file_paths']['matched']).parent
    provenance = build_script_provenance(input_paths={'source': args.result_dir / 'manifest.json', 'audit': Path(__file__)}, cwd=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in sessions.itertuples():
        record = next(r for r in records if r['dataset'] == 'pfeiffer_foster' and r['session'] == row.session)
        result = audit_session(row, record, matched)
        rows.append(result)
        print(json.dumps(result), flush=True)
    provenance.update(status='pass', results=rows,
        scope='all saved readouts, raw native counts, synthetic draws, calibration and estimates; not biological replay truth')
    (args.output_dir / 'reconstruction.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
