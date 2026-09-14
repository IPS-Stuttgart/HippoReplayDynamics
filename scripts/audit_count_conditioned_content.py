#!/usr/bin/env python3
"""Independent SciPy count-conditioning and unchanged-clock audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multinomial

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES as OLD_SOURCES, load_npz, recount, seed
from scripts.audit_encoding_uncertainty_content import check_metrics, h, metrics, scipy_posterior

METHODS = ('poisson', 'conditional_multinomial', 'poisson_entropy_matched')
ADDITIONAL = ('sim_poisson_stationary', 'sim_poisson_stationary_gain20')
SOURCES = (*OLD_SOURCES, *ADDITIONAL)


def conditional_posterior(counts, rates):
    p = (rates/rates.sum(axis=0)).T
    ll = multinomial.logpmf(counts[:, None, :], counts.sum(axis=1)[:, None], p[None])
    return np.exp(ll-logsumexp(ll, axis=1, keepdims=True))


def raw_endpoints(src):
    counts, truth, starts = [], [], []
    for i, hi in enumerate(src['offsets'][1:]):
        lo = src['offsets'][i]
        if hi-lo < 4: raise ValueError('fewer than four complete bins')
        counts.append(src['counts'][hi-4:hi].sum(axis=0))
        truth.append(src['truth_base_cm'][hi-4:hi].mean(axis=0))
        starts.append(src['starts_s'][i]+.005*(hi-lo-4))
    return np.array(counts), np.array(truth), np.array(starts)


def verify_one(row):
    folder = Path(row.artifact_dir)
    current = json.loads((folder/'outputs.json').read_text())
    if any(file_sha256(folder/k) != v for k, v in current.items()):
        raise ValueError('output changed')
    frozen = json.loads((folder/'frozen_input.json').read_text())
    prior, freeze = Path(frozen['edge_source']), frozen['freeze']
    if file_sha256(prior/'outputs.json') != frozen['source_outputs_sha256'] or any(file_sha256(prior/k) != v for k, v in frozen['source_outputs'].items()):
        raise ValueError('edge source changed')
    if file_sha256(freeze['encoding_path']) != freeze['encoding_sha256']:
        raise ValueError('RUN input changed')
    data = load_npz(freeze['encoding_path'])
    shared = load_npz(prior/'real_audit.npz')
    grid, rates, ids = (shared[k] for k in ('grid_cm', 'rates_hz', 'cell_ids'))
    frame = pd.read_csv(folder/'event_readouts.csv.gz', float_precision='round_trip')
    if len(frame) != row.rows or frame.duplicated(['source', 'split', 'method', 'event_index']).any():
        raise ValueError('row count/keys changed')
    if set(frame.source) != set(SOURCES) or set(frame.method) != set(METHODS) or set(frame.split) != {0, 1, 2}:
        raise ValueError('missing required source/split/method')
    for k in ('dataset', 'animal', 'session'):
        if not frame[k].eq(getattr(row, k)).all(): raise ValueError('recording metadata mismatch')
    checked, native, posterior_rows, unavailable, poisson_draws = 0, 0, 0, 0, 0
    for source in SOURCES:
        if source in OLD_SOURCES:
            src = load_npz(prior/f'{source}_audit.npz')
            for k, value in (('cell_ids', ids), ('rates_hz', rates), ('grid_cm', grid)):
                np.testing.assert_array_equal(src[k], value)
            counts, truth, starts = raw_endpoints(src)
            event_ids = src['event_ids']
        else:
            src = load_npz(prior/'sim_stationary_audit.npz')
            _, truth, starts = raw_endpoints(src)
            event_ids = src['event_ids']
            distances = np.linalg.norm(truth[:, None]-grid[None], axis=2)
            pos = distances.argmin(axis=1)
            if np.max(distances[np.arange(len(pos)), pos]) > 1e-8: raise ValueError('invalid frozen stationary locations')
            gain = 20. if source.endswith('gain20') else 1.
            expected = rates[:, pos].T*(gain*.02)
            counts = np.array([np.random.default_rng(seed(20260914, 'count_conditioned_poisson', row.dataset, row.session, source, int(e))).poisson(mu)
                               for e, mu in zip(event_ids, expected, strict=True)])
            generated = load_npz(folder/f'{source}_generation.npz')
            for k, value in (('expected_counts', expected), ('counts', counts), ('truth_cm', truth),
                             ('starts_s', starts), ('event_ids', event_ids), ('gain', gain), ('position_ids', pos)):
                np.testing.assert_allclose(generated[k], value, atol=1e-10, rtol=1e-12)
            poisson_draws += counts.size
        if source in ('real', 'run_q4'):
            ends = src['starts_s']+.005*np.diff(src['offsets'])
            np.testing.assert_array_equal(counts, recount(data['spikes'], ids, starts, ends))
            native += len(counts)
        for split in range(3):
            stored = load_npz(folder/f'{source}_split{split}_audit.npz')
            for k, v in (('counts', counts), ('truth_cm', truth), ('starts_s', starts), ('event_ids', event_ids)):
                np.testing.assert_allclose(stored[k], v, equal_nan=True, atol=1e-10, rtol=0)
            order = np.random.default_rng(seed(freeze['seed'], 'edge_support_partition', row.dataset, row.session, split)).permutation(len(ids))
            size = len(order)//2
            groups = (np.sort(order[:size]), np.sort(order[size:2*size]))
            view = frame.loc[frame.source.eq(source) & frame.split.eq(split)]
            reference = view.loc[view.method.eq('poisson')].set_index('event_index').loc[event_ids]
            banks = []
            for side, group in zip(('a', 'b'), groups, strict=True):
                np.testing.assert_array_equal(stored[side+'_indices'], group)
                n, mu = counts[:, group], rates[group]
                p, ll = scipy_posterior(n, mu)
                conditional = conditional_posterior(n, mu)
                log_t = reference[side+'_matched_log_temperature'].to_numpy()
                if not ((log_t >= -20) & (log_t <= 20)).all(): raise ValueError('temperature outside frozen bracket')
                tempered = (ll-ll.max(axis=1, keepdims=True))/np.exp(log_t)[:, None]
                control = np.exp(tempered-logsumexp(tempered, axis=1, keepdims=True))
                available = np.abs(h(control)-h(conditional)) <= 1e-7
                np.testing.assert_array_equal(reference[side+'_entropy_control_available'], available)
                unavailable += int((~available).sum())
                bank = dict(poisson=p, conditional_multinomial=conditional, poisson_entropy_matched=control)
                for name, posterior in bank.items():
                    np.testing.assert_allclose(stored[f'{side}_{name}_posterior'], posterior, atol=1e-8, rtol=1e-8)
                    posterior_rows += len(posterior)
                a = metrics(p, grid, truth)['regional']
                b = metrics(conditional, grid, truth)['regional']
                np.testing.assert_allclose(reference[side+'_total_count_reliance_tv'], np.abs(a-b).sum(axis=1)/2, atol=1e-8)
                np.testing.assert_array_equal(reference[side+'_spikes'], n.sum(axis=1))
                np.testing.assert_array_equal(reference[side+'_active'], np.count_nonzero(n, axis=1))
                banks.append(bank)
            for method in METHODS:
                selected = view.loc[view.method.eq(method)].set_index('event_index')
                np.testing.assert_array_equal(sorted(selected.index), sorted(event_ids))
                selected = selected.loc[event_ids]
                np.testing.assert_allclose(selected.original_start_s, starts, atol=1e-10, rtol=0)
                np.testing.assert_allclose(selected.original_end_s, starts+.02, atol=1e-10, rtol=0)
                if not selected.n_cells_per_group.eq(size).all(): raise ValueError('population size changed')
                for side in ('a', 'b'):
                    for key in ('total_count_reliance_tv', 'matched_log_temperature', 'entropy_control_available', 'spikes', 'active'):
                        np.testing.assert_array_equal(selected[side+'_'+key], reference[side+'_'+key])
                check_metrics(selected, banks[0][method], banks[1][method], grid, truth)
                checked += len(selected)
    if checked != len(frame): raise ValueError('unaudited rows')
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status='passed',
        readout_rows=checked, native_endpoint_windows=native, reconstructed_posterior_rows=posterior_rows,
        reconstructed_poisson_count_entries=poisson_draws, entropy_population_windows_unavailable=unavailable,
        readout_sha256=file_sha256(folder/'event_readouts.csv.gz'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-dir', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    inputs = {f'sessions{i}': root/'measurement_sessions.csv' for i, root in enumerate(args.measurement_dir)}
    inputs.update(auditor=Path(__file__), metrics=ROOT/'scripts/audit_encoding_uncertainty_content.py')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest['status'] = 'running'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'independent_audit.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    for root in args.measurement_dir:
        producer = json.loads((root/'manifest.json').read_text())
        if producer['status'] != 'complete' or not producer['inputs_unchanged']: raise ValueError('unfinished measurement')
        for k, v in producer['input_file_paths'].items():
            if file_sha256(v) != producer['input_file_sha256'][k]: raise ValueError('frozen source changed')
        sessions = pd.read_csv(root/'measurement_sessions.csv')
        if len(sessions) != 8 or sessions.animal.nunique() != 4: raise ValueError('incomplete cohort')
        for row in sessions.itertuples(index=False):
            try:
                if row.status != 'complete': raise ValueError('measurement failed')
                result = verify_one(row)
            except (OSError, KeyError, ValueError, AssertionError) as exc:
                result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status='failed', reason=str(exc))
            results.append(result)
            print(json.dumps(result), flush=True)
            pd.DataFrame(results).to_csv(args.output_dir/'independent_audit_sessions.csv', index=False)
    passed = bool(results) and all(r['status'] == 'passed' for r in results)
    unchanged = all(file_sha256(v) == manifest['input_file_sha256'][k] for k, v in inputs.items())
    manifest.update(status='passed' if passed and unchanged else 'failed', inputs_unchanged=unchanged, results=results)
    (args.output_dir/'independent_audit.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not passed or not unchanged: raise ValueError('independent count-conditioned audit failed')


if __name__ == '__main__':
    main()
