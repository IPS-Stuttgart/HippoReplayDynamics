#!/usr/bin/env python3
"""Fixed-time conditional-count likelihood test, with Poisson falsification controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from hipporeplayimm.replay_coverage import decode_independent
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES as OLD_SOURCES, load_npz, seed
from scripts.measure_encoding_uncertainty_content import endpoint_arrays, entropy, match_entropy, posterior_metrics
from scripts.measure_population_content_stability import tile_ids

METHODS = ('poisson', 'conditional_multinomial', 'poisson_entropy_matched')
ADDITIONAL = ('sim_poisson_stationary', 'sim_poisson_stationary_gain20')
SOURCES = (*OLD_SOURCES, *ADDITIONAL)


def poisson_sources(stationary, dataset, session):
    _, truth, starts = endpoint_arrays(stationary)
    grid, rates = stationary['grid_cm'], stationary['rates_hz']
    distances = cdist(truth, grid)
    position_ids = distances.argmin(axis=1)
    if np.max(distances[np.arange(len(truth)), position_ids]) > 1e-8:
        raise ValueError('stationary truth is not on the frozen grid')
    out = {}
    for source, gain in zip(ADDITIONAL, (1., 20.), strict=True):
        expected = .02*gain*rates[:, position_ids].T
        counts = np.array([np.random.default_rng(seed(20260914, 'count_conditioned_poisson', dataset, session, source, int(event_id))).poisson(mu)
                           for event_id, mu in zip(stationary['event_ids'], expected, strict=True)])
        out[source] = dict(counts=counts, truth_cm=truth, starts_s=starts,
            event_ids=stationary['event_ids'], expected_counts=expected, gain=np.array(gain), position_ids=position_ids)
    return out


def readout(counts, truth, rates, grid, groups):
    banks, diagnostic = [], {}
    for side, ids in zip(('a', 'b'), groups, strict=True):
        n, mu = counts[:, ids], rates[ids]
        bank = {name: decode_independent(n, mu, grid, .02, likelihood=name)['posterior']
                for name in METHODS[:2]}
        log_likelihood = n @ np.log(.02*mu) - .02*mu.sum(axis=0)
        p, log_t, available = match_entropy(log_likelihood, entropy(bank['conditional_multinomial']))
        bank['poisson_entropy_matched'] = p
        regional = [bank[name] @ np.eye(9)[tile_ids(grid)] for name in METHODS[:2]]
        diagnostic.update({f'{side}_total_count_reliance_tv': np.abs(regional[0]-regional[1]).sum(axis=1)/2,
            f'{side}_matched_log_temperature': log_t, f'{side}_entropy_control_available': available,
            f'{side}_spikes': n.sum(axis=1), f'{side}_active': np.count_nonzero(n, axis=1)})
        banks.append(bank)
    rows = []
    for method in METHODS:
        a, b = [posterior_metrics(bank[method], grid, truth) for bank in banks]
        frame = pd.DataFrame(dict(method=method, separation_cm=np.linalg.norm(a['mean']-b['mean'], axis=1),
            regional_tv=np.abs(a['regional']-b['regional']).sum(axis=1)/2, **diagnostic))
        for side, result in (('a', a), ('b', b)):
            for key in ('entropy', 'width', 'error', 'brier', 'nll'):
                frame[f'{side}_{key}'] = result[key]
            frame[f'{side}_x_cm'], frame[f'{side}_y_cm'] = result['mean'].T
            for j in range(9): frame[f'{side}_region{j}'] = result['regional'][:, j]
        rows.append(frame)
    return rows, banks


def measure_session(row, output):
    prior = Path(row.artifact_dir)
    source_hashes = json.loads((prior/'outputs.json').read_text())
    if any(file_sha256(prior/k) != h for k, h in source_hashes.items()):
        raise ValueError('source measurement changed')
    freeze = json.loads((prior/'frozen_measurement.json').read_text())
    if file_sha256(freeze['encoding_path']) != freeze['encoding_sha256']:
        raise ValueError('source encoding changed')
    training = json.loads((Path(freeze['encoding_path']).parent/'encoding_manifest.json').read_text())
    if not training['training_only'] or training['holdout_spikes_used_for_rate_or_unit_selection']:
        raise ValueError('training-only maps required')
    source = load_npz(prior/'real_audit.npz')
    ids, rates, grid = source['cell_ids'], source['rates_hz'], source['grid_cm']
    target = output/(row.animal+'__'+row.session.replace('/', '_'))
    target.mkdir(exist_ok=False)
    (target/'frozen_input.json').write_text(json.dumps(dict(edge_source=str(prior), freeze=freeze,
        source_outputs=source_hashes, source_outputs_sha256=file_sha256(prior/'outputs.json')), indent=2)+'\n')
    additional = poisson_sources(load_npz(prior/'sim_stationary_audit.npz'), row.dataset, row.session)
    frames, unavailable = [], 0
    for name in SOURCES:
        if name in OLD_SOURCES:
            arrays = load_npz(prior/f'{name}_audit.npz')
            for key, expected in (('cell_ids', ids), ('rates_hz', rates), ('grid_cm', grid)):
                np.testing.assert_array_equal(arrays[key], expected)
            counts, truth, starts = endpoint_arrays(arrays)
            event_ids = arrays['event_ids']
        else:
            arrays = additional[name]
            counts, truth, starts, event_ids = [arrays[k] for k in ('counts', 'truth_cm', 'starts_s', 'event_ids')]
            np.savez_compressed(target/f'{name}_generation.npz', **arrays)
        for part in freeze['groups']:
            groups = [np.searchsorted(ids, part[side+'_ids']) for side in ('a', 'b')]
            for side, group in zip(('a', 'b'), groups, strict=True):
                np.testing.assert_array_equal(ids[group], part[side+'_ids'])
            rows, banks = readout(counts, truth, rates, grid, groups)
            for frame in rows:
                frame['dataset'], frame['animal'], frame['session'] = row.dataset, row.animal, row.session
                frame['source'], frame['split'], frame['event_index'] = name, part['split'], event_ids
                frame['original_start_s'], frame['original_end_s'] = starts, starts+.02
                frame['n_cells_per_group'] = len(groups[0])
                frames.append(frame)
            unavailable += int((~rows[0].a_entropy_control_available | ~rows[0].b_entropy_control_available).sum())
            audit = dict(counts=counts, truth_cm=truth, starts_s=starts, event_ids=event_ids,
                         a_indices=groups[0], b_indices=groups[1])
            for side, bank in zip(('a', 'b'), banks, strict=True):
                audit.update({f'{side}_{method}_posterior': p for method, p in bank.items()})
            np.savez_compressed(target/f'{name}_split{part["split"]}_audit.npz', **audit)
    result = pd.concat(frames, ignore_index=True)
    before = pd.read_csv(prior/'edge_readouts.csv.gz')
    before = before.loc[before.policy.eq('raw_endpoint')].sort_values(['source', 'split', 'event_index'])
    after = result.loc[result.method.eq('poisson') & result.source.isin(OLD_SOURCES)].sort_values(['source', 'split', 'event_index'])
    for key in ('event_index', 'source', 'split'): np.testing.assert_array_equal(after[key], before[key])
    for key in ('separation_cm', 'regional_tv', 'a_entropy', 'b_entropy'):
        np.testing.assert_allclose(after[key], before[key], atol=1e-8, rtol=1e-9)
    result.to_csv(target/'event_readouts.csv.gz', index=False)
    (target/'outputs.json').write_text(json.dumps({p.name: file_sha256(p) for p in target.iterdir() if p.is_file()}, indent=2)+'\n')
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status='complete', rows=len(result),
        artifact_dir=str(target), selected_candidates=row.selected_candidates, entropy_control_unavailable=unavailable)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    inputs = dict(source=args.measurement_dir/'measurement_sessions.csv', script=Path(__file__),
        protocol=ROOT/'docs/count_conditioned_content_protocol.md', decoder=ROOT/'src/hipporeplayimm/replay_coverage.py',
        metrics=ROOT/'scripts/measure_encoding_uncertainty_content.py', regions=ROOT/'scripts/measure_population_content_stability.py')
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest['status'] = 'running'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    sessions, rows = pd.read_csv(inputs['source']), []
    if len(sessions) != 8 or sessions.animal.nunique() != 4 or not sessions.selected_candidates.eq(200).all():
        raise ValueError('full frozen cohort required')
    for row in sessions.itertuples(index=False):
        try:
            if row.status != 'complete': raise ValueError('source incomplete')
            result = measure_session(row, args.output_dir)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status='failed', reason=str(exc))
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir/'measurement_sessions.csv', index=False)
        print(json.dumps(result), flush=True)
    unchanged = all(file_sha256(v) == manifest['input_file_sha256'][k] for k, v in inputs.items())
    passed = len(rows) == 8 and all(r['status'] == 'complete' for r in rows)
    manifest.update(status='complete' if passed and unchanged else 'failed', inputs_unchanged=unchanged, results=rows)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not passed or not unchanged: raise ValueError('count-conditioned measurement failed')


if __name__ == '__main__':
    main()
