#!/usr/bin/env python3
"""Cross-event learned neural HMM versus frozen spatial predictive scores."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from _provenance import build_script_provenance, file_sha256
from audit_2d_count_conditioned_prediction import IDENTITY, KEYS, aggregate, folds, seed
from audit_2d_predictive_order_map import K, shuffled_indices

from hipporeplayimm.learned_assembly_prediction import LearnedAssembly, fit_learned_assembly, predict_learned_assembly

PARENT_HASH = 'd16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386'
RATE_HASH = '84418e7db33df939b9609dae312eed19d72b7dc6e8d382d7cbdc91e4c1744943'
STATES = (20, 50, 100)


def load_fit(path):
    with np.load(path) as z:
        values = {k: z[k] for k in z.files}
    for key in ('converged',):
        values[key] = bool(values[key])
    for key in ('restart', 'n_calibration_bins', 'n_calibration_events'):
        values[key] = int(values[key])
    return LearnedAssembly(**values)


def checked_manifest(root, name, expected):
    path = root / name
    if file_sha256(path) != expected:
        raise ValueError(f'parent manifest changed: {path}')
    manifest = json.loads(path.read_text())
    if manifest['status'] != 'complete':
        raise ValueError('incomplete parent')
    return manifest


def checked_file(root, manifest, name):
    path = root / name
    if file_sha256(path) != manifest['output_sha256'][name]:
        raise ValueError(f'changed source output: {name}')
    return path


def reuse_completed(previous, out, items, inputs):
    """Recover completed fold artifacts from a terminal failed run, fail closed."""
    path = previous / 'learned_assembly_manifest.json'
    old = json.loads(path.read_text())
    if old['status'] != 'failed' or old['selected_sessions'] != items:
        raise ValueError('reuse requires matching terminal failed experiment')
    for key in ('parent', 'rate', 'protocol', 'hmmlearn_hmm.py', 'hmmlearn_base.py', 'hmmlearn__emissions.py', 'hmmlearn_compiled_kernel'):
        if old['input_file_sha256'][key] != file_sha256(inputs[key]):
            raise ValueError(f'reused input changed: {key}')
    completed = []
    for item in items:
        for fold in range(5):
            for states in STATES:
                name = f"{item['tag']}__fold{fold}__k{states}_manifest.json"
                source = previous / name
                if not source.exists():
                    continue
                if file_sha256(source) != old['output_sha256'].get(name):
                    raise ValueError('unverified cached fold manifest')
                result = json.loads(source.read_text())
                if (result['tag'], result['fold'], result['n_states']) != (item['tag'], fold, states):
                    raise ValueError('cached fold identity differs')
                for filename, digest in result['output_sha256'].items():
                    if filename != Path(filename).name or digest != old['output_sha256'].get(filename) or file_sha256(previous / filename) != digest:
                        raise ValueError('cached fold output differs')
                    shutil.copy2(previous / filename, out / filename)
                result.update(reused_producer_manifest=str(path), reused_producer_sha256=file_sha256(path), reused_producer_commit=old['code_commit'])
                (out / name).write_text(json.dumps(result, indent=2) + '\n')
                completed.append(result)
    return completed


def task(job):
    item, fold_id, n_states, parent_name, out_name = job
    parent, out = Path(parent_name), Path(out_name)
    tag = item['tag']
    identity = tuple(item[k] for k in IDENTITY)
    fields = dict(zip(IDENTITY, identity, strict=True))
    selected = pd.read_csv(parent / f'{tag}_selection.csv')
    with np.load(parent / f'{tag}_cache.npz') as z:
        cache = {k: z[k] for k in z.files}
    _, test, calibration, excluded = folds(selected)[fold_id]
    fit_seed = seed(*identity, 'learned_assembly', fold_id, n_states)
    tick = time.monotonic()
    fit = fit_learned_assembly([cache[f'counts_{eid}'] for eid in calibration.event_id], n_states, fit_seed)
    fit_seconds = time.monotonic() - tick
    stem = f'{tag}__fold{fold_id}__k{n_states}'
    np.savez_compressed(out / f'{stem}_fit.npz', **asdict(fit))
    rows = []
    for e in test.itertuples(index=False):
        eid = int(e.event_id)
        counts = cache[f'counts_{eid}']
        orders = [np.arange(len(counts))] + [shuffled_indices(identity, eid, len(counts), k) for k in range(K)]
        for split in range(5):
            tr, he = cache[f'train_{split}'], cache[f'held_{split}']
            original = None
            for shuffle, order in enumerate(orders, -1):
                scores, hashes = predict_learned_assembly(counts[order], tr, he, fit)
                if not all(np.isfinite(v) and v <= 1e-8 for v in scores.values()):
                    raise ValueError('invalid held-out probability score')
                if original is None:
                    original = scores
                else:
                    for name in ('same_emissions_iid', 'nonspatial_global'):
                        if not np.isclose(scores[name], original[name], atol=1e-8, rtol=0):
                            raise ValueError('independent predictor changed under time shuffle')
                rows.append(fields | {
                    'event_id': eid, 'fold': fold_id, 'n_states': n_states, 'split': split,
                    'shuffle': shuffle, 'n_heldout_spikes': int(counts[:, he].sum()),
                    'n_train_spikes': int(counts[:, tr].sum()), 'n_time_bins': len(counts),
                    'fit_converged': fit.converged, 'heldout_used_for_inference': False,
                    'training_posterior_sha256': hashes['learned_hmm'],
                    **{f'score_{name}': value for name, value in scores.items()},
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f'{stem}_scores.csv.gz', index=False)
    result = fields | {
        'tag': tag, 'stem': stem, 'fold': fold_id, 'n_states': n_states, 'seed': fit_seed,
        'events': len(test), 'rows': len(frame), 'fit_converged': bool(fit.converged),
        'fit_iterations': len(fit.objective_trace) - 1, 'best_restart': fit.restart,
        'objective': float(fit.objective_trace[-1]), 'fit_runtime_s': fit_seconds,
        'runtime_s': time.monotonic() - tick, 'calibration_events': len(calibration),
        'calibration_bins': fit.n_calibration_bins,
        'test_ids': test.event_id.astype(int).tolist(),
        'calibration_ids': calibration.event_id.astype(int).tolist(),
        'excluded_ids': excluded.event_id.astype(int).tolist(),
        'output_sha256': {p.name: file_sha256(p) for p in out.glob(f'{stem}_*')},
    }
    (out / f'{stem}_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('tag', 'fold', 'n_states', 'events', 'fit_converged', 'fit_runtime_s', 'runtime_s')}), flush=True)
    return result


def contrasts(scores, spatial, old_order):
    keys = KEYS + ['n_states']
    if scores.empty or scores.duplicated(keys + ['shuffle']).any():
        raise ValueError('empty or duplicate assembly scores')
    if not all(set(g.shuffle) == set(range(-1, K)) for _, g in scores.groupby(keys)):
        raise ValueError('incomplete original/shuffle factors')
    original = scores[scores.shuffle.eq(-1)].set_index(keys)
    average = scores[scores.shuffle.ge(0)].groupby(keys).score_learned_hmm.mean()
    source = spatial[spatial.alpha.eq(100) & spatial['map'].eq('real')].set_index(KEYS)
    temporal = old_order[old_order.contrast.eq('alpha100__first_order_imm__real_order_advantage')].set_index(KEYS)
    if source.index.duplicated().any() or temporal.index.duplicated().any():
        raise ValueError('duplicate paired spatial source')
    joined = original.join(source[['score_first_order_imm', 'score_iid_position', 'score_event_global']], on=KEYS, validate='many_to_one')
    joined = joined.join(temporal.delta.rename('spatial_order_advantage'), on=KEYS, validate='many_to_one')
    if not np.isfinite(joined.filter(regex='^score_|spatial_order_advantage')).all().all():
        raise ValueError('missing or nonfinite paired source')
    values = {
        'spatial_imm_minus_learned_hmm': joined.score_first_order_imm - joined.score_learned_hmm,
        'spatial_iid_minus_learned_hmm': joined.score_iid_position - joined.score_learned_hmm,
        'learned_hmm_minus_nonspatial_global': joined.score_learned_hmm - joined.score_nonspatial_global,
        'learned_hmm_minus_run_shrinkage_global': joined.score_learned_hmm - joined.score_event_global,
        'learned_hmm_minus_same_emissions_iid': joined.score_learned_hmm - joined.score_same_emissions_iid,
        'learned_hmm_order_advantage': joined.score_learned_hmm - average.reindex(joined.index),
        'spatial_minus_assembly_order_advantage': joined.spatial_order_advantage - (joined.score_learned_hmm - average.reindex(joined.index)),
    }
    rows = []
    for name, delta in values.items():
        part = delta.rename('delta').to_frame()
        part['delta_per_heldout_spike'] = delta / joined.n_heldout_spikes.replace(0, np.nan)
        part = part.reset_index()
        part['contrast'] = 'k' + part.n_states.astype(str) + '__' + name
        rows.append(part)
    splits = pd.concat(rows, ignore_index=True)
    event = splits.groupby(IDENTITY + ['event_id', 'contrast'], as_index=False)[['delta', 'delta_per_heldout_spike']].median()
    return splits, event


def run(args):
    parent, rate, out = (Path(p).resolve() for p in (args.parent_dir, args.rate_dir, args.output_dir))
    pm = checked_manifest(parent, 'conditional_2d_manifest.json', PARENT_HASH)
    rm = checked_manifest(rate, 'mua_rate_transfer_manifest.json', RATE_HASH)
    items = sorted(pm['completed'], key=lambda x: x['tag'])
    if args.pilot:
        items = [next(i for i in items if i['dataset'] == d) for d in ('pfeiffer_foster', 'tanni2022')]
    if out.exists() and any(out.iterdir()):
        raise ValueError('refusing overwrite')
    out.mkdir(parents=True, exist_ok=True)
    inputs = {
        'parent': parent / 'conditional_2d_manifest.json', 'rate': rate / 'mua_rate_transfer_manifest.json',
        'protocol': ROOT / 'docs/2d_learned_assembly_protocol.md', 'script': Path(__file__),
        'fitter': ROOT / 'src/hipporeplayimm/learned_assembly_prediction.py',
        'initialization_addendum': ROOT / 'docs/2d_learned_assembly_initialization_addendum.md',
    }
    for item in items:
        tag = item['tag']
        for name in (f'{tag}_cache.npz', f'{tag}_selection.csv'):
            inputs[name] = checked_file(parent, pm, name)
        for name in (f'{tag}_original.csv.gz', f'{tag}_split_contrasts.csv.gz'):
            inputs[name] = checked_file(rate, rm, name)
    import hmmlearn
    library = Path(hmmlearn.__file__).resolve().parent
    for name in ('hmm.py', 'base.py', '_emissions.py'):
        inputs[f'hmmlearn_{name}'] = library / name
    inputs['hmmlearn_compiled_kernel'] = next(library.glob('_hmmc*.so'))
    if args.reuse_dir:
        inputs['reuse_manifest'] = Path(args.reuse_dir).resolve() / 'learned_assembly_manifest.json'
    manifest = build_script_provenance(input_paths=inputs)
    manifest.update(status='running', scope='integration_pilot' if args.pilot else 'all_9225',
                    hmmlearn_version=hmmlearn.__version__, n_states=STATES, k_shuffles=K,
                    selected_sessions=items, n_events=sum(i['events'] for i in items))
    mpath = out / 'learned_assembly_manifest.json'
    mpath.write_text(json.dumps(manifest, indent=2) + '\n')
    completed = reuse_completed(Path(args.reuse_dir).resolve(), out, items, inputs) if args.reuse_dir else []
    manifest['reused_fold_fits'] = len(completed)
    done = {(r['tag'], r['fold'], r['n_states']) for r in completed}
    tick = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(task, (i, fold, states, str(parent), str(out))) for i in items for fold in range(5) for states in STATES if (i['tag'], fold, states) not in done]
            for future in as_completed(jobs):
                completed.append(future.result())
                manifest.update(completed=completed)
                mpath.write_text(json.dumps(manifest, indent=2) + '\n')
        if len(completed) != len(items) * 5 * len(STATES):
            raise ValueError('missing fold fit')
        if sum(c['rows'] for c in completed) != manifest['n_events'] * 5 * len(STATES) * (K + 1):
            raise ValueError('missing scored events')
        events = []
        for item in items:
            tag = item['tag']
            scores = pd.concat([pd.read_csv(out / f"{x['stem']}_scores.csv.gz") for x in completed if x['tag'] == tag], ignore_index=True)
            split, event = contrasts(scores, pd.read_csv(rate / f'{tag}_original.csv.gz'), pd.read_csv(rate / f'{tag}_split_contrasts.csv.gz'))
            split.to_csv(out / f'{tag}_split_contrasts.csv.gz', index=False)
            event.to_csv(out / f'{tag}_event_contrasts.csv', index=False)
            events.append(event)
        events = pd.concat(events, ignore_index=True)
        events.to_csv(out / 'learned_assembly_event_contrasts.csv', index=False)
        fits = pd.DataFrame([{k: v for k, v in x.items() if k not in ('output_sha256', 'test_ids', 'calibration_ids', 'excluded_ids')} for x in completed])
        fits.to_csv(out / 'learned_assembly_fit_summary.csv', index=False)
        if not args.pilot:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                stats = list(pool.map(aggregate, [g for _, g in events.groupby(['dataset', 'contrast'])]))
            for index, name in enumerate(('summary', 'by_animal', 'by_session')):
                pd.concat([r[index] for r in stats], ignore_index=True).to_csv(out / f'learned_assembly_{name}.csv', index=False)
        else:
            events.groupby(['dataset', 'contrast'], as_index=False)[['delta', 'delta_per_heldout_spike']].mean().to_csv(out / 'learned_assembly_pilot_summary.csv', index=False)
        pd.DataFrame([
            {'gate': 'scoring_complete', 'passed': True},
            {'gate': 'all_fits_converged', 'passed': bool(fits.fit_converged.all())},
            {'gate': 'independent_reconstruction_required', 'passed': False},
            {'gate': 'biological_claim', 'passed': False},
        ]).to_csv(out / 'learned_assembly_gates.csv', index=False)
        manifest.update(status='complete', runtime_s=time.monotonic() - tick,
                        all_fits_converged=bool(fits.fit_converged.all()))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['completed'] = completed
        manifest['output_sha256'] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != mpath}
        mpath.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('status', 'scope', 'n_events', 'all_fits_converged', 'runtime_s')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', required=True)
    parser.add_argument('--rate-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--reuse-dir', help='Hash-checked completed folds from a matching terminal failed run')
    run(parser.parse_args())
