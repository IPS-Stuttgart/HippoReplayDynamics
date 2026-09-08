#!/usr/bin/env python3
"""Independently reconstruct learned-HMM fits and cross-cell predictions."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from _provenance import build_script_provenance, file_sha256
from audit_2d_count_conditioned_prediction import IDENTITY, KEYS
from audit_2d_learned_assembly import load_fit
from audit_2d_predictive_order_map import K, shuffled_indices


def likelihood(counts, probabilities):
    probabilities = probabilities / probabilities.sum(axis=0, keepdims=True)
    counts = np.asarray(counts)
    return (counts[:, :, None] * np.log(probabilities)[None, :, :]).sum(axis=1) + (
        gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)
    )[:, None]


def forward_backward(ll, initial, transition):
    """NumPy recurrence, separate from hmmlearn's compiled implementation."""
    maximum = np.max(ll, axis=1)
    emissions = np.exp(ll - maximum[:, None])
    forward = np.zeros_like(ll)
    backward = np.ones_like(ll)
    z = 0.0
    for t, emission in enumerate(emissions):
        prediction = initial if t == 0 else forward[t-1] @ transition
        updated = prediction * emission
        normalizer = updated.sum()
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError('invalid forward normalizer')
        z += np.log(normalizer) + maximum[t]
        forward[t] = updated / normalizer
    for t in range(len(ll)-2, -1, -1):
        backward[t] = transition @ (emissions[t+1] * backward[t+1])
        backward[t] /= backward[t].sum()
    q = forward * backward
    q /= q.sum(axis=1, keepdims=True)
    logq = np.full_like(q, -np.inf)
    np.log(q, out=logq, where=q > 0)
    return float(z), logq


def reference_prediction(counts, train, held, fit):
    ll = likelihood(counts[:, train], fit.probabilities[train])
    _, posterior = forward_backward(ll, fit.initial, fit.transition)
    independent = ll + np.log(fit.occupancy)[None, :]
    independent -= logsumexp(independent, axis=1, keepdims=True)
    hl = likelihood(counts[:, held], fit.probabilities[held])
    return {
        'learned_hmm': float(logsumexp(posterior + hl, axis=1).sum()),
        'same_emissions_iid': float(logsumexp(independent + hl, axis=1).sum()),
        'nonspatial_global': float(likelihood(counts[:, held], fit.global_probability[held, None]).sum()),
    }


def same(actual, expected, tolerance=1e-7):
    if not np.allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True):
        raise ValueError('independent reconstruction differs')


def check_task(job):
    result, parent_name, out_name = job
    parent, out = Path(parent_name), Path(out_name)
    tag, stem = result['tag'], result['stem']
    fit = load_fit(out / f'{stem}_fit.npz')
    scores = pd.read_csv(out / f'{stem}_scores.csv.gz')
    with np.load(parent / f'{tag}_cache.npz') as z:
        cache = {k: z[k] for k in z.files}
    selected = pd.read_csv(parent / f'{tag}_selection.csv').sort_values(['start_s', 'event_id'])
    test = selected.iloc[np.array_split(np.arange(len(selected)), 5)[result['fold']]]
    calibration = selected[~selected.event_id.isin(test.event_id)]
    keep = np.array([all(row.end_s + 1 <= target.start_s or row.start_s >= target.end_s + 1 for target in test.itertuples()) for row in calibration.itertuples()])
    if set(test.event_id) != set(result['test_ids']) or set(calibration[keep].event_id) != set(result['calibration_ids']) or set(calibration[~keep].event_id) != set(result['excluded_ids']):
        raise ValueError('test/calibration/guard separation differs')
    arrays = [cache[f'counts_{eid}'] for eid in result['calibration_ids']]
    informative = sum(np.count_nonzero(x.sum(axis=1)) for x in arrays)
    if bool(fit.initialization_with_replacement) != (informative < result['n_states']):
        raise ValueError('initialization support reporting differs')
    pooled = np.concatenate(arrays).sum(axis=0) + 100 / len(cache['unit_ids'])
    reference = pooled / pooled.sum()
    same(fit.global_probability, reference, 1e-12)
    if not all(np.isfinite(x).all() and (x > 0).all() for x in (fit.probabilities, fit.initial, fit.transition, fit.occupancy)):
        raise ValueError('nonpositive fit probabilities')
    same(fit.probabilities.sum(axis=0), 1, 1e-12)
    same(fit.transition.sum(axis=1), 1, 1e-12)
    same(fit.initial.sum(), 1, 1e-12)
    weights, log_likelihood = np.zeros(len(fit.initial)), 0.0
    for x in arrays:
        z, posterior = forward_backward(likelihood(x, fit.probabilities), fit.initial, fit.transition)
        log_likelihood += z
        weights += np.exp(posterior).sum(axis=0)
    weights += 1 / len(weights)
    same(fit.occupancy, weights / weights.sum(), 1e-8)
    objective = log_likelihood + (np.log(fit.initial).sum() + np.log(fit.transition).sum()) / len(fit.initial) + 10 * (reference[:, None] * np.log(fit.probabilities)).sum()
    same(objective, fit.objective_trace[-1], 1e-6)
    same(objective, max(fit.restart_objectives), 1e-6)
    if np.argmax(fit.restart_objectives) != fit.restart or fit.n_calibration_bins != sum(len(a) for a in arrays) or fit.n_calibration_events != len(arrays):
        raise ValueError('fit provenance differs')
    converged = len(fit.objective_trace) >= 3 and 0 <= fit.objective_trace[-2] - fit.objective_trace[-3] < 1e-5 * fit.n_calibration_bins
    if converged != fit.converged or bool(result['fit_converged']) != converged:
        raise ValueError('convergence reporting differs')
    if (np.diff(fit.objective_trace) < -1e-7 * (1 + np.abs(fit.objective_trace[:-1]))).any():
        raise ValueError('EM objective decreased')
    keys = ['event_id', 'split', 'shuffle']
    expected = {(int(eid), split, k) for eid in test.event_id for split in range(5) for k in range(-1, K)}
    if scores.duplicated(keys).any() or set(scores[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('score coverage differs')
    if not scores.heldout_used_for_inference.eq(False).all():
        raise ValueError('held-out target cells used for inference')
    names = ['score_learned_hmm', 'score_same_emissions_iid', 'score_nonspatial_global']
    if not np.isfinite(scores[names]).all().all() or not scores[names].le(1e-8).all().all():
        raise ValueError('invalid probability scores')
    count, max_error = 0, 0.0
    identity = tuple(result[k] for k in IDENTITY)
    for (eid, split), g in scores.groupby(['event_id', 'split']):
        x = cache[f'counts_{eid}']
        tr, he = cache[f'train_{split}'], cache[f'held_{split}']
        if sorted([*tr, *he]) != list(range(x.shape[1])):
            raise ValueError('invalid neural partition')
        same(g.n_heldout_spikes, x[:, he].sum(), 0)
        same(g.n_train_spikes, x[:, tr].sum(), 0)
        same(g.n_time_bins, len(x), 0)
        for row in g[g.shuffle.isin([-1, 0, K-1])].itertuples(index=False):
            order = np.arange(len(x)) if row.shuffle == -1 else shuffled_indices(identity, int(eid), len(x), row.shuffle)
            values = reference_prediction(x[order], tr, he, fit)
            for name, value in values.items():
                actual = getattr(row, f'score_{name}')
                same(actual, value)
                max_error = max(max_error, abs(actual - value))
                count += 1
        for name in names[1:]:
            same(g[name], g.loc[g.shuffle.eq(-1), name].iloc[0])
    return {k: result[k] for k in ('tag', 'fold', 'n_states')} | {
        'status': 'pass', 'scores_checked': count, 'rows_validated': len(scores),
        'calibration_sequences_reconstructed': len(arrays), 'max_score_error': max_error,
    }


def check_contrasts(result_items, out, rate):
    checked = 0
    for tag in sorted({r['tag'] for r in result_items}):
        score = pd.concat([pd.read_csv(out / f"{r['stem']}_scores.csv.gz") for r in result_items if r['tag'] == tag], ignore_index=True)
        original = score[score.shuffle.eq(-1)].set_index(KEYS + ['n_states'])
        null = score[score.shuffle.ge(0)].groupby(KEYS + ['n_states']).score_learned_hmm.mean()
        source = pd.read_csv(rate / f'{tag}_original.csv.gz')
        source = source[source.alpha.eq(100) & source['map'].eq('real')].set_index(KEYS)
        temporal = pd.read_csv(rate / f'{tag}_split_contrasts.csv.gz')
        temporal = temporal[temporal.contrast.eq('alpha100__first_order_imm__real_order_advantage')].set_index(KEYS)
        delta = pd.read_csv(out / f'{tag}_split_contrasts.csv.gz')
        expected_parts = []
        for n_states in (20, 50, 100):
            neural = original.xs(n_states, level='n_states').sort_index()
            baseline = source.reindex(neural.index)
            order_gain = neural.score_learned_hmm - null.xs(n_states, level='n_states').reindex(neural.index)
            expected = {
                'spatial_imm_minus_learned_hmm': baseline.score_first_order_imm - neural.score_learned_hmm,
                'spatial_iid_minus_learned_hmm': baseline.score_iid_position - neural.score_learned_hmm,
                'learned_hmm_minus_nonspatial_global': neural.score_learned_hmm - neural.score_nonspatial_global,
                'learned_hmm_minus_run_shrinkage_global': neural.score_learned_hmm - baseline.score_event_global,
                'learned_hmm_minus_same_emissions_iid': neural.score_learned_hmm - neural.score_same_emissions_iid,
                'learned_hmm_order_advantage': order_gain,
                'spatial_minus_assembly_order_advantage': temporal.delta.reindex(neural.index) - order_gain,
            }
            for axis, difference in expected.items():
                if not np.isfinite(difference).all():
                    raise ValueError('incomplete paired source')
                part = difference.rename('delta').to_frame()
                part['delta_per_heldout_spike'] = difference / neural.n_heldout_spikes.replace(0, np.nan)
                expected_parts.append(part.reset_index().assign(contrast=f'k{n_states}__{axis}'))
        index = KEYS + ['contrast']
        expected_splits = pd.concat(expected_parts).set_index(index).sort_index()
        actual_splits = delta.set_index(index).sort_index()
        if not expected_splits.index.equals(actual_splits.index):
            raise ValueError('split contrast coverage differs')
        same(actual_splits[expected_splits.columns], expected_splits)
        checked += len(expected_splits)
        expected = delta.groupby(IDENTITY + ['event_id', 'contrast'])[['delta', 'delta_per_heldout_spike']].median().sort_index()
        actual = pd.read_csv(out / f'{tag}_event_contrasts.csv').set_index(IDENTITY + ['event_id', 'contrast']).sort_index()
        if not expected.index.equals(actual.index):
            raise ValueError('event aggregation coverage differs')
        same(actual[expected.columns], expected)
    return checked


def run(args):
    out, audit = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    mpath = out / 'learned_assembly_manifest.json'
    manifest = json.loads(mpath.read_text())
    if manifest['status'] != 'complete' or not manifest['completed']:
        raise ValueError('complete nonempty scored experiment required')
    if audit.exists() and any(audit.iterdir()):
        raise ValueError('refusing overwrite')
    audit.mkdir(parents=True, exist_ok=True)
    for key, path in manifest['input_file_paths'].items():
        if file_sha256(path) != manifest['input_file_sha256'][key]:
            raise ValueError(f'changed input: {path}')
    for name, expected in manifest['output_sha256'].items():
        if file_sha256(out / name) != expected:
            raise ValueError(f'changed score artifact: {name}')
    parent = Path(manifest['input_file_paths']['parent']).parent
    rate = Path(manifest['input_file_paths']['rate']).parent
    report = build_script_provenance(input_paths={'scoring_manifest': mpath, 'verifier': Path(__file__)})
    report.update(status='running', scope='all calibration objectives; all original and two null predictions per event/split/capacity; all coverage/invariance rows; all split differences, normalizations and event medians')
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(check_task, (r, str(parent), str(out))) for r in manifest['completed']]):
                completed.append(future.result())
        contrasts_checked = check_contrasts(manifest['completed'], out, rate)
        report.update(status='pass', contrasts_checked=contrasts_checked,
                      scores_checked=sum(r['scores_checked'] for r in completed),
                      rows_validated=sum(r['rows_validated'] for r in completed),
                      max_score_error=max(r['max_score_error'] for r in completed))
    except BaseException as exc:
        report.update(status='fail', error=repr(exc))
        raise
    finally:
        report['results'] = completed
        (audit / 'learned_assembly_reconstruction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'scores_checked', 'rows_validated', 'max_score_error')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args())
