#!/usr/bin/env python3
"""Audit bilateral hc-11 sources, RUN decoders and opportunities, never replay content."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import _hc11_bilateral_feasibility as core
from _provenance import build_script_provenance, file_sha256

PREFIX = 'hc11_bilateral_'
SOURCE_SUFFIXES = ('_sessInfo.mat', '.spikes.cellinfo.mat', '.position.behavior.mat', '.SleepState.states.mat', '.sessionInfo.mat', '.xml', '.eeg')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def table(output, name, rows, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output / (PREFIX + name + '.csv'), index=False)
    return frame


def find_one(root, session, suffix):
    matches = sorted(root.rglob(session + suffix))
    if len(matches) > 1:
        raise ValueError(f'ambiguous source {session}{suffix}: {len(matches)} files')
    return matches[0] if matches else None


def sources(root, session):
    return {suffix: find_one(root, session, suffix) for suffix in SOURCE_SUFFIXES}


def identity(args):
    protocol = json.loads(args.protocol.read_text())
    if args.seed != protocol['seed']:
        raise ValueError('seed differs from frozen protocol')
    return protocol, {'dataset_root': str(args.dataset_root.resolve()), 'protocol_sha256': file_sha256(args.protocol),
                      'code_commit': build_script_provenance(cwd=ROOT)['code_commit'], 'seed': args.seed}


def checkpoint(args, stage, key, inputs):
    path = args.output_dir / f'{stage}_checkpoint.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['identity'] != key or previous['inputs'] != inputs:
            raise ValueError('checkpoint identity changed; use a new output directory')
        for name, digest in previous['outputs'].items():
            if file_sha256(args.output_dir / name) != digest:
                raise ValueError(f'checkpoint output changed: {name}')
        return previous
    return None


def save_checkpoint(args, stage, key, inputs, before):
    outputs = {str(p.relative_to(args.output_dir)): file_sha256(p) for p in args.output_dir.rglob('*')
               if p.is_file() and str(p.relative_to(args.output_dir)) not in before and not p.name.endswith('_checkpoint.json')}
    write_json(args.output_dir / f'{stage}_checkpoint.json', {'identity': key, 'inputs': inputs, 'outputs': outputs,
               'completed_at_utc': datetime.now(UTC).isoformat()})


def inventory(args, protocol):
    files, recordings, units, channels, states = [], [], [], [], []
    review_paths = []
    for session in protocol['sessions']:
        row = {'session': session, 'animal': None, 'status': 'unresolved', 'source_verified': False,
               'converted_records_present': False, 'original_session_present': False, 'eeg_present': False,
               'original_units': None, 'converted_units': None, 'timing_verified_units': None,
               'anatomy_verified_units': None, 'xml_channels': None, 'session_info_channels': None,
               'channel_count_conflict': None, 'awake_rest_verified': False, 'reason': ''}
        try:
            paths = sources(args.dataset_root, session)
            for kind, path in paths.items():
                if path:
                    files.append({'session': session, 'kind': kind, 'path': str(path), 'bytes': path.stat().st_size,
                                  'sha256': file_sha256(path), 'status': 'present'})
                else:
                    files.append({'session': session, 'kind': kind, 'path': None, 'bytes': None, 'sha256': None, 'status': 'missing'})
            row.update(original_session_present=paths['_sessInfo.mat'] is not None, eeg_present=paths['.eeg'] is not None,
                       converted_records_present=all(paths[v] is not None for v in ('.spikes.cellinfo.mat', '.position.behavior.mat', '.SleepState.states.mat')))
            src = core.load_structure(paths['_sessInfo.mat'], 'sessInfo') if paths['_sessInfo.mat'] else None
            converted = core.load_structure(paths['.spikes.cellinfo.mat'], 'spikes') if paths['.spikes.cellinfo.mat'] else None
            if converted:
                row['converted_units'] = len(np.atleast_1d(converted['UID']))
            info = core.load_structure(paths['.sessionInfo.mat'], 'sessionInfo') if paths['.sessionInfo.mat'] else None
            if info:
                row['session_info_channels'] = int(info['nChannels'])
                # This is a metadata animal identifier, not a condition inferred from a filename.
            verification_path = (paths['.spikes.cellinfo.mat'].parent / protocol['source_verification_filename']) if converted else None
            verified = None
            if verification_path and verification_path.exists():
                review_paths.append(verification_path)
                files.append({'session': session, 'kind': 'source_verification', 'path': str(verification_path),
                              'bytes': verification_path.stat().st_size, 'sha256': file_sha256(verification_path), 'status': 'present'})
                verified = core.read_verified_source(verification_path, file_sha256)
                for refs in verified['evidence'].values():
                    for ref in refs:
                        target = verification_path.parent / ref['path']
                        files.append({'session': session, 'kind': 'original_verification_evidence', 'path': str(target),
                                      'bytes': target.stat().st_size, 'sha256': file_sha256(target), 'status': 'present'})
                row['animal'] = verified['animal']
            missing = [kind for kind, path in paths.items() if path is None]
            reasons = [f'missing:{kind}' for kind in missing]
            if not verified:
                reasons.append('original_channel_order_topology_awake_rest_signoff_unavailable')
            if src:
                row['original_units'] = len(np.unique(src['Spikes']['SpikeIDs']))
                for label in ('PREEpoch', 'MazeEpoch', 'POSTEpoch', 'Wake', 'Drowsy', 'NREM', 'Intermediate', 'REM'):
                    domain = core.intervals(src['Epochs'][label])
                    states.append({'session': session, 'source': '_sessInfo.mat', 'label': label, 'intervals': len(domain),
                                   'duration_s': float(np.diff(domain, axis=1).sum()), 'awake_rest_verified': False,
                                   'reason': 'Wake denotes active waking; Drowsy includes light sleep; neither silently becomes awake rest'})
                if converted:
                    cross = core.original_crosswalk(src['Spikes'], converted, verified.get('spike_groups') if verified else None)
                    units.extend({'session': session, **r} for r in cross)
                    row['timing_verified_units'] = sum(r['timing_verified'] for r in cross)
                    row['anatomy_verified_units'] = sum(r['mapping_verified'] for r in cross)
                    if not all(r['timing_verified'] for r in cross):
                        reasons.append('unresolved_original_unit_identity')
            elif converted:
                units.extend({'session': session, 'unit_id': int(uid), 'original_cluster_id': None,
                              'spike_group': None, 'cluster_within_group': None, 'shank': None, 'hemisphere': None,
                              'ca1_pyramidal': False, 'timing_verified': False, 'mapping_verified': False,
                              'reason': 'original_cluster_records_unavailable'} for uid in np.atleast_1d(converted['UID']))
            if paths['.xml']:
                n, fs, spike_fs, groups = core.xml_layout(paths['.xml'])
                row['xml_channels'] = n
                conflict = info is not None and int(info['nChannels']) != n
                row['channel_count_conflict'] = bool(conflict)
                if conflict and not verified:
                    reasons.append('XML_vs_sessionInfo_channel_count_conflict_no_override')
                for group, ids in groups.items():
                    for channel in ids:
                        mapping = (verified or {}).get('spike_groups', {}).get(str(group), {})
                        channels.append({'session': session, 'spike_group': group, 'channel_id': channel,
                                         'shank': mapping.get('shank'), 'hemisphere': mapping.get('hemisphere'),
                                         'xml_n_channels': n, 'session_info_n_channels': row['session_info_channels'],
                                         'lfp_sampling_rate': fs, 'spike_sampling_rate': spike_fs,
                                         'mapping_verified': bool(mapping), 'reason': 'verified' if mapping else 'original_channel_order_not_verified'})
                if src and paths['.eeg']:
                    try:
                        core.verify_eeg_layout(paths['.eeg'], verified['n_channels'] if verified else n, fs, float(src['Epochs']['sessDuration']))
                    except ValueError as exc:
                        reasons.append(str(exc))
            if verified:
                core.intervals(verified['awake_rest_intervals_s'])
                row['awake_rest_verified'] = True
            row['reason'] = ';'.join(reasons)
            row['source_verified'] = bool(verified) and not reasons
            row['status'] = 'verified' if row['source_verified'] else 'unresolved'
        except (OSError, ValueError, KeyError, TypeError, NotImplementedError) as exc:
            row['reason'] = f'source_read_or_verification_failure:{exc}'
        # Dataset animal prefix is an inventory label only; never anatomical/condition evidence.
        if row['animal'] is None:
            row['animal'] = session.split('_')[0]
        recordings.append(row)
        print(json.dumps({'stage': 'inventory', **row}), flush=True)
    table(args.output_dir, 'source_inventory', files)
    table(args.output_dir, 'recording_inventory', recordings)
    table(args.output_dir, 'unit_crosswalk', units, columns=['session', 'unit_id', 'original_cluster_id', 'spike_group',
          'cluster_within_group', 'shank', 'hemisphere', 'ca1_pyramidal', 'timing_verified', 'mapping_verified', 'reason'])
    table(args.output_dir, 'channel_crosswalk', channels, columns=['session', 'spike_group', 'channel_id', 'shank', 'hemisphere', 'xml_n_channels', 'session_info_n_channels', 'lfp_sampling_rate', 'spike_sampling_rate', 'mapping_verified', 'reason'])
    table(args.output_dir, 'state_inventory', states, columns=['session', 'source', 'label', 'intervals', 'duration_s', 'awake_rest_verified', 'reason'])
    return review_paths


def verified_inputs(args, session, protocol):
    paths = sources(args.dataset_root, session)
    if any(path is None for path in paths.values()):
        raise ValueError('missing original inputs; no converted-only fallback')
    review = core.read_verified_source(paths['.spikes.cellinfo.mat'].parent / protocol['source_verification_filename'], file_sha256)
    original = core.load_structure(paths['_sessInfo.mat'], 'sessInfo')
    converted = core.load_structure(paths['.spikes.cellinfo.mat'], 'spikes')
    cross = core.original_crosswalk(original['Spikes'], converted, review['spike_groups'])
    if not all(r['mapping_verified'] and r['timing_verified'] for r in cross):
        raise ValueError('unresolved unit identities/anatomy')
    position = original['Position']
    # Scaling and origin must be documented; no range-based heuristics.
    x = np.asarray(position['OneDLocation'], float).ravel() * review['coordinate_to_cm'] - review['coordinate_origin_cm']
    t = np.asarray(position['TimeStamps'], float).ravel()
    n, fs, _, _ = core.xml_layout(paths['.xml'])
    if n != review['n_channels'] and not review.get('channel_count_discrepancy_resolution'):
        raise ValueError('channel-count discrepancy lacks original evidence')
    core.verify_eeg_layout(paths['.eeg'], review['n_channels'], fs, float(original['Epochs']['sessDuration']))
    return paths, review, original, converted, cross, t, x, fs


def run_qc(args, protocol):
    metrics, traversal_rows, status, disagreement = [], [], [], []
    for session in protocol['sessions']:
        try:
            _paths, review, original, converted, cross, t, x, _ = verified_inputs(args, session, protocol)
            train, validation, cutoff = core.chronological_split(review['traversals'], review['running_directions'], protocol['run']['train_fraction'])
            if not all(core.contains(v['start_s'], v['end_s'], original['Epochs']['MazeEpoch']) for v in train + validation):
                raise ValueError('traversal extends beyond original MAZE epoch')
            session_dir = args.output_dir / session
            session_dir.mkdir(exist_ok=True)
            caches = {}
            for side in ('left', 'right'):
                selected_units = [i for i, r in enumerate(cross) if r['hemisphere'] == side and r['ca1_pyramidal']]
                spikes = [np.asarray(converted['times'][i], float).ravel() for i in selected_units]
                maps = core.fit_maps(t, x, spikes, train, review['track_length_cm'], review['topology'], protocol['run'])
                unit_keep = np.flatnonzero(maps['selected'])
                if len(unit_keep) < protocol['run']['min_units_per_hemisphere']:
                    raise ValueError(f'{side}: fewer than 10 qualifying verified CA1 pyramidal units')
                rates = maps['rates'][unit_keep]
                chosen_spikes = [spikes[i] for i in unit_keep]
                reduced = {**maps, 'rates': rates}
                rows, cache = core.evaluate_run(t, x, chosen_spikes, validation, reduced, review['track_length_cm'], review['topology'], protocol['run'])
                caches[side] = cache
                traversal_rows.extend({'session': session, 'animal': review['animal'], 'hemisphere': side, **r} for r in rows)
                for direction in review['running_directions']:
                    subset = [r for r in rows if r['direction'] == direction]
                    available = all(r['coverage'] is not None and r['median_error_cm'] is not None for r in subset)
                    error = float(np.median([r['median_error_cm'] for r in subset])) if available else None
                    coverage = float(np.mean([r['coverage'] for r in subset])) if available else None
                    passed = available and error <= protocol['run']['max_median_error_cm'] and coverage >= protocol['run']['min_support_coverage']
                    metrics.append({'session': session, 'animal': review['animal'], 'hemisphere': side, 'direction': direction,
                                    'units': len(unit_keep), 'training_cutoff_s': cutoff, 'median_error_cm': error,
                                    'support_coverage': coverage, 'passed': passed})
                save = {k: v for k, v in maps.items()}
                save['qualified_indices'] = np.asarray(selected_units)[unit_keep]
                save['training_cutoff_s'] = np.asarray(cutoff)
                for i, c in enumerate(cache):
                    save.update({f'validation_{i}_{k}': v for k, v in c.items()})
                np.savez_compressed(session_dir / f'{side}_run_cache.npz', **save)
            for i, (left, right) in enumerate(zip(caches['left'], caches['right'])):
                common = left['supported'] & right['supported']
                disagreement.append({'session': session, 'traversal_id': validation[i]['id'], 'supported_bins_both': int(common.sum()),
                                     'median_bilateral_disagreement_cm': float(np.median(core.distance(left['predicted'][common], right['predicted'][common], review['track_length_cm'], review['topology']))) if common.any() else None,
                                     'meaning': 'heldout_RUN_decoder_noise_not_replay_content'})
            qualified = all(r['passed'] for r in metrics if r['session'] == session)
            status.append({'session': session, 'status': 'qualified' if qualified else 'decoder_failed', 'reason': ''})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            status.append({'session': session, 'status': 'not_measured', 'reason': str(exc)})
    table(args.output_dir, 'run_validation', metrics, columns=['session', 'animal', 'hemisphere', 'direction', 'units', 'training_cutoff_s', 'median_error_cm', 'support_coverage', 'passed'])
    table(args.output_dir, 'run_traversal_validation', traversal_rows, columns=['session', 'animal', 'hemisphere', 'traversal_id', 'direction', 'eligible_bins', 'supported_bins', 'coverage', 'median_error_cm', 'mean_entropy', 'zero_spike_bins'])
    table(args.output_dir, 'run_status', status)
    table(args.output_dir, 'run_bilateral_noise', disagreement, columns=['session', 'traversal_id', 'supported_bins_both', 'median_bilateral_disagreement_cm', 'meaning'])


def lfp_segments(eeg, channel, fs, domains):
    for a, b in core.intervals(domains):
        lo, hi = int(np.ceil(a * fs)), int(np.floor(b * fs))
        if lo < 0 or hi > len(eeg) or hi - lo < 100:
            raise ValueError('state interval outside common EEG clock or too short')
        yield lo / fs, np.asarray(eeg[lo:hi, channel], float)


def opportunities(args, protocol):
    counts, windows, candidates, channel_selection = [], [], [], []
    run_status = pd.read_csv(args.output_dir / (PREFIX + 'run_status.csv')).set_index('session')
    for session in protocol['sessions']:
        animal = session.split('_')[0]
        if run_status.loc[session, 'status'] != 'qualified':
            counts.extend({'session': session, 'animal': animal, 'state': state, 'decoder_qualified': False,
                           'supported_windows': None, 'coincident_windows': None, 'reason': 'RUN_or_source_not_qualified_not_measured'} for state in protocol['states'])
            continue
        paths, review, original, converted, _, _, _, fs = verified_inputs(args, session, protocol)
        states = {'awake_rest': core.intervals(review['awake_rest_intervals_s']),
                  'POST_NREM': core.intersection(original['Epochs']['POSTEpoch'], original['Epochs']['NREM'])}
        if len(core.intersection(states['awake_rest'], original['Epochs']['NREM'])):
            raise ValueError('awake-rest annotation conflicts with original NREM')
        pre = core.intersection(original['Epochs']['PREEpoch'], original['Epochs']['NREM'])
        if not len(pre):
            raise ValueError('missing PRE-NREM baseline')
        eeg = np.memmap(paths['.eeg'], mode='r', dtype='<i2').reshape(-1, review['n_channels'])
        detected, population, cutoffs = {}, {}, []
        for side in ('left', 'right'):
            cache = np.load(args.output_dir / session / f'{side}_run_cache.npz', allow_pickle=False)
            population[side] = [np.asarray(converted['times'][int(i)], float).ravel() for i in cache['qualified_indices']]
            cutoffs.append(float(cache['training_cutoff_s']))
            rows = []
            for shank, channel_ids in sorted(review['pyramidal_layer_channel_ids'][side].items()):
                choices = []
                for channel_id in channel_ids:
                    index = core.channel_rows(review['eeg_channel_ids'], [channel_id])[0]
                    envelopes = [core.ripple_envelope(lfp, fs, protocol['ripple']) for _, lfp in lfp_segments(eeg, index, fs, pre)]
                    total, n = sum(float(v.sum()) for v in envelopes), sum(len(v) for v in envelopes)
                    mean = total / n
                    sd = float(np.sqrt(sum(float(np.square(v - mean).sum()) for v in envelopes) / n))
                    peaks = [r['peak_z'] for (start, _), e in zip(lfp_segments(eeg, index, fs, pre), envelopes)
                             for r in core.envelope_events(e, fs, start, mean, sd, str(channel_id), protocol['ripple'])]
                    choices.append((float(np.mean(peaks)) if peaks else -np.inf, channel_id, int(index), mean, sd))
                if not choices or not any(np.isfinite(v[0]) for v in choices):
                    raise ValueError('no PRE-NREM eligible channel')
                # Identity tie-break; later states never contribute to channel selection.
                _, channel_id, index, mean, sd = min(choices, key=lambda v: (-v[0], v[1]))
                channel_selection.append({'session': session, 'hemisphere': side, 'shank': shank, 'channel_id': channel_id,
                                          'row_index': index, 'baseline_mean': mean, 'baseline_sd': sd, 'selection_epoch': 'PRE_NREM_only'})
                for state, domain in states.items():
                    for segment, (start, lfp) in enumerate(lfp_segments(eeg, index, fs, domain)):
                        envelope = core.ripple_envelope(lfp, fs, protocol['ripple'])
                        part = core.envelope_events(envelope, fs, start, mean, sd, f'{side}:{shank}:{state}:{segment}', protocol['ripple'])
                        # Clip/saturation is retained as an exclusion, not a clean candidate.
                        for r in part:
                            r['channel_id'] = channel_id
                            lo, hi = int((r['start_s'] - start) * fs), int((r['end_s'] - start) * fs)
                            r['artifact'] = bool(np.any(np.abs(lfp[lo:hi]) >= 32767))
                        rows.extend(part)
            detected[side] = core.merge_channels(rows)
            candidates.extend({'session': session, 'hemisphere': side, **r, 'parents': json.dumps(r['parents']),
                               'parent_channels': json.dumps(r['parent_channels'])} for r in detected[side])
        matched = core.match_ripples(detected['left'], detected['right'], protocol['ripple']['peak_separation_s'], protocol['ripple']['min_overlap_s'])
        session_windows = [core.opportunity(a, b, states, population['left'], population['right'], protocol['ripple'], max(cutoffs)) for a, b in matched]
        windows.extend({'session': session, 'animal': review['animal'], **r} for r in session_windows)
        matched_ids = {r['left_id'] for r in session_windows} | {r['right_id'] for r in session_windows}
        for r in candidates:
            if r['session'] == session:
                r['matched'] = r['id'] in matched_ids
        for state in protocol['states']:
            state_windows = [r for r in session_windows if r['state'] == state]
            counts.append({'session': session, 'animal': review['animal'], 'state': state, 'decoder_qualified': True,
                           'supported_windows': sum(r['supported'] for r in state_windows), 'coincident_windows': len(state_windows), 'reason': ''})
    table(args.output_dir, 'opportunity_counts', counts)
    table(args.output_dir, 'opportunity_inventory', windows, columns=['session', 'animal', 'left_id', 'right_id', 'start_s', 'end_s', 'state', 'supported', 'reason', 'supported_bins_both'])
    table(args.output_dir, 'ripple_inventory', candidates, columns=['session', 'hemisphere', 'id', 'start_s', 'end_s', 'peak_s', 'peak_z', 'artifact', 'parents', 'parent_channels', 'compound', 'matched'])
    table(args.output_dir, 'channel_selection', channel_selection, columns=['session', 'hemisphere', 'shank', 'channel_id', 'row_index', 'baseline_mean', 'baseline_sd', 'selection_epoch'])


def verify(args, protocol):
    rows = []
    inventory_frame = pd.read_csv(args.output_dir / (PREFIX + 'recording_inventory.csv'))
    inventory_session_set = set(inventory_frame['session'])
    rows.append({'check': 'all_eight_recordings_accounted', 'passed': inventory_session_set == set(protocol['sessions']) and len(inventory_frame) == 8})
    paths = pd.read_csv(args.output_dir / (PREFIX + 'source_inventory.csv'))
    unchanged = all(file_sha256(r.path) == r.sha256 for r in paths[paths.status == 'present'].itertuples())
    rows.append({'check': 'original_input_hashes_unchanged', 'passed': unchanged})
    cross = pd.read_csv(args.output_dir / (PREFIX + 'unit_crosswalk.csv'))
    independent_rows = []
    for session in protocol['sessions']:
        p = sources(args.dataset_root, session)
        if p['_sessInfo.mat'] and p['.spikes.cellinfo.mat']:
            # Independent exact per-unit search using original trains, not saved crosswalk rows.
            src = core.load_structure(p['_sessInfo.mat'], 'sessInfo')['Spikes']
            conv = core.load_structure(p['.spikes.cellinfo.mat'], 'spikes')
            ids, ts = np.asarray(src['SpikeIDs']).ravel(), np.asarray(src['SpikeTimes']).ravel()
            local = cross[cross.session == session]
            for r in local.itertuples():
                if r.timing_verified:
                    i = list(np.atleast_1d(conv['UID'])).index(r.unit_id)
                    a = np.asarray(conv['times'][i]).ravel()
                    b = ts[ids == r.original_cluster_id]
                    passed = len(a) == len(b) and bool(np.all(np.abs(a - b) <= 0.000025))
                    independent_rows.append({'session': session, 'unit_id': r.unit_id, 'timing_reproduced': passed})
    rows.append({'check': 'recorded_unit_timing_matches_reproduced', 'passed': bool(independent_rows) and all(r['timing_reproduced'] for r in independent_rows)})
    counts = pd.read_csv(args.output_dir / (PREFIX + 'opportunity_counts.csv'))
    windows = pd.read_csv(args.output_dir / (PREFIX + 'opportunity_inventory.csv'))
    accounting = True
    for r in counts.itertuples():
        if pd.notna(r.supported_windows):
            selected = windows[(windows.session == r.session) & (windows.state == r.state)]
            accounting &= r.supported_windows == selected.supported.sum() and r.coincident_windows == len(selected)
    rows.append({'check': 'opportunity_totals_reconstructed', 'passed': bool(accounting)})
    rows.append({'check': 'one_to_one_assignments', 'passed': not windows.duplicated(['session', 'left_id']).any() and not windows.duplicated(['session', 'right_id']).any()})
    reproduction = []
    for session in protocol['sessions']:
        for side in ('left', 'right'):
            path = args.output_dir / session / f'{side}_run_cache.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as c:
                selected = c['selected']
                rates = c['rates'][selected][:, c['support']]
                for key in c.files:
                    if key.endswith('_counts') and key.startswith('validation_'):
                        base = key[:-len('counts')]
                        logp = c[key] @ np.log(np.maximum(rates, 1e-300)) - 0.02 * np.sum(rates, axis=0)
                        logp -= np.max(logp, axis=1, keepdims=True)
                        p = np.exp(logp)
                        p /= p.sum(axis=1, keepdims=True)
                        reproduction.append({'session': session, 'hemisphere': side, 'validation': base,
                                             'posterior_reproduced': bool(np.allclose(p, c[base + 'posterior'], atol=1e-12, rtol=1e-10))})
    rows.append({'check': 'RUN_posteriors_recomputed', 'passed': bool(reproduction) and all(r['posterior_reproduced'] for r in reproduction)})
    table(args.output_dir, 'independent_unit_verification', independent_rows, columns=['session', 'unit_id', 'timing_reproduced'])
    table(args.output_dir, 'independent_run_verification', reproduction, columns=['session', 'hemisphere', 'validation', 'posterior_reproduced'])
    table(args.output_dir, 'verification', rows)


def report(args, protocol):
    recording = pd.read_csv(args.output_dir / (PREFIX + 'recording_inventory.csv'))
    counts = pd.read_csv(args.output_dir / (PREFIX + 'opportunity_counts.csv'))
    verification = pd.read_csv(args.output_dir / (PREFIX + 'verification.csv'))
    novelty = json.loads((args.protocol.parent / protocol['novelty_review']).read_text())
    # Missing measurements stay null in JSON/CSV and cannot satisfy readiness.
    records = counts.astype(object).where(pd.notna(counts), None).to_dict('records')
    verified_sources = recording[recording.source_verified]
    source_ok = len(verified_sources.animal.unique()) >= protocol['screen']['min_animals_both_states']
    status = core.decision(novelty['decision'], source_ok, bool(verification.passed.all()), records,
                           protocol['screen']['min_animals_both_states'], protocol['screen']['min_supported_windows_per_state_per_animal'])
    gates = [
        {'gate': 'all_eight_recordings_inventoried', 'status': 'pass' if len(recording) == 8 else 'fail'},
        {'gate': 'novelty_source_review_closed', 'status': 'pass' if novelty['decision'] == 'cleared' else 'fail'},
        {'gate': 'at_least_three_source_verified_animals', 'status': 'pass' if source_ok else 'fail'},
        {'gate': 'independent_verification', 'status': 'pass' if verification.passed.all() else 'fail'},
        {'gate': 'both_state_decoder_qualified_opportunity_floor', 'status': 'pass' if status == 'ready_for_calibration' else 'fail'},
        {'gate': 'overall', 'status': status}]
    table(args.output_dir, 'gate_summary', gates)
    animals = []
    for animal in sorted(counts.animal.unique()):
        for state in protocol['states']:
            local = counts[(counts.animal == animal) & (counts.state == state)]
            animals.append({'animal': animal, 'state': state, 'recordings': len(local),
                            'measured_recordings': int(local.supported_windows.notna().sum()),
                            'supported_windows': int(local.supported_windows.sum()) if local.supported_windows.notna().any() else None})
    table(args.output_dir, 'animal_summary', animals)
    text = [f'# hc-11 bilateral content feasibility: {status}', '',
            'This is a source/RUN/opportunity assessment, not a replay-content or biological comparison.',
            'Same memory is not inferred. A sequence-testing opportunity is not validated replay.', '',
            f'Inventoried recordings: {len(recording)}/8; original session bundles: {int(recording.original_session_present.sum())}/8;',
            f'raw EEG: {int(recording.eeg_present.sum())}/8; source-verified recordings: {len(verified_sources)}/8.',
            f'Novelty review: {novelty["decision"]}. {novelty["reason"]}', '',
            '## Limiting denominators',
            'Missing measurements are unavailable, not zero observations. No sleep-only or converted-unit-ID fallback.',
            'Wake in the original release denotes active waking; Drowsy includes light sleep and cannot establish awake rest.',
            'Group 16 and Gatsby channel-count discrepancies remain unqualified until original evidence resolves them.', '']
    text.extend(f'- {r.session}: {r.reason}' for r in recording.itertuples() if r.status != 'verified')
    text += ['', '## Decision',
             'No biological association, state-content contrast, trajectory reconstruction, automatic download or manuscript change was performed.',
             'The 3-animal / 30-window-per-state floors are screening rules, not a power certificate.',
             'A separately frozen calibration study is authorized only by ready_for_calibration.']
    (args.output_dir / (PREFIX + 'go_no_go.md')).write_text('\n'.join(text) + '\n')
    make_inventory_figure(args.output_dir, recording)


def make_inventory_figure(output, recording):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = recording[['converted_records_present', 'original_session_present', 'eeg_present', 'awake_rest_verified', 'source_verified']].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(values, vmin=0, vmax=1, cmap='Greys', aspect='auto')
    ax.set_yticks(range(len(recording)), recording.session)
    ax.set_xticks(range(5), ['Converted core', 'Original session', 'Raw EEG', 'Verified awake rest', 'Source qualified'])
    for i in range(len(values)):
        for j in range(5):
            ax.text(j, i, 'yes' if values[i, j] else 'no', ha='center', va='center', color='white' if values[i, j] else 'black')
    ax.set_title('Availability audit, not a biological or replay-content result')
    fig.tight_layout()
    fig.savefig(output / (PREFIX + 'source_availability.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('inventory', 'run-qc', 'opportunities', 'verify', 'report'))
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        raise ValueError('dataset root does not exist')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol, key = identity(args)
    # Root-wide identity includes small original inputs plus EEG hashes obtained by inventory.
    inputs = {'protocol': file_sha256(args.protocol), 'novelty_review': file_sha256(args.protocol.parent / protocol['novelty_review'])}
    if args.stage != 'inventory':
        inventory_path = args.output_dir / (PREFIX + 'source_inventory.csv')
        saved = pd.read_csv(inventory_path)
        for r in saved[saved.status == 'present'].itertuples():
            if file_sha256(r.path) != r.sha256:
                raise ValueError(f'input changed since inventory: {r.path}')
        inputs['source_inventory'] = file_sha256(inventory_path)
        for stage in ('inventory', 'run-qc', 'opportunities', 'verify'):
            if stage == args.stage:
                break
            dependency = args.output_dir / f'{stage}_checkpoint.json'
            if not dependency.exists():
                raise ValueError(f'missing prerequisite stage: {stage}')
            inputs[f'{stage}_checkpoint'] = file_sha256(dependency)
            upstream = json.loads(dependency.read_text())
            if upstream['identity'] != key:
                raise ValueError('upstream checkpoint code/protocol identity mismatch')
            for name, digest in upstream['outputs'].items():
                if file_sha256(args.output_dir / name) != digest:
                    raise ValueError(f'upstream checkpoint output changed: {name}')
    existing = checkpoint(args, args.stage, key, inputs)
    if existing:
        if args.stage == 'inventory':
            saved = pd.read_csv(args.output_dir / (PREFIX + 'source_inventory.csv'))
            for r in saved[saved.status == 'present'].itertuples():
                if file_sha256(r.path) != r.sha256:
                    raise ValueError('source changed since inventory checkpoint')
        print(json.dumps({'stage': args.stage, 'status': 'hash_verified_already_complete'}), flush=True)
        return
    before = {str(p.relative_to(args.output_dir)) for p in args.output_dir.rglob('*') if p.is_file()}
    if args.stage == 'inventory':
        inventory(args, protocol)
    elif args.stage == 'run-qc':
        run_qc(args, protocol)
    elif args.stage == 'opportunities':
        opportunities(args, protocol)
    elif args.stage == 'verify':
        verify(args, protocol)
    else:
        report(args, protocol)
    provenance = build_script_provenance(input_paths={'protocol': args.protocol, 'novelty_review': args.protocol.parent / protocol['novelty_review']}, cwd=ROOT)
    provenance.update(created_at_utc=datetime.now(UTC).isoformat(), stage=args.stage, seed=args.seed,
                      python_version=platform.python_version(), environment_versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'h5py', 'matplotlib')},
                      forbidden_analysis_performed=False)
    write_json(args.output_dir / f'{args.stage}_manifest.json', provenance)
    save_checkpoint(args, args.stage, key, inputs, before)
    print(json.dumps({'stage': args.stage, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
