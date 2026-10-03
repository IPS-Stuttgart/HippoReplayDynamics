"""Conservative numerical primitives for bilateral hc-11 feasibility, not replay claims."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import logsumexp


def load_structure(path, key):
    """Read v5 or v7.3 numeric MATLAB structs without executing source code."""
    if not h5py.is_hdf5(path):
        return loadmat(path, simplify_cells=True)[key]
    with h5py.File(path) as handle:
        def read(obj):
            if isinstance(obj, h5py.Group):
                return {k: read(v) for k, v in obj.items() if not k.startswith('#')}
            if h5py.check_dtype(ref=obj.dtype) is not None:
                values = [read(handle[r]) for r in obj[()].ravel()]
                return values[0] if len(values) == 1 else values
            value = obj[()].T
            if obj.attrs.get('MATLAB_class') == b'char':
                return ''.join(chr(int(c)) for c in value.ravel())
            return np.asarray(value).squeeze()
        return read(handle[key])


def intervals(value):
    x = np.asarray(value, float).reshape(-1, 2)
    if not np.isfinite(x).all() or np.any(x[:, 1] <= x[:, 0]):
        raise ValueError('nonfinite/reversed intervals')
    if len(x) > 1 and np.any(x[1:, 0] < x[:-1, 1]):
        raise ValueError('unordered/overlapping intervals; no chronology repair')
    return x


def intersection(a, b):
    return np.asarray([(max(x, u), min(y, v)) for x, y in intervals(a)
                       for u, v in intervals(b) if min(y, v) > max(x, u)], float).reshape(-1, 2)


def contains(start, end, domain):
    return bool(any(a <= start and end <= b for a, b in intervals(domain)))


def xml_layout(path):
    root = ElementTree.parse(path).getroot()
    n = int(root.findtext('acquisitionSystem/nChannels'))
    fs = float(root.findtext('fieldPotentials/lfpSamplingRate'))
    spike_fs = float(root.findtext('acquisitionSystem/samplingRate'))
    groups = {}
    for i, group in enumerate(root.findall('spikeDetection/channelGroups/group'), 1):
        groups[i] = [int(v.text) for v in group.findall('channels/channel')]
    return n, fs, spike_fs, groups


def channel_rows(channel_ids, requested):
    ids = np.asarray(channel_ids, int)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate channel identities')
    lookup = {int(v): i for i, v in enumerate(ids)}
    if not set(requested).issubset(lookup):
        raise ValueError('unknown channel ID; not a row index')
    return np.asarray([lookup[int(v)] for v in requested], int)


def verify_eeg_layout(path, n_channels, fs, duration_s):
    size = Path(path).stat().st_size
    if size % (2 * n_channels):
        raise ValueError('EEG not divisible by documented int16 frame size')
    actual = size / (2 * n_channels * fs)
    if abs(actual - duration_s) > max(1 / fs, 0.001):
        raise ValueError('EEG duration and original common clock disagree')
    return actual


def original_crosswalk(original, converted, group_map=None, tolerance_s=0.000025):
    """Match original cluster spike trains; converted UID is never an anatomical ID."""
    times = np.asarray(original['SpikeTimes'], float).ravel()
    ids = np.asarray(original['SpikeIDs']).ravel()
    if len(times) != len(ids) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('invalid original spike clock')
    pyramidal = set(np.asarray(original['PyrIDs'], int).ravel())
    trains = {int(uid): times[ids == uid] for uid in np.unique(ids)}
    counts = {}
    for uid, t in trains.items():
        counts.setdefault(len(t), []).append(uid)
    rows = []
    uids = np.atleast_1d(converted['UID'])
    converted_times = converted['times']
    if len(uids) == 1:
        converted_times = [converted_times]
    claimed = set()
    for index, uid in enumerate(uids):
        t = np.asarray(converted_times[index], float).ravel()
        if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
            raise ValueError('invalid converted spike clock')
        matches = [c for c in counts.get(len(t), []) if np.allclose(t, trains[c], rtol=0, atol=tolerance_s)]
        row = {'unit_id': int(uid), 'original_cluster_id': None, 'spike_group': None,
               'cluster_within_group': None, 'shank': None, 'hemisphere': None,
               'ca1_pyramidal': False, 'timing_verified': False, 'mapping_verified': False,
               'reason': 'missing_or_ambiguous_original_spike_match'}
        if len(matches) == 1 and matches[0] not in claimed:
            c = matches[0]
            claimed.add(c)
            group = c // 100
            row.update(original_cluster_id=c, spike_group=group, cluster_within_group=c % 100,
                       timing_verified=True, reason='original_timing_verified_anatomy_unresolved')
            entry = (group_map or {}).get(str(group))
            if entry and entry.get('hemisphere') in ('left', 'right') and entry.get('region') == 'CA1':
                row.update(shank=entry['shank'], hemisphere=entry['hemisphere'],
                           ca1_pyramidal=c in pyramidal, mapping_verified=True, reason='verified')
        rows.append(row)
    if len(set(uids)) != len(uids):
        raise ValueError('duplicate converted unit identities')
    return rows


def distance(a, b, length_cm, topology):
    d = np.abs(np.asarray(a) - np.asarray(b))
    if topology == 'circular':
        d = np.mod(d, length_cm)
        return np.minimum(d, length_cm - d)
    if topology != 'linear':
        raise ValueError('documented topology must be linear or circular')
    return d


def tracking_at(t, x, query, length_cm, topology, max_gap=0.1):
    t, x, query = np.asarray(t, float), np.asarray(x, float), np.asarray(query, float)
    if len(t) != len(x) or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('invalid tracking chronology')
    if topology == 'circular':
        # Unwrap only within valid contiguous tracking segments, never across gaps.
        breaks = np.r_[0, np.flatnonzero((np.diff(t) > max_gap) | ~np.isfinite(x[:-1]) | ~np.isfinite(x[1:])) + 1, len(t)]
        x = x.copy()
        for a, b in pairwise(breaks):
            x[a:b] = np.unwrap(x[a:b] * 2 * np.pi / length_cm) * length_cm / (2 * np.pi)
    right = np.searchsorted(t, query, side='right')
    valid = (right > 0) & (right < len(t))
    j = np.clip(right, 1, len(t) - 1)
    valid &= (t[j] - t[j - 1] <= max_gap) & np.isfinite(x[j]) & np.isfinite(x[j - 1])
    fraction = (query - t[j - 1]) / (t[j] - t[j - 1])
    out = x[j - 1] + fraction * (x[j] - x[j - 1])
    speed = distance(x[j], x[j - 1], length_cm, topology) / (t[j] - t[j - 1])
    if topology == 'circular':
        out %= length_cm
    out[~valid] = np.nan
    speed[~valid] = np.nan
    return out, speed, valid


def chronological_split(traversals, directions, fraction=0.7):
    domain = intervals([[v['start_s'], v['end_s']] for v in traversals])
    n = int(np.floor(len(traversals) * fraction))
    train, validation = traversals[:n], traversals[n:]
    if {v['direction'] for v in traversals} != set(directions):
        raise ValueError('traversal directions conflict with documented directions')
    for direction in directions:
        if sum(v['direction'] == direction for v in train) < 3 or sum(v['direction'] == direction for v in validation) < 2:
            raise ValueError('insufficient chronological traversals per direction')
    return train, validation, float(domain[n - 1, 1])


def count_bins(spikes, edges):
    edges = np.asarray(edges, float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError('invalid bin edges')
    # Half-open bins, including the last: no end-boundary spike leakage.
    trains = [np.asarray(t, float).ravel() for t in spikes]
    if any(not np.isfinite(t).all() or np.any(np.diff(t) < 0) for t in trains):
        raise ValueError('invalid spike chronology; no sorting repair')
    return np.asarray([np.diff(np.searchsorted(t, edges, side='left')) for t in trains], int).T


def fit_maps(t, x, spikes, training, length_cm, topology, params):
    domain = intervals([[v['start_s'], v['end_s']] for v in training])
    edges = np.arange(0, length_cm, params['bin_cm'])
    edges = np.r_[edges, length_cm]
    centers = (edges[1:] + edges[:-1]) / 2
    sample_times = np.concatenate([np.arange(a, b - params['decode_bin_s'] / 2, params['decode_bin_s']) + params['decode_bin_s'] / 2 for a, b in domain])
    px, speed, valid = tracking_at(t, x, sample_times, length_cm, topology, params['max_tracking_gap_s'])
    keep = valid & (speed >= params['min_speed_cm_s'])
    occupancy = np.histogram(px[keep], edges)[0] * params['decode_bin_s']
    support = occupancy >= params['min_occupancy_s']
    raw = np.zeros((len(spikes), len(centers)))
    for i, train in enumerate(spikes):
        selected = np.concatenate([np.asarray(train)[(np.asarray(train) >= a) & (np.asarray(train) < b)] for a, b in domain])
        sx, ss, sv = tracking_at(t, x, selected, length_cm, topology, params['max_tracking_gap_s'])
        raw[i] = np.histogram(sx[sv & (ss >= params['min_speed_cm_s'])], edges)[0]
    d = distance(centers[:, None], centers[None, :], length_cm, topology)
    kernel = np.exp(-0.5 * (d / params['smoothing_sigma_cm']) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    occ_smoothed = kernel @ occupancy
    spike_smoothed = raw @ kernel.T
    global_rate = raw.sum(axis=1) / max(occupancy.sum(), np.finfo(float).tiny)
    pseudo = params['global_rate_pseudocount_s']
    rates = (spike_smoothed + pseudo * global_rate[:, None]) / (occ_smoothed + pseudo)
    if not support.any():
        raise ValueError('no training-supported positions')
    p = occupancy[support] / occupancy[support].sum()
    r = rates[:, support]
    mean = r @ p
    ratio = r / np.maximum(mean[:, None], 1e-300)
    info = np.sum(p * ratio * np.log2(np.maximum(ratio, 1e-300)), axis=1)
    peak = r.max(axis=1)
    selected = (raw.sum(axis=1) >= params['min_training_spikes']) & (info >= params['min_information_bits_spike']) & (peak >= params['min_peak_rate_hz'])
    return {'centers': centers, 'edges': edges, 'occupancy': occupancy, 'raw_counts': raw,
            'rates': rates, 'support': support, 'selected': selected, 'information': info, 'peak': peak}


def flat_decode(counts, rates, support, dt=0.02):
    rates = np.asarray(rates, float)
    if not np.any(support) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError('invalid supported rate map')
    r = rates[:, support]
    logp = np.asarray(counts) @ np.log(np.maximum(r, 1e-300)) - dt * r.sum(axis=0)
    # Uniform spatial prior: occupancy never enters this calculation.
    logp -= logsumexp(logp, axis=1, keepdims=True)
    return np.exp(logp)


def evaluate_run(t, x, spikes, validation, maps, length_cm, topology, params):
    rows, cache = [], []
    for traversal in validation:
        dt = params['decode_bin_s']
        edges = np.arange(traversal['start_s'], traversal['end_s'] + 1e-12, dt)
        centers_t = (edges[1:] + edges[:-1]) / 2
        truth, speed, valid = tracking_at(t, x, centers_t, length_cm, topology, params['max_tracking_gap_s'])
        eligible = valid & (speed >= params['min_speed_cm_s'])
        counts = count_bins(spikes, edges)
        post = flat_decode(counts, maps['rates'], maps['support'], dt)
        positions = maps['centers'][maps['support']]
        predicted = positions[np.argmax(post, axis=1)]
        target = np.clip(np.searchsorted(maps['edges'], np.nan_to_num(truth), side='right') - 1, 0, len(maps['support']) - 1)
        supported = eligible & maps['support'][target]
        error = distance(predicted, truth, length_cm, topology)
        entropy = -np.sum(post * np.log(np.maximum(post, 1e-300)), axis=1)
        rows.append({'traversal_id': traversal['id'], 'direction': traversal['direction'], 'eligible_bins': int(eligible.sum()),
                     'supported_bins': int(supported.sum()), 'coverage': float(supported.sum() / eligible.sum()) if eligible.any() else None,
                     'median_error_cm': float(np.median(error[supported])) if supported.any() else None,
                     'mean_entropy': float(np.mean(entropy[supported])) if supported.any() else None,
                     'zero_spike_bins': int(np.sum(eligible & (counts.sum(axis=1) == 0)))})
        cache.append({'times': centers_t, 'truth': truth, 'counts': counts, 'eligible': eligible,
                      'supported': supported, 'error': error, 'posterior': post, 'predicted': predicted})
    return rows, cache


def ripple_envelope(lfp, fs, params):
    if fs <= 2 * params['high_hz'] or not np.isfinite(lfp).all():
        raise ValueError('invalid LFP sampling/values')
    sos = butter(params['filter_order'], [params['low_hz'], params['high_hz']], btype='bandpass', fs=fs, output='sos')
    return gaussian_filter1d(np.abs(hilbert(sosfiltfilt(sos, np.asarray(lfp, float)))), params['smooth_sigma_s'] * fs)


def envelope_events(envelope, fs, start_s, mean, sd, prefix, params):
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        raise ValueError('PRE-NREM baseline unavailable/degenerate')
    z = (np.asarray(envelope) - mean) / sd
    mask = z > params['boundary_z']
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    rows = []
    for i, (a, b) in enumerate(bounds):
        peak = a + int(np.argmax(z[a:b]))
        duration = (b - a) / fs
        if z[peak] > params['peak_z'] and params['min_duration_s'] <= duration <= params['max_duration_s']:
            rows.append({'id': f'{prefix}:{i}', 'start_s': start_s + a / fs, 'end_s': start_s + b / fs,
                         'peak_s': start_s + peak / fs, 'peak_z': float(z[peak]), 'artifact': False})
    return rows


def merge_channels(events):
    if len({e['id'] for e in events}) != len(events):
        raise ValueError('duplicate ripple identity')
    rows = []
    for event in sorted(events, key=lambda v: (v['start_s'], v['id'])):
        if not rows or event['start_s'] >= rows[-1]['end_s']:
            rows.append({**event, 'parents': [event['id']], 'parent_channels': [event.get('channel_id')],
                         'parent_peak_min_s': event['peak_s'], 'parent_peak_max_s': event['peak_s'], 'compound': False})
        else:
            last = rows[-1]
            last['end_s'] = max(last['end_s'], event['end_s'])
            last['parents'].append(event['id'])
            last['parent_channels'].append(event.get('channel_id'))
            last['parent_peak_min_s'] = min(last['parent_peak_min_s'], event['peak_s'])
            last['parent_peak_max_s'] = max(last['parent_peak_max_s'], event['peak_s'])
            # Co-detection on distinct shanks is not automatically a compound ripple.
            last['compound'] = (len(set(last['parent_channels'])) != len(last['parent_channels']) or
                                None in last['parent_channels'] or
                                last['parent_peak_max_s'] - last['parent_peak_min_s'] > 0.05 + 1e-12 or
                                last['end_s'] - last['start_s'] > 0.2 + 1e-12)
            last['artifact'] |= event.get('artifact', False)
            if (event['peak_z'], event['id']) > (last['peak_z'], last['id']):
                last['peak_z'], last['peak_s'] = event['peak_z'], event['peak_s']
    return rows


def match_ripples(left, right, max_separation=0.05, min_overlap=0.02):
    left, right = sorted(left, key=lambda e: e['id']), sorted(right, key=lambda e: e['id'])
    if len({e['id'] for e in left + right}) != len(left) + len(right):
        raise ValueError('duplicate ripple identity')
    if not left or not right:
        return []
    n, m = len(left), len(right)
    # Dummy assignment penalty dominates every possible total timing cost.
    # This enforces maximum cardinality before minimizing separation.
    penalty = (min(n, m) + 1) * (max_separation + 1)
    cost = np.full((n, m + n), penalty)
    allowed = np.zeros((n, m), bool)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            delta = abs(a['peak_s'] - b['peak_s'])
            overlap = min(a['end_s'], b['end_s']) - max(a['start_s'], b['start_s'])
            if delta <= max_separation + 1e-12 and overlap >= min_overlap - 1e-12:
                allowed[i, j] = True
                cost[i, j] = delta
            else:
                cost[i, j] = 2 * penalty
    ii, jj = linear_sum_assignment(cost)
    return [(left[i], right[j]) for i, j in zip(ii, jj) if j < m and allowed[i, j]]


def opportunity(a, b, states, left_spikes, right_spikes, params, cutoff_s):
    center = (a['peak_s'] + b['peak_s']) / 2
    start, end = center - params['common_window_s'] / 2, center + params['common_window_s'] / 2
    labels = [label for label, domain in states.items() if contains(start, end, domain)]
    row = {'left_id': a['id'], 'right_id': b['id'], 'start_s': start, 'end_s': end,
           'state': labels[0] if len(labels) == 1 else None, 'supported': False, 'reason': 'state_not_verified_or_ambiguous'}
    if len(labels) != 1:
        return row
    if labels[0] == 'awake_rest' and start < cutoff_s:
        row['reason'] = 'before_training_cutoff'
        return row
    if a.get('compound') or b.get('compound') or a.get('artifact') or b.get('artifact'):
        row['reason'] = 'compound_or_artifact'
        return row
    edges = start + np.arange(11) * 0.02
    l, r = count_bins(left_spikes, edges), count_bins(right_spikes, edges)
    both = ((l.sum(axis=1) >= params['min_spikes_per_bin']) & ((l > 0).sum(axis=1) >= params['min_active_cells_per_bin']) &
            (r.sum(axis=1) >= params['min_spikes_per_bin']) & ((r > 0).sum(axis=1) >= params['min_active_cells_per_bin']))
    row['supported_bins_both'] = int(both.sum())
    row['supported'] = int(both.sum()) >= params['min_supported_bins']
    row['reason'] = 'sequence_testing_opportunity_not_validated_replay' if row['supported'] else 'insufficient_bilateral_bin_support'
    return row


def decision(novelty, source_verified, verification_passed, qualified_counts, minimum_animals=3, minimum_windows=30):
    if novelty == 'established':
        return 'stop_prior_art'
    if novelty != 'cleared' or not source_verified or not verification_passed:
        return 'inconclusive_feasibility'
    animals = {r['animal'] for r in qualified_counts}
    passed = 0
    for animal in animals:
        totals = []
        for state in ('awake_rest', 'POST_NREM'):
            values = [r.get('supported_windows') for r in qualified_counts if r['animal'] == animal and r['state'] == state and r.get('decoder_qualified')]
            totals.append(sum(values) if values and all(v is not None and np.isfinite(v) for v in values) else None)
        passed += all(v is not None and v >= minimum_windows for v in totals)
    return 'ready_for_calibration' if passed >= minimum_animals else 'inconclusive_feasibility'


def read_verified_source(path, sha256):
    review = json.loads(Path(path).read_text())
    required = ('channel_order', 'spike_group_anatomy', 'coordinates_topology', 'traversals_directions', 'awake_rest', 'common_clock', 'pyramidal_layer')
    for key in required:
        refs = review.get('evidence', {}).get(key, [])
        if not refs:
            raise ValueError(f'unverified source requirement: {key}')
        for ref in refs:
            target = Path(path).parent / ref['path']
            if sha256(target) != ref.get('sha256') or not ref.get('sha256'):
                raise ValueError(f'source evidence hash mismatch: {key}')
    if review.get('status') != 'verified' or not review.get('reviewer'):
        raise ValueError('missing independent source sign-off')
    return review
