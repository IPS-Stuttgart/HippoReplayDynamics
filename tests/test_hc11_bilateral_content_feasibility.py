from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts import _hc11_bilateral_feasibility as c
from scripts import audit_hc11_bilateral_content_feasibility as audit

PROTOCOL = json.loads((Path(__file__).parents[1] / 'docs/hc11_bilateral_content_feasibility_protocol.json').read_text())


def event(name, peak, start=None, end=None):
    return {'id': name, 'peak_s': peak, 'start_s': peak - 0.05 if start is None else start,
            'end_s': peak + 0.05 if end is None else end, 'peak_z': 3, 'artifact': False}


def test_original_group16_not_converted_uid():
    src = {'SpikeTimes': [1, 2, 3], 'SpikeIDs': [1602, 1602, 1602], 'PyrIDs': [1602]}
    conv = {'UID': 1, 'times': [1, 2, 3]}
    unresolved = c.original_crosswalk(src, conv)[0]
    assert unresolved['spike_group'] == 16 and unresolved['hemisphere'] is None
    verified = c.original_crosswalk(src, conv, {'16': {'shank': 15, 'hemisphere': 'right', 'region': 'CA1'}})[0]
    assert verified['hemisphere'] == 'right' and verified['ca1_pyramidal']


def test_ambiguous_original_trains_do_not_qualify():
    src = {'SpikeTimes': [1, 1, 2, 2], 'SpikeIDs': [102, 1602, 102, 1602], 'PyrIDs': [102, 1602]}
    row = c.original_crosswalk(src, {'UID': 7, 'times': [1, 2]})[0]
    assert not row['timing_verified'] and not row['mapping_verified']


def test_clock_shift_not_fitted_away():
    src = {'SpikeTimes': [1, 2], 'SpikeIDs': [102, 102], 'PyrIDs': [102]}
    row = c.original_crosswalk(src, {'UID': 1, 'times': [1.01, 2.01]})[0]
    assert not row['timing_verified']


def test_channel_ids_are_not_indices():
    np.testing.assert_array_equal(c.channel_rows([20, 42, 5], [5, 20]), [2, 0])
    with pytest.raises(ValueError):
        c.channel_rows([20, 42, 5], [0])


def test_channel_count_override_not_automatic(tmp_path):
    p = tmp_path / 'Gatsby_08022013.eeg'
    np.zeros(134 * 10, dtype=np.int16).tofile(p)
    assert c.verify_eeg_layout(p, 134, 1250, 10 / 1250) == 10 / 1250
    with pytest.raises(ValueError):
        c.verify_eeg_layout(p, 128, 1250, 10 / 1250)


@pytest.mark.parametrize('domain', [[[2, 1]], [[2, 3], [1, 2]], [[1, float('nan')]], [[1, 3], [2, 4]]])
def test_invalid_intervals_not_repaired(domain):
    with pytest.raises(ValueError):
        c.intervals(domain)


def test_circular_seam_short_distance_and_interpolation():
    assert c.distance(398, 2, 400, 'circular') == 4
    x, speed, valid = c.tracking_at([0, 0.04], [398, 2], [0.02], 400, 'circular')
    assert valid[0] and x[0] == pytest.approx(0) and speed[0] == pytest.approx(100)


def test_no_gap_or_clock_repair():
    x, _, valid = c.tracking_at([0, 0.2], [0, 20], [0.1], 100, 'linear')
    assert not valid[0] and np.isnan(x[0])
    with pytest.raises(ValueError):
        c.tracking_at([1, 0], [0, 20], [0.5], 100, 'linear')


def traversals(n=10):
    return [{'id': str(i), 'start_s': i, 'end_s': i + 0.9, 'direction': 'forward'} for i in range(n)]


def test_split_chronological_and_direction_floors():
    a, b, cutoff = c.chronological_split(traversals(), ['forward'])
    assert len(a) == 7 and len(b) == 3 and cutoff < b[0]['start_s']
    with pytest.raises(ValueError):
        c.chronological_split(traversals(4), ['forward'])
    with pytest.raises(ValueError):
        c.chronological_split(traversals(), ['forward', 'reverse'])


def test_duplicate_traversals_fail():
    with pytest.raises(ValueError):
        c.chronological_split(traversals() + traversals(), ['forward'])


def test_zero_spike_poisson_not_artificially_flat():
    p = c.flat_decode([[0]], [[1, 5]], [True, True])
    np.testing.assert_allclose(p[0, 0] / p[0, 1], np.exp(0.08))
    assert p.shape == (1, 2) and np.isfinite(p).all()


def test_half_open_counts_no_endpoint_leakage():
    np.testing.assert_array_equal(c.count_bins([[0, 0.02, 0.04]], [0, 0.02, 0.04]), [[1], [1]])


def test_unsupported_targets_count_before_exclusion():
    maps = {'rates': np.asarray([[1, 5]]), 'support': np.asarray([True, False]), 'centers': np.asarray([2, 6]), 'edges': np.asarray([0, 4, 8])}
    rows, _ = c.evaluate_run([0, 0.04, 0.08], [0, 4, 8], [[]], [{'id': '1', 'start_s': 0, 'end_s': 0.08, 'direction': 'forward'}], maps, 8, 'linear', PROTOCOL['run'])
    assert rows[0]['eligible_bins'] == 4 and rows[0]['supported_bins'] == 2
    assert rows[0]['coverage'] == 0.5 and rows[0]['zero_spike_bins'] == 4


def test_future_RUN_and_replay_spikes_do_not_change_maps():
    t = np.arange(0, 12, 0.02)
    x = (t % 1) * 40
    spikes = [np.arange(0.01, 12, 0.03)]
    train = traversals()[:7]
    first = c.fit_maps(t, x, spikes, train, 40, 'linear', PROTOCOL['run'])
    other = [np.concatenate((spikes[0][spikes[0] < 6.9], np.arange(7, 12, 0.0005)))]
    second = c.fit_maps(t, x, other, train, 40, 'linear', PROTOCOL['run'])
    for key in ('rates', 'raw_counts', 'occupancy', 'selected'):
        np.testing.assert_array_equal(first[key], second[key])
    assert first['raw_counts'].sum() > 20


def test_duplicate_ripples_fail():
    with pytest.raises(ValueError):
        c.merge_channels([event('a', 1), event('a', 1)])


def test_overlap_merge_keeps_parent_identity_and_compound_flag():
    merged = c.merge_channels([event('a', 1), event('b', 1.04), event('c', 2)])
    assert len(merged) == 2 and merged[0]['parents'] == ['a', 'b'] and merged[0]['compound']


def test_maximum_cardinality_before_nearest_peak_greedy():
    # l2 only reaches r1; greedy l1->r1 would leave just one pair.
    left = [event('l1', 1), event('l2', 0.965)]
    right = [event('r1', 0.99), event('r2', 1.04)]
    pairs = c.match_ripples(left, right)
    assert {(a['id'], b['id']) for a, b in pairs} == {('l1', 'r2'), ('l2', 'r1')}


def test_matching_identity_ties_deterministic():
    left = [event('l1', 1), event('l2', 1)]
    right = [event('r1', 1), event('r2', 1)]
    a = c.match_ripples(left, right)
    b = c.match_ripples(left[::-1], right[::-1])
    assert [(x['id'], y['id']) for x, y in a] == [(x['id'], y['id']) for x, y in b]


def test_minimum_overlap_not_only_peak_distance():
    assert not c.match_ripples([event('l', 1, 0.99, 1.005)], [event('r', 1.01, 1.005, 1.02)])


def test_detector_PRE_normalization_and_duration():
    e = np.zeros(500)
    e[100:150] = 4
    rows = c.envelope_events(e, 1250, 10, 0, 1, 'channel', PROTOCOL['ripple'])
    assert len(rows) == 1 and rows[0]['start_s'] == pytest.approx(10.08)
    with pytest.raises(ValueError):
        c.envelope_events(e, 1250, 10, 0, 0, 'channel', PROTOCOL['ripple'])


def supported_population():
    ts = 0.91 + np.arange(10) * 0.02
    return [np.sort(np.r_[ts, ts + 0.001]), ts + 0.002]


def test_supported_opportunity_not_replay_and_both_states():
    a, b = event('l', 1), event('r', 1.01)
    row = c.opportunity(a, b, {'awake_rest': [[0.8, 1.2]], 'POST_NREM': [[2, 3]]}, supported_population(), supported_population(), PROTOCOL['ripple'], 0.5)
    assert row['supported'] and 'not_validated_replay' in row['reason']


@pytest.mark.parametrize('states,cutoff,reason', [({'awake_rest': [[0.96, 1.1]]}, 0, 'state_not_verified_or_ambiguous'),
                                               ({'awake_rest': [[0, 2]], 'POST_NREM': [[0, 2]]}, 0, 'state_not_verified_or_ambiguous'),
                                               ({'awake_rest': [[0, 2]]}, 1.5, 'before_training_cutoff')])
def test_state_containment_ambiguity_and_training_cutoff(states, cutoff, reason):
    row = c.opportunity(event('l', 1), event('r', 1), states, [], [], PROTOCOL['ripple'], cutoff)
    assert not row['supported'] and row['reason'] == reason


def test_missing_not_zero_and_no_sleep_only_fallback():
    counts = [{'animal': a, 'state': 'POST_NREM', 'decoder_qualified': True, 'supported_windows': 100} for a in 'abc']
    assert c.decision('cleared', True, True, counts) == 'inconclusive_feasibility'
    for a in 'abc':
        counts.append({'animal': a, 'state': 'awake_rest', 'decoder_qualified': True, 'supported_windows': None})
    assert c.decision('cleared', True, True, counts) == 'inconclusive_feasibility'


def test_nonvacuous_readiness_and_prior_art_stop():
    counts = [{'animal': a, 'state': s, 'decoder_qualified': True, 'supported_windows': 30} for a in 'abc' for s in ('awake_rest', 'POST_NREM')]
    assert c.decision('cleared', True, True, counts) == 'ready_for_calibration'
    assert c.decision('cleared', True, True, []) == 'inconclusive_feasibility'
    assert c.decision('unresolved', True, True, counts) == 'inconclusive_feasibility'
    assert c.decision('established', True, True, counts) == 'stop_prior_art'


def test_mat_v5_and_v73_loader(tmp_path):
    savemat(tmp_path / 'v5.mat', {'sessInfo': {'x': np.array([1, 2])}})
    np.testing.assert_array_equal(c.load_structure(tmp_path / 'v5.mat', 'sessInfo')['x'], [1, 2])
    import h5py
    with h5py.File(tmp_path / 'v73.mat', 'w') as f:
        g = f.create_group('sessInfo')
        g.create_dataset('x', data=np.array([[1], [2]]))
    np.testing.assert_array_equal(c.load_structure(tmp_path / 'v73.mat', 'sessInfo')['x'], [1, 2])


def test_inventory_missing_sources_preserves_all_sessions(tmp_path):
    data, out = tmp_path / 'data', tmp_path / 'output'
    data.mkdir(); out.mkdir()
    args = argparse.Namespace(dataset_root=data, output_dir=out)
    audit.inventory(args, PROTOCOL)
    frame = pd.read_csv(out / 'hc11_bilateral_recording_inventory.csv')
    assert len(frame) == 8 and not frame.source_verified.any()
    assert frame.converted_units.isna().all()


def test_source_hash_and_signoff_required(tmp_path):
    p = tmp_path / 'review.json'
    p.write_text(json.dumps({'status': 'verified', 'reviewer': 'x'}))
    with pytest.raises(ValueError, match='channel_order'):
        c.read_verified_source(p, lambda _: 'fake')


def test_checkpoint_output_tampering_rejected(tmp_path):
    args = argparse.Namespace(output_dir=tmp_path)
    (tmp_path / 'output.csv').write_text('x\n1\n')
    audit.save_checkpoint(args, 'inventory', {'commit': 'a'}, {'protocol': 'b'}, set())
    assert audit.checkpoint(args, 'inventory', {'commit': 'a'}, {'protocol': 'b'})
    (tmp_path / 'output.csv').write_text('x\n2\n')
    with pytest.raises(ValueError):
        audit.checkpoint(args, 'inventory', {'commit': 'a'}, {'protocol': 'b'})


def test_all_stages_source_limited_report_without_zero_observations(tmp_path):
    root = Path(__file__).parents[1]
    data, out = tmp_path / 'data', tmp_path / 'output'
    data.mkdir(); out.mkdir()
    args = argparse.Namespace(dataset_root=data, output_dir=out, protocol=root / 'docs/hc11_bilateral_content_feasibility_protocol.json')
    audit.inventory(args, PROTOCOL)
    audit.run_qc(args, PROTOCOL)
    audit.opportunities(args, PROTOCOL)
    audit.verify(args, PROTOCOL)
    audit.report(args, PROTOCOL)
    counts = pd.read_csv(out / 'hc11_bilateral_opportunity_counts.csv')
    assert len(counts) == 16 and counts.supported_windows.isna().all()
    gates = pd.read_csv(out / 'hc11_bilateral_gate_summary.csv').set_index('gate')
    assert gates.loc['overall', 'status'] == 'inconclusive_feasibility'
    assert (out / 'hc11_bilateral_source_availability.png').exists()
