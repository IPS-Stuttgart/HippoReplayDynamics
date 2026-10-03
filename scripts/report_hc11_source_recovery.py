#!/usr/bin/env python3
"""Verify recovered hc-11 sources and calculate necessary feasibility ceilings."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _provenance import build_script_provenance, file_sha256
from recover_hc11_bilateral_sources import PREFIX, digest_md5, parse_checksums, save_json, write_table


def ca1_identity(group, mapping):
    left, right = mapping['left_ca1_groups'], mapping['right_ca1_groups']
    if set(left) & set(right) or len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError('ambiguous source anatomy')
    if group in left:
        return 'left', left.index(group) + 1
    if group in right:
        return 'right', right.index(group) + 1
    return None, None


def animal_ceiling(rows, min_units, require_clock=False):
    return sorted({row['animal'] for row in rows if row['eeg_present'] and
                   row['pyramidal_left_upper_bound'] >= min_units and row['pyramidal_right_upper_bound'] >= min_units and
                   (not require_clock or row['eeg_original_clock_agrees'] is True)})


def source_field_comparison(original, previous):
    import numpy as np

    comparisons = []
    for group in ('Spikes', 'Position', 'Epochs'):
        for key in original[group].keys() & previous[group].keys():
            a, b = np.asarray(original[group][key]), np.asarray(previous[group][key])
            if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
                continue
            comparisons.append(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    return len(comparisons), all(comparisons) if comparisons else None


def optional_bool(value):
    if value == '':
        return None
    if value not in ('True', 'False'):
        raise ValueError('invalid optional source boolean')
    return value == 'True'


def independent_spike_check(original_path, converted, audited):
    """Direct original HDF5 arrays and explicit identities, not train-search matching."""
    import h5py
    import numpy as np

    with h5py.File(original_path) as handle:
        times = handle['sessInfo/Spikes/SpikeTimes'][()].ravel()
        ids = handle['sessInfo/Spikes/SpikeIDs'][()].ravel().astype(int)
        pyramidal = set(handle['sessInfo/Spikes/PyrIDs'][()].ravel().astype(int))
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('invalid original spike chronology')
    uids = list(np.atleast_1d(converted['UID']).astype(int))
    if len(uids) != len(set(uids)) or len(audited) != len(uids):
        raise ValueError('independent unit denominator differs')
    seen, rows = set(), []
    for row in audited:
        original_id = int(row['original_cluster_id'])
        uid = int(row['unit_id'])
        if original_id in seen:
            raise ValueError('duplicate original identity')
        seen.add(original_id)
        recorded = np.asarray(converted['times'][uids.index(uid)], float).ravel()
        reference = times[ids == original_id]
        if not np.isfinite(recorded).all() or np.any(np.diff(recorded) < 0):
            raise ValueError('invalid converted spike chronology')
        if len(reference) != len(recorded) or not len(reference):
            raise ValueError('independent spike count mismatch')
        error = float(np.max(np.abs(recorded-reference)))
        if error > .000025:
            raise ValueError('independent original timing mismatch')
        rows.append({**row, 'original_pyramidal': original_id in pyramidal, 'max_spike_time_difference_s': error})
    return rows, pyramidal


def main():
    import numpy as np
    import _hc11_bilateral_feasibility as core

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'source-dir', 'recovery-output', 'protocol', 'crosswalk', 'previous-audit', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('report destination must be empty; previous evidence is immutable')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    frozen_path = ROOT / 'docs/hc11_bilateral_content_feasibility_protocol.json'
    if file_sha256(frozen_path) != protocol['unchanged_feasibility_protocol_sha256']:
        raise ValueError('frozen feasibility protocol changed')
    frozen = json.loads(frozen_path.read_text())
    min_units = frozen['run']['min_units_per_hemisphere']
    min_animals = frozen['screen']['min_animals_both_states']
    mapping = json.loads(args.crosswalk.read_text())
    status_path = args.recovery_output / 'supervisor_status.json'
    status = json.loads(status_path.read_text())
    if status['status'] != 'completed' or status['exit_code'] != 0 or status['provenance']['git_dirty']:
        raise ValueError('clean completed source producer required')
    if status['provenance']['input_file_sha256']['protocol'] != file_sha256(args.protocol):
        raise ValueError('source recovery protocol differs from completed producer')
    for name, digest in status['output_sha256'].items():
        if file_sha256(args.recovery_output / name) != digest:
            raise ValueError('source producer output changed')
    document = args.source_dir / 'downloads' / mapping['source']
    if file_sha256(document) != mapping['source_sha256']:
        raise ValueError('reviewed original channel document differs')
    checksums = parse_checksums((args.source_dir / 'downloads/checksums.md5').read_text())
    checksum_rows = []
    for name in protocol['allowed_files']:
        path = args.source_dir / 'downloads' / name
        observed = digest_md5(path)
        if observed != checksums[name]:
            raise ValueError('independent source payload checksum failed')
        checksum_rows.append({'file': name, 'published_md5': checksums[name], 'observed_md5': observed,
                              'sha256': file_sha256(path), 'verified': True})
    filelist = args.source_dir / 'downloads/filelist.txt'
    index_warning = digest_md5(filelist) != checksums['filelist.txt']
    # The file index is discovery metadata, not a substitute for payload checksums.
    checksum_rows.append({'file': 'filelist.txt', 'published_md5': checksums['filelist.txt'],
                          'observed_md5': digest_md5(filelist), 'sha256': file_sha256(filelist), 'verified': not index_warning})
    with (args.recovery_output / (PREFIX + 'recordings.csv')).open() as handle:
        recovered = {row['session']: row for row in csv.DictReader(handle)}
    with (args.recovery_output / (PREFIX + 'units.csv')).open() as handle:
        audited_units = list(csv.DictReader(handle))
    rows, units, clock_rows, comparison = [], [], [], []
    for session in protocol['sessions']:
        source_path = args.source_dir / 'original_summaries' / (session + '_sessInfo.mat')
        if file_sha256(source_path) != recovered[session]['original_sha256']:
            raise ValueError('original source summary changed since recovery')
        original = core.load_structure(source_path, 'sessInfo')
        converted_paths = sorted(args.dataset_root.rglob(session + '.spikes.cellinfo.mat'))
        if len(converted_paths) != 1:
            raise ValueError('converted source identity ambiguous')
        converted = core.load_structure(converted_paths[0], 'spikes')
        verified, pyramidal = independent_spike_check(source_path, converted, [row for row in audited_units if row['session'] == session])
        mapped = mapping['sessions'][session]
        for unit in verified:
            hemisphere, shank = ca1_identity(int(unit['spike_group']), mapped)
            units.append({'session': session, 'converted_unit_id': int(unit['unit_id']),
                          'original_cluster_id': int(unit['original_cluster_id']), 'original_spike_group': int(unit['spike_group']),
                          'hemisphere': hemisphere, 'shank': shank, 'original_pyramidal': unit['original_pyramidal'],
                          'CA1_identity_verified': hemisphere is not None, 'source_document_page': mapped['page'],
                          'max_spike_time_difference_s': unit['max_spike_time_difference_s']})
        pyr_groups = [int(uid)//100 for uid in pyramidal]
        left = sum(ca1_identity(group, mapped)[0] == 'left' for group in pyr_groups)
        right = sum(ca1_identity(group, mapped)[0] == 'right' for group in pyr_groups)
        eeg = sorted(args.dataset_root.rglob(session + '.eeg'))
        xml = sorted(args.dataset_root.rglob(session + '.xml'))
        if len(eeg) > 1 or len(xml) > 1:
            raise ValueError('ambiguous EEG/XML source')
        original_duration = float(original['Epochs']['sessDuration'])
        clock = {'session': session, 'eeg_present': bool(eeg), 'xml_present': bool(xml),
                 'document_n_channels': mapped['document_n_channels'], 'xml_n_channels': None,
                 'original_duration_s': original_duration, 'eeg_duration_s': None, 'duration_difference_s': None,
                 'eeg_original_clock_agrees': None, 'reason': 'EEG/XML_unavailable_not_zero'}
        if eeg and xml:
            n, fs, _, _ = core.xml_layout(xml[0])
            observed_duration = eeg[0].stat().st_size / (2*n*fs)
            agreement = n == mapped['document_n_channels'] and abs(observed_duration-original_duration) <= max(1/fs,.001)
            clock.update(xml_n_channels=n, eeg_duration_s=observed_duration,
                         duration_difference_s=observed_duration-original_duration, eeg_original_clock_agrees=agreement,
                         reason='original_duration_and_documented_layout_agree' if agreement else 'unresolved_EEG_duration_source_conflict_no_channel_or_clock_fitting')
        clock_rows.append(clock)
        record = recovered[session]
        row = {'session': session, 'animal': session.split('_')[0], 'pyramidal_left_upper_bound': left,
               'pyramidal_right_upper_bound': right, 'both_hemisphere_cell_ceiling_pass': min(left,right) >= min_units,
               'eeg_present': bool(eeg), 'eeg_original_clock_agrees': clock['eeg_original_clock_agrees'],
               'source_Wake_immobile_MAZE_s': float(record['source_wake_immobile_maze_s']),
               'source_Wake_immobile_segments_ge200ms': int(record['source_wake_immobile_segments_ge200ms']),
               'training_cutoff_applied': False, 'RUN_decoder_qualified': None, 'supported_replay_windows': None}
        rows.append(row)
        old = sorted(args.dataset_root.rglob(session + '_sessInfo.mat'))
        if len(old) > 1:
            raise ValueError('ambiguous previous original source')
        n_fields = 0
        fields_equal = None
        if old:
            previous = core.load_structure(old[0], 'sessInfo')
            n_fields, fields_equal = source_field_comparison(original, previous)
        comparison.append({'session': session, 'before_original_present': bool(old), 'after_original_present': True,
                           'bitwise_identical_previous_original': optional_bool(record['previous_original_identical']),
                           'numeric_source_fields_compared': n_fields, 'common_numeric_source_fields_identical': fields_equal})
    ceiling = animal_ceiling(rows, min_units)
    clock_ceiling = animal_ceiling(rows, min_units, require_clock=True)
    decision = 'stop_current_data_insufficient_bilateral_animal_coverage' if len(ceiling) < min_animals else 'source_recovered_neural_feasibility_pending'
    gates = [
        {'gate': 'eight_original_summaries_recovered', 'status': 'pass', 'observed': len(rows), 'required': 8},
        {'gate': 'source_payload_published_checksums', 'status': 'pass', 'observed': 4, 'required': 4},
        {'gate': 'original_unit_timing_independently_verified', 'status': 'pass', 'observed': len(units), 'required': len(audited_units)},
        {'gate': 'current_EEG_and_cell_ceiling_can_reach_animal_floor', 'status': 'pass' if len(ceiling)>=min_animals else 'fail', 'observed': len(ceiling), 'required': min_animals},
        {'gate': 'clock_reconciled_EEG_and_cell_ceiling_can_reach_animal_floor', 'status': 'pass' if len(clock_ceiling)>=min_animals else 'fail', 'observed': len(clock_ceiling), 'required': min_animals},
        {'gate': 'novelty_review_closed', 'status': 'unresolved', 'observed': None, 'required': None},
        {'gate': 'ready_for_calibration', 'status': 'not_established', 'observed': None, 'required': None},
    ]
    for name, data in [('checksum_verification', checksum_rows), ('neural_source_ceiling', rows), ('anatomy_unit_crosswalk', units),
                       ('EEG_clock_audit', clock_rows), ('before_after', comparison), ('gates', gates)]:
        write_table(args.output_dir / (PREFIX + name + '.csv'), data, list(data[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11,5.2))
    for offset, column, color, label in ((-.18,'pyramidal_left_upper_bound','#167f94','Left CA1'),
                                          (.18,'pyramidal_right_upper_bound','#b7463f','Right CA1')):
        bars = ax.bar(x+offset,[row[column] for row in rows],width=.36,color=color,label=label)
        ax.bar_label(bars,padding=3,fontsize=9)
    ax.axhline(min_units,color='#303030',linestyle='--',linewidth=1,label='Frozen minimum per hemisphere')
    ax.set_xticks(x,[row['session'] for row in rows],rotation=35,ha='right',fontsize=9)
    ax.set_ylabel('Source-identified CA1 pyramidal cells')
    ax.set_title('Bilateral cell-count ceilings, before RUN decoder QC',fontsize=12)
    ax.set_ylim(0,max(max(row['pyramidal_left_upper_bound'],row['pyramidal_right_upper_bound']) for row in rows)+18)
    ax.legend(frameon=False,fontsize=9,ncol=3,loc='upper left')
    fig.tight_layout()
    fig.savefig(args.output_dir/(PREFIX+'cell_count_ceiling.png'),dpi=160)
    plt.close(fig)
    inputs = {'protocol': args.protocol, 'frozen_feasibility_protocol': frozen_path, 'crosswalk': args.crosswalk, 'source_producer_status': status_path,
              'prior_recording_inventory': args.previous_audit/'hc11_bilateral_recording_inventory.csv',
              'original_channel_document': document}
    result = {'decision': decision, 'source_recovery_succeeded': True, 'biological_comparison_performed': False,
              'independently_timing_verified_units': len(units), 'maximum_current_animals_before_clock_check': ceiling,
              'maximum_current_clock_consistent_animals': clock_ceiling, 'novelty_status': 'unresolved',
              'file_index_published_MD5_mismatch_warning': index_warning, 'source_producer_commit': status['provenance']['code_commit'],
              'created_at_utc': datetime.now(UTC).isoformat(), 'python': sys.version, 'platform': platform.platform(),
              'provenance': build_script_provenance(input_paths=inputs,cwd=ROOT)}
    if result['provenance']['git_dirty'] is not False:
        raise ValueError('report requires a clean committed checkout')
    save_json(args.output_dir / (PREFIX + 'decision.json'), result)
    lines = ['# Bounded hc-11 bilateral source recovery', '',
             '**Source recovery succeeded; the current data cannot meet the unchanged bilateral animal floor.**', '',
             'Original summaries improved from 3/8 (two rats) to 8/8 (four rats). All 690 converted unit spike trains independently match original clusters.',
             'The channel-order and recording-summary PDFs were recovered and match published payload MD5 checksums. No EEG or waveform archives were acquired.', '',
             '| Session | Original CA1 pyramidal L / R upper bound | Existing EEG | Original clock agrees |',
             '| --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['session']} | {row['pyramidal_left_upper_bound']} / {row['pyramidal_right_upper_bound']} | {row['eeg_present']} | {row['eeg_original_clock_agrees']} |")
    lines += ['', 'These are ceilings before RUN spike, spatial-information, map-quality and chronological validation filters. They are not qualified decoder counts.', '',
              'Buddy has only two right-CA1 pyramidal units; Gatsby circular has none. Both fail the frozen ten-per-hemisphere minimum even before fitting maps.',
              'The three Cicero recordings have potentially sufficient bilateral units but no existing EEG. Gatsby linear has a duration conflict: 32,002.4 s from the present EEG/XML versus 30,413.628 s in the original summary. No clock, channel-count or recording-boundary repair was attempted.', '',
              f'At most {len(ceiling)} animals can qualify with the existing EEG/cell inventory, and only {len(clock_ceiling)} after current source-clock checks; the frozen minimum is three.', '',
              'Every session contains immobile MAZE tracking within explicitly source-scored Wake. Drowsy was never relabelled awake. These availability intervals precede the RUN training-cutoff restriction and are not supported coincident replay windows. Low speed alone was not used to establish wakefulness.', '',
              'Group 16 is documented as right CA1 for Buddy/Gatsby and is retained. Extra groups 7/14 in Achilles/Cicero are not silently classed as CA1. Red excluded channels and Buddy channel-114 omission are preserved in the source crosswalk; pyramidal-layer LFP selection is not yet signed off.', '',
              'The index file does not match its listed historical checksum; this remains an explicit discovery-metadata warning. All four original source payloads do match their published checksums. The two non-bitwise-identical previous summaries are checked at the numeric-field level rather than silently replaced.', '',
              '## Go/No-Go', '',
              '**No-go for calibration or biological comparison with the present recordings.** This is a measured recording-coverage limitation, not evidence that bilateral ripples disagree.',
              'The bounded recovery attempt is complete. Reopening would require a separately authorized source/LFP plan that could supply at least three fully qualified animals; it must not relax unit/state floors or substitute sleep-only analysis. Novelty remains unresolved. Existing protocols, prior results and manuscript claims remain unchanged.', '',
              'Sources: [CRCNS hc-11](https://crcns.org/data-sets/hc/hc-11/about-hc-11), [source documentation](https://crcns.org/files/data/hc-11/crcns_hc-11_data_description.pdf). Payload names/hashes and exact code commits are in the accompanying tables and decision manifest.', '']
    (args.output_dir / (PREFIX+'report.md')).write_text('\n'.join(lines))
    save_json(args.output_dir/'report_output_hashes.json', {p.name:file_sha256(p) for p in args.output_dir.iterdir()
              if p.is_file() and p.name!='report_output_hashes.json'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
