#!/usr/bin/env python3
"""One bounded hc-11 original-source recovery; no decoder or replay comparison."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import http.cookiejar
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _provenance import build_script_provenance, file_sha256

PREFIX = 'hc11_source_recovery_'


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_table(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise ValueError('unsafe source path')
    return path


def parse_index(text):
    if text.lstrip().startswith('<') or 'dataset files' not in text:
        raise ValueError('not an authenticated CRCNS file index')
    files = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.lstrip(' +').split()
        if len(parts) < 2 or not parts[1].isdigit():
            raise ValueError('unresolved source-index row')
        name, size = parts[0], int(parts[1])
        safe_relative(name)
        if name in files or size <= 0:
            raise ValueError('duplicate or invalid source-index entry')
        files[name] = size
    return files


def parse_checksums(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*').removeprefix('./')
        safe_relative(name)
        if not re.fullmatch('[0-9a-fA-F]{32}', digest) or name in result:
            raise ValueError('invalid/duplicate checksum entry')
        result[name] = digest.lower()
    return result


def digest_md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_reserve(path, remaining, reserve):
    if shutil.disk_usage(path).free < remaining + reserve:
        raise ValueError('insufficient download space plus frozen reserve')


class Client:
    def __init__(self, timeout, deadline):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        self.timeout, self.deadline = timeout, deadline

    def open(self, request, data=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('bounded recovery deadline reached')
        return self.opener.open(request, data=data, timeout=min(self.timeout, remaining))

    def small(self, url):
        with self.open(url) as response:
            value = response.read(1024 * 1024 + 1)
        if len(value) > 1024 * 1024:
            raise ValueError('metadata response exceeds bounded size')
        return value.decode('utf-8')

    def login(self, account_file):
        # Credentials stay in memory and never enter logs, hashes or artifacts.
        account = {}
        for line in Path(account_file).read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                account[key.strip()] = ast.literal_eval(value.strip())
        form = {'__ac_name': account['crcns_username'], '__ac_password': account['crcns_password'],
                'form.submitted': '1', 'submit': 'Log in'}
        with self.open('https://crcns.org/login_form', urllib.parse.urlencode(form).encode()) as response:
            response.read(1024 * 1024)


def download(client, url, path, size, expected_md5, reserve):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != size or digest_md5(path) != expected_md5:
            raise ValueError('existing source failed published size/checksum; no overwrite')
        return 'verified_existing'
    part = path.with_suffix(path.suffix + '.partial')
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError('partial source larger than published size')
    check_reserve(path.parent, size - offset, reserve)
    if offset < size:
        request = urllib.request.Request(url)
        if offset:
            request.add_header('Range', f'bytes={offset}-')
        with client.open(request) as response:
            if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('resume range not honored; partial preserved')
            if int(response.headers.get('Content-Length', -1)) != size - offset:
                raise ValueError('source length differs from published index')
            with part.open('ab' if offset else 'wb') as handle:
                transferred = offset
                while True:
                    if time.monotonic() > client.deadline:
                        raise TimeoutError('bounded recovery deadline reached')
                    chunk = response.read(min(1024 * 1024, size - transferred + 1))
                    if not chunk:
                        break
                    transferred += len(chunk)
                    if transferred > size:
                        raise ValueError('source body exceeds published size')
                    handle.write(chunk)
    if part.stat().st_size != size or digest_md5(part) != expected_md5:
        raise ValueError('download failed published size/checksum; partial preserved')
    part.replace(path)
    return 'downloaded_verified'


def extract_summaries(archive, destination, sessions, budget):
    expected = {session + '_sessInfo.mat' for session in sessions}
    extracted = []
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        if len(members) > 100:
            raise ValueError('unexpected source bundle member count')
        files = [member for member in members if not member.isdir()]
        names = [PurePosixPath(member.name).name for member in files]
        if set(names) != expected or len(names) != len(set(names)):
            raise ValueError('bundle does not contain exactly the eight documented summaries')
        if sum(member.size for member in files) > budget:
            raise ValueError('bundle exceeds frozen unpacked budget')
        for member in files:
            safe_relative(member.name)
            if not member.isfile():
                raise ValueError('source bundle contains non-regular member')
            target = destination / PurePosixPath(member.name).name
            with handle.extractfile(member) as source:
                value = source.read(member.size + 1)
            if len(value) != member.size:
                raise ValueError('source member size mismatch')
            digest = hashlib.sha256(value).hexdigest()
            if target.exists():
                if file_sha256(target) != digest:
                    raise ValueError('existing unpacked source differs; no overwrite')
            else:
                target.write_bytes(value)
            extracted.append({'filename': target.name, 'bytes': member.size, 'sha256': digest})
    return extracted


def recover(args, protocol):
    client = Client(protocol['socket_timeout_s'], time.monotonic() + protocol['wall_time_limit_s'])
    client.login(args.account_file)
    base = protocol['crcns_base_url']
    index_text = client.small(base + 'filelist.txt')
    checksums_text = client.small(base + 'checksums.md5')
    index, checksums = parse_index(index_text), parse_checksums(checksums_text)
    requested = protocol['allowed_files']
    if set(requested) - index.keys() or set(requested) - checksums.keys():
        raise ValueError('required source absent from published index/checksums')
    if sum(index[name] for name in requested) + len(index_text) + len(checksums_text) > protocol['network_budget_bytes']:
        raise ValueError('allowlisted sources exceed frozen network budget')
    downloads = args.source_dir / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    check_reserve(downloads, sum(index[name] for name in requested) + protocol['unpacked_budget_bytes'], protocol['reserve_bytes'])
    (downloads / 'filelist.txt').write_text(index_text)
    (downloads / 'checksums.md5').write_text(checksums_text)
    rows = []
    manifest = args.output_dir / (PREFIX + 'downloads.csv')
    for name in requested:
        path = downloads / name
        row = {'file': name, 'url': base + name, 'expected_bytes': index[name], 'expected_md5': checksums[name],
               'status': None, 'bytes': None, 'sha256': None}
        row['status'] = download(client, base + name, path, index[name], checksums[name], protocol['reserve_bytes'])
        row.update(bytes=path.stat().st_size, sha256=file_sha256(path))
        rows.append(row)
        write_table(manifest, rows, list(row))
        print(json.dumps(row), flush=True)
    destination = args.source_dir / 'original_summaries'
    destination.mkdir(exist_ok=True)
    members = extract_summaries(downloads / 'data/NoveltySessInfoMatFiles.tar.gz', destination,
                                protocol['sessions'], protocol['unpacked_budget_bytes'])
    write_table(args.output_dir / (PREFIX + 'bundle_members.csv'), members, ['filename', 'bytes', 'sha256'])
    for name in requested:
        if name.endswith('.pdf'):
            target = args.output_dir / (PREFIX + Path(name).stem + '.txt')
            subprocess.run(['pdftotext', '-layout', str(downloads / name), str(target)], check=True)


def wake_immobility(original, max_gap=0.1, max_speed_cm_s=4.0):
    """Availability only: documented meters, explicit Wake, no Drowsy substitution."""
    import numpy as np
    import _hc11_bilateral_feasibility as core

    position = original['Position']
    times = np.asarray(position['TimeStamps'], float).ravel()
    xy = np.asarray(position['TwoDLocation'], float).reshape(-1, 2)
    if len(times) != len(xy) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid original tracking chronology')
    delta = np.diff(times)
    valid = (delta <= max_gap) & np.isfinite(xy[:-1]).all(axis=1) & np.isfinite(xy[1:]).all(axis=1)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * 100 / delta
    keep = valid & (speed <= max_speed_cm_s)
    segments = np.column_stack([times[:-1][keep], times[1:][keep]])
    segments = core.intersection(segments, original['Epochs']['MazeEpoch'])
    segments = core.intersection(segments, original['Epochs']['Wake'])
    merged = []
    for start, end in segments:
        if merged and start == merged[-1][1]:
            merged[-1][1] = float(end)
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, float).reshape(-1, 2)


def inspect_sources(args, protocol):
    import numpy as np
    import _hc11_bilateral_feasibility as core

    records, states, units, pauses = [], [], [], []
    for session in protocol['sessions']:
        path = args.source_dir / 'original_summaries' / (session + '_sessInfo.mat')
        original = core.load_structure(path, 'sessInfo')
        converted_paths = sorted(args.dataset_root.rglob(session + '.spikes.cellinfo.mat'))
        if len(converted_paths) != 1:
            raise ValueError('missing or ambiguous converted spike source')
        converted = core.load_structure(converted_paths[0], 'spikes')
        crosswalk = core.original_crosswalk(original['Spikes'], converted)
        units.extend({'session': session, **row} for row in crosswalk)
        previous = sorted(args.dataset_root.rglob(session + '_sessInfo.mat'))
        if len(previous) > 1:
            raise ValueError('ambiguous previous original source')
        position = original['Position']
        row = {'session': session, 'animal_inventory_label': session.split('_')[0], 'original_recovered': True,
               'original_sha256': file_sha256(path), 'previous_original_present': bool(previous),
               'previous_original_identical': file_sha256(previous[0]) == file_sha256(path) if previous else None,
               'maze_type': str(position['MazeType']), 'converted_units': len(crosswalk),
               'timing_verified_units': sum(unit['timing_verified'] for unit in crosswalk),
               'hemisphere_verified': False, 'awake_rest_verified': False,
               'source_wake_immobile_maze_s': None, 'source_wake_immobile_segments_ge200ms': None,
               'wake_tracking_check_status': None,
               'reason': 'original_sources_recovered_independent_anatomy_and_state_review_required'}
        for label in ('Wake', 'Drowsy', 'NREM', 'Intermediate', 'REM', 'PREEpoch', 'MazeEpoch', 'POSTEpoch'):
            value = np.asarray(original['Epochs'][label], float).reshape(-1, 2)
            invalid = np.flatnonzero(~np.isfinite(value).all(axis=1) | (value[:, 1] <= value[:, 0]))
            try:
                valid = core.intervals(value)
                status, total = 'valid', float(np.diff(valid, axis=1).sum())
            except ValueError as exc:
                status, total = str(exc), None
            states.append({'session': session, 'label': label, 'rows': len(value), 'duration_s': total,
                           'invalid_row_indices': ';'.join(map(str, invalid)), 'status': status,
                           'awake_rest_verified': False})
        try:
            segments = wake_immobility(original)
            row.update(source_wake_immobile_maze_s=float(np.diff(segments, axis=1).sum()),
                       source_wake_immobile_segments_ge200ms=int(np.sum(np.diff(segments, axis=1).ravel() >= 0.2)),
                       wake_tracking_check_status='availability_only_before_RUN_cutoff_and_neural_qualification')
            pauses.extend({'session': session, 'start_s': float(a), 'end_s': float(b),
                           'duration_s': float(b-a), 'source_state': 'Wake', 'replay_opportunity_qualified': False}
                          for a, b in segments)
        except ValueError as exc:
            row['wake_tracking_check_status'] = str(exc)
        records.append(row)
        print(json.dumps(row), flush=True)
    write_table(args.output_dir / (PREFIX + 'recordings.csv'), records, list(records[0]))
    write_table(args.output_dir / (PREFIX + 'states.csv'), states, list(states[0]))
    write_table(args.output_dir / (PREFIX + 'units.csv'), units, list(units[0]))
    write_table(args.output_dir / (PREFIX + 'wake_immobile_segments.csv'), pauses,
                ['session', 'start_s', 'end_s', 'duration_s', 'source_state', 'replay_opportunity_qualified'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--account-file', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    return p


def main():
    args = parser().parse_args()
    protocol = json.loads(args.protocol.read_text())
    if file_sha256(ROOT / 'docs/hc11_bilateral_content_feasibility_protocol.json') != protocol['unchanged_feasibility_protocol_sha256']:
        raise ValueError('previous feasibility protocol changed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), *[value for value in sys.argv[1:] if value != '--detach']]
        with (args.output_dir / 'supervisor.log').open('ab') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, cwd=ROOT)
        print(json.dumps({'pid': child.pid, 'log': str(args.output_dir / 'supervisor.log')}))
        return
    status_path = args.output_dir / 'supervisor_status.json'
    if status_path.exists():
        raise ValueError('immutable attempt already exists; inspect terminal status, do not restart automatically')
    provenance = build_script_provenance(input_paths={'protocol': args.protocol}, cwd=ROOT)
    if provenance['git_dirty']:
        raise ValueError('source recovery requires clean committed checkout')
    status = {'status': 'running', 'pid': os.getpid(), 'started_at_utc': datetime.now(UTC).isoformat(),
              'provenance': provenance, 'python': sys.version, 'platform': platform.platform(),
              'dataset_root': str(args.dataset_root), 'source_dir': str(args.source_dir)}
    save_json(status_path, status)
    try:
        recover(args, protocol)
        inspect_sources(args, protocol)
        status.update(status='completed', exit_code=0)
    except Exception as exc:
        # Do not emit authentication response bodies or signed redirect URLs.
        status.update(status='failed', exit_code=1, error=type(exc).__name__ + ': ' +
                      (f'HTTP {exc.code}' if isinstance(exc, urllib.error.HTTPError) else str(exc)))
    status['finished_at_utc'] = datetime.now(UTC).isoformat()
    status['output_sha256'] = {p.name: file_sha256(p) for p in args.output_dir.iterdir()
                               if p.is_file() and p != status_path and p.name != 'supervisor.log'}
    save_json(status_path, status)
    print(json.dumps({'status': status['status'], 'exit_code': status['exit_code'], 'error': status.get('error')}))
    if status['exit_code']:
        raise SystemExit(status['exit_code'])


if __name__ == '__main__':
    main()
