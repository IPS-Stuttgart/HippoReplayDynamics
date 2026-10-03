#!/usr/bin/env python3
"""Durable detached supervisor for the five non-association feasibility stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _provenance import build_script_provenance, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def status(path, **values):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'updated_at_utc': datetime.now(UTC).isoformat(), **values}, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    provenance = build_script_provenance(input_paths={'protocol': args.protocol}, cwd=ROOT)
    if provenance['git_dirty'] is not False or provenance['code_commit'] == 'unavailable':
        raise ValueError('isolated clean committed checkout required')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / 'supervisor_status.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--dataset-root', str(args.dataset_root.resolve()),
               '--protocol', str(args.protocol.resolve()), '--output-dir', str(output), '--seed', str(args.seed)]
    if args.detach:
        if terminal.exists() and json.loads(terminal.read_text()).get('status') == 'running':
            raise ValueError('existing supervisor marked running; verify process before resuming')
        with (output / 'supervisor.log').open('ab') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        print(json.dumps({'detached_pid': child.pid, 'output_dir': str(output), 'code_commit': provenance['code_commit']}))
        return
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}
    exit_code = 1
    completed = []
    stage = None
    try:
        for stage in ('inventory', 'run-qc', 'opportunities', 'verify', 'report'):
            status(terminal, status='running', stage=stage, completed=completed, pid=os.getpid(), provenance=provenance)
            stage_command = [sys.executable, str(ROOT / 'scripts/audit_hc11_bilateral_content_feasibility.py'), '--stage', stage, *command[2:]]
            with (output / f'{stage}.log').open('ab') as log:
                result = subprocess.run(stage_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                exit_code = result.returncode
                raise RuntimeError(f'{stage} failed, see {stage}.log')
            completed.append(stage)
        exit_code = 0
    except Exception as exc:
        status(terminal, status='failed', stage=stage, completed=completed, pid=os.getpid(), exit_status=exit_code,
               error=str(exc), provenance=provenance)
        raise
    finally:
        if exit_code == 0:
            status(terminal, status='completed', completed=completed, pid=os.getpid(), exit_status=0, provenance=provenance,
                   output_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p != terminal})
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
