#!/usr/bin/env python3
"""Detached, committed odor-place feasibility stages with durable terminal status."""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.odor_place_source_io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stages", nargs="+", choices=["inventory", "verify", "acquire-neural", "run-qc", "report"],
                        help="Run ordered feasibility stages using common entry-point arguments")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] not in {"gpuserver6000", "workstation2"}:
        parser.error("This study is restricted to gpuserver6000")
    if args.supervise:
        job = json.loads((args.job_dir / "job.json").read_text())
        atomic_json(args.job_dir / "status.json", {"status": "running", "pid": os.getpid()})
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
            if commit != job["code_commit"] or dirty:
                raise ValueError("Checkout changed after launch")
            commands = job.get("commands", [job["command"]])
            completed = []
            code = 0
            for command in commands:
                atomic_json(args.job_dir / "status.json", {"status": "running", "pid": os.getpid(), "command": command,
                                                           "completed_stages": completed})
                code = subprocess.call(command, cwd=ROOT, stdin=subprocess.DEVNULL)
                if code:
                    break
                completed.append(command[3] if job.get("commands") else "single_stage")
            atomic_json(args.job_dir / "status.json", {"status": "completed" if code == 0 else "failed", "exit_code": code,
                                                       "completed_stages": completed,
                                                       "finished_at_utc": datetime.now(UTC).isoformat()})
        except Exception as exc:
            atomic_json(args.job_dir / "status.json", {"status": "failed", "error": str(exc), "finished_at_utc": datetime.now(UTC).isoformat()})
            raise
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or command[0] != "audit_odor_place_post_error_feasibility.py":
        parser.error("Only the non-association feasibility entry point is allowed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Commit the isolated checkout before launching")
    args.job_dir = args.job_dir.resolve()
    args.job_dir.mkdir(parents=True, exist_ok=False)
    entry = [sys.executable, "-u", str(ROOT / "scripts" / command[0])]
    commands = [[*entry, stage, *command[1:]] for stage in args.stages] if args.stages else None
    atomic_json(args.job_dir / "job.json", {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                           "hostname": socket.gethostname(), "created_at_utc": datetime.now(UTC).isoformat(),
                                           "command": [*entry, *command[1:]], **({"commands": commands} if commands else {})})
    with (args.job_dir / "job.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", "--job-dir", str(args.job_dir)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    atomic_json(args.job_dir / "launch.json", {"supervisor_pid": process.pid})
    print(json.dumps({"job_dir": str(args.job_dir), "supervisor_pid": process.pid}))


if __name__ == "__main__":
    main()
