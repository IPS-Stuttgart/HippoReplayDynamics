#!/usr/bin/env python3
"""Launch the committed Igata feasibility audit on gpuserver6000, detached from SSH."""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", type=Path, required=True)
    p.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if socket.gethostname().split(".")[0] != "gpuserver6000":
        raise SystemExit("This experiment is restricted to gpuserver6000.")
    if a.supervise:
        job = json.loads((a.job_dir / "job.json").read_text())
        record(a.job_dir / "status.json", {"status": "running", "pid": os.getpid()})
        try:
            code = subprocess.call(job["command"], cwd=ROOT, stdin=subprocess.DEVNULL)
            record(a.job_dir / "status.json", {"status": "completed" if code == 0 else "failed", "exit_code": code, "finished_at_utc": datetime.now(UTC).isoformat()})
        except Exception as exc:
            record(a.job_dir / "status.json", {"status": "failed", "error": str(exc), "finished_at_utc": datetime.now(UTC).isoformat()})
            raise
        return
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or command[0] != "audit_igata_obsolete_route_relapse.py":
        raise SystemExit("Only the feasibility audit may run through this launcher.")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("Commit the isolated checkout before launching.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    a.job_dir = a.job_dir.resolve()
    a.job_dir.mkdir(parents=True, exist_ok=False)
    record(a.job_dir / "job.json", {"code_commit": commit, "command": [sys.executable, "-u", str(ROOT / "scripts" / command[0]), *command[1:]], "created_at_utc": datetime.now(UTC).isoformat()})
    record(a.job_dir / "status.json", {"status": "launching"})
    with (a.job_dir / "job.log").open("ab", buffering=0) as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", "--job-dir", str(a.job_dir)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record(a.job_dir / "launch.json", {"supervisor_pid": proc.pid})
    print(json.dumps({"job_dir": str(a.job_dir), "supervisor_pid": proc.pid, "log": str(a.job_dir / "job.log")}))


if __name__ == "__main__":
    main()
