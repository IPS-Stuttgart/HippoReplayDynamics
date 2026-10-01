#!/usr/bin/env python3
"""Run a committed post-error study stage on gpuserver6000, detached from SSH."""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def record(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop", action="store_true", help="Stop only this launcher's recorded process group and preserve terminal status")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] not in {"gpuserver6000", "workstation2"}:
        parser.error("This experiment is restricted to gpuserver6000")
    if args.stop:
        launch = json.loads((args.job_dir / "launch.json").read_text())
        pid = int(launch["supervisor_pid"])
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\x00", " ")
        if "launch_denovellis_post_error_content.py --supervise --job-dir " + str(args.job_dir) not in cmdline or os.getpgid(pid) != pid:
            parser.error("Recorded PID does not match this job's isolated supervisor")
        os.killpg(pid, signal.SIGTERM)
        record(args.job_dir / "status.json", {"status": "terminated_for_implementation_fix", "finished_at_utc": datetime.now(UTC).isoformat(), "reason": "restart same estimator with batched numerical evaluation; old logs preserved"})
        return
    if args.supervise:
        job = json.loads((args.job_dir / "job.json").read_text())
        record(args.job_dir / "status.json", {"status": "running", "pid": os.getpid()})
        try:
            code = subprocess.call(job["command"], cwd=ROOT, stdin=subprocess.DEVNULL)
            record(args.job_dir / "status.json", {"status": "completed" if code == 0 else "failed", "exit_code": code, "finished_at_utc": datetime.now(UTC).isoformat()})
        except Exception as exc:
            record(args.job_dir / "status.json", {"status": "failed", "error": str(exc), "finished_at_utc": datetime.now(UTC).isoformat()})
            raise
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or command[0] != "analyze_denovellis_post_error_content.py":
        parser.error("Only the staged post-error entry point may run through this launcher")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Commit the isolated checkout before launching")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.job_dir = args.job_dir.resolve()
    args.job_dir.mkdir(parents=True, exist_ok=False)
    record(args.job_dir / "job.json", {"code_commit": commit, "ssh_alias": "gpuserver6000", "os_hostname": socket.gethostname(),
                                      "command": [sys.executable, "-u", str(ROOT / "scripts" / command[0]), *command[1:]], "created_at_utc": datetime.now(UTC).isoformat()})
    record(args.job_dir / "status.json", {"status": "launching"})
    with (args.job_dir / "job.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", "--job-dir", str(args.job_dir)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record(args.job_dir / "launch.json", {"supervisor_pid": process.pid})
    print(json.dumps({"job_dir": str(args.job_dir), "supervisor_pid": process.pid}))


if __name__ == "__main__":
    main()
