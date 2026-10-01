#!/usr/bin/env python3
"""Run the three feasibility stages on gpuserver6000 independently of SSH."""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/audit_widloski_structural_updating_feasibility.py"


def record(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] not in {"gpuserver6000", "workstation2"}:
        parser.error("Feasibility execution is restricted to gpuserver6000")
    if args.supervise:
        job = json.loads((args.job_dir / "job.json").read_text())
        record(args.job_dir / "status.json", {"status": "running", "supervisor_pid": os.getpid(), "current_stage": "inventory"})
        try:
            for stage in ("inventory", "opportunities", "report"):
                record(args.job_dir / "status.json", {"status": "running", "supervisor_pid": os.getpid(), "current_stage": stage})
                code = subprocess.call([sys.executable, "-u", str(DRIVER), *job["arguments"], "--stage", stage], cwd=ROOT, stdin=subprocess.DEVNULL)
                if code:
                    record(args.job_dir / "status.json", {"status": "failed", "failed_stage": stage, "exit_code": code, "finished_at_utc": datetime.now(UTC).isoformat()})
                    return code
            record(
                args.job_dir / "status.json",
                {
                    "status": "completed",
                    "exit_code": 0,
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                    "note": "Technical completion is not a feasibility pass; read decision.json",
                },
            )
            return 0
        except Exception as exc:
            record(args.job_dir / "status.json", {"status": "failed", "error": str(exc), "finished_at_utc": datetime.now(UTC).isoformat()})
            raise
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Commit the isolated checkout before launch")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if "--output-dir" not in arguments or "--stage" in arguments:
        parser.error("Supply driver arguments with --output-dir but without --stage")
    args.job_dir = args.job_dir.resolve()
    args.job_dir.mkdir(parents=True, exist_ok=False)
    record(
        args.job_dir / "job.json",
        {
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "arguments": arguments,
            "hostname": socket.gethostname(),
            "working_directory": str(ROOT),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    record(args.job_dir / "status.json", {"status": "launching"})
    with (args.job_dir / "job.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--supervise", "--job-dir", str(args.job_dir)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    record(args.job_dir / "launch.json", {"supervisor_pid": process.pid})
    print(json.dumps({"job_dir": str(args.job_dir), "supervisor_pid": process.pid}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
