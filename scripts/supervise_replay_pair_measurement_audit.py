"""Durable terminal status wrapper for the bounded measurement audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tanni-root", type=Path)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / "scripts/audit_replay_order_run_coordination.py"),
               "--dataset-root", str(args.dataset_root), "--protocol",
               str(repo / "docs/replay_order_run_coordination_protocol.json"),
               "--output-dir", str(args.run_root / "measurement")]
    if args.tanni_root is not None:
        command.extend(["--tanni-root", str(args.tanni_root)])
    status = {"status": "running", "pid": os.getpid(), "command": command,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    path = args.run_root / "terminal_status.json"
    if path.exists():
        raise ValueError("Do not reuse a supervisor run directory")
    path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        with (args.run_root / "measurement.log").open("x") as log:
            proc = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False)
        status.update(status="completed" if proc.returncode == 0 else "failed", exit_code=proc.returncode)
    except Exception as exc:
        status.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(status, indent=2) + "\n")
    return status["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
