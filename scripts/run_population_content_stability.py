#!/usr/bin/env python3
"""Run the frozen development/external sequence under a persistent user service."""
import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    measure = [sys.executable, "-u", str(code/"measure_population_content_stability.py")]
    report = [sys.executable, "-u", str(code/"validate_population_content_stability.py")]
    model = root/"frozen"/"frozen_model.json"
    stages = [
        ("pf_measure", measure+["--input-dir", str(args.input_dir), "--dataset", "pfeiffer_foster", "--output-dir", str(root/"pf")]),
        ("freeze", report+["freeze", "--input-dir", str(root/"pf"), "--output-dir", str(root/"frozen")]),
        ("tanni_measure", measure+["--input-dir", str(args.input_dir), "--dataset", "tanni2022", "--frozen-model", str(model), "--output-dir", str(root/"tanni")]),
        ("evaluate", report+["evaluate", "--input-dir", str(root/"tanni"), "--frozen-model", str(model), "--output-dir", str(root/"validation")]),
    ]
    status = dict(started_at_utc=datetime.now(UTC).isoformat(), status="running", stages=[])
    for name, command in stages:
        status["current_stage"] = name
        (root/"status.json").write_text(json.dumps(status, indent=2)+"\n")
        print(f"START {name}", flush=True)
        result = subprocess.run(command, check=False)
        status["stages"].append(dict(name=name, command=command, exit_code=result.returncode,
                                     completed_at_utc=datetime.now(UTC).isoformat()))
        if result.returncode:
            status["status"] = "failed"
            (root/"status.json").write_text(json.dumps(status, indent=2)+"\n")
            raise SystemExit(result.returncode)
    status["status"] = "complete"
    status["completed_at_utc"] = datetime.now(UTC).isoformat()
    (root/"status.json").write_text(json.dumps(status, indent=2)+"\n")


if __name__ == "__main__":
    main()
