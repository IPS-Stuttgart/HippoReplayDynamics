"""Run frozen split/repeat jobs in bounded server subprocesses with terminal status."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def run(banks, output, workers):
    if workers < 1 or workers > 8:
        raise ValueError("use 1-8 workers")
    if output.exists() and any(output.iterdir()):
        raise ValueError("new campaign directory required")
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "docs/two_track_content_configuration.json").read_text())
    sessions = [json.loads((b / "manifest.json").read_text())["session"] for b in banks]
    if len(sessions) != len(set(sessions)):
        raise ValueError("duplicate session banks")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("commit implementation before the scientific run")
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "running",
        "pid": os.getpid(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "banks": [str(b.resolve()) for b in banks],
        "completed": [],
        "failed": [],
        "expected_jobs": len(banks) * config["cell_splits"] * config["subsample_repeats"] * 2,
    }
    write_json(output / "campaign_status.json", status)
    jobs = []
    for bank, session in zip(banks, sessions, strict=True):
        for stratum in ("ripple", "mua_only"):
            for split in range(config["cell_splits"]):
                for repeat in range(config["subsample_repeats"]):
                    jobs.append((bank, session, stratum, split, repeat))

    def execute(job):
        bank, session, stratum, split, repeat = job
        name = f"{session}_{stratum}_s{split}_r{repeat}"
        path = output / name
        cmd = [
            sys.executable,
            "-u",
            str(repo / "scripts/score_tirole_content_coverage.py"),
            "--bank-dir",
            str(bank),
            "--output-dir",
            str(path),
            "--split",
            str(split),
            "--repeat",
            str(repeat),
            "--candidate-stratum",
            stratum,
        ]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with (output / (name + ".log")).open("xb") as log:
            p = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / (name + "_job.json"), {"pid": p.pid, "command": cmd})
            code = p.wait()
        valid = False
        if code == 0 and (path / "manifest.json").exists():
            m = json.loads((path / "manifest.json").read_text())
            valid = m["status"] == "complete" and not m["technical_pilot_only"] and m["selected_events"] > 0 and m["score_rows"] == m["expected_score_rows"] and not m["git_dirty"]
        return {"job": name, "exit_code": code, "valid_terminal_manifest": valid}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(execute, j) for j in jobs]):
            item = future.result()
            status["completed" if item["valid_terminal_manifest"] else "failed"].append(item)
            write_json(output / "campaign_status.json", status)
            print(json.dumps({"completed": len(status["completed"]), "failed": len(status["failed"]), "expected": status["expected_jobs"]}), flush=True)
    status["status"] = "complete" if not status["failed"] and len(status["completed"]) == status["expected_jobs"] else "failed"
    status["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "campaign_status.json", status)
    if status["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    run(a.bank_dirs, a.output_dir, a.workers)
