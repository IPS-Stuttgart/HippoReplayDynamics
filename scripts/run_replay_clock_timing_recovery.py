#!/usr/bin/env python3
"""Fresh paired 1-ms and 20-ms oracle observations; no real replay scoring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.literal_replay_clock import bin_average
from hipporeplayimm.replay_clock_timing import MODELS, REPEATS, SUBBINS, log_scores, probabilities, rng, sample, signed_credit
from scripts._provenance import build_script_provenance, file_sha256

IDS = ["dataset", "animal", "session"]


def record_task(item, parent, output, n_paths=32, repeats=REPEATS):
    tag = item["tag"]
    rows = []
    for i in range(n_paths):
        with np.load(parent / f"{tag}_p{i:03d}.npz") as z:
            totals = z["totals"]
            n = len(totals)
            gains = z["gains"]
            rates = {model: bin_average(z[f"clock_{model}"], z["path_rates"], n * SUBBINS, 16).reshape(n, SUBBINS, -1) for model in ["physical", "neural"]}
        candidates = {k: probabilities(v) for k, v in rates.items()}
        saved = {f"rates_{k}": v for k, v in rates.items()} | {"totals": totals}
        for condition in ["exact", "gain_drift"]:
            for truth in ["physical", "neural"]:
                emitted = probabilities(rates[truth] * (1 if condition == "exact" else gains))["fine_joint"]
                expected = {k: log_scores(emitted * totals[:, None, None], v) for k, v in candidates.items()}
                draws = []
                for repeat in range(repeats):
                    counts = sample(emitted, totals, rng(tag, i, condition, truth, repeat))
                    draws.append(counts)
                    scores = {k: log_scores(counts, v) for k, v in candidates.items()}
                    for model in MODELS:
                        delta = scores["neural"][model] - scores["physical"][model]
                        expected_delta = expected["neural"][model] - expected["physical"][model]
                        rows.append(
                            {
                                **{k: item[k] for k in IDS},
                                "path_id": i,
                                "condition": condition,
                                "generator": truth,
                                "repeat": repeat,
                                "observation": model,
                                "n_bins": n,
                                "n_spikes": int(totals.sum()),
                                "delta_neural_minus_physical": delta,
                                "oracle_correct": signed_credit(delta, truth),
                                "expected_signed_margin": expected_delta if truth == "neural" else -expected_delta,
                                "log_score_neural": scores["neural"][model],
                                "log_score_physical": scores["physical"][model],
                            }
                        )
                saved[f"counts_{condition}_{truth}"] = np.array(draws)
        np.savez_compressed(output / f"{tag}_p{i:03d}.npz", **saved)
    pd.DataFrame(rows).to_csv(output / f"{tag}_scores.csv.gz", index=False)
    return {**{k: item[k] for k in IDS}, "tag": tag, "n_paths": n_paths, "rows": len(rows), "observations": n_paths * 4 * repeats}


def summarize(table):
    keys = IDS + ["path_id", "condition", "generator", "repeat", "observation"]
    if table.empty or table.duplicated(keys).any():
        raise ValueError("unique nonempty observations required")
    counts = table.groupby(IDS + ["path_id", "condition", "generator", "repeat"]).observation.agg(set)
    if not counts.map(lambda x: x == set(MODELS)).all():
        raise ValueError("all four paired scores required")
    fields = ["oracle_correct", "expected_signed_margin"]
    path = table.groupby(IDS + ["path_id", "condition", "observation"], as_index=False)[fields].mean()
    record = path.groupby(IDS + ["condition", "observation"], as_index=False)[fields].mean()
    animal = record.groupby(["dataset", "animal", "condition", "observation"], as_index=False)[fields].mean()
    summary = animal.groupby(["dataset", "condition", "observation"], as_index=False)[fields].mean()
    minimum = animal.groupby(["dataset", "condition", "observation"], as_index=False).agg(min_animal_accuracy=("oracle_correct", "min"))
    summary = summary.merge(minimum, validate="one_to_one")
    summary["practical_pass"] = (summary.oracle_correct >= 0.80) & (summary.min_animal_accuracy > 0.50)
    paired = animal.pivot(index=["dataset", "animal", "condition"], columns="observation", values="oracle_correct").reset_index()
    paired["identity_timing_gain"] = paired.fine_identity - paired.coarse_identity
    paired["joint_timing_gain"] = paired.fine_joint - paired.coarse_identity
    return path, record, animal, summary, paired


def run(args):
    parent = args.parent_dir
    mp = parent / "literal_clock_manifest.json"
    m = json.loads(mp.read_text())
    a = json.loads(args.parent_audit.read_text())
    if m["status"] != "complete" or a["status"] != "pass" or not a["numerical_readiness_pass"] or a["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("verified complete parent required")
    for f, h in m["output_sha256"].items():
        if file_sha256(parent / f) != h:
            raise ValueError("parent changed")
    p = build_script_provenance(input_paths={"parent_manifest": mp, "parent_audit": args.parent_audit, "protocol": ROOT / "docs/replay_clock_timing_protocol.md"}, cwd=ROOT)
    if p["git_dirty"] or p["code_commit"] == "unavailable":
        raise ValueError("clean frozen producer required")
    items = m["completed"]
    if len(items) != 33:
        raise ValueError("all 33 parent recordings required")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    status = p | {"status": "running", "parent_dir": str(parent), "completed": [], "real_events_rescored": False, "oracle_knows_path": True, "repeats": REPEATS, "subbins": SUBBINS}
    path = out / "clock_timing_manifest.json"
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_task, x, parent, out) for x in items]
            for f in as_completed(futures):
                status["completed"].append(f.result())
                path.write_text(json.dumps(status, indent=2) + "\n")
                print(status["completed"][-1], flush=True)
        table = pd.concat([pd.read_csv(out / f"{x['tag']}_scores.csv.gz") for x in items], ignore_index=True)
        for name, df in zip(["paths", "recordings", "animals", "summary", "paired_animals"], summarize(table), strict=True):
            df.to_csv(out / f"clock_timing_{name}.csv", index=False)
        status.update(status="complete", rows=len(table), runtime_s=time.monotonic() - start)
        status["output_sha256"] = {f.name: file_sha256(f) for f in sorted(out.iterdir()) if f != path}
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--parent-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
