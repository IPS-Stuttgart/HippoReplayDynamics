#!/usr/bin/env python3
"""Fresh simulated population mixtures; no biological replay scoring."""

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

from hipporeplayimm.conditional_spatial_prediction import identity_likelihood
from hipporeplayimm.lagged_neural_prediction import full_count_bins
from hipporeplayimm.metric_finite_recovery import reconstruct_kernel, rng_for, sample_identities, sample_path
from hipporeplayimm.metric_oracle_recovery import whole_evidence
from hipporeplayimm.metric_population_recovery import CONDITIONS, MODELS, REPEATS, SCENARIOS, SIZES, population_fit, summarize
from hipporeplayimm.physical_neural_metric import neural_cost, physical_cost
from scripts._provenance import build_script_provenance, file_sha256

IDS = ["dataset", "animal", "session"]
MAX_SIZE = max(SIZES)


def rng(*parts):
    return rng_for("population_v1", *parts)


def record_task(item, source, metric, output, repeats=REPEATS, size=MAX_SIZE):
    started = time.monotonic()
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        train, held = z["train_0"], z["held_0"]
        profiles = {}
        for key in sorted(k for k in z.files if k.startswith("counts_")):
            event = int(key.split("_")[1])
            counts, _ = full_count_bins(z[key], z[f"edges_{event}"])
            if len(counts) >= 3:
                profiles[event] = counts.astype(np.int32)
    if not profiles or not np.array_equal(np.sort(np.r_[train, held]), np.arange(len(rates))):
        raise ValueError("eligible profiles and complete source cell partition required")
    with np.load(metric / f"{tag}_kernel_parameters.npz") as z:
        physical = reconstruct_kernel(physical_cost(centers), z, "physical")
        neural = reconstruct_kernel(neural_cost(rates), z, "full_neural")
        estimated = reconstruct_kernel(neural_cost(rates[train]), z, "train_neural_0")
    teachers = {"physical": physical, "neural": neural, "stationary": np.eye(len(neural)), "iid": np.full_like(neural, 1 / len(neural))}
    gains = np.exp(0.35 * rng(tag, "gains").standard_normal(len(rates)) - 0.35**2 / 2)
    np.savez_compressed(output / f"{tag}_parameters.npz", gains=gains)
    rows = 0
    for repeat in range(repeats):
        chosen = rng(tag, repeat, "profiles").choice(sorted(profiles), size, replace=True)
        metadata, states, totals, offsets = [], [], [], [0]
        for phi in SCENARIOS:
            labels = rng(tag, repeat, phi, "labels").choice(MODELS, size, p=[0.6 * (1 - phi), 0.6 * phi, 0.2, 0.2])
            for i, (event, label) in enumerate(zip(chosen, labels, strict=True)):
                profile = profiles[int(event)]
                path = sample_path(teachers[label], len(profile), rng(tag, repeat, phi, i, "path"))
                metadata.append(
                    {
                        **{k: item[k] for k in IDS},
                        "repeat": repeat,
                        "scenario": phi,
                        "event_in_recording": i,
                        "simulation_index": len(metadata),
                        "generator": label,
                        "template_event": int(event),
                        "n_bins": len(profile),
                        "n_spikes": int(profile.sum()),
                    }
                )
                states.append(path)
                totals.append(profile)
                offsets.append(offsets[-1] + len(profile))
        states, totals, offsets = np.concatenate(states), np.concatenate(totals), np.asarray(offsets)
        observations = {"states": states, "offsets": offsets}
        scores = []
        meta = pd.DataFrame(metadata)
        for emitter in ("base", "gain"):
            truth = rates if emitter == "base" else rates * gains[:, None]
            x = sample_identities(truth, train, states, totals[:, train].sum(axis=1), rng(tag, repeat, emitter, "train"))
            y = sample_identities(truth, held, states, totals[:, held].sum(axis=1), rng(tag, repeat, emitter, "held"))
            observations[emitter + "_train"], observations[emitter + "_held"] = x, y
            ll = identity_likelihood(x, rates[train]) + identity_likelihood(y, rates[held])
            full = whole_evidence(ll, offsets, {"physical": physical, "neural": neural})
            for condition in ("exact", "train_geometry") if emitter == "base" else ("gain_drift",):
                current = full if condition != "train_geometry" else full | whole_evidence(ll, offsets, {"neural": estimated})
                table = meta.copy()
                table["condition"] = condition
                for m in MODELS:
                    table["score_" + m] = current[m]
                scores.append(table)
        prefix = output / f"{tag}_r{repeat:03d}"
        meta.to_csv(str(prefix) + "_events.csv", index=False)
        np.savez_compressed(str(prefix) + "_observations.npz", **observations)
        table = pd.concat(scores, ignore_index=True)
        values = table[["score_" + m for m in MODELS]].to_numpy()
        if not np.isfinite(values).all() or (values > 1e-8).any() or len(table) != size * len(SCENARIOS) * len(CONDITIONS):
            raise ValueError("invalid population likelihood table")
        table.to_csv(str(prefix) + "_scores.csv.gz", index=False)
        rows += len(table)
    return {**{k: item[k] for k in IDS}, "tag": tag, "rows": rows, "eligible_profiles": len(profiles), "runtime_s": time.monotonic() - started}


def fit_task(dataset, repeat, items, run):
    items = [x for x in items if x["dataset"] == dataset]
    table = pd.concat([pd.read_csv(run / f"{x['tag']}_r{repeat:03d}_scores.csv.gz") for x in items], ignore_index=True)
    keys = IDS + ["scenario", "event_in_recording", "condition"]
    if len(table) != len(items) * max(SIZES) * len(SCENARIOS) * len(CONDITIONS) or table.duplicated(keys).any() or not table.repeat.eq(repeat).all():
        raise ValueError("incomplete or duplicated population cohort")
    rows = []
    for condition in CONDITIONS:
        for phi in SCENARIOS:
            for size in SIZES:
                group = table[(table.condition == condition) & (table.scenario == phi) & (table.event_in_recording < size)]
                sizes = group.groupby(IDS).size()
                if len(sizes) != len(items) or not sizes.eq(size).all():
                    raise ValueError("unbalanced recording coverage")
                result = population_fit(group[["score_" + m for m in MODELS]].to_numpy())
                rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "scenario": phi,
                        "events_per_recording": size,
                        "condition": condition,
                        "n_events": len(group),
                        "n_recordings": len(items),
                        "n_animals": group.animal.nunique(),
                        **result,
                    }
                )
    return rows


def run(args):
    parent_path = args.oracle_dir / "metric_oracle_manifest.json"
    parent, audit = json.loads(parent_path.read_text()), json.loads(args.oracle_audit.read_text())
    if parent["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(parent_path):
        raise ValueError("verified complete oracle parent required")
    for name, digest in parent["output_sha256"].items():
        if file_sha256(args.oracle_dir / name) != digest:
            raise ValueError("changed parent artifact")
    source, metric = Path(parent["source_dir"]), Path(parent["metric_dir"])
    items = sorted(parent["completed"], key=lambda x: x["tag"])
    if len(items) != 33 or len({x["animal"] for x in items}) != 9:
        raise ValueError("all source recordings required")
    sm = json.loads((source / "conditional_2d_manifest.json").read_text())
    mm = json.loads((metric / "physical_neural_metric_manifest.json").read_text())
    for item in items:
        for directory, name, hashes in ((source, item["tag"] + "_cache.npz", sm["output_sha256"]), (metric, item["tag"] + "_kernel_parameters.npz", mm["output_sha256"])):
            if file_sha256(directory / name) != hashes[name]:
                raise ValueError("changed source cache or metric")
    provenance = build_script_provenance(
        input_paths={
            "oracle_manifest": parent_path,
            "oracle_audit": args.oracle_audit,
            "source_manifest": source / "conditional_2d_manifest.json",
            "metric_manifest": metric / "physical_neural_metric_manifest.json",
            "protocol": ROOT / "docs/metric_population_recovery_protocol.md",
        },
        cwd=ROOT,
    )
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean frozen source commit required")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = provenance | {
        "status": "running",
        "source_dir": str(source),
        "metric_dir": str(metric),
        "completed": [],
        "fit_jobs_complete": 0,
        "real_events_rescored": False,
        "biological_mechanism_established": False,
        "repeats": REPEATS,
        "sizes": SIZES,
    }
    mp = out / "metric_population_manifest.json"
    started = time.monotonic()

    def write():
        mp.write_text(json.dumps(manifest, indent=2) + "\n")

    write()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_task, x, source, metric, out) for x in items]
            for future in as_completed(futures):
                result = future.result()
                manifest["completed"].append(result)
                write()
                print(json.dumps(result), flush=True)
        fits = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_task, d, r, items, out) for d in ("pfeiffer_foster", "tanni2022") for r in range(REPEATS)]
            for future in as_completed(futures):
                fits.extend(future.result())
                manifest["fit_jobs_complete"] += 1
                write()
                print("fit jobs", manifest["fit_jobs_complete"], flush=True)
        table, summary, gates = summarize(pd.DataFrame(fits))
        for name, frame in (("fits", table), ("summary", summary), ("gates", gates)):
            frame.to_csv(out / f"metric_population_{name}.csv", index=False)
        primary = gates[gates.events_per_recording.eq(max(SIZES))]
        decision = {
            "exact_population_recovery_pass": bool(primary[primary.condition.eq("exact")].practical_pass.all()),
            "robust_population_recovery_pass": bool(primary.practical_pass.all()),
            "real_events_rescored": False,
            "biological_mechanism_established": False,
            "new_real_scoring_authorized": False,
        }
        pd.DataFrame([decision]).to_csv(out / "metric_population_decision.csv", index=False)
        manifest.update(status="complete", rows=sum(x["rows"] for x in manifest["completed"]), fit_rows=len(table), runtime_s=time.monotonic() - started)
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p != mp}
        write()
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), runtime_s=time.monotonic() - started)
        write()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--oracle-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
