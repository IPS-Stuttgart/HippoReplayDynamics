#!/usr/bin/env python3
"""Simulation-only population recovery with marginalized unknown replay paths."""

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

from hipporeplayimm.lagged_neural_prediction import full_count_bins
from hipporeplayimm.literal_replay_clock import SpatialRates, clock_coordinates, normalize, sample_geometry
from hipporeplayimm.metric_population_recovery import population_fit
from hipporeplayimm.unknown_path_clocks import EVENTS, IDS, MODELS, PATHS, REPEATS, SCENARIOS, SUPPORTS, TEACHERS, exact_bin_average, generate_counts, rng, score_batch, summarize
from scripts._provenance import build_script_provenance, file_sha256


def record_task(item, source, output, n_paths=PATHS, repeats=REPEATS, events=EVENTS, supports=SUPPORTS):
    started = time.monotonic()
    tag = item["tag"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
        profiles = {}
        for name in sorted(k for k in z.files if k.startswith("counts_")):
            event = int(name.split("_")[1])
            counts, _ = full_count_bins(z[name], z[f"edges_{event}"])
            if len(counts) >= 3:
                profiles[event] = counts.sum(axis=1).astype(np.int32)
    if not profiles:
        raise ValueError("native count profiles missing")
    spatial, libraries, saved, path_rows = SpatialRates(centers, rates), [], {}, []
    for bank in range(2):
        library = []
        for h in range(n_paths):
            points, values, attempts = sample_geometry(spatial, rng(tag, bank, h, "geometry"), curved=bool(h % 2))
            s, c, clocks = clock_coordinates(points, values)
            library.append((values, clocks))
            saved[f"points_{bank}_{h}"] = points
            for model, clock in clocks.items():
                saved[f"clock_{model}_{bank}_{h}"] = clock
            path_rows.append({"bank": bank, "path_index": h, "attempts": attempts, "length_cm": s[-1], "code_arc_length": c[-1]})
        libraries.append(library)
    np.savez_compressed(output / f"{tag}_libraries.npz", **saved)
    pd.DataFrame(path_rows).to_csv(output / f"{tag}_libraries.csv", index=False)
    del saved
    static = normalize(rates.T)
    log_static = np.log(static)
    cached = {}

    def probabilities(bank, n):
        if (bank, n) not in cached:
            cached[bank, n] = {model: np.array([normalize(exact_bin_average(clocks[model], values, n)) for values, clocks in libraries[bank]]) for model in MODELS[:2]}
        return cached[bank, n]

    n_rows, n_observations = 0, 0
    for repeat in range(repeats):
        observations, latent, metadata, offsets = [], [], [], [0]
        for scenario in SCENARIOS:
            profiles_selected = rng(tag, repeat, scenario, "profiles").choice(sorted(profiles), events, replace=True)
            labels = rng(tag, repeat, scenario, "labels").choice(len(MODELS), events, p=[0.6 * (1 - scenario), 0.6 * scenario, 0.2, 0.1, 0.1])
            for bank, teacher in enumerate(TEACHERS):
                for j, (event, label) in enumerate(zip(profiles_selected, labels, strict=True)):
                    totals = profiles[int(event)]
                    p = probabilities(bank, len(totals))
                    x, index = generate_counts(totals, MODELS[label], p["physical"], p["neural"], static, rng(tag, repeat, scenario, teacher, j, "observation"))
                    observations.append(x)
                    latent.append(index)
                    offsets.append(offsets[-1] + len(x))
                    metadata.append(
                        {
                            **{k: item[k] for k in IDS},
                            "repeat": repeat,
                            "scenario": scenario,
                            "teacher": teacher,
                            "event_in_population": j,
                            "observation_index": len(observations) - 1,
                            "template_event": int(event),
                            "generator": MODELS[label],
                            "n_bins": len(x),
                            "n_spikes": int(x.sum()),
                        }
                    )
        frame = pd.DataFrame(metadata)
        scored = {h: np.zeros((len(frame), len(MODELS))) for h in supports}
        for n, group in frame.groupby("n_bins"):
            indices = group.index.to_numpy()
            p = probabilities(0, n)
            batch = score_batch(np.stack([observations[i] for i in indices]), np.log(p["physical"]), np.log(p["neural"]), log_static, supports)
            for h in supports:
                scored[h][indices] = batch[h]
        tables = []
        for h in supports:
            table = frame.assign(support=h)
            table[["score_" + m for m in MODELS]] = scored[h]
            tables.append(table)
        table = pd.concat(tables, ignore_index=True)
        table.to_csv(output / f"{tag}_r{repeat:03d}_scores.csv.gz", index=False)
        np.savez_compressed(output / f"{tag}_r{repeat:03d}_observations.npz", counts=np.concatenate(observations), latent=np.concatenate(latent), offsets=np.array(offsets))
        n_rows += len(table)
        n_observations += len(frame)
    return {
        **{k: item[k] for k in IDS},
        "tag": tag,
        "source_events": item.get("events"),
        "n_rows": n_rows,
        "n_observations": n_observations,
        "n_library_paths": n_paths * 2,
        "runtime_s": time.monotonic() - started,
    }


def fit_repeat(repeat, items, out):
    table = pd.concat([pd.read_csv(out / f"{item['tag']}_r{repeat:03d}_scores.csv.gz") for item in items], ignore_index=True)
    rows = []
    for key, group in table.groupby(["dataset", "scenario", "teacher", "support"]):
        if group.duplicated(IDS + ["event_in_population"]).any():
            raise ValueError("duplicate population observations")
        row = dict(zip(["dataset", "scenario", "teacher", "support"], key, strict=True))
        rows.append(
            row
            | {"repeat": repeat, "n_events": len(group), "n_recordings": group.session.nunique(), "n_animals": group.animal.nunique()}
            | population_fit(group[["score_" + m for m in MODELS]].to_numpy(), MODELS, "coherent")
        )
    return rows


def run(args):
    source = args.source_dir
    mp = source / "conditional_2d_manifest.json"
    parent = json.loads(mp.read_text())
    items = sorted(parent["completed"], key=lambda x: (x["dataset"], x["animal"], x["session"]))
    if parent["status"] != "complete" or len(items) != 33 or len({x["animal"] for x in items}) != 9:
        raise ValueError("all 33 complete source recordings required")
    for item in items:
        item["tag"] = "__".join(str(item[k]).replace("/", "_") for k in IDS)
        name = item["tag"] + "_cache.npz"
        if file_sha256(source / name) != parent["output_sha256"][name]:
            raise ValueError("source cache hash mismatch")
    provenance = build_script_provenance(input_paths={"source_manifest": mp, "protocol": ROOT / "docs/unknown_path_clock_population_protocol.md"}, cwd=ROOT)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = provenance | {
        "status": "running",
        "source_dir": str(source),
        "completed": [],
        "real_events_rescored": False,
        "biological_mechanism_established": False,
        "oracle_knows_path": False,
        "library_paths": PATHS,
        "supports": SUPPORTS,
        "repeats": REPEATS,
        "events_per_recording": EVENTS,
        "models": MODELS,
    }
    path = out / "unknown_path_clock_manifest.json"
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_task, item, source, out) for item in items]
            for future in as_completed(futures):
                result = future.result()
                manifest["completed"].append(result)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        fits = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_repeat, r, items, out) for r in range(REPEATS)]
            for future in as_completed(futures):
                fits.extend(future.result())
                print(f"population fits {len(fits)}/1200", flush=True)
        for name, frame in zip(("fits", "summary", "gates"), summarize(pd.DataFrame(fits)), strict=True):
            frame.to_csv(out / f"unknown_path_clock_{name}.csv", index=False)
        manifest.update(
            status="complete",
            n_rows=sum(x["n_rows"] for x in manifest["completed"]),
            n_observations=sum(x["n_observations"] for x in manifest["completed"]),
            n_fits=len(fits),
            runtime_s=time.monotonic() - started,
        )
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p != path}
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
