#!/usr/bin/env python3
"""Exact finite-prior clock calibration on two predeclared recording maps."""

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

from hipporeplayimm.exact_path_clocks import merge_blocks, score_geometry_block
from hipporeplayimm.literal_replay_clock import SpatialRates
from hipporeplayimm.metric_population_recovery import population_fit
from hipporeplayimm.unknown_path_clocks import MODELS, summarize
from scripts._provenance import build_script_provenance, file_sha256
from scripts.run_dense_path_clock_recovery import load_observations

TAGS = ("pfeiffer_foster__Rat1__Rat1_Open1", "tanni2022__R2470__2018-08-10_18-11-37")
BLOCK = 16
SUPPORTS = tuple(f"mc_{h}_b{b}" for h in (1024, 4096, 8192) for b in (0, 1)) + ("exhaustive",)
TEACHERS = ("original_bank_0", "original_bank_1")


def load_groups(run, tag):
    with np.load(run / f"{tag}_counts.npz") as z:
        return {int(k.split("_")[1]): (z[k.replace("counts_", "index_")], z[k]) for k in z.files if k.startswith("counts_")}


def block_task(tag, start, stop, source, output):
    t = time.monotonic()
    groups = load_groups(output, tag)
    with np.load(source / f"{tag}_cache.npz") as z:
        spatial = SpatialRates(z["centers"], z["rates"])
    result = score_geometry_block(spatial, groups, start, stop)
    name = f"{tag}_block_{start:04d}.npz"
    np.savez(output / name, **result)
    return {
        "tag": tag,
        "start": start,
        "stop": stop,
        "file": name,
        "accepted": result["accepted"].tolist(),
        "proposed": result["proposed"].tolist(),
        "runtime_s": time.monotonic() - t,
    }


def fit_repeat(repeat, output, dense, tags=TAGS):
    rows = []
    for tag in tags:
        metadata = pd.read_csv(output / f"{tag}_observations.csv.gz")
        metadata = metadata[metadata.repeat.eq(repeat)]
        exact = np.load(output / f"{tag}_exact_scores.npy", mmap_mode="r")
        mc = np.load(dense / f"{tag}_scores.npy", mmap_mode="r")
        for (dataset, animal, session, teacher, scenario), group in metadata.groupby(["dataset", "animal", "session", "source_teacher", "scenario"]):
            index = group.row_index.to_numpy()
            matrices = {"exhaustive": exact[index]}
            matrices.update({f"mc_{h}_b{b}": mc[b, hi, index] for hi, h in enumerate((1024, 4096, 8192)) for b in (0, 1)})
            for support, ll in matrices.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "animal": animal,
                        "session": session,
                        "teacher": teacher,
                        "scenario": scenario,
                        "repeat": repeat,
                        "support": support,
                        "n_events": len(group),
                    }
                    | population_fit(ll, MODELS, "coherent")
                )
    return rows


def run(args):
    parent, dense, output = args.parent_dir, args.dense_dir, args.output_dir
    pp, dp, ap = parent / "unknown_path_clock_manifest.json", dense / "dense_path_clock_manifest.json", args.dense_audit_dir / "dense_path_clock_audit.json"
    pm, dm, audit = [json.loads(p.read_text()) for p in (pp, dp, ap)]
    if (
        pm["status"] != "complete"
        or dm["status"] != "complete"
        or audit["status"] != "pass"
        or audit["input_file_sha256"]["run_manifest"] != file_sha256(dp)
        or dm["input_file_sha256"]["parent_manifest"] != file_sha256(pp)
    ):
        raise ValueError("audited frozen parent and dense-path run required")
    source = Path(pm["source_dir"])
    sm = source / "conditional_2d_manifest.json"
    if file_sha256(sm) != pm["input_file_sha256"]["source_manifest"]:
        raise ValueError("encoding source manifest differs")
    source_hashes = json.loads(sm.read_text())["output_sha256"]
    for tag in TAGS:
        name = tag + "_cache.npz"
        if file_sha256(source / name) != source_hashes[name]:
            raise ValueError("source encoder hash differs")
        for repeat in range(50):
            for tail in ("scores.csv.gz", "observations.npz"):
                name = f"{tag}_r{repeat:03d}_{tail}"
                if file_sha256(parent / name) != pm["output_sha256"][name]:
                    raise ValueError("frozen observations differ")
        for tail in ("scores.npy", "observations.csv.gz"):
            name = f"{tag}_{tail}"
            if file_sha256(dense / name) != dm["output_sha256"][name]:
                raise ValueError("dense comparison input differs")
    manifest = build_script_provenance(input_paths={"parent": pp, "dense": dp, "dense_audit": ap, "protocol": ROOT / "docs/exact_path_clock_recovery_protocol.md"}, cwd=ROOT)
    if manifest["git_dirty"] or manifest["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    output.mkdir(parents=True, exist_ok=False)
    manifest.update(
        status="running",
        tags=TAGS,
        source_dir=str(source),
        parent_dir=str(parent),
        dense_dir=str(dense),
        blocks=[],
        records=[],
        exhaustive_finite_prior=True,
        all_33_encoders=False,
        real_events_rescored=False,
        oracle_knows_path=False,
        workers=args.workers,
        block_starts=BLOCK,
    )
    path = output / "exact_path_clock_manifest.json"
    t = time.monotonic()
    try:
        tasks = []
        for tag in TAGS:
            metadata, groups = load_observations(tag, parent)
            reference = pd.read_csv(dense / f"{tag}_observations.csv.gz")
            pd.testing.assert_frame_equal(metadata, reference, check_dtype=False)
            metadata.to_csv(output / f"{tag}_observations.csv.gz", index=False)
            arrays = {f"{label}_{n}": value for n, pair in groups.items() for label, value in zip(("index", "counts"), pair, strict=True)}
            np.savez(output / f"{tag}_counts.npz", **arrays)
            with np.load(source / f"{tag}_cache.npz") as z:
                n = len(z["centers"])
            tasks.extend((tag, a, min(a + BLOCK, n), source, output) for a in range(0, n, BLOCK))
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(block_task, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                manifest["blocks"].append(result)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        for tag in TAGS:
            groups = load_groups(output, tag)
            blocks = sorted((b for b in manifest["blocks"] if b["tag"] == tag), key=lambda b: b["start"])
            with np.load(source / f"{tag}_cache.npz") as z:
                rates = z["rates"]

            def partials(blocks=blocks):
                for b in blocks:
                    with np.load(output / b["file"]) as partial:
                        yield partial

            scores, totals = merge_blocks(partials(), groups, rates, sum(len(x) for _index, x in groups.values()))
            np.save(output / f"{tag}_exact_scores.npy", scores)
            manifest["records"].append({"tag": tag, "observations": len(scores), "accepted_straight": int(totals[0]), "accepted_curved": int(totals[1]), "n_blocks": len(blocks)})
            print(json.dumps(manifest["records"][-1]), flush=True)
        fits = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(fit_repeat, range(50), [output] * 50, [dense] * 50):
                fits.extend(rows)
        for name, frame in zip(("fits", "summary", "gates"), summarize(pd.DataFrame(fits), supports=SUPPORTS, teachers=TEACHERS), strict=True):
            frame.to_csv(output / f"exact_path_clock_{name}.csv", index=False)
        manifest.update(status="complete", n_observations=sum(r["observations"] for r in manifest["records"]), n_fits=len(fits), runtime_s=time.monotonic() - t)
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p != path}
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--dense-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
