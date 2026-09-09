#!/usr/bin/env python3
"""Fresh-event-path calibration with the exhaustive scorer held unchanged."""

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

from hipporeplayimm.exact_path_clocks import merge_blocks
from hipporeplayimm.fresh_path_clocks import family_indices, generate_event, rng
from hipporeplayimm.literal_replay_clock import SpatialRates
from hipporeplayimm.metric_population_recovery import population_fit
from hipporeplayimm.unknown_path_clocks import MODELS, summarize
from scripts._provenance import build_script_provenance, file_sha256
from scripts.run_exact_path_clock_recovery import BLOCK, TAGS, block_task, load_groups

TEACHERS = ("fresh_stream_0", "fresh_stream_1")


def generate_repeat(tag, repeat, previous, source, output):
    metadata = pd.read_csv(output / f"{tag}_observations.csv.gz")
    meta = metadata[metadata.repeat.eq(repeat)].sort_values("row_index")
    prior = json.loads((previous / "exact_path_clock_manifest.json").read_text())
    with np.load(Path(prior["parent_dir"]) / f"{tag}_r{repeat:03d}_observations.npz") as z:
        old_counts, old_offsets = z["counts"], z["offsets"]
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
    spatial = SpatialRates(centers, rates)
    descriptors = np.load(output / f"{tag}_prior_descriptors.npy", mmap_mode="r")
    families = family_indices(descriptors)
    counts, latent, offsets = [], [], [0]
    for row in meta.itertuples():
        a, b = old_offsets[row.observation_index : row.observation_index + 2]
        totals = old_counts[a:b].sum(axis=1)
        generated, ids = generate_event(totals, row.generator, spatial, rates, descriptors, families, rng(tag, repeat, row.source_teacher, row.event_in_population, row.scenario))
        counts.append(generated)
        latent.append(ids)
        offsets.append(offsets[-1] + len(ids))
    name = f"{tag}_fresh_r{repeat:03d}.npz"
    np.savez(output / name, counts=np.concatenate(counts), latent=np.concatenate(latent), offsets=np.array(offsets), row_index=meta.row_index.to_numpy())
    return {"tag": tag, "repeat": repeat, "file": name, "n_observations": len(meta)}


def collect_counts(tag, output):
    records = {}
    for repeat in range(50):
        with np.load(output / f"{tag}_fresh_r{repeat:03d}.npz") as z:
            for row, a, b in zip(z["row_index"], z["offsets"][:-1], z["offsets"][1:], strict=True):
                records[int(row)] = z["counts"][a:b].copy()
    if set(records) != set(range(38400)):
        raise ValueError("complete fresh observations required")
    groups = {}
    for row in sorted(records):
        n = len(records[row])
        groups.setdefault(n, ([], []))
        groups[n][0].append(row)
        groups[n][1].append(records[row])
    np.savez(output / f"{tag}_counts.npz", **{f"{label}_{n}": np.asarray(value) for n, pair in groups.items() for label, value in zip(("index", "counts"), pair, strict=True)})


def fit_populations(repeat, output):
    rows = []
    for tag in TAGS:
        metadata = pd.read_csv(output / f"{tag}_observations.csv.gz")
        if repeat >= 0:
            metadata = metadata[metadata.repeat.eq(repeat)]
        scores = np.load(output / f"{tag}_exact_scores.npy", mmap_mode="r")
        for key, group in metadata.groupby(["dataset", "animal", "session", "source_teacher", "scenario"]):
            expected = 128 if repeat >= 0 else 6400
            if len(group) != expected:
                raise ValueError("complete source population required")
            rows.append(
                dict(zip(("dataset", "animal", "session", "teacher", "scenario"), key, strict=True))
                | {"repeat": repeat, "support": "exhaustive", "n_events": len(group)}
                | population_fit(scores[group.row_index.to_numpy()], MODELS, "coherent")
            )
    return rows


def run(args):
    previous, source_audit, output = args.previous_dir, args.previous_audit_dir, args.output_dir
    mp, ap = previous / "exact_path_clock_manifest.json", source_audit / "exact_path_clock_audit.json"
    old, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if old["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp) or tuple(old["tags"]) != TAGS:
        raise ValueError("matching audited exhaustive pilot required")
    for name, digest in old["output_sha256"].items():
        if file_sha256(previous / name) != digest:
            raise ValueError("prior run hash mismatch")
    source = Path(old["source_dir"])
    source_manifest = source / "conditional_2d_manifest.json"
    hashes = json.loads(source_manifest.read_text())["output_sha256"]
    parent = Path(old["parent_dir"])
    pm = parent / "unknown_path_clock_manifest.json"
    if file_sha256(pm) != old["input_file_sha256"]["parent"]:
        raise ValueError("source observation provenance changed")
    parent_manifest = json.loads(pm.read_text())
    if file_sha256(source_manifest) != parent_manifest["input_file_sha256"]["source_manifest"]:
        raise ValueError("source encoder manifest changed")
    parent_hashes = parent_manifest["output_sha256"]
    for tag in TAGS:
        p = source / f"{tag}_cache.npz"
        if file_sha256(p) != hashes[p.name]:
            raise ValueError("encoding map changed")
        for repeat in range(50):
            p = parent / f"{tag}_r{repeat:03d}_observations.npz"
            if file_sha256(p) != parent_hashes[p.name]:
                raise ValueError("source count profile changed")
    manifest = build_script_provenance(
        input_paths={"previous": mp, "previous_audit": ap, "source": source_manifest, "parent": pm, "protocol": ROOT / "docs/fresh_path_clock_recovery_protocol.md"}, cwd=ROOT
    )
    if manifest["git_dirty"] or manifest["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    output.mkdir(parents=True, exist_ok=False)
    manifest.update(
        status="running",
        tags=TAGS,
        source_dir=str(source),
        previous_dir=str(previous),
        parent_dir=str(parent),
        generations=[],
        blocks=[],
        records=[],
        fresh_path_per_observation=True,
        finite_teacher_bank=False,
        real_events_rescored=False,
        all_33_encoders=False,
        oracle_knows_path=False,
        workers=args.workers,
    )
    path = output / "fresh_path_clock_manifest.json"
    start = time.monotonic()
    try:
        for tag in TAGS:
            descriptors = []
            for block in sorted((b for b in old["blocks"] if b["tag"] == tag), key=lambda b: b["start"]):
                with np.load(previous / block["file"]) as z:
                    descriptors.append(z["descriptors"])
            np.save(output / f"{tag}_prior_descriptors.npy", np.concatenate(descriptors))
            metadata = pd.read_csv(previous / f"{tag}_observations.csv.gz")
            metadata["source_teacher"] = metadata.source_teacher.map({"original_bank_0": TEACHERS[0], "original_bank_1": TEACHERS[1]})
            if metadata.source_teacher.isna().any():
                raise ValueError("unexpected source stream")
            metadata.to_csv(output / f"{tag}_observations.csv.gz", index=False)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_repeat, tag, repeat, previous, source, output) for tag in TAGS for repeat in range(50)]
            for f in as_completed(futures):
                record = f.result()
                manifest["generations"].append(record)
                print(json.dumps(record), flush=True)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
        tasks = []
        for tag in TAGS:
            collect_counts(tag, output)
            with np.load(source / f"{tag}_cache.npz") as z:
                n = len(z["centers"])
            tasks.extend((tag, a, min(a + BLOCK, n), source, output) for a in range(0, n, BLOCK))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(block_task, *task) for task in tasks]):
                record = f.result()
                manifest["blocks"].append(record)
                print(json.dumps(record), flush=True)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
        for tag in TAGS:
            groups = load_groups(output, tag)
            blocks = sorted((b for b in manifest["blocks"] if b["tag"] == tag), key=lambda b: b["start"])
            with np.load(source / f"{tag}_cache.npz") as z:
                rates = z["rates"]

            def partials(blocks=blocks):
                for b in blocks:
                    with np.load(output / b["file"]) as z:
                        yield z

            scores, totals = merge_blocks(partials(), groups, rates, 38400)
            np.save(output / f"{tag}_exact_scores.npy", scores)
            manifest["records"].append({"tag": tag, "observations": len(scores), "accepted_straight": int(totals[0]), "accepted_curved": int(totals[1]), "n_blocks": len(blocks)})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fits = [row for batch in pool.map(fit_populations, range(50), [output] * 50) for row in batch]
        for name, frame in zip(("fits", "summary", "gates"), summarize(pd.DataFrame(fits), supports=("exhaustive",), teachers=TEACHERS), strict=True):
            frame.to_csv(output / f"fresh_path_clock_{name}.csv", index=False)
        pooled = pd.DataFrame(fit_populations(-1, output))
        pooled.to_csv(output / "fresh_path_clock_pooled_fits.csv", index=False)
        manifest.update(status="complete", n_observations=76800, n_fits=len(fits), n_pooled_fits=len(pooled), runtime_s=time.monotonic() - start)
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p != path}
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--previous-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    run(parser.parse_args())
