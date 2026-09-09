#!/usr/bin/env python3
"""Rescore frozen simulated populations with dense independent path quadrature."""

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
from scipy.special import logsumexp

from hipporeplayimm.dense_path_clocks import BANKS, CHUNK, SUPPORTS, PathAccumulator, RateIntegral, rng, summarize
from hipporeplayimm.literal_replay_clock import SpatialRates, clock_coordinates, normalize, sample_geometry
from hipporeplayimm.metric_population_recovery import population_fit
from hipporeplayimm.unknown_path_clocks import MODELS
from scripts._provenance import build_script_provenance, file_sha256

IDS = ["dataset", "animal", "session"]


def load_observations(tag, parent, repeats=50):
    tables, observations = [], []
    for repeat in range(repeats):
        table = pd.read_csv(parent / f"{tag}_r{repeat:03d}_scores.csv.gz")
        table = table[table.support.eq(256)].sort_values("observation_index").drop(columns=[c for c in table if c.startswith("score_") or c == "support"])
        with np.load(parent / f"{tag}_r{repeat:03d}_observations.npz") as z:
            counts, offsets = z["counts"], z["offsets"]
            if len(offsets) != len(table) + 1 or not table.repeat.eq(repeat).all():
                raise ValueError("unaligned frozen observations")
            for i in range(len(table)):
                observations.append(counts[offsets[i] : offsets[i + 1]])
        tables.append(table)
    metadata = pd.concat(tables, ignore_index=True).rename(columns={"teacher": "source_teacher"})
    metadata["source_teacher"] = metadata.source_teacher.map({"matched": "original_bank_0", "independent": "original_bank_1"})
    metadata["row_index"] = np.arange(len(metadata))
    if metadata.source_teacher.isna().any() or metadata.duplicated(["repeat", "source_teacher", "scenario", "event_in_population"]).any():
        raise ValueError("unique known source population identifiers required")
    groups = {}
    for n, group in metadata.groupby("n_bins"):
        index = group.index.to_numpy()
        x = np.stack([observations[i] for i in index])
        if x.shape[1] != n or not np.array_equal(x.sum(axis=(1, 2)), group.n_spikes):
            raise ValueError("source counts differ from metadata")
        groups[n] = index, x
    return metadata, groups


def record_task(item, source, parent, output, supports=SUPPORTS, banks=BANKS, chunk=CHUNK, repeats=50):
    started = time.monotonic()
    if any(h % chunk for h in supports) or chunk % 2 or tuple(sorted(supports)) != tuple(supports):
        raise ValueError("increasing supports and even dividing path chunks required")
    tag = item["tag"]
    metadata, groups = load_observations(tag, parent, repeats)
    metadata.to_csv(output / f"{tag}_observations.csv.gz", index=False)
    with np.load(source / f"{tag}_cache.npz") as z:
        rates, centers = z["rates"], z["centers"]
    shape = (len(banks), len(supports), len(metadata), len(MODELS))
    result = np.lib.format.open_memmap(output / f"{tag}_scores.npy", mode="w+", dtype="float64", shape=shape)
    result[:] = np.nan
    spatial, descriptors = SpatialRates(centers, rates), []
    static = np.log(normalize(rates.T))
    for index, counts in groups.values():
        for start in range(0, len(index), 1024):
            selected = index[start : start + 1024]
            x = counts[start : start + 1024].sum(axis=1)
            result[:, :, selected, 2] = logsumexp(x @ static.T, axis=1) - np.log(len(static))
    for bi, bank in enumerate(banks):
        accumulators = {n: [PathAccumulator(counts), PathAccumulator(counts)] for n, (_index, counts) in groups.items()}
        for start in range(0, max(supports), chunk):
            integrals = []
            for h in range(start, start + chunk):
                points, values, attempts = sample_geometry(spatial, rng(tag, bank, h, "geometry"), curved=bool(h % 2))
                s, c, clocks = clock_coordinates(points, values)
                integrals.append([RateIntegral(clocks[m], values) for m in MODELS[:2]])
                delta = points[-1] - points[0]
                normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
                amplitude = np.dot(points[len(points) // 2] - (points[0] + points[-1]) / 2, normal)
                descriptors.append(
                    {
                        "candidate_bank": bank,
                        "path_index": h,
                        "start_x": points[0, 0],
                        "start_y": points[0, 1],
                        "end_x": points[-1, 0],
                        "end_y": points[-1, 1],
                        "amplitude": amplitude,
                        "attempts": attempts,
                        "arc_cm": s[-1],
                        "code_arc": c[-1],
                    }
                )
            for n, (index, _counts) in groups.items():
                for mi in range(2):
                    logp = np.log(np.array([normalize(pair[mi].average(n)) for pair in integrals]))
                    accumulators[n][mi].add(logp)
                    if start + chunk in supports:
                        si = supports.index(start + chunk)
                        coherent, reset = accumulators[n][mi].scores()
                        result[bi, si, index, mi] = coherent
                        result[bi, si, index, mi + 3] = reset
            if start + chunk in supports:
                result.flush()
                print(json.dumps({"tag": tag, "bank": bank, "paths": start + chunk, "runtime_s": time.monotonic() - started}), flush=True)
        del accumulators
    if not np.isfinite(result).all():
        raise ValueError("missing or nonfinite scored likelihoods")
    result.flush()
    pd.DataFrame(descriptors).to_csv(output / f"{tag}_paths.csv.gz", index=False)
    return {
        **{k: item[k] for k in IDS},
        "tag": tag,
        "observations": len(metadata),
        "score_rows": int(np.prod(shape[:-1])),
        "likelihoods": int(np.prod(shape)),
        "library_paths": len(descriptors),
        "runtime_s": time.monotonic() - started,
    }


def fit_repeat(repeat, items, output, supports=SUPPORTS, banks=BANKS):
    tables = []
    for item in items:
        table = pd.read_csv(output / f"{item['tag']}_observations.csv.gz")
        table = table[table.repeat.eq(repeat)].copy()
        scores = np.load(output / f"{item['tag']}_scores.npy", mmap_mode="r")
        index = table.row_index.to_numpy()
        for bi, bank in enumerate(banks):
            for si, h in enumerate(supports):
                frame = table.assign(candidate_bank=bank, support=h)
                frame[["score_" + m for m in MODELS]] = scores[bi, si, index]
                tables.append(frame)
    full = pd.concat(tables, ignore_index=True)
    rows = []
    keys = ["dataset", "source_teacher", "scenario", "candidate_bank", "support"]
    for key, group in full.groupby(keys):
        fit = population_fit(group[["score_" + m for m in MODELS]].to_numpy(), MODELS, "coherent")
        rows.append(
            dict(zip(keys, key, strict=True)) | {"repeat": repeat, "n_events": len(group), "n_animals": group.animal.nunique(), "n_recordings": group.session.nunique()} | fit
        )
    return rows


def run(args):
    parent, output = args.parent_dir, args.output_dir
    mp = parent / "unknown_path_clock_manifest.json"
    ap = args.audit_dir / "unknown_path_clock_audit.json"
    m, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("complete audited source observations required")
    if m["n_observations"] != 1267200 or len(m["completed"]) != 33:
        raise ValueError("all frozen observations and 33 encoders required")
    source = Path(m["source_dir"])
    sm = source / "conditional_2d_manifest.json"
    if file_sha256(sm) != m["input_file_sha256"]["source_manifest"]:
        raise ValueError("source encoding manifest differs")
    hashes = json.loads(sm.read_text())["output_sha256"]
    items = sorted(m["completed"], key=lambda x: x["tag"])
    for item in items:
        name = item["tag"] + "_cache.npz"
        if file_sha256(source / name) != hashes[name]:
            raise ValueError("source encoding hash differs")
        for repeat in range(50):
            for tail in ("scores.csv.gz", "observations.npz"):
                name = f"{item['tag']}_r{repeat:03d}_{tail}"
                if file_sha256(parent / name) != m["output_sha256"][name]:
                    raise ValueError("frozen observation/metadata hash differs")
    provenance = build_script_provenance(input_paths={"parent_manifest": mp, "parent_audit": ap, "protocol": ROOT / "docs/dense_path_clock_recovery_protocol.md"}, cwd=ROOT)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    output.mkdir(parents=True, exist_ok=False)
    manifest = provenance | {
        "status": "running",
        "completed": [],
        "parent_dir": str(parent),
        "source_dir": str(source),
        "supports": SUPPORTS,
        "candidate_banks": BANKS,
        "models": MODELS,
        "all_scorer_libraries_independent": True,
        "oracle_knows_path": False,
        "real_events_rescored": False,
        "observations_regenerated": False,
        "score_axes": ["candidate_bank", "support", "observation_row", "model"],
    }
    path = output / "dense_path_clock_manifest.json"
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(record_task, item, source, parent, output) for item in items]
            for f in as_completed(futures):
                result = f.result()
                manifest["completed"].append(result)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        fits = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_repeat, r, items, output) for r in range(50)]
            for f in as_completed(futures):
                fits.extend(f.result())
                print(f"population fits {len(fits)}/3600", flush=True)
        for name, frame in zip(("fits", "summary", "gates", "convergence"), summarize(pd.DataFrame(fits)), strict=True):
            frame.to_csv(output / f"dense_path_clock_{name}.csv", index=False)
        manifest.update(
            status="complete",
            observations=sum(x["observations"] for x in manifest["completed"]),
            score_rows=sum(x["score_rows"] for x in manifest["completed"]),
            n_fits=len(fits),
            runtime_s=time.monotonic() - started,
        )
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p != path}
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
