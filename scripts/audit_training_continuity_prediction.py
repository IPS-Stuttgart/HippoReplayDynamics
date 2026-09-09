#!/usr/bin/env python3
"""Decode training-only geometric labels; never rescore held-out predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _provenance import build_script_provenance, file_sha256

from hipporeplayimm.training_continuity import (
    CRITERIA,
    IDENTITY,
    classification_group,
    classify_training,
    nested_half,
    predictive_contrasts,
    spatial_supports,
    validate_labels,
)

PARENT_HASH = "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"
ORDER_HASH = "40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae"
PARENT_AUDIT_HASH = "63df5acf95eda445de9c32e3a137d918e515eb1f405773f00f2032af9802a07e"
REFERENCE_HASH = "0a8d9762b2559fc007881b0213cb50c91e5a548212b90d6b2cef54206a0bbee3"


def checked(path, digest):
    if not digest or file_sha256(path) != digest:
        raise ValueError(f"missing/changed input: {path}")
    return Path(path)


def load_source(parent, tag):
    selection = pd.read_csv(parent / f"{tag}_selection.csv")
    with np.load(parent / f"{tag}_cache.npz", allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    if selection.source_cache_path.nunique() != 1 or selection.source_cache_sha256.nunique() != 1:
        raise ValueError("ambiguous raw cache")
    raw_path = checked(selection.iloc[0].source_cache_path, selection.iloc[0].source_cache_sha256)
    with np.load(raw_path, allow_pickle=False) as z:
        needed = [
            "unit_qc_mask",
            "cell_ids",
            "rates_hz",
            "valid_spatial_bins",
            "bin_centers_cm",
            "candidate_event_indices",
            "candidate_offsets",
            "candidate_base_counts",
            "candidate_base_durations_s",
            "candidate_start_s",
            "candidate_end_s",
        ]
        source = {k: z[k] for k in needed}
    units, states = source["unit_qc_mask"].astype(bool), source["valid_spatial_bins"].astype(bool)
    if (
        not np.array_equal(source["cell_ids"][units], cache["unit_ids"])
        or not np.array_equal(source["rates_hz"][units][:, states], cache["rates"])
        or not np.array_equal(source["bin_centers_cm"][states], cache["centers"])
    ):
        raise ValueError("raw/parent encoding mismatch")
    return selection, cache, source


def process(record, parent, order, output):
    tick = time.monotonic()
    tag = record["tag"]
    identity = {k: record[k] for k in IDENTITY}
    selected, cache, raw = load_source(parent, tag)
    if len(selected) != record["events"] or selected.event_id.duplicated().any():
        raise ValueError("source event denominator mismatch")
    qc = raw["unit_qc_mask"].astype(bool)
    lookup = {int(e): j for j, e in enumerate(raw["candidate_event_indices"])}
    bounds, grid = cache["arena_bounds_cm"], cache["centers"]
    supports, bounds_available = spatial_supports(grid, bounds)
    archive = {"event_ids": selected.event_id.to_numpy(int), "unit_ids": cache["unit_ids"], **{f"mask_{k}": v for k, v in supports.items()}}
    offsets, rows, path_parts, splits = [0], [], {}, []
    for split in range(5):
        tr, held = cache[f"train_{split}"], cache[f"held_{split}"]
        if set(tr) & set(held) or sorted(np.r_[tr, held]) != list(range(len(cache["unit_ids"]))):
            raise ValueError("invalid parent neuron partition")
        half = nested_half(tr, tuple(identity.values()), split)
        splits.append((tr, half, held))
        archive.update({f"train_{split}": tr, f"half_{split}": half, f"held_{split}": held})
        for support in supports:
            for level in ("full", "half"):
                path_parts[f"path_{support}_{level}_{split}"] = []
    for e in selected.itertuples(index=False):
        j = lookup[int(e.event_id)]
        a, b = raw["candidate_offsets"][j : j + 2]
        base = raw["candidate_base_counts"][a:b][:, qc]
        dt = raw["candidate_base_durations_s"][a:b]
        pooled = np.add.reduceat(base, np.arange(0, len(base), 4), axis=0)
        if not np.array_equal(pooled, cache[f"counts_{e.event_id}"]) or not np.allclose(
            [raw["candidate_start_s"][j], raw["candidate_end_s"][j]], [e.start_s, e.end_s], atol=1e-8, rtol=0
        ):
            raise ValueError("raw event count/time alignment mismatch")
        nframes = None
        for split, (tr, half, held) in enumerate(splits):
            for support, mask in supports.items():
                results = []
                for level, cells in [("full", tr), ("half", half)]:
                    path, _, metrics = classify_training(base, dt, cache["rates"], grid, cells, mask)
                    path_parts[f"path_{support}_{level}_{split}"].append(path)
                    results.append(metrics)
                    nframes = len(path)
                for ci, (filtered, minimum) in enumerate(CRITERIA):
                    full, reduced = results[0][ci], results[1][ci]
                    rows.append(
                        identity
                        | {
                            "event_id": e.event_id,
                            "split": split,
                            "support": support,
                            "bin_filter": "bin_support" if filtered else "edge_only",
                            "min_frames": minimum,
                            "full_training_pass": full["geometric_pass"],
                            "nested_half_pass": reduced["geometric_pass"],
                            "group": classification_group(full["geometric_pass"], reduced["geometric_pass"]),
                            "n_train_cells": len(tr),
                            "arena_bounds_available": bounds_available,
                            "n_nested_cells": len(half),
                            "n_heldout_cells": len(held),
                            "heldout_used_for_classification": False,
                        }
                        | {"full_" + k: v for k, v in full.items() if k != "geometric_pass"}
                        | {"half_" + k: v for k, v in reduced.items() if k != "geometric_pass"}
                    )
        offsets.append(offsets[-1] + nframes)
    labels = pd.DataFrame(rows)
    validate_labels(labels, selected)
    archive.update({k: np.concatenate(v) for k, v in path_parts.items()})
    archive["frame_offsets"] = np.asarray(offsets)
    np.savez_compressed(output / f"{tag}_paths.npz", **archive)
    labels.to_csv(output / f"{tag}_labels.csv.gz", index=False)
    predictions = predictive_contrasts(pd.read_csv(parent / f"{tag}_scores.csv"), pd.read_csv(order / f"{tag}_split_contrasts.csv.gz"))
    if len(predictions) != len(selected) * 5:
        raise ValueError("missing predictive splits")
    predictions.to_csv(output / f"{tag}_predictions.csv.gz", index=False)
    return identity | {
        "tag": tag,
        "events": len(selected),
        "label_rows": len(labels),
        "prediction_rows": len(predictions),
        "outside_arena_centres": int((~supports["arena_clipped"]).sum()) if bounds_available else None,
        "arena_bounds_available": bounds_available,
        "total_centres": len(grid),
        "raw_cache_path": selected.iloc[0].source_cache_path,
        "raw_cache_sha256": selected.iloc[0].source_cache_sha256,
        "runtime_s": time.monotonic() - tick,
    }


def run(args):
    parent, order, output = args.parent_dir.resolve(), args.order_dir.resolve(), args.output_dir.resolve()
    inputs = {
        "parent_manifest": checked(parent / "conditional_2d_manifest.json", PARENT_HASH),
        "order_manifest": checked(order / "predictive_order_map_manifest.json", ORDER_HASH),
        "parent_audit": checked(args.parent_audit, PARENT_AUDIT_HASH),
        "order_audit": args.order_audit,
        "geometric_reference": checked(args.geometric_reference, REFERENCE_HASH),
        "protocol": ROOT / "docs/training_continuity_prediction_protocol.md",
        "bounds_addendum": ROOT / "docs/training_continuity_missing_bounds_addendum.md",
        "producer": Path(__file__),
        "library": ROOT / "src/hipporeplayimm/training_continuity.py",
        "provenance_helper": ROOT / "scripts/_provenance.py",
    }
    pm, om = json.loads(inputs["parent_manifest"].read_text()), json.loads(inputs["order_manifest"].read_text())
    for key in ("parent_audit", "order_audit"):
        audit = json.loads(Path(inputs[key]).read_text())
        if audit.get("status") != "pass":
            raise ValueError("passing parent audits required")
    oa = json.loads(Path(inputs["order_audit"]).read_text())
    if ORDER_HASH not in oa.get("input_file_sha256", {}).values():
        raise ValueError("factorial audit does not link to the frozen run")
    if (
        pm.get("status") != "complete"
        or om.get("status") != "complete"
        or pm["n_events"] != 9225
        or len(pm["completed"]) != 33
        or {r["tag"] for r in pm["completed"]} != {r["tag"] for r in om["completed"]}
    ):
        raise ValueError("frozen whole-cohort inputs incomplete")
    for root, manifest in ((parent, pm), (order, om)):
        for name, digest in manifest["output_sha256"].items():
            checked(root / name, digest)
    meta = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if meta["git_dirty"] or meta["code_commit"] == "unavailable":
        raise ValueError("freeze code in a clean commit before classification")
    output.mkdir(parents=True, exist_ok=False)
    meta.update(
        status="running",
        created_at_utc=datetime.now(UTC).isoformat(),
        workers=args.workers,
        n_events=9225,
        n_splits=5,
        predictive_scores_recomputed=False,
        claim_boundary="training-only geometric rejection with independent predictive endpoints; not replay ground truth",
    )
    manifest_path = output / "training_continuity_manifest.json"
    manifest_path.write_text(json.dumps(meta, indent=2) + "\n")
    tick, results, errors = time.monotonic(), [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r, parent, order, output): r for r in pm["completed"]}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:  # noqa: BLE001 -- retain every failed session in the manifest.
                errors.append({"tag": futures[future]["tag"], "error": repr(exc)})
    meta.update(completed=sorted(results, key=lambda r: r["tag"]), errors=errors, runtime_s=time.monotonic() - tick)
    try:
        for name, digest in meta["input_file_sha256"].items():
            checked(inputs[name], digest)
        if len(results) != 33 or sum(r["label_rows"] for r in results) != 9225 * 40 or errors:
            raise ValueError("incomplete classification run")
        meta["status"] = "complete"
        meta["output_sha256"] = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p != manifest_path}
    except Exception as exc:  # noqa: BLE001 -- a failed final check must leave an explicit failed run.
        meta["status"] = "failed"
        errors.append({"error": repr(exc)})
    manifest_path.write_text(json.dumps(meta, indent=2) + "\n")
    if errors:
        raise RuntimeError("classification failed; inspect the manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/mnt/seagate10tb/florianpfaff")
    parser.add_argument("--parent-dir", type=Path, default=root / "conditional-2d-mua-pf-tanni-all33-20260908")
    parser.add_argument("--order-dir", type=Path, default=root / "2d-predictive-order-map-all9225-k20-20260908")
    parser.add_argument("--parent-audit", type=Path, default=root / "conditional-2d-mua-pf-tanni-all33-20260908-audit/conditional_2d_audit.json")
    parser.add_argument("--order-audit", type=Path, default=root / "2d-predictive-order-map-all9225-k20-20260908-audit/predictive_order_map_audit.json")
    parser.add_argument(
        "--geometric-reference", type=Path, default=Path("/home/florianpfaff/HippoReplayDynamics-recording-coverage/src/hipporeplayimm/replay_coverage_shuffle_baseline.py")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
