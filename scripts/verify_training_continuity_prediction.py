#!/usr/bin/env python3
"""Independently reconstruct training labels and their predictive joins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _provenance import build_script_provenance
from audit_training_continuity_prediction import checked, load_source

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split"]


def reference_metrics(path, grid, counts, filtered, minimum):
    totals = counts.sum(axis=1)
    endpoints = np.where(totals >= 2)[0]
    valid = np.zeros(len(path), bool)
    if len(endpoints):
        valid[endpoints[0] : endpoints[-1] + 1] = True
    edge_count = int(valid.sum())
    if filtered:
        valid &= (totals >= 3) & ((counts != 0).sum(axis=1) >= 2)
    runs, current = [], []
    for i, v in enumerate(valid):
        separated = not v or (len(current) and np.linalg.norm(grid[path[i]] - grid[path[current[-1]]]) >= 20 - 1e-9)
        if separated and current:
            runs.append(current)
            current = []
        if v:
            current.append(i)
    if current:
        runs.append(current)
    longest = max(runs, key=len) if runs else []
    distance = float(np.linalg.norm(grid[path[longest[-1]]] - grid[path[longest[0]]])) if longest else 0.0
    passed = len(longest) >= minimum and distance >= 40 - 1e-9
    reason = "pass"
    if not passed:
        if not len(path):
            reason = "no_complete_windows"
        elif not edge_count:
            reason = "no_supported_edges"
        elif valid.sum() < minimum:
            reason = "insufficient_supported_frames"
        elif len(longest) < minimum:
            reason = "jump_fragmentation"
        else:
            reason = "insufficient_displacement"
    return {
        "geometric_pass": passed,
        "failure_reason": reason,
        "decoding_frames": len(path),
        "edge_supported_frames": edge_count,
        "valid_frames": int(valid.sum()),
        "longest_run_frames": len(longest),
        "run_start_frame": longest[0] if longest else -1,
        "run_displacement_cm": distance,
    }


def independent_predictions(parent, order, tag):
    scores = pd.read_csv(parent / f"{tag}_scores.csv").set_index(KEY + ["map"])
    shuffle = pd.read_csv(order / f"{tag}_scores.csv.gz")
    if shuffle.duplicated(KEY + ["map", "shuffle"]).any() or shuffle.groupby(KEY + ["map"]).size().ne(20).any():
        raise ValueError("incomplete predictive shuffle source")
    means = shuffle.groupby(KEY + ["map"])[["score_first_order_imm", "score_diffusion"]].mean()
    real, wrong = scores.xs("real", level="map"), scores.xs("population_code_permuted", level="map")
    sr, sw = means.xs("real", level="map"), means.xs("population_code_permuted", level="map")
    result = real[["n_heldout_spikes", "n_train_spikes", "duration_s"]].copy()
    for model, prefix in [("first_order_imm", "imm"), ("diffusion", "diffusion")]:
        c = "score_" + model
        for other, suffix in [("iid_position", "iid"), ("static_location", "static"), ("event_global", "composition")]:
            result[prefix + "_minus_" + suffix] = real[c] - real["score_" + other]
        result[prefix + "_order_advantage"] = real[c] - sr[c]
        result[prefix + "_order_map_interaction"] = (real[c] - sr[c]) - (wrong[c] - sw[c])
    return result


def audit_record(record, root, parent, order, reference_path):
    tag = record["tag"]
    selected, cache, raw = load_source(parent, tag)
    labels = pd.read_csv(root / f"{tag}_labels.csv.gz")
    expected_keys = {
        (int(e), s, support, filtering, frames)
        for e in selected.event_id
        for s in range(5)
        for support in ("parent", "arena_clipped")
        for filtering in ("edge_only", "bin_support")
        for frames in (10, 11)
    }
    keys = ["event_id", "split", "support", "bin_filter", "min_frames"]
    if labels.duplicated(keys).any() or set(labels[keys].itertuples(index=False, name=None)) != expected_keys or labels.heldout_used_for_classification.any():
        raise ValueError("missing/duplicate/leaking classification labels")
    for k in ID:
        if not labels[k].eq(record[k]).all():
            raise ValueError("session identity mismatch")
    saved = labels.set_index(keys).to_dict("index")
    with np.load(root / f"{tag}_paths.npz", allow_pickle=False) as z:
        paths = {k: z[k] for k in z.files}
    if not np.array_equal(paths["event_ids"], selected.event_id) or not np.array_equal(paths["unit_ids"], cache["unit_ids"]):
        raise ValueError("archive event/cell identities differ")
    spec = importlib.util.spec_from_file_location("frozen_geometric_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    lookup = {int(e): j for j, e in enumerate(raw["candidate_event_indices"])}
    masks = {
        "parent": np.ones(len(cache["centers"]), bool),
        "arena_clipped": ((cache["centers"] >= cache["arena_bounds_cm"][0]) & (cache["centers"] <= cache["arena_bounds_cm"][1])).all(axis=1),
    }
    bounds_available = not np.isnan(cache["arena_bounds_cm"]).all()
    if not bounds_available:
        masks["arena_clipped"] = masks["parent"].copy()
    elif not np.isfinite(cache["arena_bounds_cm"]).all():
        raise ValueError("partially missing bounds")
    if not labels.arena_bounds_available.eq(bounds_available).all() or record["arena_bounds_available"] != bounds_available:
        raise ValueError("incorrect clipping-availability flag")
    expected_outside = int((~masks["arena_clipped"]).sum()) if bounds_available else None
    if record["outside_arena_centres"] != expected_outside:
        raise ValueError("incorrect out-of-arena denominator")
    for name, mask in masks.items():
        if not np.array_equal(mask, paths["mask_" + name]):
            raise ValueError("changed spatial support")
    label_checks, path_checks, tie_indices, max_gap, offset = 0, 0, 0, 0.0, 0
    for ei, e in enumerate(selected.itertuples(index=False)):
        j = lookup[int(e.event_id)]
        a, b = raw["candidate_offsets"][j : j + 2]
        base = raw["candidate_base_counts"][a:b][:, raw["unit_qc_mask"].astype(bool)]
        dt = raw["candidate_base_durations_s"][a:b]
        n = sum(abs(float(d) - 0.005) <= 1e-9 for d in dt)
        windows = np.array([base[i : i + 4].sum(axis=0) for i in range(max(0, n - 3))], dtype=np.int64).reshape(-1, base.shape[1])
        lo, hi = paths["frame_offsets"][ei : ei + 2]
        if lo != offset or hi - lo != len(windows):
            raise ValueError("overlapping-window clock mismatch")
        offset = int(hi)
        for s in range(5):
            tr, held = cache[f"train_{s}"], cache[f"held_{s}"]
            key = "|".join(map(str, ("training_continuity_v1", 20260910, *[record[k] for k in ID], s)))
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
            half = np.sort(rng.permutation(np.sort(tr))[: len(tr) // 2])
            if set(tr) & set(held) or set(half) & set(held):
                raise ValueError("classification includes held-out cells")
            for name, cells in [("train", tr), ("half", half), ("held", held)]:
                if not np.array_equal(cells, paths[f"{name}_{s}"]):
                    raise ValueError("wrong saved neuron partition")
            for support, mask in masks.items():
                grid = cache["centers"][mask]
                passes = {}
                for level, cells in [("full", tr), ("half", half)]:
                    cc, rates = windows[:, cells], cache["rates"][cells][:, mask]
                    ll = cc.astype(float) @ np.log(rates) - 0.02 * rates.sum(axis=0)
                    path = paths[f"path_{support}_{level}_{s}"][lo:hi]
                    if len(path) and ((path < 0).any() or (path >= len(grid)).any()):
                        raise ValueError("invalid saved MAP indices")
                    gap = ll.max(axis=1) - ll[np.arange(len(path)), path]
                    if (gap > 1e-10).any():
                        raise ValueError("saved path differs from independent MAP beyond rounding")
                    max_gap = max(max_gap, float(gap.max(initial=0)))
                    tie_indices += int((ll.argmax(axis=1) != path).sum())
                    old = reference.screen_maps(path[None], grid, cc)[0]
                    for ci, (filtered, minimum) in enumerate(reference.CRITERIA):
                        row_key = (e.event_id, s, support, "bin_support" if filtered else "edge_only", minimum)
                        row = saved[row_key]
                        metrics = reference_metrics(path, grid, cc, filtered, minimum)
                        flag = row["full_training_pass" if level == "full" else "nested_half_pass"]
                        if bool(flag) != bool(old[ci]) or bool(flag) != metrics.pop("geometric_pass"):
                            raise ValueError("geometric reference mismatch")
                        for field, value in metrics.items():
                            actual = row[level + "_" + field]
                            if isinstance(value, float):
                                if not np.isclose(value, actual, atol=1e-9, rtol=0):
                                    raise ValueError("geometric displacement mismatch")
                            elif value != actual:
                                raise ValueError("geometric metric mismatch: " + field)
                        passes[level, filtered, minimum] = bool(flag)
                        label_checks += 1
                    path_checks += 1
                for filtered, minimum in reference.CRITERIA:
                    pair = (passes["full", filtered, minimum], passes["half", filtered, minimum])
                    group = {(True, True): "retained_with_thinning", (True, False): "lost_with_thinning", (False, True): "gained_with_thinning", (False, False): "rejected_both"}[
                        pair
                    ]
                    row = saved[e.event_id, s, support, "bin_support" if filtered else "edge_only", minimum]
                    if row["group"] != group or (row["n_train_cells"], row["n_nested_cells"], row["n_heldout_cells"]) != (len(tr), len(half), len(held)):
                        raise ValueError("group/cell-count mismatch")
    for name, value in paths.items():
        if name.startswith("path_") and len(value) != offset:
            raise ValueError("unaccounted saved path frames")
    predicted = pd.read_csv(root / f"{tag}_predictions.csv.gz").set_index(KEY).sort_index()
    independent = independent_predictions(parent, order, tag).sort_index()
    if not predicted.index.equals(independent.index) or list(predicted.columns) != list(independent.columns) or not np.allclose(predicted, independent, atol=1e-9, rtol=0):
        raise ValueError("independent predictive contrasts mismatch")
    return {
        "tag": tag,
        "events": len(selected),
        "paths_checked": path_checks,
        "label_metrics_checked": label_checks,
        "tied_map_indices": tie_indices,
        "maximum_map_log_gap": max_gap,
        "prediction_rows_checked": len(predicted),
    }


def run(args):
    root, out = args.run_dir.resolve(), args.output_dir.resolve()
    path = root / "training_continuity_manifest.json"
    m = json.loads(path.read_text())
    if m.get("status") != "complete" or len(m.get("completed", [])) != 33 or m.get("predictive_scores_recomputed") is not False:
        raise ValueError("complete non-rescoring run required")
    for k, digest in m["input_file_sha256"].items():
        checked(m["input_file_paths"][k], digest)
    for name, digest in m["output_sha256"].items():
        checked(root / name, digest)
    parent = Path(m["input_file_paths"]["parent_manifest"]).parent
    order = Path(m["input_file_paths"]["order_manifest"]).parent
    for directory, name in [(parent, "conditional_2d_manifest.json"), (order, "predictive_order_map_manifest.json")]:
        pm = json.loads((directory / name).read_text())
        for file, digest in pm["output_sha256"].items():
            checked(directory / file, digest)
    out.mkdir(parents=True, exist_ok=False)
    result = build_script_provenance(input_paths={"run_manifest": path, "auditor": Path(__file__)}, cwd=ROOT)
    records, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = {pool.submit(audit_record, r, root, parent, order, m["input_file_paths"]["geometric_reference"]): r for r in m["completed"]}
        for f in as_completed(fs):
            try:
                record = f.result()
                records.append(record)
                print(json.dumps(record), flush=True)
            except Exception as exc:  # noqa: BLE001 -- retain failures rather than certifying a subset.
                errors.append({"tag": fs[f]["tag"], "error": repr(exc)})
    expected = (
        sum(r["events"] for r in records) == 9225 and sum(r["paths_checked"] for r in records) == 9225 * 20 and sum(r["prediction_rows_checked"] for r in records) == 9225 * 5
    )
    result.update(
        status="pass" if not errors and len(records) == 33 and expected else "fail",
        records=records,
        errors=errors,
        all_paths_independently_checked=True,
        heldout_scores_recomputed=False,
    )
    (out / "training_continuity_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise RuntimeError("training-continuity audit failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    run(p.parse_args())
