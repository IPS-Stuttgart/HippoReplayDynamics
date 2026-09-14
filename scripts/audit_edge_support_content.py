#!/usr/bin/env python3
"""Independent count, clock, partition, posterior and paired-readout audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256

POLICIES = ("raw_endpoint", "pooled_two_spike_edge", "a_supported_edge", "joint_supported_edge")
SOURCES = ("real", "run_q4", "sim_stationary", "sim_moving", "sim_moving_gain", "sim_late_jump")
KEYS = ["dataset", "animal", "session", "source", "split", "event_index", "policy"]


def seed(*items):
    return int.from_bytes(hashlib.sha256("|".join(map(str, items)).encode()).digest()[:8], "little")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_manifest(path):
    value = json.loads(path.read_text())
    if value["status"] != "complete" or not value["inputs_unchanged"]:
        raise ValueError(f"unfinished or changed inputs: {path}")
    for name, filename in value["input_file_paths"].items():
        if file_sha256(filename) != value["input_file_sha256"][name]:
            raise ValueError(f"manifest input differs: {filename}")
    return value


def recount(spikes, ids, starts, ends):
    """Independent half-open interval recount from unbinned observations."""
    counts = np.empty((len(starts), len(ids)), np.int64)
    for j, cell in enumerate(ids):
        t = np.sort(spikes[spikes[:, 1] == cell, 0])
        counts[:, j] = np.searchsorted(t, ends, side="left") - np.searchsorted(t, starts, side="left")
    return counts


def policies(base, groups):
    if len(base) < 4 or base.ndim != 2 or np.any(base < 0) or np.any(base != np.floor(base)):
        raise ValueError("invalid base observations")
    windows = np.stack([base[i:i+4].sum(axis=0) for i in range(len(base)-3)])
    left, right = (windows[:, g] for g in groups)
    a_good = (left.sum(axis=1) >= 3) & (np.count_nonzero(left, axis=1) >= 2)
    b_good = (right.sum(axis=1) >= 3) & (np.count_nonzero(right, axis=1) >= 2)
    masks = [np.ones(len(windows), bool), left.sum(axis=1)+right.sum(axis=1) >= 2, a_good, a_good & b_good]
    chosen = [max(np.where(m)[0], default=-1) for m in masks]
    return windows, chosen


def dense_metrics(counts, rates, grid, groups, selected_truth, original_truth):
    out, decoded = {}, []
    # Normalize before multiplying: reassociation can move exact thirds across
    # tile boundaries on decimal-origin grids. Preserve the frozen convention.
    normalized = (grid-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)
    tile_xy = np.clip(np.floor(3*normalized).astype(int), 0, 2)
    tile = 3*tile_xy[:, 0]+tile_xy[:, 1]
    for side, group in zip(("a", "b"), groups, strict=True):
        n, r = counts[:, group], rates[group]
        log_likelihood = n @ np.log(r) - .02*np.sum(r, axis=0)
        p = np.exp(log_likelihood-logsumexp(log_likelihood, axis=1)[:, None])
        mean = p @ grid
        regional = np.column_stack([p[:, tile == k].sum(axis=1) for k in range(9)])
        out.update({f"{side}_spikes": n.sum(axis=1), f"{side}_active": np.count_nonzero(n, axis=1),
                    f"{side}_entropy": -(p*np.log(np.maximum(p, 1e-300))).sum(axis=1)/np.log(len(grid)),
                    f"{side}_width_cm": np.sqrt(np.maximum(p @ np.sum(grid**2, axis=1)-np.sum(mean**2, axis=1), 0)),
                    f"{side}_x_cm": mean[:, 0], f"{side}_y_cm": mean[:, 1],
                    f"{side}_selected_truth_error_cm": np.linalg.norm(mean-selected_truth, axis=1),
                    f"{side}_original_truth_error_cm": np.linalg.norm(mean-original_truth, axis=1)})
        out.update({f"{side}_region{k}": regional[:, k] for k in range(9)})
        decoded.append((mean, regional))
    out["separation_cm"] = np.linalg.norm(decoded[0][0]-decoded[1][0], axis=1)
    out["regional_tv"] = np.sum(np.abs(decoded[0][1]-decoded[1][1]), axis=1)/2
    return out


def reconstruct(arrays, groups):
    selected_counts, raw_counts, selected_truth, raw_truth, meta = [], [], [], [], []
    for j, event in enumerate(arrays["event_ids"]):
        lo, hi = arrays["offsets"][j:j+2]
        base = arrays["counts"][lo:hi]
        truth = arrays["truth_base_cm"][lo:hi]
        windows, choices = policies(base, groups)
        last, start = len(windows)-1, arrays["starts_s"][j]
        for policy, index in zip(POLICIES, choices, strict=True):
            valid = index >= 0
            selected_counts.append(windows[index] if valid else np.zeros(base.shape[1]))
            raw_counts.append(windows[-1])
            selected_truth.append(truth[index:index+4].mean(axis=0) if valid else np.full(2, np.nan))
            raw_truth.append(truth[-4:].mean(axis=0))
            meta.append(dict(event_index=int(event), policy=policy, raw_window_index=last,
                selected_window_index=index, status="available" if valid else "abstain",
                candidate_start_s=start, raw_start_s=start+last*.005, raw_end_s=start+last*.005+.02,
                selected_start_s=start+index*.005 if valid else np.nan,
                selected_end_s=start+index*.005+.02 if valid else np.nan,
                shift_earlier_ms=(last-index)*5 if valid else np.nan, n_cells_per_group=len(groups[0])))
    frame = pd.DataFrame(meta)
    raw_truth, selected_truth = np.asarray(raw_truth), np.asarray(selected_truth)
    baseline = dense_metrics(np.asarray(raw_counts), arrays["rates_hz"], arrays["grid_cm"], groups, raw_truth, raw_truth)
    selected = dense_metrics(np.asarray(selected_counts), arrays["rates_hz"], arrays["grid_cm"], groups, selected_truth, raw_truth)
    missing = frame.status.eq("abstain").to_numpy()
    for key, value in baseline.items():
        frame[f"raw_{key}"] = value
        frame[key] = np.where(missing, np.nan, selected[key])
    frame["raw_a_activity_inadequate"] = (baseline["a_spikes"] < 3) | (baseline["a_active"] < 2)
    frame["truth_shift_cm"] = np.linalg.norm(selected_truth-raw_truth, axis=1)
    return frame


def compare_rows(actual, expected):
    keys = ["event_index", "policy"]
    if actual.duplicated(keys).any() or len(actual) != len(expected):
        raise ValueError("missing or duplicated readout rows")
    a, b = (f.sort_values(keys).reset_index(drop=True) for f in (actual, expected))
    for name in expected:
        if name in ("status", "policy"):
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        else:
            np.testing.assert_allclose(a[name].to_numpy(float), b[name].to_numpy(float),
                                       atol=2e-7, rtol=2e-9, equal_nan=True, err_msg=name)


def verify_one(row):
    folder = Path(row.artifact_dir)
    outputs = json.loads((folder/"outputs.json").read_text())
    for name, sha in outputs.items():
        if file_sha256(folder/name) != sha:
            raise ValueError(f"changed output: {folder/name}")
    freeze = json.loads((folder/"frozen_measurement.json").read_text())
    path = Path(freeze["encoding_path"])
    if file_sha256(path) != freeze["encoding_sha256"]:
        raise ValueError("encoding checksum differs")
    data = load_npz(path)
    if not json.loads((path.parent/"encoding_manifest.json").read_text())["training_only"]:
        raise ValueError("encoding not marked training-only")
    mask, support = data["unit_qc_mask"].astype(bool), data["valid_spatial_bins"].astype(bool)
    ids, grid = data["cell_ids"][mask], data["bin_centers_cm"][support]
    rates = np.maximum(data["rates_hz"][mask][:, support], 1e-4)
    sampling = json.loads((path.parent/"sampling_manifest.json").read_text())
    if file_sha256(path.parent/"frozen_candidates.csv") != sampling["candidate_sha256"]:
        raise ValueError("frozen candidate checksum differs")
    expected_metadata = (row.dataset, row.animal, row.session)
    if tuple(freeze[k] for k in ("dataset", "animal", "session")) != expected_metadata:
        raise ValueError("frozen recording mismatch")
    frame = pd.read_csv(folder/"edge_readouts.csv.gz", float_precision="round_trip")
    if frame.duplicated(KEYS).any() or len(frame) != row.rows:
        raise ValueError("duplicate or missing rows")
    for k, value in zip(("dataset", "animal", "session"), expected_metadata, strict=True):
        if not frame[k].eq(value).all():
            raise ValueError(f"wrong recording key: {k}")
    if set(frame.source) != set(SOURCES) or set(frame.split) != {0, 1, 2}:
        raise ValueError("missing sources or partitions")
    raw = load_npz(folder/"real_audit.npz")
    np.testing.assert_array_equal(raw["event_ids"], data["candidate_event_indices"])
    np.testing.assert_array_equal(raw["starts_s"], data["candidate_start_s"])
    real_counts, native_windows, truth_windows, checked = 0, 0, 0, 0
    for source in SOURCES:
        arrays = load_npz(folder/f"{source}_audit.npz")
        for k, value in (("cell_ids", ids), ("grid_cm", grid), ("rates_hz", rates)):
            np.testing.assert_array_equal(arrays[k], value, err_msg=k)
        offsets, counts = arrays["offsets"], arrays["counts"]
        if offsets[0] != 0 or offsets[-1] != len(counts) or np.any(np.diff(offsets) < 4):
            raise ValueError("invalid ragged count support")
        if len(offsets) != len(arrays["event_ids"])+1 or len(np.unique(arrays["event_ids"])) != len(offsets)-1:
            raise ValueError("invalid event identities")
        if source == "real":
            for j in range(len(raw["event_ids"])):
                lo, hi = data["candidate_offsets"][j:j+2]
                n = int(np.sum(np.isclose(data["candidate_base_durations_s"][lo:hi], .005, atol=1e-9, rtol=0)))
                a, b = offsets[j:j+2]
                if n != b-a:
                    raise ValueError("raw candidate clock changed")
                np.testing.assert_array_equal(counts[a:b], data["candidate_base_counts"][lo:lo+n][:, mask])
            if not np.isnan(arrays["truth_base_cm"]).all():
                raise ValueError("real replay cannot have supplied truth")
        if source in ("real", "run_q4"):
            # Preserve exact original floating-point edges at spike timestamps.
            boundaries = [s+.005*np.arange(int(n)+1) for s, n in zip(arrays["starts_s"], np.diff(offsets), strict=True)]
            starts, ends = np.concatenate([b[:-1] for b in boundaries]), np.concatenate([b[1:] for b in boundaries])
            np.testing.assert_array_equal(recount(data["spikes"], ids, starts, ends), counts)
            native_windows += len(starts)
            if source == "real":
                real_counts = counts.size
            else:
                run_start, run_end = data["run_bounds_s"]
                if not (arrays["starts_s"] >= run_start+.75*(run_end-run_start)).all():
                    raise ValueError("RUN control overlaps training")
                truths = []
                for start in arrays["starts_s"]:
                    t = start+(np.arange(200)+.5)*.001
                    xy = np.column_stack([np.interp(t, data["position"][:, 0], data["position"][:, d]) for d in (1, 2)])
                    truths.append(xy.reshape(40, 5, 2).mean(axis=1))
                np.testing.assert_allclose(arrays["truth_base_cm"], np.concatenate(truths), atol=1e-9, rtol=1e-10)
                truth_windows += len(starts)
        else:
            for key in ("event_ids", "offsets", "starts_s"):
                np.testing.assert_array_equal(arrays[key], raw[key])
            np.testing.assert_array_equal(counts.sum(axis=1), raw["counts"].sum(axis=1))
            if not np.isfinite(arrays["truth_base_cm"]).all():
                raise ValueError("simulation truth missing")
            for j in range(len(arrays["event_ids"])):
                lo, hi = offsets[j:j+2]
                truth = arrays["truth_base_cm"][lo:hi]
                if source == "sim_stationary":
                    np.testing.assert_allclose(truth, np.broadcast_to(truth[0], truth.shape), atol=1e-10)
                elif source == "sim_late_jump":
                    np.testing.assert_allclose(truth[-4:], np.broadcast_to(truth[-1], (4, 2)), atol=1e-10)
                elif (np.linalg.norm(np.diff(truth, axis=0), axis=1) > 5+1e-7).any():
                    raise ValueError("moving truth violates frozen 1000cm/s bound")
        for split in range(3):
            p = np.random.default_rng(seed(freeze["seed"], "edge_support_partition", row.dataset, row.session, split)).permutation(len(ids))
            n = len(ids)//2
            groups = [np.sort(p[:n]), np.sort(p[n:2*n])]
            if n < 5 or np.intersect1d(*groups).size:
                raise ValueError("invalid independent groups")
            for side, group in zip(("a", "b"), groups, strict=True):
                np.testing.assert_array_equal(freeze["groups"][split][f"{side}_ids"], ids[group])
            rebuilt = reconstruct(arrays, groups)
            compare_rows(frame.loc[frame.source.eq(source) & frame.split.eq(split)], rebuilt)
            checked += len(rebuilt)
    if checked != len(frame):
        raise ValueError("unaudited extra rows")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status="passed",
                rows=checked, native_base_windows=native_windows, real_cell_base_counts=real_counts,
                run_truth_base_windows=truth_windows, sources=len(SOURCES),
                readouts_sha256=file_sha256(folder/"edge_readouts.csv.gz"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {f"sessions{i}": d/"measurement_sessions.csv" for i, d in enumerate(args.measurement_dir)}
    inputs["auditor"] = Path(__file__)
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", started_at_utc=datetime.now(UTC).isoformat(),
        scope="independent count/clock/partition/flat-Poisson/readout reconstruction; RUN truth from native tracking; simulation totals and path invariants, not independent path regeneration")
    (args.output_dir/"independent_audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    results = []
    for folder in args.measurement_dir:
        measure = check_manifest(folder/"manifest.json")
        encoding = Path(measure["input_file_paths"]["catalog"]).parent
        check_manifest(encoding/"manifest.json")
        rows = pd.read_csv(folder/"measurement_sessions.csv")
        if len(rows) != 8 or rows.animal.nunique() != 4 or rows.duplicated(["dataset", "session"]).any():
            raise ValueError("incomplete frozen recording cohort")
        for row in rows.itertuples(index=False):
            try:
                if row.status != "complete":
                    raise ValueError("measurement has missing sources")
                result = verify_one(row)
            except (ValueError, KeyError, OSError, AssertionError) as exc:
                result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc))
            results.append(result)
            print(json.dumps(result), flush=True)
            pd.DataFrame(results).to_csv(args.output_dir/"independent_audit_sessions.csv", index=False)
    passed = bool(results) and all(r["status"] == "passed" for r in results)
    unchanged = all(file_sha256(v) == manifest["input_file_sha256"][k] for k, v in inputs.items())
    manifest.update(status="passed" if passed and unchanged else "failed", inputs_unchanged=unchanged,
                    finished_at_utc=datetime.now(UTC).isoformat(), results=results)
    (args.output_dir/"independent_audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not passed or not unchanged:
        raise ValueError("edge-support independent audit failed")


if __name__ == "__main__":
    main()
