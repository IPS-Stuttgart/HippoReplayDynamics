#!/usr/bin/env python3
"""Generate A-only predictors and disjoint B outcomes without selecting events."""
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
from scipy.spatial import cKDTree
from scipy.special import softmax

from hipporeplayimm.encoding import _speed_cm_s
from scripts._provenance import build_script_provenance, file_sha256


def seed_for(*values):
    value = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def partition(n, seed):
    if n < 20:
        raise ValueError("fewer than ten units per disjoint population")
    ix = np.random.default_rng(seed).permutation(n)
    k = n // 2
    return np.sort(ix[:k]), np.sort(ix[k:2*k])


def tile_ids(grid):
    scaled = (grid - grid.min(axis=0)) / np.maximum(np.ptp(grid, axis=0), 1)
    tile = np.clip((3 * scaled).astype(int), 0, 2)
    return tile[:, 0] * 3 + tile[:, 1]


def decode(counts, rates, grid):
    counts, rates = np.asarray(counts), np.asarray(rates)
    if (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("counts must be nonnegative integers")
    if counts.shape[1] != len(rates) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("invalid encoding")
    p = softmax(counts @ np.log(rates) - .02 * rates.sum(axis=0), axis=1)
    mean = p @ grid
    width = np.sqrt(np.maximum(p @ (grid**2).sum(axis=1) - (mean**2).sum(axis=1), 0))
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1) / np.log(len(grid))
    tile = tile_ids(grid)
    regional = p @ np.eye(9)[tile]
    return dict(p=p, mean=mean, width=width, entropy=entropy, regional=regional,
                map_region=tile[p.argmax(axis=1)], peak=p.max(axis=1))


def run_windows(data, ids, start, end):
    position, intervals = data["position"], data["supported_run_intervals"]
    t, xy = position[:, 0], position[:, 1:3]
    speed = _speed_cm_s(t, xy)
    rows = []
    for a in np.arange(start, end - .02, .25):
        b = a + .02
        left, right = np.searchsorted(t, [a, b])
        left, right = max(0, left-1), min(len(t), right+1)
        if right-left < 2 or t[left] > a or t[right-1] < b or np.max(np.diff(t[left:right])) > .1:
            continue
        if not np.isfinite(xy[left:right]).all() or not ((speed[left:right] >= 10) & (speed[left:right] <= 200)).all():
            continue
        if not any(a >= x and b <= y for x, y in intervals):
            continue
        rows.append([a, b, *[np.interp(a+.01, t, xy[:, d]) for d in range(2)]])
    if len(rows) < 100:
        raise ValueError(f"insufficient supported RUN windows: {len(rows)}")
    windows = np.array(rows)[np.linspace(0, len(rows)-1, min(len(rows), 1024), dtype=int)]
    counts = np.empty((len(windows), len(ids)), dtype=int)
    for j, cell in enumerate(ids):
        ts = np.sort(data["spikes"][data["spikes"][:, 1] == cell, 0])
        counts[:, j] = np.searchsorted(ts, windows[:, 1]) - np.searchsorted(ts, windows[:, 0])
    return windows, counts


def endpoint_counts(data, mask):
    rows, counts, excluded = [], [], []
    for e, event_id in enumerate(data["candidate_event_indices"]):
        left, right = data["candidate_offsets"][e:e+2]
        durations = data["candidate_base_durations_s"][left:right]
        full = np.flatnonzero(np.isclose(durations, .005, atol=1e-9, rtol=0))
        if len(full) < 4 or not np.array_equal(full, np.arange(len(full))):
            excluded.append(dict(event_index=int(event_id), reason="fewer_than_four_full_base_bins"))
            continue
        indices = left + full[-4:]
        counts.append(data["candidate_base_counts"][indices][:, mask].sum(axis=0))
        rows.append(dict(event_index=int(event_id), start_s=float(data["candidate_base_starts_s"][indices[0]]),
                         end_s=float(data["candidate_base_starts_s"][indices[-1]] + .005)))
    if not rows:
        raise ValueError("no complete candidate endpoints")
    return pd.DataFrame(rows), np.array(counts), excluded


def local_run_errors(means, calibration_xy, errors):
    dist, indices = cKDTree(calibration_xy).query(means, k=min(20, len(errors)))
    out = np.full(len(means), np.nan)
    for i in range(len(means)):
        keep = dist[i] <= 40
        if keep.sum() >= 5:
            out[i] = np.median(errors[indices[i][keep]])
    return out


def readouts(ca, cb, ra, rb, grid, calibration_xy, calibration_errors, truth=None):
    a, b = decode(ca, ra, grid), decode(cb, rb, grid)
    near = cKDTree(grid).query(a["mean"])[1]
    coverage = ra.sum(axis=0)
    frame = pd.DataFrame(dict(
        a_spikes=ca.sum(axis=1), a_active=(ca > 0).sum(axis=1),
        b_spikes=cb.sum(axis=1), b_active=(cb > 0).sum(axis=1), n_cells=ca.shape[1],
        a_entropy=a["entropy"], a_width_cm=a["width"], a_peak=a["peak"],
        b_entropy=b["entropy"], b_width_cm=b["width"],
        a_global_run_error_cm=float(np.median(calibration_errors)),
        a_local_run_error_cm=local_run_errors(a["mean"], calibration_xy, calibration_errors),
        a_coverage=coverage[near] / coverage.mean(), grid_diagonal_cm=float(np.linalg.norm(np.ptp(grid, axis=0))),
        endpoint_separation_cm=np.linalg.norm(a["mean"]-b["mean"], axis=1),
        regional_tv=.5*np.abs(a["regional"]-b["regional"]).sum(axis=1),
        map_region_agreement=a["map_region"] == b["map_region"],
        a_x_cm=a["mean"][:, 0], a_y_cm=a["mean"][:, 1], b_x_cm=b["mean"][:, 0], b_y_cm=b["mean"][:, 1],
    ))
    for region in range(9):
        frame[f"a_region_{region}"] = a["regional"][:, region]
        frame[f"b_region_{region}"] = b["regional"][:, region]
    for side, result in (("a", a), ("b", b)):
        frame[f"{side}_truth_error_cm"] = np.linalg.norm(result["mean"]-truth, axis=1) if truth is not None else np.nan
    return frame


def simulate_counts(totals, rates, states, rng):
    probabilities = rates[:, states].T
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    result = np.array([rng.multinomial(int(n), p) for n, p in zip(totals, probabilities, strict=True)])
    if not np.array_equal(result.sum(axis=1), totals):
        raise ValueError("conditional simulation changed spike counts")
    return result


def measure_session(row, output, seed, dataset_pass):
    path = Path(row.artifact_path)
    if file_sha256(path) != row.artifact_sha256:
        raise ValueError(f"cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    mask = data["unit_qc_mask"].astype(bool)
    ids = data["cell_ids"][mask]
    support = data["valid_spatial_bins"] & (data["occupancy_first_half_s"] >= .05)
    grid = data["bin_centers_cm"][support]
    rates = np.maximum(data["rates_first_half_hz"][mask][:, support], 1e-4)
    drift = np.maximum(data["rates_second_half_hz"][mask][:, support], 1e-4)
    if len(grid) < 2:
        raise ValueError("insufficient first-half spatial support")
    quarter = np.linspace(data["supported_run_intervals"][:, 0].min(), data["supported_run_intervals"][:, 1].max(), 5)
    cal_w, cal_c = run_windows(data, ids, quarter[2], quarter[3])
    test_w, test_c = run_windows(data, ids, quarter[3], quarter[4])
    anchor, counts, excluded = endpoint_counts(data, mask)
    parts = []
    for split in range(3):
        a, b = partition(len(ids), seed_for(seed, row.dataset, row.animal, row.session, split))
        parts.append(dict(split=split, a=a.tolist(), b=b.tolist(), a_ids=ids[a].tolist(), b_ids=ids[b].tolist()))
    freeze = dict(dataset=row.dataset, animal=row.animal, session=row.session, parts=parts,
                  cache_path=str(path), cache_sha256=row.artifact_sha256, seed=seed,
                  created_at_utc=datetime.now(UTC).isoformat(), no_outcome_selection=True)
    output.mkdir(parents=True, exist_ok=False)
    (output/"frozen_populations.json").write_text(json.dumps(freeze, indent=2)+"\n")
    np.savez_compressed(output/"audit_arrays.npz", grid_cm=grid, rates_hz=rates, drift_hz=drift, cell_ids=ids,
                        calibration_windows=cal_w, calibration_counts=cal_c, test_windows=test_w, test_counts=test_c,
                        endpoint_counts=counts, event_indices=anchor.event_index.to_numpy(),
                        endpoint_start_s=anchor.start_s.to_numpy(), endpoint_end_s=anchor.end_s.to_numpy())
    frames, run_rows = [], []
    for part in parts:
        split, a, b = part["split"], part["a"], part["b"]
        calibration = decode(cal_c[:, a], rates[a], grid)
        errors = np.linalg.norm(calibration["mean"]-cal_w[:, 2:], axis=1)
        run_rows.append(dict(split=split, n_cells=len(a), cal_windows=len(cal_w), test_windows=len(test_w),
                             cal_error_cm=float(np.median(errors))))
        cases = [("real", -1, counts[:, a], counts[:, b], None, anchor),
                 ("run_test", -1, test_c[:, a], test_c[:, b], test_w[:, 2:],
                  pd.DataFrame(dict(event_index=np.arange(len(test_w)), start_s=test_w[:, 0], end_s=test_w[:, 1])))]
        for generator in ("sim_matched", "sim_drift"):
            for draw in range(2):
                rng = np.random.default_rng(seed_for(seed, row.dataset, row.animal, row.session, split, generator, draw))
                states = rng.integers(len(grid), size=len(counts))
                source = rates if generator == "sim_matched" else drift * rng.lognormal(0, .4, (len(ids), 1))
                ca = simulate_counts(counts[:, a].sum(axis=1), source[a], states, rng)
                cb = simulate_counts(counts[:, b].sum(axis=1), source[b], states, rng)
                cases.append((generator, draw, ca, cb, grid[states], anchor))
        for source, draw, ca, cb, truth, identity in cases:
            local = readouts(ca, cb, rates[a], rates[b], grid, cal_w[:, 2:], errors, truth)
            for col in identity:
                local[col] = identity[col].to_numpy()
            local["dataset"], local["animal"], local["session"] = row.dataset, row.animal, row.session
            local["split"], local["source"], local["draw"] = split, source, draw
            frames.append(local)
    joined = pd.concat(frames, ignore_index=True)
    joined.to_csv(output/"event_readouts.csv.gz", index=False)
    pd.DataFrame(run_rows).to_csv(output/"run_qc.csv", index=False)
    (output/"excluded_endpoints.json").write_text(json.dumps(excluded, indent=2)+"\n")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status="complete",
                candidates=len(anchor), excluded=len(excluded), input_candidates=int(row.candidates),
                artifact_dir=str(output), input_sha256=row.artifact_sha256,
                readouts_sha256=file_sha256(output/"event_readouts.csv.gz"), dataset_pass=dataset_pass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["pfeiffer_foster", "tanni2022"], required=True)
    parser.add_argument("--frozen-model", type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.dataset == "tanni2022" and (args.frozen_model is None or not args.frozen_model.is_file()):
        raise ValueError("freeze PF model before generating external outcomes")
    inputs = dict(sessions=args.input_dir/"coverage_input_sessions.csv", producer=Path(__file__),
                  protocol=ROOT/"docs/content_stability_protocol.md", frozen_model=args.frozen_model)
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(created_at_utc=datetime.now(UTC).isoformat(), status="running", dataset=args.dataset, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    sessions = pd.read_csv(inputs["sessions"])
    sessions = sessions.loc[sessions.dataset.eq(args.dataset)]
    if sessions.empty or sessions.duplicated(["animal", "session"]).any():
        raise ValueError("empty/duplicate session catalog")
    results = []
    for row in sessions.itertuples(index=False):
        name = f"{row.animal}__{row.session.replace('/', '_')}"
        try:
            result = measure_session(row, args.output_dir/name, args.seed, args.dataset)
        except (ValueError, OSError, KeyError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc))
        results.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(results).to_csv(args.output_dir/"sessions.csv", index=False)
    unchanged = all(file_sha256(value) == manifest["input_file_sha256"][key]
                    for key, value in inputs.items() if value is not None)
    manifest.update(status="complete" if unchanged and all(r["status"] == "complete" for r in results) else "failed",
                    inputs_unchanged=unchanged,
                    results=results, completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if manifest["status"] != "complete":
        raise RuntimeError("session failures; inspect sessions.csv")


if __name__ == "__main__":
    main()
