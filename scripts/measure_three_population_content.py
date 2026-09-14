#!/usr/bin/env python3
"""Measure A/B-only diagnostic features and independent C validation outcomes."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_population_content_stability import (
    decode,
    endpoint_counts,
    run_windows,
    seed_for,
    simulate_counts,
)

POOLED_FEATURES = ["pooled_spikes", "pooled_active", "pooled_entropy", "pooled_width_scaled",
                   "pooled_local_mass", "n_observed_cells"]
FULL_FEATURES = POOLED_FEATURES + [f"{side}_{metric}" for side in ("a", "b") for metric in
    ("spikes", "active", "entropy", "width_scaled", "local_mass", "x_scaled", "y_scaled")] + [
        "b_mass_near_a", "ab_regional_tv", "ab_separation_scaled"]


def partition_three(n, seed):
    if n < 15:
        raise ValueError("fewer than five eligible cells per disjoint group")
    indices = np.random.default_rng(seed).permutation(n)
    k = n//3
    return tuple(np.sort(indices[i*k:(i+1)*k]) for i in range(3))


def mass_near(posterior, grid, mean, radius):
    distances = np.linalg.norm(grid[None, :, :] - mean[:, None, :], axis=2)
    return np.sum(posterior * (distances <= radius), axis=1)


def observed_features(ca, cb, ra, rb, grid):
    """C counts, maps and outcomes cannot enter this function."""
    if ca.shape != cb.shape or ra.shape != rb.shape:
        raise ValueError("A and B require equal cell counts and aligned states")
    diagonal = float(np.linalg.norm(np.ptp(grid, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("degenerate spatial support")
    radius = .15*diagonal
    a, b = decode(ca, ra, grid), decode(cb, rb, grid)
    pooled = decode(np.column_stack([ca, cb]), np.concatenate([ra, rb]), grid)
    frame = pd.DataFrame(dict(
        pooled_spikes=(ca+cb).sum(axis=1), pooled_active=(ca > 0).sum(axis=1)+(cb > 0).sum(axis=1),
        pooled_entropy=pooled["entropy"], pooled_width_scaled=pooled["width"]/diagonal,
        pooled_local_mass=mass_near(pooled["p"], grid, pooled["mean"], radius),
        n_observed_cells=2*ca.shape[1], grid_diagonal_cm=diagonal, support_radius_cm=radius,
        b_mass_near_a=mass_near(b["p"], grid, a["mean"], radius),
        ab_regional_tv=.5*np.abs(a["regional"]-b["regional"]).sum(axis=1),
        ab_separation_scaled=np.linalg.norm(a["mean"]-b["mean"], axis=1)/diagonal))
    for side, result, counts in (("a", a, ca), ("b", b, cb)):
        frame[f"{side}_spikes"] = counts.sum(axis=1)
        frame[f"{side}_active"] = (counts > 0).sum(axis=1)
        frame[f"{side}_entropy"] = result["entropy"]
        frame[f"{side}_width_scaled"] = result["width"]/diagonal
        frame[f"{side}_local_mass"] = mass_near(result["p"], grid, result["mean"], radius)
        for dim, name in enumerate(("x", "y")):
            frame[f"{side}_{name}_scaled"] = (result["mean"][:, dim]-grid[:, dim].min())/diagonal
    return frame, a


def validation_outcomes(cc, rc, grid, a, truth=None):
    c = decode(cc, rc, grid)
    radius = .15*float(np.linalg.norm(np.ptp(grid, axis=0)))
    mass = mass_near(c["p"], grid, a["mean"], radius)
    return pd.DataFrame(dict(
        c_support=mass >= .5, c_mass_near_a=mass, c_spikes=cc.sum(axis=1), c_active=(cc > 0).sum(axis=1),
        c_entropy=c["entropy"], ac_regional_tv=.5*np.abs(a["regional"]-c["regional"]).sum(axis=1),
        ac_separation_cm=np.linalg.norm(a["mean"]-c["mean"], axis=1),
        c_width_cm=c["width"], a_width_cm=a["width"],
        a_truth_error_cm=np.linalg.norm(a["mean"]-truth, axis=1) if truth is not None else np.nan,
        c_truth_error_cm=np.linalg.norm(c["mean"]-truth, axis=1) if truth is not None else np.nan,
        a_x_cm=a["mean"][:, 0], a_y_cm=a["mean"][:, 1], c_x_cm=c["mean"][:, 0], c_y_cm=c["mean"][:, 1]))


def measure_session(row, output, seed):
    path = Path(row.artifact_path)
    if file_sha256(path) != row.artifact_sha256:
        raise ValueError("source cache checksum differs")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    mask = data["unit_qc_mask"].astype(bool)
    ids = data["cell_ids"][mask]
    parts = []
    for split in range(3):
        a, b, c = partition_three(len(ids), seed_for(seed, "three_population", row.dataset, row.animal, row.session, split))
        parts.append(dict(split=split, a=a.tolist(), b=b.tolist(), c=c.tolist(),
                          a_ids=ids[a].tolist(), b_ids=ids[b].tolist(), c_ids=ids[c].tolist()))
    support = data["valid_spatial_bins"] & (data["occupancy_first_half_s"] >= .05)
    grid = data["bin_centers_cm"][support]
    if len(grid) < 2:
        raise ValueError("insufficient first-half spatial support")
    rates = np.maximum(data["rates_first_half_hz"][mask][:, support], 1e-4)
    drift = np.maximum(data["rates_second_half_hz"][mask][:, support], 1e-4)
    quarter = np.linspace(data["supported_run_intervals"][:, 0].min(), data["supported_run_intervals"][:, 1].max(), 5)
    cal_w, cal_c = run_windows(data, ids, quarter[2], quarter[3])
    test_w, test_c = run_windows(data, ids, quarter[3], quarter[4])
    anchors, counts, excluded = endpoint_counts(data, mask)
    output.mkdir(parents=True, exist_ok=False)
    freeze = dict(dataset=row.dataset, animal=row.animal, session=row.session, parts=parts,
                  cache_path=str(path), cache_sha256=row.artifact_sha256, seed=seed,
                  created_at_utc=datetime.now(UTC).isoformat(), no_population_outcome_selection=True)
    (output/"frozen_populations.json").write_text(json.dumps(freeze, indent=2)+"\n")
    arrays = dict(grid_cm=grid, rates_hz=rates, drift_hz=drift, cell_ids=ids,
                  calibration_windows=cal_w, calibration_counts=cal_c, test_windows=test_w, test_counts=test_c,
                  endpoint_counts=counts, event_indices=anchors.event_index.to_numpy(),
                  endpoint_start_s=anchors.start_s.to_numpy(), endpoint_end_s=anchors.end_s.to_numpy())
    frames, run_rows = [], []
    for part in parts:
        split, a, b, c = part["split"], part["a"], part["b"], part["c"]
        for side, indices in (("a", a), ("b", b), ("c", c)):
            cal = decode(cal_c[:, indices], rates[indices], grid)
            run_rows.append(dict(split=split, population=side, n_cells=len(indices),
                calibration_error_cm_median=float(np.median(np.linalg.norm(cal["mean"]-cal_w[:, 2:], axis=1))),
                cal_windows=len(cal_w), test_windows=len(test_w)))
        cases = [("real", -1, counts, None, anchors),
                 ("run_test", -1, test_c, test_w[:, 2:], pd.DataFrame(dict(
                     event_index=np.arange(len(test_w)), start_s=test_w[:, 0], end_s=test_w[:, 1])))]
        for generator in ("sim_matched", "sim_drift"):
            for draw in range(2):
                rng = np.random.default_rng(seed_for(seed, "three_population_sim", row.dataset, row.animal,
                                                    row.session, split, generator, draw))
                states = rng.integers(len(grid), size=len(counts))
                source = rates if generator == "sim_matched" else drift*rng.lognormal(0, .4, (len(ids), 1))
                simulated = simulate_counts(counts.sum(axis=1), source, states, rng)
                prefix = f"split{split}_{generator}_{draw}"
                arrays[f"{prefix}_counts"] = simulated
                arrays[f"{prefix}_states"] = states
                cases.append((generator, draw, simulated, grid[states], anchors))
        for source, draw, observed, truth, identity in cases:
            features, a_decoded = observed_features(observed[:, a], observed[:, b], rates[a], rates[b], grid)
            outcomes = validation_outcomes(observed[:, c], rates[c], grid, a_decoded, truth)
            local = pd.concat([features, outcomes], axis=1)
            for col in identity:
                local[col] = identity[col].to_numpy()
            local["dataset"], local["animal"], local["session"] = row.dataset, row.animal, row.session
            local["split"], local["source"], local["draw"] = split, source, draw
            local["eligible_cells"], local["n_cells_per_group"] = len(ids), len(a)
            frames.append(local)
    np.savez_compressed(output/"audit_arrays.npz", **arrays)
    pd.concat(frames, ignore_index=True).to_csv(output/"event_readouts.csv.gz", index=False)
    pd.DataFrame(run_rows).to_csv(output/"run_qc.csv", index=False)
    (output/"excluded_endpoints.json").write_text(json.dumps(excluded, indent=2)+"\n")
    if file_sha256(path) != row.artifact_sha256:
        raise ValueError("cache changed during measurement")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status="complete",
                candidates=len(anchors), excluded=len(excluded), input_candidates=int(row.candidates),
                artifact_dir=str(output), input_sha256=row.artifact_sha256,
                readouts_sha256=file_sha256(output/"event_readouts.csv.gz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["pfeiffer_foster", "autopi_ca1"], required=True)
    parser.add_argument("--frozen-model", type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.dataset == "autopi_ca1":
        if args.frozen_model is None or not args.frozen_model.is_file():
            raise ValueError("freeze PF model before generating external outcomes")
        frozen = json.loads(args.frozen_model.read_text())
        if frozen.get("training_dataset") != "pfeiffer_foster" or frozen.get("status") != "frozen":
            raise ValueError("invalid PF freeze")
    inputs = dict(sessions=args.input_dir/"coverage_input_sessions.csv", producer=Path(__file__),
                  protocol=ROOT/"docs/three_population_content_protocol.md", frozen_model=args.frozen_model,
                  shared_measure=ROOT/"scripts/measure_population_content_stability.py")
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
            if hasattr(row, "status") and row.status not in ("cached", "complete"):
                raise ValueError(f"source_not_cached:{row.status}:{getattr(row, 'reason', '')}")
            result = measure_session(row, args.output_dir/name, args.seed)
        except (ValueError, OSError, KeyError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc),
                          input_candidates=getattr(row, "candidates", np.nan), candidates=0)
        results.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(results).to_csv(args.output_dir/"sessions.csv", index=False)
    unchanged = all(file_sha256(path) == manifest["input_file_sha256"][key]
                    for key, path in inputs.items() if path is not None)
    manifest.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged,
                    all_sessions_attempted=True, failed_sessions=sum(row["status"] != "complete" for row in results),
                    results=results, completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("input changed")


if __name__ == "__main__":
    main()
