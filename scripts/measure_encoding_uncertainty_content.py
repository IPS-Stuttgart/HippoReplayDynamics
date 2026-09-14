#!/usr/bin/env python3
"""Fixed-time empirical rate-uncertainty decoding with an entropy control."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, softmax

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz
from scripts.measure_population_content_stability import tile_ids

METHODS = ("poisson", "gamma_exposure", "gamma_training_drift", "poisson_entropy_matched")


def effective_exposure(occupancy, shape, sigma=1.5):
    occupancy = np.asarray(occupancy, float)
    if occupancy.ndim != 1 or occupancy.size != np.prod(shape) or not np.isfinite(occupancy).all() or np.any(occupancy < 0):
        raise ValueError("invalid raw occupancy")
    kernel = gaussian_filter(np.eye(len(occupancy)).reshape((len(occupancy), *shape)),
                             sigma=(0, sigma, sigma), mode="constant", truncate=4).reshape(len(occupancy), -1)
    smoothed, squared = kernel.T @ occupancy, (kernel**2).T @ occupancy
    return np.divide(smoothed*np.maximum(smoothed, .05), squared,
                     out=np.zeros_like(smoothed), where=squared > 0)


def rate_uncertainty(data):
    support, keep = data["valid_spatial_bins"].astype(bool), data["unit_qc_mask"].astype(bool)
    grid_shape = (len(data["x_edges_cm"])-1, len(data["y_edges_cm"])-1)
    exposure = [effective_exposure(data[name], grid_shape)[support] for name in
                ("occupancy_s", "occupancy_first_half_s", "occupancy_second_half_s")]
    means = [np.maximum(data[name][keep][:, support], 1e-4) for name in
             ("rates_hz", "rates_first_half_hz", "rates_second_half_hz")]
    if not np.all(exposure[0] > 0):
        raise ValueError("no effective training exposure at a decoded bin")
    variance = [mu/np.maximum(t, 1e-12)[None, :] for mu, t in zip(means, exposure, strict=True)]
    both = (data["occupancy_first_half_s"][support] >= .05) & (data["occupancy_second_half_s"][support] >= .05)
    temporal = np.maximum(0, .5*(means[1]-means[2])**2 - .5*(variance[1]+variance[2]))
    temporal[:, ~both] = 0
    return dict(mean_rates=means[0], quarter1_rates=means[1], quarter2_rates=means[2],
                effective_exposure_s=exposure[0], quarter1_exposure_s=exposure[1], quarter2_exposure_s=exposure[2],
                sampling_variance=variance[0], temporal_variance=temporal,
                exposure_shape=means[0]**2/variance[0], drift_shape=means[0]**2/(variance[0]+temporal),
                quarter_common_support=both, cell_ids=data["cell_ids"][keep], grid_cm=data["bin_centers_cm"][support])


def log_predictive(counts, means, shapes=None, dt=.02):
    """Stable integer-count Gamma-Poisson PMF including the Poisson limit."""
    counts, means = np.asarray(counts), np.asarray(means, float)
    if counts.ndim != 2 or means.ndim != 2 or counts.shape[1] != len(means):
        raise ValueError("incompatible count/rate arrays")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be nonnegative integers")
    if not np.isfinite(means).all() or np.any(means <= 0) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid rates or bin duration")
    if shapes is None:
        return counts @ np.log(means*dt) - dt*means.sum(axis=0) - gammaln(counts+1).sum(axis=1)[:, None]
    shapes = np.asarray(shapes, float)
    if shapes.shape != means.shape or not np.isfinite(shapes).all() or np.any(shapes <= 0):
        raise ValueError("invalid Gamma shapes")
    out = np.zeros((len(counts), means.shape[1]))
    for cell in range(len(means)):
        maximum = int(counts[:, cell].max(initial=0))
        n = np.arange(maximum+1)[:, None]
        alpha, mu = shapes[cell], means[cell]*dt
        rising = np.vstack([np.zeros_like(alpha), np.cumsum(np.log1p(np.arange(maximum)[:, None]/alpha), axis=0)])
        lookup = rising + n*np.log(mu) - (n+alpha)*np.log1p(mu/alpha) - gammaln(n+1)
        out += lookup[counts[:, cell].astype(int)]
    return out


def entropy(p):
    return -(p*np.log(np.maximum(p, 1e-300))).sum(axis=1)/np.log(p.shape[1])


def match_entropy(log_likelihood, target):
    log_likelihood = log_likelihood-log_likelihood.max(axis=1, keepdims=True)
    lo, hi = np.full(len(target), -20.), np.full(len(target), 20.)
    for _ in range(70):
        mid = (lo+hi)/2
        current = entropy(softmax(log_likelihood/np.exp(mid)[:, None], axis=1))
        lo = np.where(current < target, mid, lo)
        hi = np.where(current < target, hi, mid)
    log_temp = (lo+hi)/2
    p = softmax(log_likelihood/np.exp(log_temp)[:, None], axis=1)
    achieved = np.abs(entropy(p)-target) <= 1e-7
    return p, log_temp, achieved


def posterior_metrics(p, grid, truth):
    mean = p @ grid
    regional = p @ np.eye(9)[tile_ids(grid)]
    result = dict(mean=mean, regional=regional, entropy=entropy(p),
                  width=np.sqrt(np.maximum(p @ np.sum(grid**2, axis=1)-np.sum(mean**2, axis=1), 0)),
                  error=np.linalg.norm(mean-truth, axis=1))
    if np.isfinite(truth).all():
        normalized = (truth-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)
        bins = np.clip((3*normalized).astype(int), 0, 2)
        truth_region = 3*bins[:, 0]+bins[:, 1]
        result["brier"] = np.sum((regional-np.eye(9)[truth_region])**2, axis=1)
        result["nll"] = -np.log(np.maximum(regional[np.arange(len(p)), truth_region], 1e-300))
    elif np.isnan(truth).all():
        result["brier"] = result["nll"] = np.full(len(p), np.nan)
    else:
        raise ValueError("partially missing truth")
    return result


def endpoint_arrays(source):
    counts, truth, starts = [], [], []
    for j in range(len(source["event_ids"])):
        lo, hi = source["offsets"][j:j+2]
        if hi-lo < 4:
            raise ValueError("fewer than four endpoint base bins")
        counts.append(source["counts"][hi-4:hi].sum(axis=0))
        truth.append(source["truth_base_cm"][hi-4:hi].mean(axis=0))
        starts.append(source["starts_s"][j]+.005*(hi-lo-4))
    return np.asarray(counts), np.asarray(truth), np.asarray(starts)


def readout(counts, truth, means, uncertainty, grid, groups):
    populations, diagnostics = [], {}
    for side, ids in zip(("a", "b"), groups, strict=True):
        n, mu = counts[:, ids], means[ids]
        logs = {"poisson": log_predictive(n, mu),
                "gamma_exposure": log_predictive(n, mu, uncertainty["exposure_shape"][ids]),
                "gamma_training_drift": log_predictive(n, mu, uncertainty["drift_shape"][ids])}
        probs = {method: softmax(value, axis=1) for method, value in logs.items()}
        p, log_t, good = match_entropy(logs["poisson"], entropy(probs["gamma_training_drift"]))
        probs["poisson_entropy_matched"] = p
        diagnostics[f"{side}_entropy_control_available"] = good
        diagnostics[f"{side}_matched_log_temperature"] = log_t
        for quarter in (1, 2):
            pq = softmax(log_predictive(n, uncertainty[f"quarter{quarter}_rates"][ids]), axis=1)
            probs[f"quarter{quarter}"] = pq
        quarter_regions = [probs[f"quarter{q}"] @ np.eye(9)[tile_ids(grid)] for q in (1, 2)]
        diagnostics[f"{side}_encoding_sensitivity_tv"] = np.abs(quarter_regions[0]-quarter_regions[1]).sum(axis=1)/2
        diagnostics[f"{side}_spikes"] = n.sum(axis=1)
        diagnostics[f"{side}_active"] = np.count_nonzero(n, axis=1)
        populations.append(probs)
    rows = []
    for method in METHODS:
        pa, pb = (p[method] for p in populations)
        a, b = posterior_metrics(pa, grid, truth), posterior_metrics(pb, grid, truth)
        frame = pd.DataFrame(dict(method=method, separation_cm=np.linalg.norm(a["mean"]-b["mean"], axis=1),
            regional_tv=np.abs(a["regional"]-b["regional"]).sum(axis=1)/2, **diagnostics))
        for side, result in (("a", a), ("b", b)):
            for key in ("entropy", "width", "error", "brier", "nll"):
                frame[f"{side}_{key}"] = result[key]
            frame[f"{side}_x_cm"], frame[f"{side}_y_cm"] = result["mean"].T
            for j in range(9):
                frame[f"{side}_region{j}"] = result["regional"][:, j]
        rows.append(frame)
    return rows, populations


def measure_session(row, output):
    folder = Path(row.artifact_dir)
    recorded = json.loads((folder/"outputs.json").read_text())
    if any(file_sha256(folder/k) != v for k, v in recorded.items()):
        raise ValueError("edge-support source changed")
    freeze = json.loads((folder/"frozen_measurement.json").read_text())
    if file_sha256(freeze["encoding_path"]) != freeze["encoding_sha256"]:
        raise ValueError("encoding changed")
    data = load_npz(freeze["encoding_path"])
    training_meta = json.loads((Path(freeze["encoding_path"]).parent/"encoding_manifest.json").read_text())
    if not training_meta["training_only"] or training_meta["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("training-only encoder required")
    uncertain = rate_uncertainty(data)
    target = output/(row.animal+"__"+row.session.replace("/", "_"))
    target.mkdir(exist_ok=False)
    np.savez_compressed(target/"uncertainty.npz", **uncertain)
    (target/"frozen_input.json").write_text(json.dumps(dict(edge_source=str(folder), freeze=freeze,
        source_outputs=recorded, source_outputs_sha256=file_sha256(folder/"outputs.json"),
        uncertainty_sha256=file_sha256(target/"uncertainty.npz")), indent=2)+"\n")
    rows, unavailable = [], 0
    for source in SOURCES:
        arrays = load_npz(folder/f"{source}_audit.npz")
        counts, truth, starts = endpoint_arrays(arrays)
        for name, expected in (("cell_ids", uncertain["cell_ids"]), ("grid_cm", uncertain["grid_cm"]), ("rates_hz", uncertain["mean_rates"])):
            np.testing.assert_array_equal(arrays[name], expected)
        for part in freeze["groups"]:
            groups = [np.searchsorted(uncertain["cell_ids"], part[f"{s}_ids"]) for s in ("a", "b")]
            for side, group in zip(("a", "b"), groups, strict=True):
                np.testing.assert_array_equal(uncertain["cell_ids"][group], part[f"{side}_ids"])
            local, posteriors = readout(counts, truth, uncertain["mean_rates"], uncertain, uncertain["grid_cm"], groups)
            for frame in local:
                frame["event_index"] = arrays["event_ids"]
                frame["original_start_s"], frame["original_end_s"] = starts, starts+.02
                frame["dataset"], frame["animal"], frame["session"] = row.dataset, row.animal, row.session
                frame["source"], frame["split"] = source, part["split"]
                frame["n_cells_per_group"] = len(groups[0])
                rows.append(frame)
            unavailable += int((~local[0].a_entropy_control_available | ~local[0].b_entropy_control_available).sum())
            audit = dict(counts=counts, truth_cm=truth, starts_s=starts, event_ids=arrays["event_ids"],
                         a_indices=groups[0], b_indices=groups[1])
            for side, bank in zip(("a", "b"), posteriors, strict=True):
                audit.update({f"{side}_{name}_posterior": p for name, p in bank.items()})
            np.savez_compressed(target/f"{source}_split{part['split']}_audit.npz", **audit)
    result = pd.concat(rows, ignore_index=True)
    # The original Poisson baseline must reproduce the prior independently
    # audited raw endpoints, not a new window or new candidate subset.
    prior = pd.read_csv(folder/"edge_readouts.csv.gz")
    prior = prior.loc[prior.policy.eq("raw_endpoint")].sort_values(["source", "split", "event_index"])
    current = result.loc[result.method.eq("poisson")].sort_values(["source", "split", "event_index"])
    for key in ("event_index", "source", "split"):
        np.testing.assert_array_equal(current[key], prior[key])
    for name, before in (("separation_cm", "separation_cm"), ("regional_tv", "regional_tv"),
                         ("a_error", "a_original_truth_error_cm"), ("b_error", "b_original_truth_error_cm"),
                         ("a_entropy", "a_entropy"), ("b_entropy", "b_entropy")):
        np.testing.assert_allclose(current[name], prior[before], rtol=1e-9, atol=1e-8, equal_nan=True)
    result.to_csv(target/"event_readouts.csv.gz", index=False)
    (target/"outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in target.iterdir() if p.is_file()}, indent=2)+"\n")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status="complete", rows=len(result),
                entropy_control_unavailable=unavailable, cells=len(uncertain["cell_ids"]),
                spatial_bins=len(uncertain["grid_cm"]), artifact_dir=str(target), selected_candidates=row.selected_candidates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    inputs = dict(source=args.measurement_dir/"measurement_sessions.csv", script=Path(__file__),
                  protocol=ROOT/"docs/encoding_uncertainty_content_protocol.md")
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", started_at_utc=datetime.now(UTC).isoformat())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    table, rows = pd.read_csv(inputs["source"]), []
    if len(table) != 8 or table.animal.nunique() != 4:
        raise ValueError("eight sessions/four rats required")
    for row in table.itertuples(index=False):
        try:
            if row.status != "complete":
                raise ValueError("upstream input incomplete")
            result = measure_session(row, args.output_dir)
        except (OSError, KeyError, ValueError, AssertionError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc))
        rows.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir/"measurement_sessions.csv", index=False)
    unchanged = all(file_sha256(v) == manifest["input_file_sha256"][k] for k, v in inputs.items())
    complete = all(r["status"] == "complete" for r in rows)
    manifest.update(status="complete" if unchanged and complete else "failed", inputs_unchanged=unchanged,
                    finished_at_utc=datetime.now(UTC).isoformat(), results=rows)
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged or not complete:
        raise ValueError("encoding-uncertainty measurement incomplete")


if __name__ == "__main__":
    main()
