#!/usr/bin/env python3
"""Independent spatial-kernel, SciPy predictive PMF and posterior reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import nbinom, poisson

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz, recount, seed, SOURCES

METHODS = ("poisson", "gamma_exposure", "gamma_training_drift", "poisson_entropy_matched")


def exposure_from_kernel(occ, shape):
    coords = np.array([(i, j) for i in range(shape[0]) for j in range(shape[1])])
    distances = np.abs(coords[:, None]-coords[None])
    kernel = np.exp(-.5*(np.arange(-6, 7)/1.5)**2)
    kernel /= kernel.sum()
    weights = np.prod(kernel[np.minimum(distances, 6)+6], axis=2)*np.all(distances <= 6, axis=2)
    s, q = weights @ occ, (weights*weights) @ occ
    return np.divide(s*np.maximum(s, .05), q, out=np.zeros_like(s), where=q > 0)


def reconstruct_uncertainty(data):
    shape = (len(data["x_edges_cm"])-1, len(data["y_edges_cm"])-1)
    good, keep = data["valid_spatial_bins"].astype(bool), data["unit_qc_mask"].astype(bool)
    t = [exposure_from_kernel(data[k], shape)[good] for k in ("occupancy_s", "occupancy_first_half_s", "occupancy_second_half_s")]
    mu = [np.maximum(data[k][keep][:, good], 1e-4) for k in ("rates_hz", "rates_first_half_hz", "rates_second_half_hz")]
    v = [m/np.maximum(e, 1e-12)[None] for m, e in zip(mu, t, strict=True)]
    common = (data["occupancy_first_half_s"][good] >= .05) & (data["occupancy_second_half_s"][good] >= .05)
    temporal = np.clip(((mu[1]-mu[2])**2-v[1]-v[2])/2, 0, np.inf)
    temporal[:, ~common] = 0
    return dict(mean_rates=mu[0], quarter1_rates=mu[1], quarter2_rates=mu[2],
                effective_exposure_s=t[0], quarter1_exposure_s=t[1], quarter2_exposure_s=t[2],
                sampling_variance=v[0], temporal_variance=temporal,
                exposure_shape=mu[0]**2/v[0], drift_shape=mu[0]**2/(v[0]+temporal),
                quarter_common_support=common, cell_ids=data["cell_ids"][keep], grid_cm=data["bin_centers_cm"][good])


def scipy_posterior(counts, means, shape=None):
    log_likelihood = np.zeros((len(counts), means.shape[1]))
    for j in range(len(means)):
        if shape is None:
            log_likelihood += poisson.logpmf(counts[:, j, None], .02*means[j])
        else:
            probability = shape[j]/(shape[j]+.02*means[j])
            log_likelihood += nbinom.logpmf(counts[:, j, None], shape[j], probability)
    return np.exp(log_likelihood-logsumexp(log_likelihood, axis=1)[:, None]), log_likelihood


def h(p):
    return -(p*np.log(np.maximum(p, 1e-300))).sum(axis=1)/np.log(p.shape[1])


def metrics(p, grid, truth):
    normalized = (grid-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)
    cells = np.minimum(np.floor(3*normalized).astype(int), 2)
    ids = 3*cells[:, 0]+cells[:, 1]
    regions = np.column_stack([p[:, ids == i].sum(axis=1) for i in range(9)])
    mean = p @ grid
    result = dict(entropy=h(p), width=np.sqrt(np.maximum(np.sum(p*np.sum(grid**2, axis=1), axis=1)-np.sum(mean**2, axis=1), 0)),
                  x_cm=mean[:, 0], y_cm=mean[:, 1], error=np.linalg.norm(mean-truth, axis=1), regional=regions)
    if np.isfinite(truth).all():
        normalized = (truth-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)
        bins = np.clip(np.floor(3*normalized).astype(int), 0, 2)
        target = 3*bins[:, 0]+bins[:, 1]
        result["brier"] = np.sum((regions-np.eye(9)[target])**2, axis=1)
        result["nll"] = -np.log(np.maximum(regions[np.arange(len(p)), target], 1e-300))
    else:
        result["brier"] = result["nll"] = np.full(len(p), np.nan)
    return result


def check_metrics(frame, pa, pb, grid, truth):
    summaries = []
    for side, p in (("a", pa), ("b", pb)):
        values = metrics(p, grid, truth)
        for name, value in values.items():
            if name == "regional":
                for j in range(9):
                    np.testing.assert_allclose(frame[f"{side}_region{j}"], value[:, j], atol=3e-8, rtol=3e-8)
            else:
                np.testing.assert_allclose(frame[f"{side}_{name}"], value, atol=2e-6, rtol=2e-8, equal_nan=True, err_msg=side+name)
        summaries.append(values)
    distance = np.hypot(summaries[0]["x_cm"]-summaries[1]["x_cm"], summaries[0]["y_cm"]-summaries[1]["y_cm"])
    tv = np.sum(np.abs(summaries[0]["regional"]-summaries[1]["regional"]), axis=1)/2
    np.testing.assert_allclose(frame.separation_cm, distance, atol=2e-6, rtol=2e-8)
    np.testing.assert_allclose(frame.regional_tv, tv, atol=3e-8, rtol=3e-8)


def verify_one(row):
    folder = Path(row.artifact_dir)
    output_hashes = json.loads((folder/"outputs.json").read_text())
    if any(file_sha256(folder/k) != v for k, v in output_hashes.items()):
        raise ValueError("measurement output changed")
    frozen = json.loads((folder/"frozen_input.json").read_text())
    prior = Path(frozen["edge_source"])
    if file_sha256(prior/"outputs.json") != frozen["source_outputs_sha256"]:
        raise ValueError("source manifest changed")
    if any(file_sha256(prior/k) != v for k, v in frozen["source_outputs"].items()):
        raise ValueError("edge-support source changed")
    freeze = frozen["freeze"]
    if file_sha256(freeze["encoding_path"]) != freeze["encoding_sha256"]:
        raise ValueError("RUN source changed")
    data = load_npz(freeze["encoding_path"])
    uncertain = load_npz(folder/"uncertainty.npz")
    rebuilt = reconstruct_uncertainty(data)
    for key, value in rebuilt.items():
        np.testing.assert_allclose(uncertain[key], value, atol=1e-8, rtol=1e-9, err_msg=key)
    frame = pd.read_csv(folder/"event_readouts.csv.gz", float_precision="round_trip")
    if frame.duplicated(["source", "split", "method", "event_index"]).any() or len(frame) != row.rows:
        raise ValueError("missing or repeated readouts")
    if set(frame.source) != set(SOURCES) or set(frame.split) != {0, 1, 2} or set(frame.method) != set(METHODS):
        raise ValueError("incomplete sources/splits/methods")
    for name in ("dataset", "animal", "session"):
        if not frame[name].eq(getattr(row, name)).all():
            raise ValueError("recording key mismatch")
    checked, native_windows, posterior_rows, unavailable = 0, 0, 0, 0
    for source in SOURCES:
        src = load_npz(prior/f"{source}_audit.npz")
        counts = np.array([src["counts"][hi-4:hi].sum(0) for hi in src["offsets"][1:]])
        truth = np.array([src["truth_base_cm"][hi-4:hi].mean(0) for hi in src["offsets"][1:]])
        starts = src["starts_s"] + .005*(np.diff(src["offsets"])-4)
        ends = src["starts_s"] + .005*np.diff(src["offsets"])
        if source in ("real", "run_q4"):
            np.testing.assert_array_equal(recount(data["spikes"], uncertain["cell_ids"], starts, ends), counts)
            native_windows += len(counts)
        for split in range(3):
            stored = load_npz(folder/f"{source}_split{split}_audit.npz")
            for name, value in (("counts", counts), ("truth_cm", truth), ("event_ids", src["event_ids"]), ("starts_s", starts)):
                np.testing.assert_allclose(stored[name], value, atol=1e-10, rtol=0, equal_nan=True)
            order = np.random.default_rng(seed(freeze["seed"], "edge_support_partition", row.dataset, row.session, split)).permutation(len(uncertain["cell_ids"]))
            size = len(order)//2
            groups = [np.sort(order[:size]), np.sort(order[size:2*size])]
            rebuilt_p = []
            view = frame.loc[frame.source.eq(source) & frame.split.eq(split)]
            reference = view.loc[view.method.eq("poisson")].set_index("event_index").loc[src["event_ids"]]
            for side, indices in zip(("a", "b"), groups, strict=True):
                np.testing.assert_array_equal(stored[f"{side}_indices"], indices)
                n, mu = counts[:, indices], uncertain["mean_rates"][indices]
                np.testing.assert_array_equal(reference[f"{side}_spikes"], n.sum(1))
                np.testing.assert_array_equal(reference[f"{side}_active"], np.count_nonzero(n, axis=1))
                bank, lp = {}, None
                for name, shape in (("poisson", None), ("gamma_exposure", uncertain["exposure_shape"][indices]),
                                    ("gamma_training_drift", uncertain["drift_shape"][indices])):
                    bank[name], likelihood = scipy_posterior(n, mu, shape)
                    if name == "poisson": lp = likelihood
                target = h(bank["gamma_training_drift"])
                log_t = reference[f"{side}_matched_log_temperature"].to_numpy()
                if not ((log_t >= -20) & (log_t <= 20)).all():
                    raise ValueError("temperature outside frozen search bracket")
                tempered = (lp-lp.max(axis=1, keepdims=True))/np.exp(log_t)[:, None]
                bank["poisson_entropy_matched"] = np.exp(tempered-logsumexp(tempered, axis=1)[:, None])
                success = np.abs(h(bank["poisson_entropy_matched"])-target) <= 1e-7
                np.testing.assert_array_equal(reference[f"{side}_entropy_control_available"], success)
                if not success.all():
                    unavailable += int((~success).sum())
                    centered = lp-lp.max(axis=1, keepdims=True)
                    limits = []
                    for t in (-20, 20):
                        value = centered/np.exp(t)
                        limits.append(h(np.exp(value-logsumexp(value, axis=1)[:, None])))
                    if np.any((~success) & (target >= limits[0]+1e-7) & (target <= limits[1]-1e-7)):
                        raise ValueError("reachable entropy target was missed")
                quarter = []
                for q in (1, 2):
                    bank[f"quarter{q}"], _ = scipy_posterior(n, uncertain[f"quarter{q}_rates"][indices])
                    quarter.append(metrics(bank[f"quarter{q}"], uncertain["grid_cm"], truth)["regional"])
                sensitivity = np.abs(quarter[0]-quarter[1]).sum(1)/2
                np.testing.assert_allclose(reference[f"{side}_encoding_sensitivity_tv"], sensitivity, atol=3e-8)
                for name, p in bank.items():
                    np.testing.assert_allclose(stored[f"{side}_{name}_posterior"], p, atol=3e-8, rtol=3e-8, err_msg=name)
                    posterior_rows += len(p)
                rebuilt_p.append(bank)
            for method in METHODS:
                selected = view.loc[view.method.eq(method)].set_index("event_index")
                np.testing.assert_array_equal(sorted(selected.index), sorted(src["event_ids"]))
                selected = selected.loc[src["event_ids"]]
                np.testing.assert_allclose(selected.original_start_s, starts, atol=1e-10, rtol=0)
                np.testing.assert_allclose(selected.original_end_s, starts+.02, atol=1e-10, rtol=0)
                for name in ("a_encoding_sensitivity_tv", "b_encoding_sensitivity_tv", "a_matched_log_temperature", "b_matched_log_temperature"):
                    np.testing.assert_array_equal(selected[name], reference[name])
                check_metrics(selected, rebuilt_p[0][method], rebuilt_p[1][method], uncertain["grid_cm"], truth)
                checked += len(selected)
    if checked != len(frame):
        raise ValueError("unaudited rows")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status="passed", readout_rows=checked,
                native_endpoint_windows=native_windows, reconstructed_posterior_rows=posterior_rows,
                entropy_population_windows_unavailable=unavailable, readout_sha256=file_sha256(folder/"event_readouts.csv.gz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {f"sessions{i}": p/"measurement_sessions.csv" for i, p in enumerate(args.measurement_dir)}
    inputs["auditor"] = Path(__file__)
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest["status"] = "running"
    (args.output_dir/"independent_audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    results = []
    for folder in args.measurement_dir:
        source = json.loads((folder/"manifest.json").read_text())
        if source["status"] != "complete" or not source["inputs_unchanged"]:
            raise ValueError("unfinished source")
        for key, filename in source["input_file_paths"].items():
            if file_sha256(filename) != source["input_file_sha256"][key]:
                raise ValueError("frozen producer/protocol/input changed")
        sessions = pd.read_csv(folder/"measurement_sessions.csv")
        if len(sessions) != 8 or sessions.animal.nunique() != 4:
            raise ValueError("incomplete cohort")
        for row in sessions.itertuples(index=False):
            try:
                if row.status != "complete": raise ValueError("measurement incomplete")
                result = verify_one(row)
            except (ValueError, AssertionError, OSError, KeyError) as exc:
                result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc))
            results.append(result)
            print(json.dumps(result), flush=True)
            pd.DataFrame(results).to_csv(args.output_dir/"independent_audit_sessions.csv", index=False)
    passed = bool(results) and all(r["status"] == "passed" for r in results)
    unchanged = all(file_sha256(v) == manifest["input_file_sha256"][k] for k, v in inputs.items())
    manifest.update(status="passed" if passed and unchanged else "failed", inputs_unchanged=unchanged, results=results)
    (args.output_dir/"independent_audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not passed or not unchanged: raise ValueError("independent uncertainty audit failed")


if __name__ == "__main__":
    main()
