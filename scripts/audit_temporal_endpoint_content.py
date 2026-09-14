#!/usr/bin/env python3
"""Independently reconstruct temporal endpoint posteriors with dense filtering."""

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

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz, recount
from scripts.audit_encoding_uncertainty_content import check_metrics, h, scipy_posterior

METHODS = ("independent", "diffusion_reset", "diffusion_no_reset", "pooled_static", "entropy_matched")


def dense_gaussian(grid):
    points = np.asarray(grid, dtype=float)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("finite nonempty grid required")
    left, right = points[:, None, :], points[None, :, :]
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 20.0)
    # The runtime wrapper scales coordinates before subtracting. Preserve that
    # cutoff convention: raw squared distance differs at some radius boundaries.
    standardized = np.abs(left / scale - right / scale) / (20.0 / scale)
    distance = np.hypot.reduce(standardized, axis=2)
    kernel = np.exp(-0.5 * distance * distance) * (distance <= 4.0)
    return kernel / kernel.sum(axis=0)


def native_context_intervals(src, event, native, source):
    lo, hi = src["offsets"][event : event + 2]
    n_base = int(hi - lo)
    if source == "real":
        found = np.flatnonzero(native["candidate_event_indices"] == src["event_ids"][event])
        if len(found) != 1:
            raise ValueError("native event identity missing or duplicated")
        begin = int(native["candidate_offsets"][found[0]])
        left = native["candidate_base_starts_s"][begin : begin + n_base]
        end = left[-1] + native["candidate_base_durations_s"][begin + n_base - 1]
        edges = np.r_[left, end]
    elif source == "run_q4":
        edges = src["starts_s"][event] + np.arange(n_base + 1) * 0.005
        edges[-1] = src["starts_s"][event] + n_base * 0.005
    else:
        raise ValueError("no native spike intervals for simulations")
    index = np.arange(n_base - 4 * min(10, n_base // 4), n_base, 4)
    # Read both edges from the source clock; left+20ms can include an edge spike.
    return edges[index], edges[index + 4]


def dense_bank(events, rates, grid):
    """Batch independent SciPy PMFs and probability prediction/log Bayes updates."""
    matrix = dense_gaussian(grid)
    result = {m: np.empty((len(events), len(grid))) for m in METHODS[:-1]}
    last_logs = np.empty_like(result["independent"])
    for length in sorted({len(x) for x in events}):
        index = np.array([i for i, x in enumerate(events) if len(x) == length])
        counts = np.stack([events[i] for i in index])
        independent, ll = scipy_posterior(counts.reshape(-1, rates.shape[0]), rates)
        independent = independent.reshape(len(index), length, len(grid))
        ll = ll.reshape(len(index), length, len(grid))
        last_logs[index] = ll[:, -1]
        result["independent"][index] = independent[:, -1]
        joint = ll.sum(axis=1)
        result["pooled_static"][index] = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        for name, reset in (("diffusion_reset", 0.25), ("diffusion_no_reset", 0.0)):
            alpha = independent[:, 0]
            for t in range(1, length):
                predicted = (1 - reset) * (alpha @ matrix.T) + reset / len(grid)
                with np.errstate(divide="ignore"):
                    weights = np.log(predicted) + ll[:, t]
                alpha = np.exp(weights - logsumexp(weights, axis=1, keepdims=True))
            result[name][index] = alpha
    return result, last_logs


def verify_one(row):
    folder = Path(row.artifact_dir)
    outputs = json.loads((folder / "outputs.json").read_text())
    if any(file_sha256(folder / k) != v for k, v in outputs.items()):
        raise ValueError("output changed")
    frozen = json.loads((folder / "frozen_input.json").read_text())
    prior, freeze = Path(frozen["edge_source"]), frozen["freeze"]
    if file_sha256(prior / "outputs.json") != frozen["source_outputs_sha256"] or any(file_sha256(prior / k) != v for k, v in frozen["source_outputs"].items()):
        raise ValueError("source changed")
    if file_sha256(freeze["encoding_path"]) != freeze["encoding_sha256"]:
        raise ValueError("native cache changed")
    native = load_npz(freeze["encoding_path"])
    frame = pd.read_csv(folder / "event_readouts.csv.gz", float_precision="round_trip")
    if len(frame) != row.rows or frame.duplicated(["source", "split", "method", "event_index"]).any():
        raise ValueError("duplicate or missing measurements")
    if set(frame.source) != set(SOURCES) or set(frame.method) != set(METHODS) or set(frame.split) != {0, 1, 2}:
        raise ValueError("incomplete sources/methods/splits")
    for name in ("dataset", "animal", "session"):
        if not frame[name].eq(getattr(row, name)).all():
            raise ValueError("identity mismatch")
    verified, native_blocks, posterior_count = 0, 0, 0
    for source in SOURCES:
        src = load_npz(prior / f"{source}_audit.npz")
        rates, grid, ids = (src[k] for k in ("rates_hz", "grid_cm", "cell_ids"))
        blocks, truth, starts, context_ms, raw_starts, raw_ends = [], [], [], [], [], []
        for j, (lo, hi) in enumerate(zip(src["offsets"][:-1], src["offsets"][1:], strict=True)):
            n = min(10, (hi - lo) // 4)
            if n < 1:
                raise ValueError("missing fixed endpoint")
            blocks.append(np.stack([src["counts"][i : i + 4].sum(axis=0) for i in range(hi - n * 4, hi, 4)]))
            starts.append(src["starts_s"][j] + 0.005 * (hi - lo - 4))
            truth.append(src["truth_base_cm"][hi - 4 : hi].mean(axis=0))
            context_ms.append(20 * n)
            if source in ("real", "run_q4"):
                left, right = native_context_intervals(src, j, native, source)
                raw_starts.append(left)
                raw_ends.append(right)
                native_blocks += int(n)
        if raw_starts:
            np.testing.assert_array_equal(np.concatenate(blocks), recount(native["spikes"], ids, np.concatenate(raw_starts), np.concatenate(raw_ends)))
        counts, truth, starts = np.stack([x[-1] for x in blocks]), np.asarray(truth), np.asarray(starts)
        for part in freeze["groups"]:
            split = part["split"]
            groups = []
            for side in ("a", "b"):
                lookup = {int(k): i for i, k in enumerate(ids)}
                groups.append(np.array([lookup[int(k)] for k in part[side + "_ids"]]))
            if len(groups[0]) != len(groups[1]) or np.intersect1d(*groups).size:
                raise ValueError("invalid population pair")
            stored = load_npz(folder / f"{source}_split{split}_audit.npz")
            for key, expected in (
                ("counts", counts),
                ("truth_cm", truth),
                ("starts_s", starts),
                ("context_ms", context_ms),
                ("event_ids", src["event_ids"]),
                ("a_indices", groups[0]),
                ("b_indices", groups[1]),
            ):
                np.testing.assert_allclose(stored[key], expected, atol=1e-10, rtol=0, equal_nan=True)
            view = frame.loc[frame.source.eq(source) & frame.split.eq(split)]
            reference = view.loc[view.method.eq("independent")].set_index("event_index").loc[src["event_ids"]]
            banks = []
            for side, group in zip(("a", "b"), groups, strict=True):
                bank, ll = dense_bank([x[:, group] for x in blocks], rates[group], grid)
                temperature = reference[side + "_matched_log_temperature"].to_numpy()
                if not ((temperature >= -20) & (temperature <= 20)).all():
                    raise ValueError("temperature outside frozen bracket")
                tempered = (ll - ll.max(axis=1, keepdims=True)) / np.exp(temperature)[:, None]
                bank["entropy_matched"] = np.exp(tempered - logsumexp(tempered, axis=1, keepdims=True))
                good = np.abs(h(bank["entropy_matched"]) - h(bank["diffusion_reset"])) <= 1e-7
                np.testing.assert_array_equal(reference[side + "_entropy_control_available"], good)
                np.testing.assert_array_equal(reference[side + "_spikes"], counts[:, group].sum(axis=1))
                np.testing.assert_array_equal(reference[side + "_active"], np.count_nonzero(counts[:, group], axis=1))
                for method, posterior in bank.items():
                    np.testing.assert_allclose(stored[f"{side}_{method}_posterior"], posterior, atol=5e-9, rtol=2e-8, err_msg=method)
                    posterior_count += len(posterior)
                banks.append(bank)
            for method in METHODS:
                selected = view.loc[view.method.eq(method)].set_index("event_index")
                np.testing.assert_array_equal(sorted(selected.index), sorted(src["event_ids"]))
                selected = selected.loc[src["event_ids"]]
                np.testing.assert_allclose(selected.original_start_s, starts, atol=1e-10, rtol=0)
                np.testing.assert_allclose(selected.original_end_s, starts + 0.02, atol=1e-10, rtol=0)
                np.testing.assert_array_equal(selected.context_ms, context_ms)
                if not selected.n_cells_per_group.eq(len(groups[0])).all():
                    raise ValueError("population size changed")
                for side in ("a", "b"):
                    for key in ("spikes", "active", "entropy_control_available", "matched_log_temperature"):
                        np.testing.assert_array_equal(selected[side + "_" + key], reference[side + "_" + key])
                check_metrics(selected, banks[0][method], banks[1][method], grid, truth)
                verified += len(selected)
        print(json.dumps(dict(session=row.session, source=source, status="verified")), flush=True)
    if verified != len(frame):
        raise ValueError("unaudited rows")
    return dict(
        dataset=row.dataset,
        animal=row.animal,
        session=row.session,
        status="passed",
        rows=verified,
        native_context_blocks=native_blocks,
        posterior_rows=posterior_count,
        readout_sha256=file_sha256(folder / "event_readouts.csv.gz"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.measurement_dir / "manifest.json").read_text())
    if source["status"] != "complete" or not source["inputs_unchanged"]:
        raise ValueError("source measurement incomplete")
    for key, value in source["input_file_paths"].items():
        if file_sha256(value) != source["input_file_sha256"][key]:
            raise ValueError("producer input changed")
    inputs = dict(
        producer=args.measurement_dir / "manifest.json",
        sessions=args.measurement_dir / "measurement_sessions.csv",
        auditor=Path(__file__),
        metric_auditor=ROOT / "scripts/audit_encoding_uncertainty_content.py",
        runtime_gaussian_wrapper=ROOT / "src/hipporeplayimm/candidate_active_support_validation.py",
        runtime_scalar_wrapper=ROOT / "src/hipporeplayimm/state_space_gaussian_scalar_validation.py",
        runtime_transition_base=ROOT / "src/hipporeplayimm/state_space_utils.py",
    )
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(inputs["sessions"])
    if len(table) != 8 or table.animal.nunique() != 4 or not table.status.eq("complete").all():
        raise ValueError("incomplete recording cohort")
    results = []
    for row in table.itertuples(index=False):
        results.append(verify_one(row))
        pd.DataFrame(results).to_csv(args.output_dir / "independent_audit_sessions.csv", index=False)
        print(json.dumps(results[-1]), flush=True)
    if not all(file_sha256(v) == manifest["input_file_sha256"][k] for k, v in inputs.items()):
        raise ValueError("audit inputs changed")
    manifest.update(status="passed", results=results, all_endpoint_posteriors_independently_rebuilt=True)
    (args.output_dir / "independent_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
