#!/usr/bin/env python3
"""Evaluate fixed-endpoint temporal context without forcing population agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator
from scipy.special import softmax

from hipporeplayimm.state_space_first_order import _forward_backward_first_order
from hipporeplayimm.state_space_utils import _gaussian_transition_matrix
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz
from scripts.measure_encoding_uncertainty_content import (
    entropy,
    log_predictive,
    match_entropy,
    posterior_metrics,
)

METHODS = ("independent", "diffusion_reset", "diffusion_no_reset", "pooled_static", "entropy_matched")
PRIMARY = "diffusion_reset"


def context_blocks(base):
    base = np.asarray(base)
    if base.ndim != 2 or len(base) < 4 or not np.isfinite(base).all() or np.any(base < 0) or np.any(base != np.floor(base)):
        raise ValueError("need at least four finite integer 5ms count bins")
    n = min(len(base) // 4, 10)
    return base[-4 * n :].reshape(n, 4, base.shape[1]).sum(axis=1)


def reset_transition(gaussian, probability):
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("reset probability outside [0,1]")
    n = gaussian.shape[0]
    if gaussian.shape != (n, n) or n < 2:
        raise ValueError("invalid transition shape")
    # Rank-one reset keeps the exact all-state transition without a dense matrix.
    return LinearOperator(
        (n, n),
        matvec=lambda x: (1 - probability) * (gaussian @ x) + probability * np.sum(x) / n,
        rmatvec=lambda x: (1 - probability) * (gaussian.T @ x) + probability * np.sum(x) / n,
        dtype=float,
    )


def endpoint_bank(blocks, rates, gaussian):
    ll = log_predictive(blocks, rates)
    output = {"independent": softmax(ll[-1]), "pooled_static": softmax(ll.sum(axis=0))}
    for name, reset in ((PRIMARY, 0.25), ("diffusion_no_reset", 0.0)):
        _, log_posterior = _forward_backward_first_order(ll, reset_transition(gaussian, reset))
        output[name] = np.exp(log_posterior[-1])
    return output, ll[-1]


def source_readouts(arrays, groups, gaussian):
    rates, grid = arrays["rates_hz"], arrays["grid_cm"]
    n_events = len(arrays["event_ids"])
    endpoints, truth, context_ms, counts = [], [], [], []
    banks, last_logs = [{m: [] for m in METHODS[:-1]} for _ in groups], [[], []]
    for j, (lo, hi) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)):
        base = arrays["counts"][lo:hi]
        blocks = context_blocks(base)
        endpoints.append(arrays["starts_s"][j] + 0.005 * (hi - lo - 4))
        truth.append(arrays["truth_base_cm"][hi - 4 : hi].mean(axis=0))
        context_ms.append(20 * len(blocks))
        counts.append(blocks[-1])
        for side, group in enumerate(groups):
            bank, ll = endpoint_bank(blocks[:, group], rates[group], gaussian)
            for name in bank:
                banks[side][name].append(bank[name])
            last_logs[side].append(ll)
    truth, counts = np.asarray(truth), np.asarray(counts)
    diagnostics = {}
    for side, group in enumerate(groups):
        banks[side] = {m: np.asarray(p) for m, p in banks[side].items()}
        matched, temperature, available = match_entropy(np.asarray(last_logs[side]), entropy(banks[side][PRIMARY]))
        banks[side]["entropy_matched"] = matched
        label = "ab"[side]
        diagnostics[label + "_spikes"] = counts[:, group].sum(axis=1)
        diagnostics[label + "_active"] = np.count_nonzero(counts[:, group], axis=1)
        diagnostics[label + "_entropy_control_available"] = available
        diagnostics[label + "_matched_log_temperature"] = temperature
    records = []
    for method in METHODS:
        a, b = (posterior_metrics(bank[method], grid, truth) for bank in banks)
        frame = pd.DataFrame(
            dict(
                method=method,
                event_index=arrays["event_ids"],
                original_start_s=endpoints,
                original_end_s=np.asarray(endpoints) + 0.02,
                context_ms=context_ms,
                separation_cm=np.linalg.norm(a["mean"] - b["mean"], axis=1),
                regional_tv=0.5 * np.abs(a["regional"] - b["regional"]).sum(axis=1),
                n_cells_per_group=len(groups[0]),
                **diagnostics,
            )
        )
        for side, value in (("a", a), ("b", b)):
            for metric in ("entropy", "width", "error", "brier", "nll"):
                frame[side + "_" + metric] = value[metric]
            frame[side + "_x_cm"], frame[side + "_y_cm"] = value["mean"].T
            for region in range(9):
                frame[f"{side}_region{region}"] = value["regional"][:, region]
        records.append(frame)
    if not all(len(b[PRIMARY]) == n_events for b in banks):
        raise ValueError("endpoint loss")
    audit = dict(truth_cm=truth, starts_s=endpoints, event_ids=arrays["event_ids"], counts=counts, context_ms=context_ms)
    for side, group, bank in zip(("a", "b"), groups, banks, strict=True):
        audit[side + "_indices"] = group
        audit.update({f"{side}_{method}_posterior": p for method, p in bank.items()})
    return pd.concat(records, ignore_index=True), audit


def measure_session(row, output):
    prior = Path(row.artifact_dir)
    hashes = json.loads((prior / "outputs.json").read_text())
    if any(file_sha256(prior / k) != h for k, h in hashes.items()):
        raise ValueError("source changed")
    frozen = json.loads((prior / "frozen_measurement.json").read_text())
    if file_sha256(frozen["encoding_path"]) != frozen["encoding_sha256"]:
        raise ValueError("RUN source changed")
    training = json.loads((Path(frozen["encoding_path"]).parent / "encoding_manifest.json").read_text())
    if not training["training_only"] or training["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("first-half-only training required")
    target = output / (row.animal + "__" + row.session.replace("/", "_"))
    target.mkdir(exist_ok=False)
    (target / "frozen_input.json").write_text(
        json.dumps(dict(edge_source=str(prior), source_outputs=hashes, source_outputs_sha256=file_sha256(prior / "outputs.json"), freeze=frozen), indent=2) + "\n"
    )
    shared = load_npz(prior / "real_audit.npz")
    grid, rates, ids = (shared[k] for k in ("grid_cm", "rates_hz", "cell_ids"))
    gaussian = _gaussian_transition_matrix(grid, 20.0, 4.0)
    rows = []
    for source in SOURCES:
        arrays = load_npz(prior / f"{source}_audit.npz")
        for key, value in (("grid_cm", grid), ("rates_hz", rates), ("cell_ids", ids)):
            np.testing.assert_array_equal(arrays[key], value)
        for part in frozen["groups"]:
            groups = [np.searchsorted(ids, part[side + "_ids"]) for side in ("a", "b")]
            for side, group in zip(("a", "b"), groups, strict=True):
                np.testing.assert_array_equal(ids[group], part[side + "_ids"])
            if len(groups[0]) != len(groups[1]) or np.intersect1d(*groups).size:
                raise ValueError("populations must be equal and disjoint")
            frame, audit = source_readouts(arrays, groups, gaussian)
            frame["dataset"], frame["animal"], frame["session"] = row.dataset, row.animal, row.session
            frame["source"], frame["split"] = source, part["split"]
            rows.append(frame)
            np.savez_compressed(target / f"{source}_split{part['split']}_audit.npz", **audit)
        print(json.dumps(dict(session=row.session, source=source, status="scored")), flush=True)
    result = pd.concat(rows, ignore_index=True)
    original = pd.read_csv(prior / "edge_readouts.csv.gz")
    original = original.loc[original.policy.eq("raw_endpoint")].sort_values(["source", "split", "event_index"])
    current = result.loc[result.method.eq("independent")].sort_values(["source", "split", "event_index"])
    for key in ("source", "split", "event_index"):
        np.testing.assert_array_equal(current[key], original[key])
    for after, before in (
        ("separation_cm", "separation_cm"),
        ("regional_tv", "regional_tv"),
        ("a_error", "a_original_truth_error_cm"),
        ("b_error", "b_original_truth_error_cm"),
        ("original_start_s", "raw_start_s"),
        ("original_end_s", "raw_end_s"),
    ):
        np.testing.assert_allclose(current[after], original[before], atol=1e-8, rtol=1e-10, equal_nan=True)
    result.to_csv(target / "event_readouts.csv.gz", index=False)
    (target / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in target.iterdir() if p.is_file()}, indent=2) + "\n")
    if any(file_sha256(prior / k) != h for k, h in hashes.items()):
        raise ValueError("source changed during measurement")
    return dict(
        dataset=row.dataset, animal=row.animal, session=row.session, status="complete", rows=len(result), artifact_dir=str(target), selected_candidates=row.selected_candidates
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--development-gates", type=Path)
    args = p.parse_args()
    source_manifest = json.loads((args.measurement_dir / "manifest.json").read_text())
    if source_manifest["status"] != "complete" or not source_manifest["inputs_unchanged"]:
        raise ValueError("incomplete source manifest")
    table = pd.read_csv(args.measurement_dir / "measurement_sessions.csv")
    if len(table) != 8 or table.animal.nunique() != 4 or table.dataset.nunique() != 1 or table.duplicated(["animal", "session"]).any() or not table.status.eq("complete").all():
        raise ValueError("all eight unique sessions/four rats required")
    if table.dataset.iloc[0] != "pfeiffer_foster":
        if args.development_gates is None:
            raise ValueError("external run requires audited PF development gates")
        gates = pd.read_csv(args.development_gates)
        advance = gates.loc[gates.gate.eq("advance_external_validation")]
        if len(advance) != 1 or str(advance.iloc[0]["passed"]).lower() != "true":
            raise ValueError("PF development failed; no external advancement")
    inputs = dict(
        source=args.measurement_dir / "measurement_sessions.csv",
        script=Path(__file__),
        protocol=ROOT / "docs/temporal_endpoint_content_protocol.md",
        recursion=ROOT / "src/hipporeplayimm/state_space_first_order.py",
        transition=ROOT / "src/hipporeplayimm/state_space_utils.py",
        metrics=ROOT / "scripts/measure_encoding_uncertainty_content.py",
        development_gates=args.development_gates,
    )
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", primary_method=PRIMARY, sigma_cm=20, reset_probability=0.25, context_cap_ms=200, scientific_goal_achieved=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for row in table.itertuples(index=False):
        try:
            result = measure_session(row, args.output_dir)
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc))
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir / "measurement_sessions.csv", index=False)
        print(json.dumps(result), flush=True)
    unchanged = all(file_sha256(v) == manifest["input_file_sha256"][k] for k, v in inputs.items() if v is not None)
    complete = unchanged and all(r["status"] == "complete" for r in rows)
    manifest.update(status="complete" if complete else "failed", inputs_unchanged=unchanged, results=rows)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not complete:
        raise RuntimeError("temporal endpoint measurement incomplete")


if __name__ == "__main__":
    main()
