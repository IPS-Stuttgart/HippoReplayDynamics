#!/usr/bin/env python3
"""Choose endpoint context using own-population cell-held-out prediction only."""

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

from hipporeplayimm.state_space_first_order import _forward_backward_first_order
from hipporeplayimm.state_space_utils import _gaussian_transition_matrix
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz, seed
from scripts.measure_encoding_uncertainty_content import entropy, log_predictive, match_entropy, posterior_metrics
from scripts.measure_temporal_endpoint_content import context_blocks, reset_transition

METHODS = ("independent", "unconditional_context", "predictive_context", "entropy_matched")
PRIMARY = "predictive_context"


def cell_folds(cell_ids, session, split, side):
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or len(ids) < 6 or len(np.unique(ids)) != len(ids):
        raise ValueError("at least six unique own-population cells required")
    if side not in ("a", "b"):
        raise ValueError("invalid population side")
    order = np.random.default_rng(seed(20260914, "predictive_context", session, split, side)).permutation(len(ids))
    fold = np.empty(len(ids), int)
    fold[order] = np.arange(len(ids)) % 3
    return fold


def infer_two(blocks, rates, transition):
    ll = log_predictive(blocks, rates)
    if not len(ll):
        raise ValueError("missing endpoint")
    if len(ll) == 1:
        lp = ll[-1] - logsumexp(ll[-1])
        return np.stack([lp, lp]), ll[-1]
    _, lp = _forward_backward_first_order(ll, transition)
    return np.stack([ll[-1] - logsumexp(ll[-1]), lp[-1]]), ll[-1]


def integrated_score(log_posterior, joint_log_likelihood):
    lp, ll = np.asarray(log_posterior), np.asarray(joint_log_likelihood)
    if lp.ndim != 1 or ll.shape != lp.shape or not np.isfinite(lp).all() or not np.isfinite(ll).all():
        raise ValueError("finite aligned log posterior and likelihood required")
    if not np.isclose(logsumexp(lp), 0, atol=1e-8, rtol=0):
        raise ValueError("posterior must be normalized")
    return float(logsumexp(lp + ll))


def choose_context(fold_delta, counts):
    delta, n = np.asarray(fold_delta), np.asarray(counts)
    if delta.shape != (3,) or not np.isfinite(delta).all() or n.ndim != 1 or not np.isfinite(n).all() or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("three finite predictive differences and integer endpoint counts required")
    return bool(np.median(delta) > 1e-10 and n.sum() >= 3 and np.count_nonzero(n) >= 2)


def own_population(blocks, rates, transition, fold):
    fold = np.asarray(fold)
    if fold.shape != (len(rates),) or set(fold) != {0, 1, 2} or any((fold == k).sum() < 2 for k in range(3)):
        raise ValueError("three valid cell folds required")
    scores, sizes = [], []
    for k in range(3):
        training, validation = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        lp, _ = infer_two(blocks[:, training], rates[training], transition)
        heldout = log_predictive(blocks[-1:, validation], rates[validation])[0]
        scores.append([integrated_score(p, heldout) for p in lp])
        sizes.append(len(validation))
    scores, sizes = np.asarray(scores), np.asarray(sizes)
    delta = (scores[:, 1] - scores[:, 0]) / sizes
    use_context = choose_context(delta, blocks[-1])
    lp, ll = infer_two(blocks, rates, transition)
    bank = dict(independent=np.exp(lp[0]), unconditional_context=np.exp(lp[1]), predictive_context=np.exp(lp[int(use_context)]))
    return bank, dict(fold_scores=scores, fold_delta=delta, fold_sizes=sizes, use_context=use_context, median_delta=float(np.median(delta))), ll


def source_readouts(arrays, groups, folds, transition):
    rates, grid = arrays["rates_hz"], arrays["grid_cm"]
    banks = [{m: [] for m in METHODS[:-1]} for _ in range(2)]
    decisions, logs = [[], []], [[], []]
    counts, truth, starts, context = [], [], [], []
    for j, (lo, hi) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)):
        blocks = context_blocks(arrays["counts"][lo:hi])
        counts.append(blocks[-1])
        truth.append(arrays["truth_base_cm"][hi - 4 : hi].mean(axis=0))
        starts.append(arrays["starts_s"][j] + 0.005 * (hi - lo - 4))
        context.append(20 * len(blocks))
        for i, group in enumerate(groups):
            bank, diagnostic, ll = own_population(blocks[:, group], rates[group], transition, folds[i])
            for m, value in bank.items():
                banks[i][m].append(value)
            decisions[i].append(diagnostic)
            logs[i].append(ll)
    counts, truth, starts = np.asarray(counts), np.asarray(truth), np.asarray(starts)
    audit = dict(counts=counts, truth_cm=truth, starts_s=starts, context_ms=context, event_ids=arrays["event_ids"])
    diagnostics = {}
    for i, (side, group) in enumerate(zip(("a", "b"), groups, strict=True)):
        banks[i] = {m: np.asarray(v) for m, v in banks[i].items()}
        matched, temperature, good = match_entropy(np.asarray(logs[i]), entropy(banks[i][PRIMARY]))
        banks[i]["entropy_matched"] = matched
        diagnostics[side + "_entropy_control_available"] = good
        diagnostics[side + "_matched_log_temperature"] = temperature
        diagnostics[side + "_spikes"] = counts[:, group].sum(axis=1)
        diagnostics[side + "_active"] = np.count_nonzero(counts[:, group], axis=1)
        diagnostics[side + "_use_context"] = [d["use_context"] for d in decisions[i]]
        diagnostics[side + "_median_predictive_delta"] = [d["median_delta"] for d in decisions[i]]
        audit[side + "_indices"], audit[side + "_fold_ids"] = group, folds[i]
        for key in ("fold_scores", "fold_delta", "fold_sizes"):
            audit[side + "_" + key] = np.asarray([d[key] for d in decisions[i]])
        audit.update({f"{side}_{m}_posterior": p for m, p in banks[i].items()})
    frames = []
    for method in METHODS:
        a, b = (posterior_metrics(p[method], grid, truth) for p in banks)
        frame = pd.DataFrame(
            dict(
                method=method,
                event_index=arrays["event_ids"],
                original_start_s=starts,
                original_end_s=starts + 0.02,
                context_ms=context,
                n_cells_per_group=len(groups[0]),
                separation_cm=np.linalg.norm(a["mean"] - b["mean"], axis=1),
                regional_tv=0.5 * np.abs(a["regional"] - b["regional"]).sum(axis=1),
                **diagnostics,
            )
        )
        for side, p in (("a", a), ("b", b)):
            for key in ("entropy", "width", "error", "brier", "nll"):
                frame[side + "_" + key] = p[key]
            frame[side + "_x_cm"], frame[side + "_y_cm"] = p["mean"].T
            for region in range(9):
                frame[f"{side}_region{region}"] = p["regional"][:, region]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), audit


def measure_session(row, output):
    source = Path(row.artifact_dir)
    hashes = json.loads((source / "outputs.json").read_text())
    if any(file_sha256(source / k) != v for k, v in hashes.items()):
        raise ValueError("source changed")
    freeze = json.loads((source / "frozen_measurement.json").read_text())
    if file_sha256(freeze["encoding_path"]) != freeze["encoding_sha256"]:
        raise ValueError("encoding changed")
    training_path = Path(freeze["encoding_path"]).parent / "encoding_manifest.json"
    training = json.loads(training_path.read_text())
    if not training["training_only"] or training["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("first-half training required")
    target = output / (row.animal + "__" + row.session.replace("/", "_"))
    target.mkdir(exist_ok=False)
    (target / "frozen_input.json").write_text(
        json.dumps(
            dict(
                edge_source=str(source),
                source_outputs=hashes,
                source_outputs_sha256=file_sha256(source / "outputs.json"),
                freeze=freeze,
                training_manifest_sha256=file_sha256(training_path),
            ),
            indent=2,
        )
        + "\n"
    )
    shared = load_npz(source / "real_audit.npz")
    grid, rates, ids = (shared[k] for k in ("grid_cm", "rates_hz", "cell_ids"))
    transition = reset_transition(_gaussian_transition_matrix(grid, 20.0, 4.0), 0.25)
    rows = []
    for name in SOURCES:
        arrays = load_npz(source / f"{name}_audit.npz")
        for key, expected in (("grid_cm", grid), ("rates_hz", rates), ("cell_ids", ids)):
            np.testing.assert_array_equal(arrays[key], expected)
        for part in freeze["groups"]:
            groups = [np.searchsorted(ids, part[side + "_ids"]) for side in ("a", "b")]
            for side, group in zip(("a", "b"), groups, strict=True):
                np.testing.assert_array_equal(ids[group], part[side + "_ids"])
            if len(groups[0]) != len(groups[1]) or np.intersect1d(*groups).size:
                raise ValueError("populations not equal and disjoint")
            folds = [cell_folds(ids[g], row.session, part["split"], side) for side, g in zip(("a", "b"), groups, strict=True)]
            frame, audit = source_readouts(arrays, groups, folds, transition)
            frame["dataset"], frame["animal"], frame["session"] = row.dataset, row.animal, row.session
            frame["source"], frame["split"] = name, part["split"]
            rows.append(frame)
            np.savez_compressed(target / f"{name}_split{part['split']}_audit.npz", **audit)
        print(json.dumps(dict(session=row.session, source=name, status="scored")), flush=True)
    frame = pd.concat(rows, ignore_index=True)
    previous = pd.read_csv(source / "edge_readouts.csv.gz")
    previous = previous.loc[previous.policy.eq("raw_endpoint")].sort_values(["source", "split", "event_index"])
    current = frame.loc[frame.method.eq("independent")].sort_values(["source", "split", "event_index"])
    for key in ("source", "split", "event_index"):
        np.testing.assert_array_equal(current[key], previous[key])
    for key, old in (
        ("separation_cm", "separation_cm"),
        ("regional_tv", "regional_tv"),
        ("a_error", "a_original_truth_error_cm"),
        ("b_error", "b_original_truth_error_cm"),
        ("original_start_s", "raw_start_s"),
        ("original_end_s", "raw_end_s"),
    ):
        np.testing.assert_allclose(current[key], previous[old], rtol=1e-10, atol=1e-8, equal_nan=True)
    frame.to_csv(target / "event_readouts.csv.gz", index=False)
    (target / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in target.iterdir() if p.is_file()}, indent=2) + "\n")
    if any(file_sha256(source / k) != v for k, v in hashes.items()):
        raise ValueError("source changed during scoring")
    return dict(
        dataset=row.dataset, animal=row.animal, session=row.session, status="complete", rows=len(frame), artifact_dir=str(target), selected_candidates=row.selected_candidates
    )


def validate_development(gates_path):
    if gates_path is None:
        raise ValueError("independent validation requires predictive-context development pass")
    path = Path(gates_path)
    report = json.loads((path.parent / "manifest.json").read_text())
    if report.get("status") != "complete" or report.get("primary_method") != PRIMARY or report.get("primary_advanced") is not True:
        raise ValueError("wrong or failed development method")
    if report["output_sha256"].get(path.name) != file_sha256(path):
        raise ValueError("changed development gates")
    for key, value in report["input_file_paths"].items():
        if file_sha256(value) != report["input_file_sha256"][key]:
            raise ValueError("changed development report input")
    producer_path = report["input_file_paths"]["producer"]
    producer = json.loads(Path(producer_path).read_text())
    audit = json.loads(Path(report["input_file_paths"]["audit"]).read_text())
    if producer.get("primary_method") != PRIMARY or producer.get("status") != "complete" or set(r["dataset"] for r in producer["results"]) != {"pfeiffer_foster"}:
        raise ValueError("not a completed PF predictive-context development run")
    if audit.get("status") != "passed" or audit["input_file_sha256"]["producer"] != file_sha256(producer_path):
        raise ValueError("development audit not matched")
    checks = pd.read_csv(path)
    selected = checks.loc[checks.gate.eq("advance_external_validation")]
    diagnostic = {f"{source}_{side}_predicts_known_gain" for source in ("run_q4", "sim_late_jump") for side in ("a", "b")}
    if len(selected) != 1 or checks.gate.duplicated().any() or not checks.passed.eq(True).all() or not diagnostic.issubset(checks.gate):
        raise ValueError("development did not advance with diagnostic gates")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--development-gates", type=Path)
    args = p.parse_args()
    source = json.loads((args.measurement_dir / "manifest.json").read_text())
    table = pd.read_csv(args.measurement_dir / "measurement_sessions.csv")
    if (
        source["status"] != "complete"
        or not source["inputs_unchanged"]
        or len(table) != 8
        or table.animal.nunique() != 4
        or table.dataset.nunique() != 1
        or table.duplicated(["animal", "session"]).any()
        or not table.status.eq("complete").all()
    ):
        raise ValueError("complete eight-session/four-animal input required")
    if table.dataset.iloc[0] != "pfeiffer_foster":
        validate_development(args.development_gates)
    inputs = dict(
        source=args.measurement_dir / "manifest.json",
        sessions=args.measurement_dir / "measurement_sessions.csv",
        producer=Path(__file__),
        protocol=ROOT / "docs/predictive_context_content_protocol.md",
        temporal=ROOT / "scripts/measure_temporal_endpoint_content.py",
        likelihood=ROOT / "scripts/measure_encoding_uncertainty_content.py",
        recursion=ROOT / "src/hipporeplayimm/state_space_first_order.py",
        transition_wrapper=ROOT / "src/hipporeplayimm/candidate_active_support_validation.py",
        development_gates=args.development_gates,
    )
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(
        status="running",
        primary_method=PRIMARY,
        folds=3,
        seed=20260914,
        minimum_endpoint_spikes=3,
        minimum_endpoint_active=2,
        minimum_median_predictive_delta=1e-10,
        scientific_goal_achieved=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for row in table.itertuples(index=False):
        try:
            result = measure_session(row, args.output_dir)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc))
        results.append(result)
        pd.DataFrame(results).to_csv(args.output_dir / "measurement_sessions.csv", index=False)
        print(json.dumps(result), flush=True)
    unchanged = all(file_sha256(v) == manifest["input_file_sha256"][k] for k, v in inputs.items() if v is not None)
    complete = unchanged and all(r["status"] == "complete" for r in results)
    manifest.update(status="complete" if complete else "failed", inputs_unchanged=unchanged, results=results)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not complete:
        raise RuntimeError("predictive context measurement incomplete")


if __name__ == "__main__":
    main()
