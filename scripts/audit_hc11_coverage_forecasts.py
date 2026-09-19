#!/usr/bin/env python3
"""Frozen-operator hc-11 coverage test with genuine future-cell prediction."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import build_script_provenance, file_sha256
from audit_2d_count_conditioned_prediction import folds

from hipporeplayimm.independent_rejected_forecast import BASELINES, forecast_distributions, score_predictions, seed
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.learned_assembly_prediction import fit_learned_assembly
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from hipporeplayimm.training_continuity import geometry, overlapping_counts, poisson_map

ID = ["dataset", "animal", "session"]
SCORE_COLUMNS = ["score_dynamic"] + ["score_" + b for b in BASELINES]


def classify_native(base, durations, rates, centers, train, topology, length):
    centers = np.asarray(centers, float)
    if topology not in ("linear", "circular") or not np.isfinite(length) or length <= 0:
        raise ValueError("declared native topology and positive track length required")
    if centers.ndim != 1 or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise ValueError("increasing native 1D centers required")
    windows = overlapping_counts(base[:, train], durations)
    path = poisson_map(windows, rates[train])
    locations = centers[path]
    jumps = np.diff(locations)
    if topology == "circular":
        jumps = (jumps + length / 2) % length - length / 2
    unwrapped = np.r_[0.0, np.cumsum(jumps)] if len(path) else np.empty(0)
    grid = np.column_stack((unwrapped, np.zeros(len(unwrapped))))
    result = geometry(np.arange(len(path)), grid, windows)
    if topology == "circular" and result["longest_run_frames"]:
        start = result["run_start_frame"]
        end = start + result["longest_run_frames"] - 1
        result["run_displacement_cm"] = float(abs((locations[end] - locations[start] + length / 2) % length - length / 2))
        result["geometric_pass"] = result["longest_run_frames"] >= 10 and result["run_displacement_cm"] >= 40 - 1e-9
        if result["longest_run_frames"] >= 10:
            result["failure_reason"] = "pass" if result["geometric_pass"] else "insufficient_displacement"
    return result, path


def process(record, bank, destination):
    identity = {k: record[k] for k in ID}
    output = Path(destination) / record["session"]
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = identity | {"status": "running"}
    source = Path(bank) / record["session"]
    try:
        for name, digest in record["output_sha256"].items():
            if file_sha256(source / name) != digest:
                raise ValueError("bank output hash mismatch")
        cache = np.load(source / "cache.npz", allow_pickle=False)
        selected = pd.read_csv(source / "selection.csv")
        selected.to_csv(output / "selection.csv", index=False)
        labels, paths = [], {}
        for event in selected.itertuples(index=False):
            eid = int(event.event_id)
            for split in range(5):
                row = identity | {"event_id": eid, "split": split, "topology": record["topology"]}
                for level, key in (("full", "train"), ("half", "half")):
                    label, path = classify_native(
                        cache[f"base_{eid}"],
                        cache[f"durations_{eid}"],
                        cache["rates"],
                        cache["centers"],
                        cache[f"{key}_{split}"],
                        record["topology"],
                        record["track_length_cm"],
                    )
                    row.update({level + "_" + k: v for k, v in label.items()})
                    paths[f"{level}_{eid}_{split}"] = path
                row["lost_with_thinning"] = row["full_geometric_pass"] and not row["half_geometric_pass"]
                row["lost_supported"] = row["lost_with_thinning"] and row["half_valid_frames"] >= 10
                labels.append(row)
        table = pd.DataFrame(labels)
        table.to_csv(output / "labels.csv", index=False)
        np.savez_compressed(output / "paths.npz", **paths)
        indexed = table.set_index(["event_id", "split"])
        rows, fit_rows = [], []
        for fold, test, calibration, excluded in folds(selected):
            fit = fit_learned_assembly([cache[f"counts_{eid}"] for eid in calibration.event_id], 50, seed(*identity.values(), "fit", fold))
            fit_rows.append(
                {
                    "fold": fold,
                    "test_ids": test.event_id.tolist(),
                    "calibration_ids": calibration.event_id.tolist(),
                    "excluded_ids": excluded.event_id.tolist(),
                    "converged": bool(fit.converged),
                    "restart": fit.restart,
                    "restart_objectives": fit.restart_objectives,
                    "restart_converged": fit.restart_converged,
                    "n_calibration_bins": fit.n_calibration_bins,
                    "initialization_with_replacement": fit.initialization_with_replacement,
                }
            )
            (output / "folds.json").write_text(json.dumps(fit_rows, indent=2) + "\n")
            np.savez_compressed(
                output / f"fit_{fold}.npz",
                probabilities=fit.probabilities,
                initial=fit.initial,
                transition=fit.transition,
                occupancy=fit.occupancy,
                global_probability=fit.global_probability,
                objective_trace=fit.objective_trace,
            )
            if not fit.converged:
                raise ValueError("unconverged frozen K50 fit")
            operator = NeuralOperator(fit.initial, fit.transition, fit.occupancy)
            null = OccupancyMatchedNull.from_operator(operator)
            (output / f"null_{fold}.json").write_text(json.dumps(null.diagnostics(), indent=2) + "\n")
            for event in test.itertuples(index=False):
                eid = int(event.event_id)
                counts = cache[f"counts_{eid}"]
                for split in range(5):
                    held = cache[f"held_{split}"]
                    for level, key in (("full", "train"), ("half", "half")):
                        train = cache[f"{key}_{split}"]
                        forecasts = forecast_distributions(counts[:, train], fit.probabilities[train], operator, null, horizons=(2,))
                        scores = score_predictions(forecasts, counts[:, held], fit.probabilities[held], fit.global_probability[held])
                        common = indexed.loc[(eid, split)].to_dict() | {
                            "event_id": eid,
                            "split": split,
                            "level": level,
                            "fold": fold,
                            "horizon": 2,
                            "model": "learned_hmm",
                            "arm": "frozen_operator",
                            "n_train_cells": len(train),
                            "n_heldout_cells": len(held),
                            "duration_s": event.duration_s,
                            "forecast_uses_future": False,
                            "forecast_uses_heldout": False,
                            "classification_uses_heldout": False,
                            "detector_uses_heldout": False,
                        }
                        if scores:
                            rows.append(common | scores[0] | {"status": "scored"})
                        else:
                            rows.append(common | {"status": "insufficient_full_bins", "n_target_bins": 0, "n_heldout_target_spikes": 0, **{c: np.nan for c in SCORE_COLUMNS}})
            pd.DataFrame(rows).to_csv(output / "scores.csv.gz", index=False)
            print(json.dumps(identity | {"fold": fold, "cumulative_rows": len(rows)}), flush=True)
        if len(rows) != len(selected) * 10:
            raise ValueError("incomplete event/split/coverage rows")
        result.update(status="complete", rows=len(rows), events=len(selected))
    except (ValueError, OSError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        result.update(status="failed", error=repr(exc))
    finally:
        result["runtime_s"] = time.monotonic() - started
        result["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p.name != "session_manifest.json"}
        (output / "session_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(scores):
    keys = ID + ["event_id", "split", "level"]
    if scores.empty or scores.duplicated(keys).any():
        raise ValueError("nonempty unique forecast rows required")
    finite = scores.status.eq("scored")
    if not np.isfinite(scores.loc[finite, SCORE_COLUMNS]).all().all():
        raise ValueError("nonfinite forecast scores")
    parts = []
    groups = {
        "all": np.ones(len(scores), bool),
        "lost_with_thinning": scores.lost_with_thinning,
        "lost_supported": scores.lost_supported,
        "full_geometric_pass": scores.full_geometric_pass,
    }
    for group, mask in groups.items():
        local = scores[finite & mask]
        for baseline in BASELINES:
            table = local[keys + ["n_heldout_target_spikes", "n_target_bins"]].copy()
            table["group"], table["contrast"] = group, "dynamic_minus_" + baseline
            table["delta"] = (local.score_dynamic - local["score_" + baseline]).to_numpy()
            table["delta_per_spike"] = table.delta / table.n_heldout_target_spikes.replace(0, np.nan)
            parts.append(table)
    split_table = pd.concat(parts, ignore_index=True)
    factors = ["level", "group", "contrast"]
    events = split_table.groupby(ID + ["event_id"] + factors, as_index=False).agg(
        delta=("delta", "median"),
        delta_per_spike=("delta_per_spike", "median"),
        qualifying_splits=("split", "nunique"),
        informative_splits=("delta_per_spike", "count"),
    )
    sessions = events.groupby(ID + factors, as_index=False).agg(
        delta=("delta", "mean"),
        delta_per_spike=("delta_per_spike", "mean"),
        events=("event_id", "nunique"),
        informative_events=("delta_per_spike", "count"),
    )
    animals = sessions.groupby(["dataset", "animal"] + factors, as_index=False).agg(
        delta=("delta", "mean"),
        delta_per_spike=("delta_per_spike", "mean"),
        sessions=("session", "nunique"),
        informative_sessions=("delta_per_spike", "count"),
        events=("events", "sum"),
    )
    rows = []
    for key, group in animals.groupby(["dataset"] + factors):
        ids = dict(zip(["dataset"] + factors, key, strict=True))
        for metric in ("delta", "delta_per_spike"):
            v = group.loc[np.isfinite(group[metric]), metric].to_numpy()
            if not len(v):
                continue
            samples = np.asarray(list(itertools.product(range(len(v)), repeat=len(v))))
            lo, hi = np.quantile(v[samples].mean(axis=1), [0.025, 0.975])
            rows.append(
                ids
                | {
                    "metric": metric,
                    "mean": float(v.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "animals": len(v),
                    "positive_animals": int((v > 0).sum()),
                    "interval_scope": "descriptive_animal_bootstrap_fixed_events",
                }
            )
    return split_table, events, sessions, animals, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--bank-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("use one to eight workers")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("clean committed source required")
    source = args.bank_dir / "preflight_manifest.json"
    if file_sha256(source) != args.bank_manifest_sha256:
        raise ValueError("pinned bank manifest mismatch")
    bank = json.loads(source.read_text())
    if bank["status"] != "complete" or not bank["all_sessions_eligible"] or len(bank["sessions"]) != 8:
        raise ValueError("frozen all-eight-session preflight required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_script_provenance(input_paths={"bank_manifest": source, "protocol": ROOT / "docs/hc11_coverage_forecast_protocol.md"})
    manifest.update(
        status="running",
        experiment="hc11_independent_coverage_forecasts",
        arm="frozen_operator",
        reduced_population_refit="not_run",
        created_at_utc=datetime.now(UTC).isoformat(),
        sessions=[],
    )
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process, r, args.bank_dir, output) for r in bank["sessions"]]
        for job in as_completed(jobs):
            record = job.result()
            manifest["sessions"].append(record)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    complete = all(r["status"] == "complete" for r in manifest["sessions"])
    gates = [{"gate": "all_sessions_complete", "passed": complete}]
    if complete:
        scores = pd.concat([pd.read_csv(output / r["session"] / "scores.csv.gz") for r in bank["sessions"]], ignore_index=True)
        expected = sum(r["n_selected"] for r in bank["sessions"]) * 10
        keys = ID + ["event_id", "split", "level"]
        gates += [
            {"gate": "nonempty_exact_rows", "passed": expected > 0 and len(scores) == expected and not scores.duplicated(keys).any()},
            {"gate": "all_four_animals", "passed": scores.animal.nunique() == 4},
            {"gate": "each_session_has_forecasts", "passed": scores.loc[scores.status.eq("scored"), "session"].nunique() == 8},
            {"gate": "only_scored_or_explicit_short", "passed": scores.status.isin(["scored", "insufficient_full_bins"]).all()},
            {
                "gate": "no_future_or_evaluation_input",
                "passed": not scores[["forecast_uses_future", "forecast_uses_heldout", "classification_uses_heldout", "detector_uses_heldout"]].any().any(),
            },
        ]
        if not all(g["passed"] for g in gates):
            raise ValueError("technical completeness gate failed")
        scores.to_csv(output / "event_scores.csv.gz", index=False)
        tables = summarize(scores)
        for name, table in zip(("split_contrasts", "event_contrasts", "by_session", "by_animal", "summary"), tables, strict=True):
            table.to_csv(output / (name + ".csv"), index=False)
        labels = scores[scores.level.eq("full")]
        label_summary = labels.groupby(ID, as_index=False).agg(
            events=("event_id", "nunique"),
            full_pass_fraction=("full_geometric_pass", "mean"),
            half_pass_fraction=("half_geometric_pass", "mean"),
            lost_fraction=("lost_with_thinning", "mean"),
            lost_supported_fraction=("lost_supported", "mean"),
        )
        label_summary.to_csv(output / "classification_summary.csv", index=False)
    pd.DataFrame(gates).to_csv(output / "technical_gates.csv", index=False)
    manifest.update(status="complete" if complete else "failed", completed_at_utc=datetime.now(UTC).isoformat())
    manifest["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
