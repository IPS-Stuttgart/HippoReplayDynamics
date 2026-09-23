#!/usr/bin/env python3
"""Match fine-bin spike counts while varying sampled neural coverage."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.conditional_spatial_prediction import identity_likelihood
from hipporeplayimm.independent_rejected_forecast import ID, forecast_distributions, full_counts, score_predictions, seed
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from hipporeplayimm.training_continuity import geometry, overlapping_counts, poisson_map
from scripts._provenance import build_script_provenance, file_sha256

DRAWS = 5
ARMS = ("full", "half_neurons", "matched_spikes")
KEY = ID + ["event_id", "split"]


def match_counts(counts, targets, rng):
    counts, targets = np.asarray(counts), np.asarray(targets)
    if counts.ndim != 2 or targets.shape != (len(counts),):
        raise ValueError("aligned counts and targets required")
    if not np.isfinite(counts).all() or not np.isfinite(targets).all():
        raise ValueError("finite counts required")
    if (counts < 0).any() or (targets < 0).any() or (counts != np.floor(counts)).any() or (targets != np.floor(targets)).any():
        raise ValueError("nonnegative integer counts required")
    if (targets > counts.sum(axis=1)).any():
        raise ValueError("cannot draw more spikes than observed")
    output = np.zeros_like(counts, dtype=np.int64)
    for t, (row, n) in enumerate(zip(counts, targets, strict=True)):
        if n:
            output[t] = rng.multivariate_hypergeometric(row.astype(np.int64), int(n), method="marginals")
    if not np.array_equal(output.sum(axis=1), targets) or (output > counts).any():
        raise AssertionError("finite-population count matching failed")
    return output


def conditional_path(counts, rates):
    if len(counts) == 0:
        return np.empty(0, dtype=np.int32)
    return identity_likelihood(counts, rates).argmax(axis=1).astype(np.int32)


def contrasts(rows):
    if rows.empty or rows.duplicated(KEY + ["arm", "draw"]).any():
        raise ValueError("unique nonempty rows required")
    expected = {(a, -1) for a in ARMS[:2]} | {("matched_spikes", d) for d in range(DRAWS)}
    for _, g in rows.groupby(KEY, sort=False):
        if set(g[["arm", "draw"]].itertuples(index=False, name=None)) != expected:
            raise ValueError("incomplete matched arms")
        if g.n_heldout_target_spikes.nunique() != 1 or g.n_target_bins.nunique() != 1:
            raise ValueError("unequal evaluation targets")
        half = g[g.arm.eq("half_neurons")].iloc[0]
        matched = g[g.arm.eq("matched_spikes")]
        if not matched.n_inference_spikes.eq(half.n_inference_spikes).all() or not matched.edge_supported_frames.eq(half.edge_supported_frames).all():
            raise ValueError("count/support mismatch")
    value = rows.copy()
    value["geometric_pass"] = value.geometric_pass.astype(float)
    denominator = value.n_heldout_target_spikes.replace(0, np.nan)
    value["predictive_per_spike"] = value.score_dynamic / denominator
    value["advantage_per_spike"] = (value.score_dynamic - value.score_matched_own) / denominator
    metrics = ["geometric_pass", "predictive_per_spike", "advantage_per_spike", "n_active_inference_cells"]
    averaged = value.groupby(KEY + ["arm"], as_index=False)[metrics].mean()
    wide = averaged.pivot(index=KEY, columns="arm", values=metrics)
    result = wide.index.to_frame(index=False)
    for metric in metrics:
        for arm in ARMS:
            result[metric + "__" + arm] = wide[metric][arm].to_numpy()
        result[metric + "__matched_minus_half"] = (wide[metric].matched_spikes - wide[metric].half_neurons).to_numpy()
    result["group"] = "all"
    lost = (wide.geometric_pass.full == 1) & (wide.geometric_pass.half_neurons == 0)
    return pd.concat([result, result.loc[lost.to_numpy()].assign(group="conditional_full_pass_half_fail")], ignore_index=True)


def summarize(paired):
    metrics = [c for c in paired if "__" in c]
    events = paired.groupby(ID + ["event_id", "group"], as_index=False)[metrics].mean()
    sessions = events.groupby(ID + ["group"], as_index=False)[metrics].mean()
    animals = sessions.groupby(["dataset", "animal", "group"], as_index=False)[metrics].mean()
    summaries, loo = [], []
    for (dataset, group), g in animals.groupby(["dataset", "group"]):
        for metric in metrics:
            local = g[np.isfinite(g[metric])].sort_values("animal")
            v = local[metric].to_numpy()
            if not len(v):
                continue
            indices = np.array(list(itertools.product(range(len(v)), repeat=len(v))))
            lo, hi = np.quantile(v[indices].mean(axis=1), [0.025, 0.975])
            summaries.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "metric": metric,
                    "mean": float(v.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "animals": len(v),
                    "positive_animals": int((v > 0).sum()),
                }
            )
            for animal in local.animal:
                other = local[local.animal.ne(animal)][metric]
                loo.append({"dataset": dataset, "group": group, "metric": metric, "omitted_animal": animal, "mean": float(other.mean()), "animals": len(other)})
    return events, sessions, animals, pd.DataFrame(summaries), pd.DataFrame(loo)


def process(record, source_dir, output_dir):
    start = time.monotonic()
    identity = {k: record[k] for k in ID}
    source, output = Path(source_dir) / record["tag"], Path(output_dir) / record["tag"]
    output.mkdir(exist_ok=False)
    result = identity | {"tag": record["tag"], "status": "running"}
    try:
        required = ["cache.npz", "folds.json", "selection.csv", "labels.csv.gz"] + [f"fit_{i}.npz" for i in range(5)]
        for name in required:
            if file_sha256(source / name) != record["output_sha256"][name]:
                raise ValueError("source hash mismatch: " + name)
        with np.load(source / "cache.npz", allow_pickle=False) as z:
            cache = {k: z[k] for k in z.files}
        selected = pd.read_csv(source / "selection.csv")
        legacy = pd.read_csv(source / "labels.csv.gz").set_index(["event_id", "split"])
        folds = json.loads((source / "folds.json").read_text())
        tests = [int(e) for f in folds for e in f["test_ids"]]
        if sorted(tests) != sorted(selected.event_id.tolist()) or len(tests) != len(set(tests)):
            raise ValueError("fold/event coverage mismatch")
        rows, samples = [], 0
        for fold in folds:
            if set(fold["test_ids"]) & set(fold["calibration_ids"]):
                raise ValueError("test-event calibration leakage")
            with np.load(source / f"fit_{fold['fold']}.npz", allow_pickle=False) as fit:
                fit = dict(fit)
            operator = NeuralOperator(fit["initial"], fit["transition"], fit["occupancy"])
            null = OccupancyMatchedNull.from_operator(operator)
            for eid in fold["test_ids"]:
                base, durations = cache[f"base_{eid}"], cache[f"durations_{eid}"]
                original, _ = full_counts(base, durations)
                if not np.array_equal(original, cache[f"counts_{eid}"]):
                    raise ValueError("forecast clock changed")
                for split in range(5):
                    train, half, held = (cache[f"{name}_{split}"] for name in ("train", "half", "held"))
                    if set(train) & set(held) or not set(half) <= set(train):
                        raise ValueError("invalid inference/evaluation partition")
                    target = base[:, half].sum(axis=1)
                    half_windows = overlapping_counts(base[:, half], durations)
                    half_counts, _ = full_counts(base[:, half], durations)
                    configurations = [("full", -1, train, base[:, train]), ("half_neurons", -1, half, base[:, half])]
                    for draw in range(DRAWS):
                        rng = np.random.default_rng(seed("count_matched_v1", *identity.values(), eid, split, draw))
                        matched = match_counts(base[:, train], target, rng)
                        configurations.append(("matched_spikes", draw, train, matched))
                    for arm, draw, cells, observed in configurations:
                        windows = overlapping_counts(observed, durations)
                        counts, _ = full_counts(observed, durations)
                        if arm == "matched_spikes":
                            if not np.array_equal(windows.sum(axis=1), half_windows.sum(axis=1)) or not np.array_equal(counts.sum(axis=1), half_counts.sum(axis=1)):
                                raise ValueError("overlapping/coarse count mismatch")
                            samples += 1
                        path = conditional_path(windows, cache["rates"][cells])
                        label = geometry(path, cache["centers"], windows)
                        poisson_pass = None
                        if arm != "matched_spikes":
                            poisson_pass = geometry(poisson_map(windows, cache["rates"][cells]), cache["centers"], windows)["geometric_pass"]
                            old_column = "full_geometric_pass" if arm == "full" else "half_geometric_pass"
                            if poisson_pass != bool(legacy.loc[(eid, split), old_column]):
                                raise ValueError("legacy Poisson classification changed")
                        score = {
                            "n_target_bins": max(0, len(counts) - 2),
                            "n_heldout_target_spikes": int(original[2:, held].sum()),
                            "score_dynamic": np.nan,
                            "score_matched_own": np.nan,
                        }
                        status = "insufficient_bins"
                        if len(counts) > 2:
                            pred = forecast_distributions(counts, fit["probabilities"][cells], operator, null, (2,))
                            score = score_predictions(pred, original[:, held], fit["probabilities"][held], fit["global_probability"][held])[0]
                            if not np.isfinite([score[k] for k in score if k.startswith("score_")]).all():
                                raise ValueError("nonfinite predictive score")
                            status = "scored"
                        rows.append(
                            identity
                            | {
                                "event_id": eid,
                                "split": split,
                                "fold": fold["fold"],
                                "arm": arm,
                                "draw": draw,
                                "status": status,
                                "n_inference_cells": len(cells),
                                "n_inference_spikes": int(observed.sum()),
                                "n_active_inference_cells": int((observed.sum(axis=0) > 0).sum()),
                                "legacy_poisson_pass": poisson_pass,
                            }
                            | label
                            | score
                        )
        frame = pd.DataFrame(rows)
        if len(frame) != len(selected) * 5 * (2 + DRAWS):
            raise ValueError("missing score rows")
        paired = contrasts(frame)
        frame.to_csv(output / "scores.csv.gz", index=False)
        paired.to_csv(output / "paired.csv.gz", index=False)
        result.update(
            status="complete",
            events=len(selected),
            rows=len(frame),
            matched_samples=samples,
            insufficient_bin_rows=int(frame.status.eq("insufficient_bins").sum()),
            runtime_s=time.monotonic() - start,
        )
        result["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir()}
    except Exception as exc:  # noqa: BLE001 - persist each failed session; fail the entire run.
        import traceback

        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), runtime_s=time.monotonic() - start)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    source_manifest = args.source_dir / "independent_rejected_forecast_manifest.json"
    source = json.loads(source_manifest.read_text())
    records = source["completed"]
    if source["status"] != "complete" or len(records) != 33 or any(r["status"] != "complete" for r in records):
        raise ValueError("complete frozen 33-session cohort required")
    provenance = build_script_provenance(
        cwd=ROOT, input_paths={"source_manifest": source_manifest, "protocol": ROOT / "docs/count_matched_coverage_protocol.md", "runner": Path(__file__)}
    )
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed implementation required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = provenance | {"status": "running", "draws": DRAWS, "workers": args.workers, "completed": []}
    manifest_path = args.output_dir / "count_matched_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process, r, str(args.source_dir), str(args.output_dir)) for r in records]
        for job in as_completed(jobs):
            result = job.result()
            manifest["completed"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "output_sha256"}), flush=True)
    if any(r["status"] != "complete" for r in manifest["completed"]):
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit("session failure; no scientific interpretation")
    paired = pd.concat([pd.read_csv(args.output_dir / r["tag"] / "paired.csv.gz") for r in records], ignore_index=True)
    for name, table in zip(("events", "sessions", "animals", "summary", "leave_one_animal_out"), summarize(paired), strict=True):
        table.to_csv(args.output_dir / f"count_matched_{name}.csv", index=False)
    gates = [
        ("sessions_complete", len(manifest["completed"]) == 33),
        ("fine_bin_and_window_counts_equal", True),
        ("original_poisson_labels_unchanged", True),
        ("targets_and_partitions_unchanged", True),
        ("required_rows_complete", True),
        ("no_scoring_failures", True),
        ("independent_verification", False),
    ]
    pd.DataFrame([{"gate": g, "passed": v} for g, v in gates]).to_csv(args.output_dir / "count_matched_gates.csv", index=False)
    manifest.update(
        status="complete_pending_independent_verification", runtime_s=time.monotonic() - start, output_sha256={p.name: file_sha256(p) for p in args.output_dir.glob("*.csv")}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["status", "runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
