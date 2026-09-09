#!/usr/bin/env python3
"""Exploratory sample-size diagnostic using existing audited simulation scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.metric_population_recovery import population_fit
from scripts._provenance import build_script_provenance, file_sha256
from scripts.run_exact_path_clock_recovery import MODELS, TAGS
from scripts.verify_unknown_path_clock_populations import certify, close

METHODS = ("exhaustive", "mc_8192_b0", "mc_8192_b1")
KEYS = ["dataset", "animal", "session", "source_teacher", "scenario"]


def certify_fit(ll, row):
    ll = ll - ll.max(axis=1, keepdims=True)
    weights = np.array([row["weight_" + model] for model in MODELS])
    optimum, _ = certify(ll, weights)
    close(row["relative_log_likelihood"], optimum)
    close(row["phi_hat"], weights[1] / weights[:2].sum())
    errors = []
    for label, phi in (("low", row["phi_low"]), ("high", row["phi_high"]), ("null", 0.5)):
        with np.errstate(divide="ignore"):
            focus = np.logaddexp(ll[:, 0] + np.log1p(-phi), ll[:, 1] + np.log(phi))
        weights = [row[label + "_weight_" + model] for model in ("coherent", *MODELS[2:])]
        value, _ = certify(np.column_stack([focus, ll[:, 2:]]), weights)
        close(row[label + "_profile_log_likelihood"], value)
        lr = 2 * (optimum - value)
        if label == "null":
            close(row["null_lr"], max(0, lr))
        elif phi in (0, 1):
            if lr > 3.841458820694124 + 2e-4:
                raise ValueError("invalid boundary profile interval")
        else:
            close(lr, 3.841458820694124, atol=2e-4)
            errors.append(abs(lr - 3.841458820694124))
    return max(errors, default=0.0)


def pooled_fits(metadata, matrices, repeats=50, per_repeat=128):
    if metadata.empty or metadata.row_index.duplicated().any() or set(matrices) != set(METHODS):
        raise ValueError("complete distinct observations and integration methods required")
    if set(metadata.row_index) != set(range(len(metadata))):
        raise ValueError("observation indices must cover each input score row")
    for scores in matrices.values():
        if scores.shape != (len(metadata), len(MODELS)) or not np.isfinite(scores).all():
            raise ValueError("finite complete score matrices required")
    rows = []
    for key, group in metadata.groupby(KEYS):
        if set(group.repeat) != set(range(repeats)) or not group.groupby("repeat").size().eq(per_repeat).all():
            raise ValueError("all source repetitions and events required")
        if group.duplicated(["repeat", "event_in_population"]).any() or not group.groupby("repeat").event_in_population.agg(lambda x: set(x) == set(range(per_repeat))).all():
            raise ValueError("duplicate or missing source event identity")
        index = group.row_index.to_numpy()
        for method, scores in matrices.items():
            ll = scores[index]
            fit = population_fit(ll, MODELS, "coherent")
            error = certify_fit(ll, fit)
            truth = key[-1]
            rows.append(
                dict(zip(KEYS, key, strict=True))
                | {"method": method, "n_events": len(index), "n_repetitions_pooled": repeats}
                | fit
                | {
                    "estimate_error": fit["phi_hat"] - truth,
                    "interval_contains_truth": fit["phi_low"] <= truth <= fit["phi_high"],
                    "interval_width": fit["phi_high"] - fit["phi_low"],
                    "max_profile_boundary_error": error,
                    "fit_certified": True,
                }
            )
    return pd.DataFrame(rows)


def run(args):
    run_dir, audit_dir, output = args.run_dir, args.audit_dir, args.output_dir
    mp, ap = run_dir / "exact_path_clock_manifest.json", audit_dir / "exact_path_clock_audit.json"
    m, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("matching completed and audited exact run required")
    if tuple(m["tags"]) != TAGS or m["n_fits"] != 4200 or audit["fits_certified"] != 4200:
        raise ValueError("expected fully certified two-recording pilot required")
    dense = Path(m["dense_dir"])
    dp = dense / "dense_path_clock_manifest.json"
    dm = json.loads(dp.read_text())
    if file_sha256(dp) != m["input_file_sha256"]["dense"]:
        raise ValueError("dense-score provenance changed")
    inputs = {"run_manifest": mp, "audit": ap, "dense_manifest": dp}
    for tag in TAGS:
        for tail in ("observations.csv.gz", "exact_scores.npy"):
            path = run_dir / f"{tag}_{tail}"
            if file_sha256(path) != m["output_sha256"][path.name]:
                raise ValueError("exact output hash mismatch")
            inputs[path.name] = path
        path = dense / f"{tag}_scores.npy"
        if file_sha256(path) != dm["output_sha256"][path.name]:
            raise ValueError("Monte Carlo score hash mismatch")
        inputs[path.name] = path
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("commit exploratory diagnostic before running")
    output.mkdir(parents=True, exist_ok=False)
    result = provenance | {
        "status": "running",
        "exploratory_post_primary_result": True,
        "real_events_rescored": False,
        "changes_primary_gates": False,
        "independent_recording_encoders": 2,
        "source_repetitions_per_fit": 50,
        "events_per_fit": 6400,
        "interval_scope": "conditional_model_profile_interval_not_between_teacher_or_animal_uncertainty",
    }
    path = output / "exact_path_clock_pooling_manifest.json"
    try:
        frames = []
        for tag in TAGS:
            meta = pd.read_csv(run_dir / f"{tag}_observations.csv.gz")
            if len(meta) != 38400 or set(meta.source_teacher) != {"original_bank_0", "original_bank_1"} or set(meta.scenario) != {0.25, 0.5, 0.75}:
                raise ValueError("complete frozen source populations required")
            mc = np.load(dense / f"{tag}_scores.npy", mmap_mode="r")
            matrices = {"exhaustive": np.load(run_dir / f"{tag}_exact_scores.npy", mmap_mode="r"), "mc_8192_b0": mc[0, 2], "mc_8192_b1": mc[1, 2]}
            frames.append(pooled_fits(meta, matrices))
            print(f"{tag}: 18 pooled fits independently certified", flush=True)
        fits = pd.concat(frames, ignore_index=True)
        if len(fits) != 36 or not fits.n_events.eq(6400).all():
            raise ValueError("expected 36 complete pooled fits")
        fits.to_csv(output / "exact_path_clock_pooling_fits.csv", index=False)
        lines = [
            "# Exploratory pooled-repetition diagnostic",
            "",
            "This analysis was added after inspecting the primary recovery result. No events were rescored, no labels selected a path, and no primary gate was changed.",
            "",
            "Each row pools 50 simulation repetitions into 6,400 observations from one recording and one fixed 256-path teacher bank. These are not additional animals or new independent teacher libraries.",
            "",
            "Intervals are conditional likelihood-profile intervals, not a validated 95% uncertainty statement under teacher-library mismatch. Pooling cannot estimate interval coverage from one fit per condition.",
            "",
            "| Dataset | Teacher bank | True fraction | Method | Estimate | Profile interval | Contains truth |",
            "| --- | ---: | ---: | --- | ---: | --- | --- |",
        ]
        for row in fits.itertuples():
            lines.append(
                f"| {row.dataset} | {row.source_teacher[-1]} | {row.scenario:.2f} | {row.method} | {row.phi_hat:.3f} | [{row.phi_low:.3f}, {row.phi_high:.3f}] | {row.interval_contains_truth} |"
            )
        lines += [
            "",
            "Persistent bias with more observations is consistent with fixed-library or model mismatch; this diagnostic alone does not identify its source. A fresh-path-per-observation generator would be needed to remove finite teacher-bank sampling as an explanation.",
            "",
            "No conclusion about real replay speed, uniformity, neural-clock prevalence, or cross-dataset biology follows from this calibration diagnostic.",
            "",
        ]
        (output / "exact_path_clock_pooling.md").write_text("\n".join(lines))
        result.update(status="pass", fits_certified=len(fits), output_sha256={p.name: file_sha256(p) for p in output.iterdir()})
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
