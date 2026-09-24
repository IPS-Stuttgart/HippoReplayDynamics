#!/usr/bin/env python3
"""Calibrate a conditional ripple/spatial-alignment association on frozen banks."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import ndtri
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from audit_kleinman_ripple_run_opportunities import counts_in_intervals, duration, intersect_intervals
from calibrate_kleinman_spatial_expression import estimate_reference, key_seed
from kleinman_conditional_spatial_score import anchor_score, reference_feature, spatial_score
from validate_kleinman_run_decoder import align_behavior, interval_data, make_traversals, split_units

PARAMETERS = {
    "seed": 20260924,
    "replicates": 128,
    "reference_replicates_reused": 16,
    "reference_gain": 1.0,
    "predictor_pseudocount": 0.5,
    "gain_coupling": 0.7,
    "shared_gain_sd": 0.4,
    "spatial_fluctuation_sd": 0.6,
    "planted_beta": 0.35,
    "nominal_alpha": 0.05,
    "maximum_false_flag_fraction": 0.10,
    "maximum_false_flag_wilson_upper": 0.15,
    "minimum_detection_fraction": 0.80,
}
CASES = ("unchanged", "gain_correlated", "burst2", "burst2_gain", "spatial_fluctuation_gain", "positive_coupling", "negative_coupling")
BANKS = None


def recruitment(folder, anchor, unit_keys):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, _, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    _, valid, _, _, _, _ = interval_data(t, x, speed, runs, edges)
    immobile = valid & (np.maximum(speed[:-1], speed[1:]) <= 8)
    exposure = intersect_intervals(np.column_stack((t[:-1][immobile], t[1:][immobile])), [[v["start_s"], min(v["end_s"], v["start_s"] + 10)] for v in visits])
    exposure = intersect_intervals(exposure, [[anchor.baseline_end_s, anchor.target_start_s]])
    raw_ripples = np.asarray(loadmat(folder / "ripple_events.mat", simplify_cells=True)["ripple_events"]).reshape(-1, 4)
    ripples = intersect_intervals(raw_ripples[:, :2], exposure)
    keys, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    indices = {tuple(k): i for i, k in enumerate(keys)}
    subset = [trains[indices[tuple(k)]] for k in unit_keys]
    all_counts = counts_in_intervals(subset, exposure)
    rc = counts_in_intervals(subset, ripples)
    rt, bt = duration(ripples), duration(exposure) - duration(ripples)
    np.testing.assert_allclose([rt, bt], [anchor.eligible_ripple_s, anchor.eligible_background_s], atol=1e-9, rtol=0)
    predictor = np.log((rc + 0.5) / rt) - np.log((all_counts - rc + 0.5) / bt) if rt > 0 and bt > 0 else np.full(len(rc), np.nan)
    return {
        "ripple_counts": rc,
        "background_counts": all_counts - rc,
        "predictor": predictor,
        "ripple_seconds": rt,
        "background_seconds": bt,
        "eligible_intervals": exposure.tolist(),
        "ripple_intervals": ripples.tolist(),
    }


def init_worker(banks):
    global BANKS
    BANKS = banks


def animal_inference(rows):
    data = pd.DataFrame(rows)
    usable = data.loc[data.information > 1e-10]
    animals = []
    for animal, g in usable.groupby("animal"):
        animals.append({"animal": animal, "score": g.score.sum(), "information": g.information.sum(), "effect": g.score.sum() / g.information.sum(), "n_anchors": len(g)})
    effects = np.array([r["effect"] for r in animals])
    if len(effects) != 6 or not np.isfinite(effects).all():
        return animals, {"status": "missing_animals", "n_animals": len(effects), "mean_effect": np.nan, "ci_low": np.nan, "ci_high": np.nan, "flag": False, "correct_flag": False}
    effect = float(effects.mean())
    half = float(student_t.ppf(0.975, 5) * effects.std(ddof=1) / np.sqrt(6))
    return animals, {"status": "scored", "n_animals": 6, "mean_effect": effect, "ci_low": effect - half, "ci_high": effect + half, "flag": bool(abs(effect) > half)}


def simulate_replica(rep):
    rows = []
    for bank in BANKS:
        identity, animal = bank["identity"], bank["animal"]
        truth, ref_t, base_t, target_t = [bank[k] for k in ("generating_rates", "reference_exposure", "baseline_exposure", "target_exposure")]
        common = (base_t > 0) & (target_t > 0)
        models, included = [], []
        for u, rate in zip(bank["unit_keys"], truth, strict=True):
            uid = "_".join(map(str, u.astype(int)))
            count = np.random.default_rng(key_seed(identity, uid, 1.0, rep % 16)).poisson(ref_t * rate)
            fitted, keep = estimate_reference(count, ref_t)
            models.append(fitted)
            included.append(keep)
        included = np.asarray(included)
        n_included = int(included.sum())
        available = np.isfinite(bank["predictor"]).all() and n_included >= 2
        x = bank["predictor"][included]
        if available:
            available = x.std() > 1e-10
            if available:
                x = (x - x.mean()) / x.std()
        for case in CASES:
            entry = {k: bank[k] for k in ("identity", "animal", "session", "drug", "novel", "direction")}
            entry.update(
                replicate=rep,
                case=case,
                n_bank_units=len(truth),
                n_reference_included=n_included,
                n_informative_cells=0,
                before_common_spikes=0,
                after_common_spikes=0,
                excluded_spikes=0,
                score=0.0,
                information=0.0,
                effect=np.nan,
                status="no_predictor_or_insufficient_reference",
            )
            if not available:
                rows.append(entry)
                continue
            shared = np.random.default_rng(key_seed("coupling", rep, animal, "shared"))
            gains = np.exp(shared.normal(0, PARAMETERS["shared_gain_sd"], 2))
            coeff = shared.normal(size=6)
            space = np.linspace(0, 1, len(base_t))
            wiggle = sum(coeff[2 * k] * np.sin((k + 1) * 2 * np.pi * space) + coeff[2 * k + 1] * np.cos((k + 1) * 2 * np.pi * space) for k in range(3))
            wiggle = PARAMETERS["spatial_fluctuation_sd"] * wiggle / max(wiggle.std(), 1e-10)
            score, info = [], []
            for local, idx in enumerate(np.flatnonzero(included)):
                rate, model = truth[idx], models[idx]
                uid = "_".join(map(str, bank["unit_keys"][idx].astype(int)))
                rng = np.random.default_rng(key_seed("coupling", rep, identity, uid, case))
                target = rate.copy()
                if case == "spatial_fluctuation_gain":
                    target *= np.exp(wiggle)
                if case in ("positive_coupling", "negative_coupling"):
                    sign = 1 if case == "positive_coupling" else -1
                    target *= np.exp(sign * PARAMETERS["planted_beta"] * x[local] * reference_feature(rate, common))
                target *= (rate @ target_t) / (target @ target_t)
                gain = np.exp(PARAMETERS["gain_coupling"] * x[local]) if case in ("gain_correlated", "burst2_gain", "spatial_fluctuation_gain") else 1.0
                multiplier = 2 if case in ("burst2", "burst2_gain") else 1
                b = multiplier * rng.poisson(base_t * rate * gains[0] / multiplier)
                y = multiplier * rng.poisson(target_t * target * gains[1] * gain / multiplier)
                s = spatial_score(b, y, base_t, target_t, reference_feature(model, common))
                score.append(s["score"])
                info.append(s["information"])
                entry["n_informative_cells"] += s["informative"]
                for k in ("before_common_spikes", "after_common_spikes", "excluded_spikes"):
                    entry[k] += s[k]
            u, v = anchor_score(x, score, info)
            entry.update(score=u, information=v, effect=u / v if v > 1e-10 else np.nan, status="scored" if v > 1e-10 else "no_conditional_information")
            rows.append(entry)
    animal_rows, estimates = [], []
    for case in CASES:
        animals, inference = animal_inference([r for r in rows if r["case"] == case])
        for a in animals:
            animal_rows.append({**a, "case": case, "replicate": rep})
        inference.update(case=case, replicate=rep)
        expected_sign = -1 if case == "negative_coupling" else 1
        inference["correct_flag"] = bool(inference["flag"] and inference["mean_effect"] * expected_sign > 0)
        estimates.append(inference)
    return rows, animal_rows, estimates


def wilson(k, n):
    z = ndtri(0.975)
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    return center - half, center + half


def run(args):
    source = args.calibration_dir
    previous = json.loads((source / "manifest.json").read_text())
    verification = json.loads((source / "verification.json").read_text())
    if verification["status"] != "passed" or verification["manifest_sha256"] != file_sha256(source / "manifest.json"):
        raise ValueError("verified fixed bank required")
    for name, digest in previous["outputs"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed bank")
    anchors = pd.read_csv(source / "selected_anchors.csv", float_precision="round_trip")
    inputs = {name: source / name for name in ("manifest.json", "verification.json", "selected_anchors.csv")}
    inputs.update(producer=Path(__file__), helper=ROOT / "scripts/kleinman_conditional_spatial_score.py", protocol=ROOT / "docs/kleinman_conditional_coupling_protocol.md")
    for name in ("calibrate_kleinman_spatial_expression.py", "audit_kleinman_ripple_run_opportunities.py", "validate_kleinman_run_decoder.py"):
        inputs[name] = ROOT / "scripts" / name
    for idx, a in anchors.iterrows():
        inputs[f"bank_{idx}"] = source / f"banks/{idx:03}.npz"
        for name in ("session_info.mat", "spike_data.mat", "ripple_events.mat"):
            inputs[f"{a.animal}/{a.session}/{name}"] = args.dataset_root / "Experiment_1" / a.animal / a.session / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed calibration required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    banks, predictors, intervals = [], [], []
    for idx, a in anchors.iterrows():
        bank = dict(np.load(inputs[f"bank_{idx}"]))
        identity = f"{a.animal}/{a.session}/{a.opportunity_id}"
        rec = recruitment(args.dataset_root / "Experiment_1" / a.animal / a.session, a, bank["unit_keys"])
        bank.update({k: getattr(a, k) for k in ("animal", "session", "drug", "novel", "direction")})
        bank.update(identity=identity, predictor=rec["predictor"])
        banks.append(bank)
        intervals.append({"identity": identity, "eligible": rec["eligible_intervals"], "ripples": rec["ripple_intervals"]})
        for idx_unit, unit in enumerate(bank["unit_keys"]):
            predictors.append(
                {k: bank[k] for k in ("identity", "animal", "session", "drug", "novel", "direction")}
                | {
                    "unit_id": "_".join(map(str, unit.astype(int))),
                    "ripple_spikes": int(rec["ripple_counts"][idx_unit]),
                    "background_spikes": int(rec["background_counts"][idx_unit]),
                    "ripple_s": rec["ripple_seconds"],
                    "background_s": rec["background_seconds"],
                    "log_rate_enrichment": rec["predictor"][idx_unit],
                }
            )
    pd.DataFrame(predictors).to_csv(args.output_dir / "predictors.csv", index=False)
    (args.output_dir / "exposure_intervals.json").write_text(json.dumps(intervals) + "\n")
    anchor_rows, animal_rows, estimates = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(banks,)) as pool:
        for rep, (a, b, c) in enumerate(pool.map(simulate_replica, range(PARAMETERS["replicates"]))):
            anchor_rows.extend(a)
            animal_rows.extend(b)
            estimates.extend(c)
            print(json.dumps({"completed_replicas": rep + 1}), flush=True)
    pd.DataFrame(anchor_rows).to_csv(args.output_dir / "anchor_scores.csv.gz", index=False)
    pd.DataFrame(animal_rows).to_csv(args.output_dir / "animal_scores.csv.gz", index=False)
    e = pd.DataFrame(estimates)
    e.to_csv(args.output_dir / "replicate_estimates.csv", index=False)
    summaries = []
    for case, g in e.groupby("case", sort=False):
        positive = case in ("positive_coupling", "negative_coupling")
        n_flag = int(g.correct_flag.sum() if positive else g.flag.sum())
        low, high = wilson(n_flag, len(g))
        complete = bool(g.n_animals.eq(6).all() and g.status.eq("scored").all())
        passed = complete and (n_flag / len(g) >= 0.8 if positive else n_flag / len(g) <= 0.1 and high <= 0.15)
        summaries.append(
            {
                "case": case,
                "n_replicates": len(g),
                "n_flags": n_flag,
                "flag_fraction": n_flag / len(g),
                "wilson_low": low,
                "wilson_high": high,
                "median_effect": g.mean_effect.median(),
                "mean_effect": g.mean_effect.mean(),
                "complete_six_animal_replicas": int(g.n_animals.eq(6).sum()),
                "planted": positive,
                "engineering_screen_passed": bool(passed),
            }
        )
    pd.DataFrame(summaries).to_csv(args.output_dir / "calibration_summary.csv", index=False)
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("source changed during run")
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parameters": PARAMETERS,
        "cases": CASES,
        "n_anchors": len(banks),
        "real_future_outcome_scored": False,
        "real_recruitment_predictors_used": True,
        "biological_association_scored": False,
        "drug_effect_scored": False,
        "engineering_screen_passed": all(r["engineering_screen_passed"] for r in summaries),
        "outputs": {p.name: file_sha256(p) for p in args.output_dir.iterdir()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    run(p.parse_args())
