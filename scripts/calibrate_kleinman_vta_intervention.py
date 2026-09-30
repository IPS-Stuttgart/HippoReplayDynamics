#!/usr/bin/env python3
"""Frozen, cohort-specific difference-of-differences simulation; no native outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_kleinman_joint_temporal_specificity import build_bank, paired_scores
from calibrate_kleinman_spatial_expression import estimate_reference, key_seed
from kleinman_conditional_spatial_score import reference_feature
from kleinman_vta_intervention import EFFECTS, NULLS, animal_contrasts, calibration_summary, contrast_inference

SCENARIOS = [(s, 0.0) for s in NULLS] + [("intervention", e) for e in EFFECTS]
BANKS = None
BANK_KIND = None


def rng_for(kind, *keys):
    if kind not in ("development", "validation"):
        raise ValueError("invalid_bank_kind")
    return np.random.default_rng(key_seed("vta-did-v1", kind, *keys))


def simulate_packet(bank, kind, scenario, effect, rep):
    """One shared R4 draw, reference-only inclusion, triple spatial support."""
    rng = rng_for(kind, bank["identity"], scenario, effect, rep)
    shared = rng_for(kind, bank["animal"], scenario, effect, rep)
    rates = bank["generating_rates"]
    exposure = bank["readout_exposure"].copy()
    reference = rng.poisson(rates * bank["reference_exposure"])
    fitted = [estimate_reference(c, bank["reference_exposure"]) for c in reference]
    keep = np.array([k for _, k in fitted])
    models = np.array([m for m, _ in fitted])[keep]
    strength = int(bank["drug"]) * (1.5 if bank["group"] == "experimental" else 0.5)
    rt, bt = bank["ripple_exposure_s"], bank["background_exposure_s"]
    rmean, bmean = bank["ripple_counts"].astype(float) + 0.5, bank["background_counts"].astype(float) + 0.5
    if scenario == "drug_recruitment_exposure":
        rt *= np.exp(-0.7 * strength)
        exposure *= np.exp(shared.normal(0, 0.5 + strength * 0.2, (3, 1)))
        rmean *= np.exp(strength * rng.normal(0, 0.8, len(rmean))) * rt / bank["ripple_exposure_s"]
    rc, bc = rng.poisson(rmean), rng.poisson(bmean)
    x = (np.log((rc + 0.5) / rt) - np.log((bc + 0.5) / bt))[keep]
    row = {k: bank[k] for k in ("animal", "session", "packet_id", "physical_track_id", "drug")}
    row.update(preceding_score=0.0, prospective_score=0.0, preceding_information=0.0, prospective_information=0.0)
    if len(x) < 2 or not np.isfinite(x).all() or x.std() <= 1e-10:
        return row
    x = (x - x.mean()) / x.std()
    common = np.all(exposure > 0, axis=0)
    if not np.any(common):
        return row
    shared_gain = np.exp(shared.normal(0, 0.4, 3))
    counts = []
    for cell, rate, model, value in zip(np.flatnonzero(keep), rates[keep], models, x, strict=True):
        feature = reference_feature(model, common)
        expected = np.broadcast_to(rate, (3, len(rate))).copy()
        if scenario == "intervention" and bank["group"] == "experimental" and bank["drug"] == 1:
            expected[2] *= np.exp(effect * value * feature)
        elif scenario == "elapsed_time_drift":
            # Continuous pre-existing drift, scaled by actual elapsed minutes, not lap index.
            elapsed = (bank["readout_midpoints_s"] - bank["readout_midpoints_s"][1]) / 60
            slope = (0.04 + 0.02 * strength) * value
            expected *= np.exp(np.clip(elapsed[:, None] * slope * feature[None, :], -4, 4))
        totals = (exposure * rate).sum(axis=1)
        new_totals = (exposure * expected).sum(axis=1)
        if np.any(new_totals <= 0):
            return row
        expected *= (totals / new_totals)[:, None]
        expected *= shared_gain[:, None]
        if scenario in ("scalar_gain", "bursting"):
            expected *= np.exp(0.7 * (1 + strength) * value * np.array([-1, 0, 1]))[:, None]
        if scenario == "cell_expression":
            expected *= np.exp(rng.normal(0, 0.8 + 0.3 * strength, (3, 1)))
        if scenario == "shared_baseline":
            # R4 is reused on opposite sides; recruitment depends on its shared excitability.
            expected[1] *= np.exp((0.5 + 0.5 * strength) * value)
        compound = 3 if scenario == "bursting" else 1
        counts.append(compound * rng.poisson(exposure * expected / compound))
    row.update(paired_scores(np.asarray(counts), exposure, models, x))
    return row


def init_worker(banks, kind):
    global BANKS, BANK_KIND
    BANKS, BANK_KIND = banks, kind


def simulate_replica(rep):
    estimates = []
    for scenario, effect in SCENARIOS:
        rows = pd.DataFrame([simulate_packet(b, BANK_KIND, scenario, effect, rep) for b in BANKS])
        _, animals = animal_contrasts(rows)
        estimate = contrast_inference(animals)
        estimates.append({"scenario": scenario, "effect": effect, "replicate": rep, **estimate})
    return estimates


def run(args):
    from run_kleinman_vta_intervention import checked_freeze, finish, lock_stage, provenance

    frozen, cohort = checked_freeze(args.frozen_cohort, args.dataset_root)
    inputs = {"freeze": args.frozen_cohort / "freeze.json", "cohort": args.frozen_cohort / "frozen_cohort.csv"}
    p = provenance(inputs, committed=True)
    lock_stage(args.frozen_cohort, args.bank, args.output_dir, p)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    banks = []
    for a in cohort.itertuples():
        bank = build_bank(args.dataset_root / "Experiment_1" / a.animal / a.session, a)
        bank.update({k: getattr(a, k) for k in ("animal", "session", "packet_id", "physical_track_id", "group", "drug")})
        bank.update(
            identity=f"{a.animal}/{a.session}/{a.packet_id}",
            ripple_exposure_s=a.eligible_ripple_s,
            background_exposure_s=a.eligible_background_s,
            readout_midpoints_s=np.array([(getattr(a, s + "_start_s") + getattr(a, s + "_end_s")) / 2 for s in ("past", "baseline", "target")]),
        )
        banks.append(bank)
    reps = 1000 if args.bank == "validation" else 64
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(banks, args.bank)) as pool:
        for rep, rows in enumerate(pool.map(simulate_replica, range(reps))):
            records.extend(rows)
            with (args.output_dir / "progress.jsonl").open("a") as h:
                h.write(json.dumps({"replicate_completed": rep, "bank": args.bank}) + "\n")
            print(json.dumps({"replicate_completed": rep}), flush=True)
    estimates = pd.DataFrame(records)
    summary = calibration_summary(estimates, validation=args.bank == "validation")
    estimates.to_csv(args.output_dir / "replicate_contrasts.csv", index=False)
    summary.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    recovery = summary.loc[summary.effect.ne(0)].copy()
    recovery.to_csv(args.output_dir / "recovery_curve.csv", index=False)
    detectable = []
    for sign in (-1, 1):
        candidates = recovery.loc[recovery.effect.mul(sign).gt(0) & recovery.frequency.ge(0.8) & recovery.complete]
        detectable.append(
            {
                "direction": sign,
                "minimum_tested_abs_effect_at_80pct_power": float(candidates.effect.abs().min()) if len(candidates) else None,
                "interpretation": "engineering_grid_only_not_biological_threshold",
            }
        )
    pd.DataFrame(detectable).to_csv(args.output_dir / "minimum_detectable_effect.csv", index=False)
    passed = bool(summary.complete.all() and summary.loc[summary.effect.eq(0), "null_pass"].all())
    status = (
        ("passed" if passed else "stopped_calibration")
        if args.bank == "validation"
        else ("development_complete" if summary.complete.all() else "stopped_missing_calibration_information")
    )
    finish(
        args.output_dir,
        {
            **p,
            "bank": args.bank,
            "replicates_per_scenario": reps,
            "freeze_sha256": frozen["freeze_sha256"],
            "validation_passed": passed and args.bank == "validation",
            "status": status,
            "native_outcomes_scored": False,
            "drug_effect_scored": False,
            "generating_maps": "existing_full_RUN_map_for_simulation_only_reference_fit_and_selection_from_R0_R2",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frozen-cohort", type=Path, required=True)
    parser.add_argument("--bank", choices=("development", "validation"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
