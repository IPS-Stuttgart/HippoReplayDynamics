#!/usr/bin/env python3
"""Calibrate a paired prospective-minus-preceding score, using simulated outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import itertools

from _provenance import build_script_provenance, file_sha256
from audit_kleinman_ripple_run_opportunities import interval_run_mask
from calibrate_kleinman_conditional_coupling import recruitment, wilson
from calibrate_kleinman_replay_content import full_map
from calibrate_kleinman_spatial_expression import estimate_reference, key_seed
from kleinman_conditional_spatial_score import anchor_score, reference_feature, spatial_score
from validate_kleinman_run_decoder import align_behavior, interval_data, make_traversals

PARAMETERS = {
    "seed": 20260924,
    "replicates": 128,
    "reference_repeats": 16,
    "packets_per_condition": 2,
    "beta": 0.35,
    "gain_coupling": 0.7,
    "shared_gain_sd": 0.4,
    "spatial_drift_sd": 0.6,
}
CASES = ("unchanged", "scalar_gain", "burst_gain", "spatial_drift", "continuous_positive_drift", "continuous_negative_drift", "future_only", "past_only")
EXPECTED = {"future_only": 1, "past_only": -1}
BANKS = None


def select_packets(frame):
    if frame.duplicated(["animal", "session", "packet_id"]).any():
        raise ValueError("duplicate_packet")
    f = frame.loc[frame.selected & frame.availability_descriptor].copy()
    f["selection_key"] = [hashlib.sha256(f"{PARAMETERS['seed']}|{r.animal}|{r.session}|{r.packet_id}".encode()).hexdigest() for r in f.itertuples()]
    keys = ["animal", "drug", "novel"]
    f = f.sort_values("selection_key").groupby(keys, sort=True).head(2)
    if len(f) != 48 or f.animal.nunique() != 6 or not f.groupby(keys).size().eq(2).all():
        raise ValueError("all_24_conditions_need_two_packets")
    return f.sort_values(keys + ["selection_key"]).reset_index(drop=True)


def build_bank(folder, row):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    lookup = {r["traversal"]: r for r in runs}
    ref = [lookup[i] for i in json.loads(row.reference_traversals)]
    selected = [lookup[getattr(row, n + "_traversal")] for n in ("past", "baseline", "target")]
    chain = ref + selected
    if not all(a["end_s"] < b["start_s"] for a, b in itertools.pairwise(chain)):
        raise ValueError("invalid_chronology")
    model = full_map(folder)
    nb = len(model["edges"]) - 1
    dt, _, train, _, _, bins = interval_data(t, x, speed, runs, model["edges"])
    readout = train & ((speed[:-1] + speed[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > ends[0] + 20) & ((x[:-1] + x[1:]) / 2 < ends[1] - 20)
    masks = [train & interval_run_mask(t, ref), *[readout & interval_run_mask(t, [r]) for r in selected]]
    occ = np.array([np.bincount(bins[m], weights=dt[m], minlength=nb) for m in masks])
    rec = recruitment(folder, row, model["units"])
    return {
        "reference_exposure": occ[0],
        "readout_exposure": occ[1:],
        "generating_rates": model["rates"].reshape(2, nb, -1)[row.direction].T,
        "unit_keys": model["units"],
        "predictor": rec["predictor"],
        "ripple_counts": rec["ripple_counts"],
        "background_counts": rec["background_counts"],
        "edges": model["edges"],
    }


def expected_rates(rate, exposure, x, case, wiggle, gains):
    if case not in CASES:
        raise ValueError("unknown_case")
    common = np.all(exposure > 0, axis=0)
    feature = reference_feature(rate, common)
    coeff = np.zeros(3)
    if case == "continuous_positive_drift":
        coeff = np.array([-1.0, 0.0, 1.0]) * PARAMETERS["beta"]
    elif case == "continuous_negative_drift":
        coeff = np.array([1.0, 0.0, -1.0]) * PARAMETERS["beta"]
    elif case == "future_only":
        coeff[2] = PARAMETERS["beta"]
    elif case == "past_only":
        coeff[0] = -PARAMETERS["beta"]
    target = rate[None, :] * np.exp(coeff[:, None] * x * feature[None, :])
    if case == "spatial_drift":
        target *= np.exp(np.array([-1.0, 0.0, 1.0])[:, None] * wiggle[None, :])
    totals = (exposure * rate).sum(axis=1)
    new_totals = (exposure * target).sum(axis=1)
    if np.any(new_totals <= 0):
        raise ValueError("empty_readout")
    target *= (totals / new_totals)[:, None]
    target *= gains[:, None]
    if case in ("scalar_gain", "burst_gain"):
        target *= np.exp(PARAMETERS["gain_coupling"] * x * np.array([-1.0, 0.0, 1.0]))[:, None]
    return target


def paired_scores(counts, exposure, models, predictor):
    common = np.all(exposure > 0, axis=0)
    result = {}
    for name, before, after in (("preceding", 0, 1), ("prospective", 1, 2)):
        scores, infos = [], []
        informative, excluded = 0, 0
        for c, model in zip(counts, models, strict=True):
            feature = reference_feature(model, common)[common]
            s = spatial_score(c[before, common], c[after, common], exposure[before, common], exposure[after, common], feature)
            scores.append(s["score"])
            infos.append(s["information"])
            informative += s["informative"]
            excluded += int((c[before] + c[after])[~common].sum())
        u, v = anchor_score(predictor, scores, infos)
        result[name + "_score"], result[name + "_information"] = u, v
        result[name + "_informative_cells"] = informative
        result[name + "_excluded_spikes"] = excluded
    result["common_bins"] = int(common.sum())
    return result


def infer(rows):
    frame = pd.DataFrame(rows)
    animals = []
    for animal, g in frame.groupby("animal"):
        paired = g.loc[g.preceding_information.gt(1e-10) & g.prospective_information.gt(1e-10)]
        if paired.empty:
            continue
        row = {"animal": animal, "n_packets": len(paired)}
        for name in ("preceding", "prospective"):
            u, v = paired[name + "_score"].sum(), paired[name + "_information"].sum()
            row.update({name + "_score": u, name + "_information": v, name + "_effect": u / v})
        row["contrast"] = row["prospective_effect"] - row["preceding_effect"]
        animals.append(row)
    d = np.array([a["contrast"] for a in animals])
    if len(d) != 6 or not np.isfinite(d).all():
        return animals, {
            "n_animals": len(d),
            "status": "missing_information",
            "mean_contrast": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "positive_flag": False,
            "negative_flag": False,
            "flag": False,
        }
    mean = float(d.mean())
    half = float(student_t.ppf(0.975, 5) * d.std(ddof=1) / np.sqrt(6))
    return animals, {
        "n_animals": 6,
        "status": "scored",
        "mean_contrast": mean,
        "mean_preceding": float(np.mean([a["preceding_effect"] for a in animals])),
        "mean_prospective": float(np.mean([a["prospective_effect"] for a in animals])),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "positive_flag": mean - half > 0,
        "negative_flag": mean + half < 0,
        "flag": abs(mean) > half,
    }


def init_worker(banks):
    global BANKS
    BANKS = banks


def simulate_replica(rep):
    rows = []
    for bank in BANKS:
        identity, animal = bank["identity"], bank["animal"]
        rates, ref_t, exposure = (bank[k] for k in ("generating_rates", "reference_exposure", "readout_exposure"))
        models, included = [], []
        for key, rate in zip(bank["unit_keys"], rates, strict=True):
            uid = "_".join(map(str, key.astype(int)))
            rng = np.random.default_rng(key_seed("joint-reference", identity, uid, rep % PARAMETERS["reference_repeats"]))
            model, keep = estimate_reference(rng.poisson(ref_t * rate), ref_t)
            models.append(model)
            included.append(keep)
        included = np.array(included)
        x = bank["predictor"][included]
        ready = len(x) >= 2 and np.isfinite(x).all() and x.std() > 1e-10
        if ready:
            x = (x - x.mean()) / x.std()
        shared = np.random.default_rng(key_seed("joint-shared", animal, rep))
        gains = np.exp(shared.normal(0, PARAMETERS["shared_gain_sd"], 3))
        coef = shared.normal(size=6)
        space = np.linspace(0, 1, len(ref_t))
        wiggle = sum(coef[2 * k] * np.sin(2 * np.pi * (k + 1) * space) + coef[2 * k + 1] * np.cos(2 * np.pi * (k + 1) * space) for k in range(3))
        wiggle *= PARAMETERS["spatial_drift_sd"] / max(wiggle.std(), 1e-10)
        for case in CASES:
            row = {k: bank[k] for k in ("identity", "animal", "session", "drug", "novel", "direction")}
            row.update(
                case=case,
                replicate=rep,
                n_reference_units=int(included.sum()),
                preceding_score=0.0,
                prospective_score=0.0,
                preceding_information=0.0,
                prospective_information=0.0,
                status="no_predictor_or_reference",
            )
            if ready:
                counts = []
                for local, idx in enumerate(np.flatnonzero(included)):
                    uid = "_".join(map(str, bank["unit_keys"][idx].astype(int)))
                    rng = np.random.default_rng(key_seed("joint-readout", identity, uid, case, rep))
                    expected = exposure * expected_rates(rates[idx], exposure, x[local], case, wiggle, gains)
                    compound = 2 if case == "burst_gain" else 1
                    counts.append(compound * rng.poisson(expected / compound))
                row.update(paired_scores(np.asarray(counts), exposure, np.asarray(models)[included], x))
                row["status"] = "scored" if min(row["preceding_information"], row["prospective_information"]) > 1e-10 else "no_paired_information"
            rows.append(row)
    by_animal, estimates = [], []
    for case in CASES:
        animals, estimate = infer([r for r in rows if r["case"] == case])
        by_animal.extend({**a, "case": case, "replicate": rep} for a in animals)
        estimates.append({**estimate, "case": case, "replicate": rep})
    return rows, by_animal, estimates


def summarize(estimates):
    summaries = []
    for case, g in estimates.groupby("case", sort=False):
        direction = EXPECTED.get(case, 0)
        flags = g.positive_flag if direction == 1 else g.negative_flag if direction == -1 else g.flag
        n, total = int(flags.sum()), len(g)
        low, high = wilson(n, total)
        complete = bool(g.n_animals.eq(6).all() and g.status.eq("scored").all())
        passed = complete and (n / total >= 0.8 if direction else n / total <= 0.1 and high <= 0.15)
        summaries.append(
            {
                "case": case,
                "expected_contrast_sign": direction,
                "n_replicates": total,
                "n_flags": n,
                "flag_fraction": n / total,
                "wilson_low": low,
                "wilson_high": high,
                "mean_preceding": g.mean_preceding.mean(),
                "mean_prospective": g.mean_prospective.mean(),
                "mean_contrast": g.mean_contrast.mean(),
                "median_contrast": g.mean_contrast.median(),
                "n_positive_flags": int(g.positive_flag.sum()),
                "n_negative_flags": int(g.negative_flag.sum()),
                "all_six_animals": complete,
                "engineering_screen_passed": passed,
            }
        )
    return pd.DataFrame(summaries)


def run(args):
    source, out = args.audit_dir, args.output_dir
    old = json.loads((source / "manifest.json").read_text())
    verified = json.loads((source / "verification.json").read_text())
    if verified["status"] != "pass" or verified["manifest_sha256"] != file_sha256(source / "manifest.json"):
        raise ValueError("verified_audit_required")
    for name, digest in old["outputs"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed_audit")
    selected = select_packets(pd.read_csv(source / "packets.csv", float_precision="round_trip"))
    inputs = {name: source / name for name in ("manifest.json", "verification.json", "packets.csv")}
    inputs.update(producer=Path(__file__), protocol=ROOT / "docs/kleinman_joint_temporal_protocol.md")
    for name in (
        "_provenance.py",
        "audit_kleinman_ripple_run_opportunities.py",
        "calibrate_kleinman_conditional_coupling.py",
        "calibrate_kleinman_replay_content.py",
        "calibrate_kleinman_spatial_expression.py",
        "kleinman_conditional_spatial_score.py",
        "validate_kleinman_run_decoder.py",
    ):
        inputs[name] = ROOT / "scripts" / name
    for a in selected.itertuples():
        for name in ("session_info.mat", "spike_data.mat", "ripple_events.mat"):
            inputs[f"{a.animal}/{a.session}/{name}"] = args.dataset_root / "Experiment_1" / a.animal / a.session / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable" or any(x is None for x in provenance["input_file_sha256"].values()):
        raise ValueError("clean_committed_inputs_required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "banks").mkdir()
    selected.to_csv(out / "selected_packets.csv", index=False)
    banks = []
    for i, a in enumerate(selected.itertuples()):
        bank = build_bank(args.dataset_root / "Experiment_1" / a.animal / a.session, a)
        np.savez_compressed(out / f"banks/{i:03}.npz", **bank)
        bank.update({k: getattr(a, k) for k in ("animal", "session", "drug", "novel", "direction")})
        bank["identity"] = f"{a.animal}/{a.session}/{a.packet_id}"
        banks.append(bank)
    packet_rows, animal_rows, estimates = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(banks,)) as pool:
        for rep, (a, b, c) in enumerate(pool.map(simulate_replica, range(PARAMETERS["replicates"]))):
            packet_rows.extend(a)
            animal_rows.extend(b)
            estimates.extend(c)
            print(json.dumps({"completed_replicates": rep + 1}), flush=True)
    pd.DataFrame(packet_rows).to_csv(out / "packet_scores.csv.gz", index=False)
    pd.DataFrame(animal_rows).to_csv(out / "animal_scores.csv.gz", index=False)
    estimates = pd.DataFrame(estimates)
    estimates.to_csv(out / "replicate_estimates.csv", index=False)
    summary = summarize(estimates)
    summary.to_csv(out / "calibration_summary.csv", index=False)
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("input_changed")
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "parameters": PARAMETERS,
        "cases": CASES,
        "native_outcomes_scored": False,
        "drug_effect_scored": False,
        "engineering_screen_passed": bool(summary.engineering_screen_passed.all()),
        "outputs": {str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*") if p.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    run(p.parse_args())
