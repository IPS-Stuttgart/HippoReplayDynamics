#!/usr/bin/env python3
"""Exact conditional-score calibration on fixed data-shaped RUN maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from audit_kleinman_ripple_run_opportunities import interval_run_mask, opportunity_pairs
from calibrate_kleinman_replay_content import full_map
from validate_kleinman_run_decoder import align_behavior, interval_data, make_traversals

PARAMETERS = {
    "seed": 20260924,
    "reference_repeats": 16,
    "reference_gains": [0.25, 1.0, 4.0],
    "readout_counts": [5, 20, 80],
    "smooth_bins": 2.0,
    "rate_floor_hz": 1e-5,
    "shift_cm": 20.0,
}
CASES = ("unchanged", "gain_only_4x", "shift_20cm", "sharpen_1p5", "broaden_0p5", "uniform")


def key_seed(*parts):
    h = hashlib.sha256("|".join(map(str, (PARAMETERS["seed"], *parts))).encode()).digest()
    return int.from_bytes(h[:8], "little")


def select_anchors(opportunities):
    p = opportunities.loc[opportunities.original_run_pass & opportunities.minimum_units_descriptor & opportunities.status.eq("audited")].copy()
    p["selection_key"] = [hashlib.sha256(f"{PARAMETERS['seed']}|{r.animal}|{r.session}|{r.opportunity_id}".encode()).hexdigest() for r in p.itertuples()]
    keys = ["animal", "drug", "novel", "direction"]
    selected = p.sort_values("selection_key").groupby(keys, sort=True).head(1)
    if len(selected) != 48 or selected.animal.nunique() != 6:
        raise ValueError("all 48 animal/condition/direction strata required")
    return selected.sort_values(keys).reset_index(drop=True)


def probability(rate, exposure):
    rate, exposure = np.asarray(rate, float), np.asarray(exposure, float)
    if rate.shape != exposure.shape or not np.isfinite(rate).all() or np.any(rate <= 0):
        raise ValueError("invalid_rate")
    if not np.isfinite(exposure).all() or np.any(exposure < 0) or exposure.sum() <= 0:
        raise ValueError("invalid_exposure")
    weighted = rate * exposure
    return weighted / weighted.sum()


def conditional_moments(generating, reference, exposure):
    truth = probability(generating, exposure)
    fitted = probability(reference, exposure)
    log_rate = np.log(reference)
    center = fitted @ log_rate
    expected = truth @ log_rate
    variance = truth @ (log_rate - expected) ** 2
    return float(expected - center), float(variance)


def change_moments(generating_before, generating_after, reference, before, after):
    b, vb = conditional_moments(generating_before, reference, before)
    a, va = conditional_moments(generating_after, reference, after)
    return a - b, va + vb


def score(counts, reference, exposure):
    c = np.asarray(counts, float)
    if np.any(c < 0) or not np.isfinite(c).all() or np.any((exposure == 0) & (c > 0)):
        raise ValueError("invalid_spatial_counts")
    if c.sum() == 0:
        return np.nan
    log_rate = np.log(reference)
    return float((c / c.sum() - probability(reference, exposure)) @ log_rate)


def estimate_reference(counts, exposure):
    n = gaussian_filter1d(np.asarray(counts, float), PARAMETERS["smooth_bins"], mode="constant", truncate=4)
    t = gaussian_filter1d(np.asarray(exposure, float), PARAMETERS["smooth_bins"], mode="constant", truncate=4)
    rate = n / np.maximum(t, 1e-12)
    included = bool(np.sum(counts) >= 10 and rate.max() >= 1.0)
    return np.maximum(rate, PARAMETERS["rate_floor_hz"]), included


def target_profile(rate, case, shift_sign):
    if case == "unchanged":
        return rate.copy()
    if case == "gain_only_4x":
        return rate * 4
    if case == "shift_20cm":
        b = np.arange(len(rate))
        return np.interp(b - shift_sign * PARAMETERS["shift_cm"] / 2, b, rate, left=PARAMETERS["rate_floor_hz"], right=PARAMETERS["rate_floor_hz"])
    if case == "sharpen_1p5":
        return (rate / rate.max()) ** 1.5
    if case == "broaden_0p5":
        return (rate / rate.max()) ** 0.5
    if case == "uniform":
        return np.ones_like(rate)
    raise ValueError("unknown_case")


def build_bank(folder, row):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, _, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    pair = next(p for p in opportunity_pairs(runs) if p["opportunity_id"] == row.opportunity_id)
    model = full_map(folder)
    edges = model["edges"]
    dt, _, trainable, _, _, spatial = interval_data(t, x, speed, runs, edges)
    nb = len(edges) - 1
    ref = trainable & interval_run_mask(t, pair["reference"])
    readout = trainable & ((speed[:-1] + speed[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > model["ends"][0] + 20) & ((x[:-1] + x[1:]) / 2 < model["ends"][1] - 20)
    masks = {"reference": ref, "baseline": readout & interval_run_mask(t, [pair["baseline"]]), "target": readout & interval_run_mask(t, [pair["target"]])}
    result = {name + "_exposure": np.bincount(spatial[mask], weights=dt[mask], minlength=nb) for name, mask in masks.items()}
    if any(e.sum() <= 0 for e in result.values()):
        raise ValueError("empty_bank_exposure")
    result.update(generating_rates=model["rates"].reshape(2, nb, -1)[row.direction].T, unit_keys=model["units"], edges=edges)
    return result


def calibrate_bank(bank, identity):
    records, fit_records = [], []
    ref_t, base_t, target_t = [bank[k + "_exposure"] for k in ("reference", "baseline", "target")]
    for unit, truth in zip(bank["unit_keys"], bank["generating_rates"], strict=True):
        uid = "_".join(map(str, unit.astype(int)))
        shift_sign = 1 if key_seed(identity, uid, "shift") % 2 else -1
        targets = {case: target_profile(truth, case, shift_sign) for case in CASES}
        for gain in PARAMETERS["reference_gains"]:
            for rep in range(PARAMETERS["reference_repeats"]):
                rng = np.random.default_rng(key_seed(identity, uid, gain, rep))
                count = rng.poisson(ref_t * truth * gain)
                fitted, included = estimate_reference(count, ref_t)
                fit_records.append({"anchor": identity, "unit_id": uid, "reference_gain": gain, "replicate": rep, "reference_spikes": int(count.sum()), "included": included})
                for arm, reference in (("known_map", truth), ("estimated_map", fitted)):
                    for case, target in targets.items():
                        expected, variance = change_moments(truth, target, reference, base_t, target_t)
                        row = {
                            "anchor": identity,
                            "unit_id": uid,
                            "reference_gain": gain,
                            "replicate": rep,
                            "included": included,
                            "arm": arm,
                            "case": case,
                            "expected_change_nats_per_spike": expected,
                            "variance_at_one_spike_per_readout": variance,
                            "shift_sign": shift_sign,
                        }
                        row.update({f"sd_at_{n}_spikes": float(np.sqrt(variance / n)) for n in PARAMETERS["readout_counts"]})
                        records.append(row)
    return pd.DataFrame(records), pd.DataFrame(fit_records)


def aggregate(frame):
    keys = ["animal", "drug", "novel", "reference_gain", "arm", "case"]
    included = frame.loc[frame.included].copy()
    return (
        included.groupby(keys)
        .agg(
            n_rows=("expected_change_nats_per_spike", "size"),
            n_anchors=("anchor", "nunique"),
            mean_expected_change=("expected_change_nats_per_spike", "mean"),
            median_expected_change=("expected_change_nats_per_spike", "median"),
            p10_expected_change=("expected_change_nats_per_spike", lambda x: x.quantile(0.1)),
            p90_expected_change=("expected_change_nats_per_spike", lambda x: x.quantile(0.9)),
            fraction_expected_change_positive=("expected_change_nats_per_spike", lambda x: (x > 1e-10).mean()),
            median_sd_at_5_spikes=("sd_at_5_spikes", "median"),
            median_sd_at_20_spikes=("sd_at_20_spikes", "median"),
            median_sd_at_80_spikes=("sd_at_80_spikes", "median"),
        )
        .reset_index()
    )


def run(args):
    previous = json.loads((args.opportunity_dir / "manifest.json").read_text())
    verification = json.loads((args.opportunity_dir / "verification.json").read_text())
    if verification["status"] != "passed" or verification["manifest_sha256"] != file_sha256(args.opportunity_dir / "manifest.json"):
        raise ValueError("verified opportunity bank required")
    for name, digest in previous["outputs"].items():
        if file_sha256(args.opportunity_dir / name) != digest:
            raise ValueError("changed opportunity input")
    opportunities = pd.read_csv(args.opportunity_dir / "opportunities.csv", float_precision="round_trip")
    selected = select_anchors(opportunities)
    inputs = {
        "opportunities": args.opportunity_dir / "opportunities.csv",
        "opportunity_manifest": args.opportunity_dir / "manifest.json",
        "opportunity_verification": args.opportunity_dir / "verification.json",
        "producer": Path(__file__),
        "protocol": ROOT / "docs/kleinman_spatial_expression_protocol.md",
    }
    for f in ("validate_kleinman_run_decoder.py", "calibrate_kleinman_replay_content.py", "audit_kleinman_ripple_run_opportunities.py"):
        inputs[f] = ROOT / "scripts" / f
    for row in selected.itertuples():
        folder = args.dataset_root / "Experiment_1" / row.animal / row.session
        for name in ("session_info.mat", "spike_data.mat"):
            inputs[f"{row.animal}/{row.session}/{name}"] = folder / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "banks").mkdir()
    selected.to_csv(args.output_dir / "selected_anchors.csv", index=False)
    frames, fits, inventory = [], [], []
    for index, row in selected.iterrows():
        identity = f"{row.animal}/{row.session}/{row.opportunity_id}"
        bank = build_bank(args.dataset_root / "Experiment_1" / row.animal / row.session, row)
        np.savez_compressed(args.output_dir / "banks" / f"{index:03}.npz", **bank)
        metadata = {
            "anchor": identity,
            "animal": row.animal,
            "session": row.session,
            "drug": int(row.drug),
            "novel": int(row.novel),
            "direction": int(row.direction),
            "epoch": int(row.epoch),
        }
        calibration, fitted = calibrate_bank(bank, identity)
        for k, value in metadata.items():
            calibration[k] = value
            fitted[k] = value
        frames.append(calibration)
        fits.append(fitted)
        rt = bank["reference_exposure"]
        st = gaussian_filter1d(rt, PARAMETERS["smooth_bins"], mode="constant", truncate=4)
        inventory.append(
            {
                **metadata,
                "bank_file": f"banks/{index:03}.npz",
                "n_units": len(bank["unit_keys"]),
                "raw_target_support_fraction": float(bank["target_exposure"][rt >= 0.1].sum() / bank["target_exposure"].sum()),
                "smoothed_target_nonzero_fraction": float(bank["target_exposure"][st > 0].sum() / bank["target_exposure"].sum()),
            }
        )
        print(json.dumps({"anchor": identity, "units": len(bank["unit_keys"]), "rows": len(calibration)}), flush=True)
    frame, fitted = pd.concat(frames, ignore_index=True), pd.concat(fits, ignore_index=True)
    frame.to_csv(args.output_dir / "calibration_rows.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    fitted.to_csv(args.output_dir / "reference_fit_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(inventory).to_csv(args.output_dir / "bank_inventory.csv", index=False)
    aggregate(frame).to_csv(args.output_dir / "calibration_summary.csv", index=False)
    fitted.groupby(["animal", "drug", "novel", "reference_gain"]).agg(
        n_fits=("included", "size"), included_fits=("included", "sum"), median_reference_spikes=("reference_spikes", "median")
    ).to_csv(args.output_dir / "reference_inclusion_summary.csv")
    known = frame.loc[frame.arm.eq("known_map")]
    same = frame.loc[frame.case.isin(["unchanged", "gain_only_4x"])].pivot(
        index=["anchor", "unit_id", "reference_gain", "replicate", "arm"], columns="case", values="expected_change_nats_per_spike"
    )
    checks = {
        "complete_48_anchors": len(selected) == 48,
        "finite_moments": bool(np.isfinite(frame[["expected_change_nats_per_spike", "variance_at_one_spike_per_readout"]]).all().all()),
        "oracle_unchanged_zero": bool(known.loc[known.case.eq("unchanged"), "expected_change_nats_per_spike"].abs().max() < 1e-10),
        "gain_invariance": bool((same.unchanged - same.gain_only_4x).abs().max() < 1e-10),
        "oracle_sharpen_nonnegative": bool(known.loc[known.case.eq("sharpen_1p5"), "expected_change_nats_per_spike"].min() >= -1e-10),
        "oracle_broaden_nonpositive": bool(known.loc[known.case.eq("broaden_0p5"), "expected_change_nats_per_spike"].max() <= 1e-10),
    }
    pd.DataFrame([{"gate": k, "passed": v} for k, v in checks.items()]).to_csv(args.output_dir / "technical_checks.csv", index=False)
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("input changed")
    manifest = {
        **provenance,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "parameters": PARAMETERS,
        "cases": CASES,
        "real_future_outcome_scored": False,
        "recruitment_association_scored": False,
        "drug_effect_scored": False,
        "technical_checks_passed": all(checks.values()),
        "n_calibration_rows": len(frame),
        "n_reference_fits": len(fitted),
        "outputs": {str(p.relative_to(args.output_dir)): file_sha256(p) for p in args.output_dir.rglob("*") if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--opportunity-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
