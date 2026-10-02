"""Development-only conditional-independence stress of RUN rate adjustment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import i0
from scipy.stats import t

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.measure_run_pair_coordination_endpoint import crossfit_rates, residual_coordination, strata
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from measure_run_pair_coordination_endpoint import crossfit_rates, residual_coordination, strata


def check(value, message):
    if not value:
        raise ValueError(message)


def known_intensity(theta, references, period, width, p):
    theta, references = np.asarray(theta), np.asarray(references, int)
    check(theta.ndim == 2 and np.isfinite(theta).all() and np.all(references >= 0)
          and np.all(references < theta.shape[1]), "Missing or ambiguous source LFP reference")
    preferences = np.arange(len(references)) * 2 * np.pi / len(references)
    if period == "post":
        preferences[::2] += p["post_even_cell_phase_shift_rad"]
    else:
        check(period == "pre", "Unknown chronological period")
    return width * p["base_rate_hz"] * np.exp(p["theta_concentration"] *
        np.cos(theta[:, references] - preferences[None, :])) / i0(p["theta_concentration"])


def summarize(frame, replicas):
    groups = frame.groupby(["animal", "session", "pause_id", "unit_a", "unit_b"], sort=True)
    check(len(groups) > 0, "Empty stress family")
    critical = t.ppf(1 - .05 / (2 * len(groups)), replicas - 1)
    rows = []
    for identity, d in groups:
        check(len(d) == replicas and d.replicate.nunique() == replicas, "Incomplete stress replicates")
        mean = float(d.discrepancy.mean())
        half_width = float(critical * d.discrepancy.std(ddof=1) / np.sqrt(replicas))
        rows.append({**dict(zip(["animal", "session", "pause_id", "unit_a", "unit_b"], identity, strict=True)),
                     "replicates": replicas, "mean_fitted_change": float(d.fitted_change.mean()),
                     "mean_oracle_change": float(d.oracle_change.mean()), "mean_discrepancy": mean,
                     "discrepancy_sd": float(d.discrepancy.std(ddof=1)),
                     "simultaneous_mc_interval_low": mean - half_width,
                     "simultaneous_mc_interval_high": mean + half_width,
                     "systematic_nuisance_bias_detected": bool(mean - half_width > 0 or mean + half_width < 0),
                     "biological_inference": False})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--endpoint-verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.endpoint_dir
    m = json.loads((root / "manifest.json").read_text())
    v = json.loads(args.endpoint_verification.read_text())
    check(v.get("verified") and v["input_file_sha256"]["endpoint_manifest"] == file_sha256(root / "manifest.json"),
          "Identity-matched independent endpoint verification required")
    for name, digest in m["outputs_sha256"].items():
        check(file_sha256(root / name) == digest, f"Changed endpoint: {name}")
    endpoint_protocol = Path(m["input_file_paths"]["protocol"])
    check(file_sha256(endpoint_protocol) == m["input_file_sha256"]["protocol"], "Changed estimator protocol")
    estimator = json.loads(endpoint_protocol.read_text())
    p = json.loads(args.protocol.read_text())
    check(p["replicates"] >= 2 and p["cells_per_pause"] >= 2 and p["base_rate_hz"] > 0,
          "Positive generator dimensions required")
    provenance = build_script_provenance(input_paths={"endpoint_manifest": root / "manifest.json",
        "endpoint_verification": args.endpoint_verification, "estimator_protocol": endpoint_protocol, "stress_protocol": args.protocol})
    check(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed checkout required")
    source_path = Path(m["input_file_paths"]["measurement_manifest"])
    check(file_sha256(source_path) == m["input_file_sha256"]["measurement_manifest"], "Changed source measurement")
    source_manifest = json.loads(source_path.read_text())
    for name in ("banks.csv", "pauses.csv"):
        check(file_sha256(source_path.parent / name) == source_manifest["outputs_sha256"][name], "Changed source table")
    banks = pd.read_csv(source_path.parent / "banks.csv")
    source_pauses = pd.read_csv(source_path.parent / "pauses.csv")
    pauses = pd.read_csv(root / "pauses.csv")
    selected = source_pauses.merge(pauses[["session", "pause_id", "status"]], on=["session", "pause_id"], validate="one_to_one")
    selected = selected[selected.status == "candidate_endpoint_measured"].sort_values(["animal", "session", "start_s"])
    selected = selected.groupby("animal", sort=True).head(1)
    check(len(selected) > 0, "No fixed eligible animal pauses")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, inventory = [], []
    for pause in selected.itertuples(index=False):
        bank_row = banks[(banks.session == pause.session) & (banks.pause_id == pause.pause_id)]
        check(len(bank_row) == 1, "Ambiguous bank identity")
        bank_row = bank_row.iloc[0]
        check(file_sha256(bank_row.bank_path) == bank_row.bank_sha256, "Changed stress bank")
        with np.load(bank_row.bank_path, allow_pickle=False) as bank:
            n = p["cells_per_pause"]
            ids, references = bank["unit_ids"][:n], bank["unit_theta_reference_index"][:n]
            check(len(ids) == n, "Insufficient frozen cells; do not substitute")
            a, b = np.triu_indices(n, 1)
            periods = {}
            for period in ("pre", "post"):
                valid = np.isfinite(bank[f"{period}_theta_phase_rad"]).all(axis=1)
                theta = bank[f"{period}_theta_phase_rad"][valid]
                times = bank[f"{period}_time_s"][valid]
                labels = strata(bank[f"{period}_position_cm"][valid], bank[f"{period}_direction_rad"][valid],
                                bank[f"{period}_speed_cm_s"][valid], theta, estimator)
                periods[period] = (times, labels, known_intensity(theta, references, period, estimator["run_bin_s"], p))
            salt = int.from_bytes(hashlib.sha256(f"{pause.session}:{pause.pause_id}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(np.random.SeedSequence([p["seed"], salt]))
            for replicate in range(p["replicates"]):
                scores = {}
                for period, (times, labels, mu) in periods.items():
                    counts = rng.poisson(mu)
                    adjusted, _ = crossfit_rates(counts, times, labels, estimator["run_bin_s"], estimator)
                    usable = np.isfinite(adjusted).all(axis=1)
                    check(usable.any(), "Missing crossfit stress predictions")
                    oracle = (counts[usable] - mu[usable]) / np.sqrt(mu[usable])
                    fitted, fitted_n = residual_coordination(adjusted[usable], times[usable], estimator["run_bin_s"],
                                                            estimator["lag_min_s"], estimator["lag_max_s"])
                    truth, truth_n = residual_coordination(oracle, times[usable], estimator["run_bin_s"],
                                                          estimator["lag_min_s"], estimator["lag_max_s"])
                    check(fitted_n == truth_n and fitted_n > 0 and np.isfinite(fitted).all(), "Stress support differs")
                    scores[period] = (fitted[a, b], truth[a, b])
                fitted_change = scores["post"][0] - scores["pre"][0]
                oracle_change = scores["post"][1] - scores["pre"][1]
                for j in range(len(a)):
                    rows.append({"animal": pause.animal, "session": pause.session, "pause_id": pause.pause_id,
                        "unit_a": int(ids[a[j]]), "unit_b": int(ids[b[j]]), "replicate": replicate,
                        "fitted_change": float(fitted_change[j]), "oracle_change": float(oracle_change[j]),
                        "discrepancy": float(fitted_change[j] - oracle_change[j])})
                if replicate % 25 == 0:
                    print(f"STRESS {pause.animal} {replicate}/{p['replicates']}", flush=True)
        inventory.append({"animal": pause.animal, "session": pause.session, "pause_id": pause.pause_id,
                          "bank_path": bank_row.bank_path, "bank_sha256": bank_row.bank_sha256,
                          "cells": n, "replicates": p["replicates"]})
    frame = pd.DataFrame(rows)
    summary = summarize(frame, p["replicates"])
    outputs = {"replicates.csv": frame, "pair_stress_summary.csv": summary, "inventory.csv": pd.DataFrame(inventory)}
    for name, output in outputs.items():
        output.to_csv(args.output_dir / name, index=False)
    biased = int(summary.systematic_nuisance_bias_detected.sum())
    result = {**provenance, "protocol_id": p["protocol_id"], "replicates_per_animal": p["replicates"],
              "selected_animals": len(inventory), "fixed_animal_pair_family": len(summary),
              "biased_animal_pairs": biased,
              "development_stress_status": "nuisance_bias_detected" if biased else "no_bias_detected_in_this_generator",
              "association_fit": False, "full_procedure_calibrated": False, "goal_complete": False,
              "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in outputs},
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE development nuisance bias={biased}/{len(summary)}; association_fit=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
