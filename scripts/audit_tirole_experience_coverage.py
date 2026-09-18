"""RUN-only, rate-paired coverage intervention with fixed independent rest readouts."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import classify_sequence, stable_seed
from scripts.score_tirole_content_coverage import validate_bank


def run_information(rates, occupancy, valid):
    rates, occupancy, valid = np.asarray(rates, float), np.asarray(occupancy, float), np.asarray(valid, bool)
    if rates.ndim != 3 or rates.shape[0] != 2 or occupancy.shape != valid.shape or valid.shape != (2, rates.shape[2]):
        raise ValueError("two-track RUN map dimensions required")
    if not np.isfinite(rates).all() or (rates < 0).any() or not np.isfinite(occupancy).all() or (occupancy < 0).any():
        raise ValueError("invalid RUN rates/occupancy")
    weighted = occupancy * valid
    if not (weighted.sum(axis=1) > 0).all() or not (occupancy.sum(axis=1) > 0).all():
        raise ValueError("missing occupied RUN support")
    weights = weighted / weighted.sum(axis=1, keepdims=True)
    mean = (rates * weights[:, None]).sum(axis=2)
    relative = np.divide(rates, mean[:, :, None], out=np.zeros_like(rates), where=mean[:, :, None] > 0)
    terms = np.zeros_like(relative)
    nonzero = relative > 0
    terms[nonzero] = relative[nonzero] * np.log2(relative[nonzero])
    information = np.maximum((weights[:, None] * terms).sum(axis=2), 0.0)
    # Match the RUN-rate definition already frozen for the independent B control.
    run_rate = (rates * (occupancy / occupancy.sum(axis=1, keepdims=True))[:, None]).sum(axis=2).mean(axis=0)
    return information, run_rate


def coverage_subsets(inference, information, run_rate, session, split):
    ids = np.asarray(inference, int)
    if len(ids) < 5 or len(set(ids)) != len(ids) or not (run_rate[ids] > 0).all():
        raise ValueError("positive-rate distinct inference units required")
    ordered = np.array(sorted(ids, key=lambda i: (run_rate[i], stable_seed(20260918, session, split, int(i), "RUN-rate-pair-tie"))))
    pairs = ordered[: len(ordered) // 2 * 2].reshape(-1, 2)
    singleton = ordered[len(pairs) * 2 :]
    difference = information[0] - information[1]
    first, second = [], []
    for pair in pairs:
        ranked = sorted(pair, key=lambda i: (-difference[i], stable_seed(20260918, session, split, int(i), "RUN-information-tie")))
        first.append(ranked[0])
        second.append(ranked[1])
    arms = {"full": np.sort(ids), "track1_information": np.sort(np.r_[first, singleton]), "track2_information": np.sort(np.r_[second, singleton])}
    for repeat in range(5):
        flips = np.random.default_rng(stable_seed(20260918, session, split, repeat, "rate-paired-random-half-v1")).integers(0, 2, len(pairs))
        arms[f"random_{repeat}_a"] = np.sort(np.r_[pairs[np.arange(len(pairs)), flips], singleton])
        arms[f"random_{repeat}_b"] = np.sort(np.r_[pairs[np.arange(len(pairs)), 1 - flips], singleton])
    return arms, pairs, singleton


def load_fixed_evaluation(folder, bank_hash, session, parts):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["git_dirty"] or manifest["selection_or_thresholds_changed"]:
        raise ValueError("invalid frozen independent evaluation control")
    for name, h in manifest["output_sha256"].items():
        if file_sha256(folder / name) != h:
            raise ValueError("changed independent readout")
    matches = [x for x in manifest["bank_manifests"] if x["session"] == session]
    if len(matches) != 1 or matches[0]["manifest_sha256"] != bank_hash:
        raise ValueError("independent readout used a different candidate bank")
    matching = pd.read_csv(folder / "evaluation_RUN_rate_matching.csv")
    for s, part in enumerate(parts["splits"]):
        row = matching[(matching.session == session) & matching.split.eq(s)]
        if len(row) != 1 or json.loads(row.evaluation_indices.iloc[0]) != part["evaluation"]:
            raise ValueError("evaluation cells changed")
    readout = pd.read_csv(folder / "evaluation_identity_by_event_split.csv")
    readout = readout[readout.session == session]
    if readout.duplicated(["event_id", "split"]).any():
        raise ValueError("duplicate independent readout")
    return readout.set_index(["event_id", "split"])


def run(bank, evaluation, output):
    if output.exists():
        raise ValueError("new immutable coverage intervention required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze RUN-only intervention before outcome scoring")
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    if info["cohort_stratum"] != "strict_RUN_pass":
        raise ValueError("frozen primary sessions only")
    session = info["session"]
    config_path = ROOT / "docs/two_track_content_configuration.json"
    config = json.loads(config_path.read_text())
    if (
        config["n_sequence_shuffles"] != 499
        or config["alpha_per_track_per_null"] != 0.025
        or parts["seed"] != config["seed"]
        or len(parts["splits"]) != 5
        or session not in config["primary_cohort"]
    ):
        raise ValueError("sequence protocol mismatch")
    fixed = load_fixed_evaluation(evaluation, file_sha256(bank / "manifest.json"), session, parts)
    chosen = events[events.ripple_supported].copy()
    if chosen.empty:
        raise ValueError("no covered events")
    information, rates = run_information(maps["rates"], maps["occupancy_s"], maps["valid_bins"])
    arms_by_split, selections, matching = {}, [], []
    for split, part in enumerate(parts["splits"]):
        arms, pairs, singleton = coverage_subsets(part["inference"], information, rates, session, split)
        arms_by_split[split] = arms
        for name, ids in arms.items():
            if set(ids) & (set(parts["detector"]) | set(part["evaluation"])):
                raise ValueError("cell-role leakage")
            selections.append(
                {
                    "session": session,
                    "split": split,
                    "arm": name,
                    "n_units": len(ids),
                    "unit_indices": json.dumps(ids.tolist()),
                    "mean_RUN_rate_hz": float(rates[ids].mean()),
                    "mean_track1_information_bits_per_spike": float(information[0, ids].mean()),
                    "mean_track2_information_bits_per_spike": float(information[1, ids].mean()),
                    "mean_track1_minus_track2_information": float((information[0, ids] - information[1, ids]).mean()),
                }
            )
        for k, pair in enumerate(pairs):
            matching.append(
                {
                    "session": session,
                    "split": split,
                    "pair": k,
                    "first_unit": pair[0],
                    "second_unit": pair[1],
                    "log_RUN_rate_gap": float(abs(np.log(rates[pair[0]]) - np.log(rates[pair[1]]))),
                    "shared_singleton": json.dumps(singleton.tolist()),
                }
            )
    output.mkdir(parents=True)
    pd.DataFrame(selections).to_csv(output / "RUN_only_subsets.csv", index=False)
    pd.DataFrame(matching).to_csv(output / "RUN_rate_pairs.csv", index=False)
    pd.DataFrame(
        {"unit_index": np.arange(len(rates)), "RUN_rate_hz": rates, "track1_information_bits_per_spike": information[0], "track2_information_bits_per_spike": information[1]}
    ).to_csv(output / "RUN_unit_information.csv", index=False)
    rows = []
    started = time.monotonic()
    for count, event in enumerate(chosen.to_dict("records"), 1):
        eid = int(event["event_id"])
        c = counts[offsets[eid] : offsets[eid + 1]]
        for split, part in enumerate(parts["splits"]):
            if (eid, split) not in fixed.index:
                raise ValueError("missing frozen evaluation row")
            b = fixed.loc[(eid, split)]
            rng = np.random.default_rng(stable_seed(config["seed"], session, eid, split, "sequence-nulls"))
            shifts = rng.integers(0, len(maps["bin_centers_cm"]), (499, 2, counts.shape[1]))
            permutations = np.argsort(rng.random((499, len(c))), axis=1)
            base = {
                **event,
                "session": session,
                "animal": info["animal"],
                "split": split,
                "evaluation_conditional_track2_probability": float(expit(-b.conditional_observed_log_odds)),
                "evaluation_conditional_track2_identity_z": float(-b.conditional_identity_z_log_odds),
                "n_evaluation_spikes": int(c[:, part["evaluation"]].sum()),
                "n_evaluation_active_units": int((c[:, part["evaluation"]].sum(axis=0) > 0).sum()),
            }
            for name, ids in arms_by_split[split].items():
                result = classify_sequence(c[:, ids], maps["rates"][:, ids], maps["valid_bins"], maps["bin_centers_cm"], shifts[:, :, ids], permutations)
                rows.append({**base, "arm": name, "n_inference_units": len(ids), "n_inference_spikes": int(c[:, ids].sum()), **result})
        if count % 25 == 0 or count == len(chosen):
            print(json.dumps({"session": session, "completed_events": count, "total_events": len(chosen), "elapsed_s": time.monotonic() - started}), flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != len(chosen) * 5 * 13 or frame.duplicated(["event_id", "split", "arm"]).any():
        raise ValueError("incomplete intervention")
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during intervention")
    frame.to_csv(output / "event_coverage_scores.csv", index=False)
    manifest = {
        "status": "complete",
        "session": session,
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "bank_dir": str(bank.resolve()),
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "evaluation_control_dir": str(evaluation.resolve()),
        "evaluation_manifest_sha256": file_sha256(evaluation / "manifest.json"),
        "configuration_sha256": file_sha256(config_path),
        "selected_event_ids": chosen.event_id.tolist(),
        "n_score_rows": len(frame),
        "elapsed_s": time.monotonic() - started,
        "subset_selection_RUN_only": True,
        "evaluation_readout_reused_unchanged": True,
        "n_splits": 5,
        "n_arms": 13,
        "natural_random_sampling_bias_confirmed": False,
        "true_experience_fractions_estimated": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--evaluation-control-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.evaluation_control_dir, a.output_dir)
