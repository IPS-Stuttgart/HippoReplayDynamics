#!/usr/bin/env python3
"""Source-verified CA1/PFC RUN validation, with whole-epoch holdouts; no replay."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance
from acquire_dandi000978 import destination, write_json
from audit_dandi000978_cohort import metadata_fingerprint, verify_acquisition_record
from dandi000978_verified_units import (
    SUPPORTED_ANIMALS,
    read_author_crosswalks,
    verified_unit_table,
)
from preflight_dandi000978 import inspect, text

PARAMETERS = {
    "bin_s": 0.25,
    "minimum_speed_cm_s": 5.0,
    "min_frames": 5,
    "max_frame_gap_s": 0.1,
    "min_training_spikes": 20,
    "min_encoding_units": 2,
    "spatial_bin_cm": 8.0,
    "smoothing_sigma_cm": 8.0,
    "smoothing_cutoff_cm": 24.0,
    "min_occupancy_s": 0.5,
    "prior_seconds": 0.25,
    "min_training_bins": 200,
    "min_test_bins": 40,
    "min_trial_bins_for_spatial_null": 5,
    "min_training_trials_per_route": 3,
    "shuffles": 99,
    "seed": 20260923,
}
ROUTES = ((1, 2), (1, 3), (2, 1), (3, 1))
READOUTS = ("poisson", "composition", "count_only")


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def prepare_bins(arrays, trials):
    p = PARAMETERS
    clock, data = arrays["clock"], arrays["position_and_speed"]
    ends = arrays["spike_ends"]
    spike_lists = [arrays["spikes"][a:b] for a, b in zip(np.r_[0, ends[:-1]], ends, strict=True)]
    rows, count_parts, audit = [], [], []
    ordered = trials.sort_values("start_time")
    if ordered.id.duplicated().any() or (ordered.start_time.to_numpy()[1:] < ordered.stop_time.to_numpy()[:-1] - 1e-6).any():
        raise ValueError("Trial identities or nonoverlap invalid")
    for trial in ordered.itertuples(index=False):
        route = (int(trial.start_well), int(trial.end_well))
        if trial.start_well not in (1, 2, 3) or trial.end_well not in (1, 2, 3):
            raise ValueError("Unrecognized well labels")
        route_id = ROUTES.index(route) if route in ROUTES else -1
        n = int(np.floor((trial.stop_time - trial.start_time) / p["bin_s"]))
        edges = trial.start_time + np.arange(n + 1) * p["bin_s"]
        frame_index = np.searchsorted(clock, edges, side="left")
        eligible, means, speeds = [], [], []
        for j, (a, b) in enumerate(itertools.pairwise(frame_index)):
            block = data[a:b]
            if b - a < p["min_frames"] or not np.isfinite(block).all():
                continue
            if np.diff(clock[a:b]).max(initial=0) > p["max_frame_gap_s"]:
                continue
            if (block[:, 2] <= p["minimum_speed_cm_s"]).any():
                continue
            eligible.append(j)
            means.append(block[:, :2].mean(axis=0))
            speeds.append(block[:, 2].mean())
        if eligible:
            counts = np.column_stack([np.diff(np.searchsorted(s, edges, side="left")) for s in spike_lists])
            count_parts.append(counts[eligible])
            for j, xy, speed in zip(eligible, means, speeds, strict=True):
                rows.append(
                    {
                        "trial_id": int(trial.id),
                        "epoch": int(trial.epoch_index),
                        "route": route_id,
                        "start_s": float(edges[j]),
                        "stop_s": float(edges[j + 1]),
                        "x_cm": xy[0],
                        "y_cm": xy[1],
                        "speed_cm_s": speed,
                    }
                )
        expected_type = 0 if route[0] == 1 else 1
        audit.append(
            {
                "trial_id": int(trial.id),
                "epoch": int(trial.epoch_index),
                "start_well": route[0],
                "end_well": route[1],
                "correct": trial.correct,
                "canonical_route": route_id,
                "native_trajectory_type": trial.trajectory_type,
                "native_direction_disagrees_with_endpoint_rule": trial.trajectory_type != expected_type,
                "possible_bins": n,
                "eligible_bins": len(eligible),
                "route_exclusion": "noncanonical_endpoints" if route_id < 0 else ("no_movement_bins" if not eligible else ""),
            }
        )
    if not rows:
        raise ValueError("No eligible RUN bins")
    return pd.DataFrame(rows), np.concatenate(count_parts), pd.DataFrame(audit)


def epoch_masks(bins, epoch):
    test = bins.epoch.to_numpy() == epoch
    train = ~test
    if not test.any() or not train.any():
        raise ValueError("Empty epoch holdout")
    if set(bins.loc[train, "trial_id"]) & set(bins.loc[test, "trial_id"]):
        raise ValueError("A trial crosses training and test epochs")
    return train, test


def fit_spatial(counts, xy, train):
    p = PARAMETERS
    if train.sum() < p["min_training_bins"]:
        raise ValueError("Too few training RUN bins")
    use = counts[train].sum(axis=0) >= p["min_training_spikes"]
    if use.sum() < p["min_encoding_units"]:
        raise ValueError("Too few training-active encoding units")
    keys, inverse, number = np.unique(np.floor(xy[train] / p["spatial_bin_cm"]).astype(int), axis=0, return_inverse=True, return_counts=True)
    occupancy = number * p["bin_s"]
    centers = (keys + 0.5) * p["spatial_bin_cm"]
    sums = np.zeros((len(keys), int(use.sum())))
    np.add.at(sums, inverse, counts[train][:, use])
    support = occupancy >= p["min_occupancy_s"]
    if support.sum() < 4:
        raise ValueError("Too few training-supported spatial bins")
    distances = cdist(centers[support], centers)
    kernel = np.exp(-0.5 * (distances / p["smoothing_sigma_cm"]) ** 2)
    kernel[distances > p["smoothing_cutoff_cm"]] = 0
    global_rate = counts[train][:, use].sum(axis=0) / (train.sum() * p["bin_s"])
    rates = (kernel @ sums + p["prior_seconds"] * global_rate) / (kernel @ occupancy + p["prior_seconds"])[:, None]
    return np.maximum(rates, 1e-10), use, centers[support], keys[support]


def posterior(counts, rates, exposure, kind):
    counts, rates = np.asarray(counts), np.asarray(rates)
    exposure = np.asarray(exposure)
    if kind == "composition":
        logp = counts @ np.log(rates / rates.sum(axis=1, keepdims=True)).T
    elif kind == "poisson":
        logp = counts @ np.log(rates).T - exposure.reshape(-1, 1) * rates.sum(axis=1)
    elif kind == "count_only":
        totals = rates.sum(axis=1)
        logp = counts.sum(axis=1)[:, None] * np.log(totals) - exposure.reshape(-1, 1) * totals
    else:
        raise ValueError("Unknown readout")
    logp -= logsumexp(logp, axis=1, keepdims=True)
    return np.exp(logp)


def trial_weighted_mean(values, trials):
    return float(pd.Series(values).groupby(np.asarray(trials)).mean().mean())


def spatial_scores(bins, counts, train, test, key):
    p = PARAMETERS
    xy = bins[["x_cm", "y_cm"]].to_numpy()
    rates, use, centers, keys = fit_spatial(counts, xy, train)
    # A within-trial phase null requires multiple eligible samples per trial.
    trial_sizes = bins.loc[test].groupby("trial_id").size()
    allowed = set(trial_sizes[trial_sizes >= p["min_trial_bins_for_spatial_null"]].index)
    tested = test & bins.trial_id.isin(allowed).to_numpy()
    if tested.sum() < p["min_test_bins"]:
        raise ValueError("Too few held-out RUN bins for within-trial null")
    truth = xy[tested]
    trial_ids = bins.loc[tested, "trial_id"].to_numpy()
    groups = [np.flatnonzero(trial_ids == i) for i in np.unique(trial_ids)]
    rng = np.random.default_rng(seed_for(p["seed"], *key, "spatial_null"))
    shifts = np.array([[rng.integers(max(1, int(np.ceil(0.2 * len(g)))), len(g) - max(1, int(np.ceil(0.2 * len(g)))) + 1) for g in groups] for _ in range(p["shuffles"])])
    support = {tuple(k) for k in keys}
    covered = [tuple(k) in support for k in np.floor(truth / p["spatial_bin_cm"]).astype(int)]
    output, predictions = [], []
    audit = {"train_indices": np.flatnonzero(train), "test_indices": np.flatnonzero(tested), "use": use, "rates": rates, "spatial_centers_cm": centers, "null_shifts": shifts}
    for kind in ("poisson", "composition"):
        prob = posterior(counts[tested][:, use], rates, np.full(tested.sum(), p["bin_s"]), kind)
        mean, maximum = prob @ centers, centers[prob.argmax(axis=1)]
        errors = np.linalg.norm(mean - truth, axis=1)
        map_errors = np.linalg.norm(maximum - truth, axis=1)
        mae = trial_weighted_mean(errors, trial_ids)
        null = []
        for draw in shifts:
            shuffled = mean.copy()
            for group, shift in zip(groups, draw, strict=True):
                shuffled[group] = np.roll(mean[group], int(shift), axis=0)
            null.append(trial_weighted_mean(np.linalg.norm(shuffled - truth, axis=1), trial_ids))
        null = np.asarray(null)
        output.append(
            {
                "task": "position",
                "readout": kind,
                "n_training_bins": int(train.sum()),
                "n_test_bins": int(tested.sum()),
                "n_test_trials": len(groups),
                "n_test_bins_excluded_short_trial": int(test.sum() - tested.sum()),
                "n_encoding_units": int(use.sum()),
                "n_spatial_bins": len(centers),
                "trial_balanced_mean_error_cm": mae,
                "trial_balanced_map_error_cm": trial_weighted_mean(map_errors, trial_ids),
                "trial_median_error_cm": float(pd.Series(errors).groupby(trial_ids).median().median()),
                "training_median_baseline_error_cm": trial_weighted_mean(np.linalg.norm(truth - np.median(xy[train], axis=0), axis=1), trial_ids),
                "test_train_support_fraction": float(np.mean(covered)),
                "test_nonzero_spike_fraction": float((counts[tested][:, use].sum(axis=1) > 0).mean()),
                "null_median_error_cm": float(np.median(null)),
                "gain_over_null_cm": float(np.median(null) - mae),
                "empirical_p": float((1 + (null <= mae).sum()) / (1 + len(null))),
            }
        )
        frame = bins.loc[tested].copy()
        frame["readout"] = kind
        frame["mean_x_cm"], frame["mean_y_cm"] = mean.T
        frame["map_x_cm"], frame["map_y_cm"] = maximum.T
        frame["mean_error_cm"], frame["map_error_cm"] = errors, map_errors
        frame["posterior_entropy_nats"] = -(prob * np.log(np.maximum(prob, 1e-300))).sum(axis=1)
        predictions.append(frame)
        audit[f"posterior_{kind}"] = prob
        audit[f"null_error_cm_{kind}"] = null
    return output, pd.concat(predictions, ignore_index=True), audit


def aggregate_trials(bins, counts):
    labels, totals, exposures = [], [], []
    for tid, ix in bins.groupby("trial_id", sort=True).indices.items():
        b = bins.iloc[ix]
        if b.epoch.nunique() != 1 or b.route.nunique() != 1:
            raise ValueError("Inconsistent labels within trial")
        if b.route.iloc[0] < 0:
            continue
        labels.append({"trial_id": tid, "epoch": b.epoch.iloc[0], "route": b.route.iloc[0]})
        totals.append(counts[ix].sum(axis=0))
        exposures.append(len(ix) * PARAMETERS["bin_s"])
    if not labels:
        raise ValueError("No canonical route trials")
    return pd.DataFrame(labels), np.asarray(totals), np.asarray(exposures)


def fit_routes(counts, exposure, labels):
    global_rate = counts.sum(axis=0) / exposure.sum()
    return np.array(
        [(counts[labels == k].sum(axis=0) + PARAMETERS["prior_seconds"] * global_rate) / (exposure[labels == k].sum() + PARAMETERS["prior_seconds"]) for k in range(4)]
    ).clip(1e-10)


def balanced_accuracy(truth, prediction):
    if set(truth) != set(range(4)):
        raise ValueError("All four routes required in held-out epoch")
    return float(np.mean([np.mean(prediction[truth == k] == k) for k in range(4)]))


def permute_training_routes(labels, epochs, rng):
    permuted = labels.copy()
    for epoch in np.unique(epochs):
        ix = np.flatnonzero(epochs == epoch)
        permuted[ix] = rng.permutation(labels[ix])
    return permuted


def route_scores(bins, counts, epoch, key):
    labels, totals, exposures = aggregate_trials(bins, counts)
    train, test = epoch_masks(labels, epoch)
    use = totals[train].sum(axis=0) >= PARAMETERS["min_training_spikes"]
    if use.sum() < PARAMETERS["min_encoding_units"]:
        raise ValueError("Too few route encoding units")
    ytrain, ytest = labels.route.to_numpy()[train], labels.route.to_numpy()[test]
    if any((ytrain == k).sum() < PARAMETERS["min_training_trials_per_route"] for k in range(4)):
        raise ValueError("Too few training trials for one or more routes")
    if set(ytest) != set(range(4)):
        raise ValueError("Held-out epoch lacks one or more canonical routes")
    training, testing = totals[train][:, use], totals[test][:, use]
    rates = fit_routes(training, exposures[train], ytrain)
    rng = np.random.default_rng(seed_for(PARAMETERS["seed"], *key, "route_null"))
    null_rates = [fit_routes(training, exposures[train], permute_training_routes(ytrain, labels.epoch.to_numpy()[train], rng)) for _ in range(PARAMETERS["shuffles"])]
    rows, frames, null_rows = [], [], []
    for kind in READOUTS:
        prob = posterior(testing, rates, exposures[test], kind)
        pred = prob.argmax(axis=1)
        accuracy = balanced_accuracy(ytest, pred)
        null = np.array([balanced_accuracy(ytest, posterior(testing, r, exposures[test], kind).argmax(axis=1)) for r in null_rates])
        log_loss = -np.log(prob[np.arange(len(ytest)), ytest].clip(1e-300))
        rows.append(
            {
                "task": "route",
                "readout": kind,
                "n_encoding_units": int(use.sum()),
                "n_training_trials": int(train.sum()),
                "n_test_trials": int(test.sum()),
                "balanced_accuracy": accuracy,
                "class_balanced_log_loss": float(np.mean([log_loss[ytest == k].mean() for k in range(4)])),
                "null_median_balanced_accuracy": float(np.median(null)),
                "null_p95_balanced_accuracy": float(np.quantile(null, 0.95)),
                "gain_over_null_balanced_accuracy": float(accuracy - np.median(null)),
                "above_null_p95": bool(accuracy > np.quantile(null, 0.95)),
                "empirical_p": float((1 + (null >= accuracy).sum()) / (1 + len(null))),
                "test_nonzero_spike_fraction": float((testing.sum(axis=1) > 0).mean()),
            }
        )
        frame = labels.loc[test].copy()
        frame["readout"], frame["predicted_route"] = kind, pred
        frame["n_spikes"], frame["movement_exposure_s"] = testing.sum(axis=1), exposures[test]
        for k in range(4):
            frame[f"posterior_route_{k}"] = prob[:, k]
        frames.append(frame)
        null_rows.extend({"readout": kind, "shuffle": i, "balanced_accuracy": a} for i, a in enumerate(null))
    return rows, pd.concat(frames, ignore_index=True), pd.DataFrame(null_rows)


def summarize(out, rows, eligibility):
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "fold_metrics.csv", index=False)
    status = pd.DataFrame(eligibility)
    status.to_csv(out / "cohort_eligibility.csv", index=False)
    ok = scores[scores.status == "scored"]
    metrics = [
        "trial_balanced_mean_error_cm",
        "trial_balanced_map_error_cm",
        "gain_over_null_cm",
        "test_train_support_fraction",
        "n_encoding_units",
        "balanced_accuracy",
        "gain_over_null_balanced_accuracy",
        "above_null_p95",
    ]
    for col in metrics:
        if col not in ok:
            ok = ok.assign(**{col: np.nan})
    file_summary = ok.groupby(["animal", "file", "region", "task", "readout"])[metrics].median()
    file_summary["folds_scored"] = ok.groupby(["animal", "file", "region", "task", "readout"]).size()
    file_summary.to_csv(out / "file_summary.csv")
    animal = file_summary.reset_index().groupby(["animal", "region", "task", "readout"])[metrics].median()
    animal.to_csv(out / "animal_summary.csv")
    readiness = []
    for (who, filename, region), group in scores.groupby(["animal", "file", "region"]):
        r = group[(group.task == "route") & (group.readout == "composition")]
        n = len(r)
        supported = (r.status == "scored") & r.get("above_null_p95", pd.Series(False, index=r.index)).fillna(False).astype(bool)
        gain = pd.to_numeric(r.get("gain_over_null_balanced_accuracy", pd.Series(np.nan, index=r.index)))
        readiness.append(
            {
                "animal": who,
                "file": filename,
                "region": region,
                "expected_route_folds": n,
                "successful_route_folds": int((r.status == "scored").sum()),
                "route_folds_above_null_p95": int(supported.sum()),
                "median_composition_route_gain": float(gain.median()),
                "RUN_route_content_supported": bool(n > 0 and (r.status == "scored").all() and supported.sum() / n >= 0.5 and gain.median() > 0),
            }
        )
    pd.DataFrame(readiness).to_csv(out / "readiness_by_file_region.csv", index=False)
    gates = [
        {"gate": "supported_files_present", "passed": bool((status.status == "processed").sum() == 3)},
        {"gate": "all_expected_readouts_scored", "passed": bool(len(scores) == 160 and (scores.status == "scored").all())},
        {"gate": "all_supported_animals_represented", "passed": set(ok.animal) == {"JS14", "ZT2"}},
        {"gate": "both_regions_present", "passed": set(ok.region) == {"CA1", "PFC"}},
        {"gate": "no_replay_scoring", "passed": True},
    ]
    technical = all(g["passed"] for g in gates)
    gates.append({"gate": "supported_subset_technical_overall", "passed": technical})
    gates.append({"gate": "full_eight_animal_cohort_ready", "passed": False})
    pd.DataFrame(gates).to_csv(out / "gate_summary.csv", index=False)
    lines = [
        "# DANDI000978 held-out RUN readouts",
        "",
        f"Supported-subset technical completion: {technical}.",
        "Two source-supported animals, three files; six animals still have unresolved anatomy.",
        "ZT2 files are aggregated within one animal, not two replicates.",
        "",
        "## Per-file route feasibility",
        "",
        "```csv",
        pd.DataFrame(readiness).to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation limits",
        "",
        "These are RUN readouts, not replay results. Source mapping was not chosen from decoding performance.",
        "Count-conditioned route decoding cannot use total spike rate alone, but can use position/direction covariates.",
        "Passing this check does not prove abstract cortical content or independent replay ground truth.",
        "Sparse held-out route counts, learning across epochs and two-animal coverage limit generalization.",
        "All eligible windows, including zero-spike windows, remain in testing; technical failures remain visible.",
        "Private crosswalks and source correspondence must not be redistributed with these outputs.",
        "",
    ]
    (out / "run_readout_summary.md").write_text("\n".join(lines))
    return technical


def run(args):
    manifest_path = args.dataset_root / "metadata/asset_manifest.json"
    status_path = args.dataset_root / "download_status.json"
    source = json.loads(manifest_path.read_text())
    verify_acquisition_record(args.dataset_root, source, json.loads(status_path.read_text()))
    crosswalks = read_author_crosswalks(args.author_archive)
    provenance = build_script_provenance(input_paths={"assets": manifest_path, "acquisition": status_path, "private_author_archive": args.author_archive}, cwd=ROOT)
    if provenance["git_dirty"] is not False or len(provenance["code_commit"]) != 40:
        raise ValueError("Clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.output_dir
    record = {
        **provenance,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "parameters": PARAMETERS,
        "routes": ROUTES,
        "scope": "RUN_only_no_replay",
        "full_raw_rehash": False,
        "raw_hash_basis": "verified_acquisition_plus_current_size",
        "supported_animals": SUPPORTED_ANIMALS,
        "private_source_not_for_redistribution": True,
        "source_files": [],
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
    }
    write_json(out / "manifest.json", record)
    rows, eligibility, all_units, trial_audits = [], [], [], []
    for asset in source["assets"]:
        filename = Path(asset["path"]).name
        animal = filename.split("SingleDay-")[1].split("_")[0]
        item = {"animal": animal, "file": filename, "asset_id": asset["asset_id"], "status": "pending_source_anatomy", "failure_reason": ""}
        if animal not in SUPPORTED_ANIMALS:
            eligibility.append(item)
            continue
        started = time.monotonic()
        try:
            with h5py.File(destination(args.dataset_root, asset), "r") as f:
                _report, _epochs, trials, _, arrays = inspect(f)
                pos = f["processing/behavior/Position/SpatialSeries/data"]
                if text(pos.attrs["unit"]) != "centimeters; centimeters/second" or pos.attrs.get("conversion", 1) != 1 or pos.attrs.get("offset", 0) != 0:
                    raise ValueError("Unrecognized physical coordinate convention")
                table = f[f["units/electrodes"].attrs["table"]]
                units = verified_unit_table(
                    animal, filename, arrays["unit_ids"], f["units/electrodes"][()], table["location"].asstr()[()], table["group_name"].asstr()[()], crosswalks.get(filename)
                )
                fingerprint = metadata_fingerprint(f)
            units.insert(0, "file", filename)
            units.insert(0, "animal", animal)
            all_units.append(units)
            record["source_files"].append(
                {
                    "asset_id": asset["asset_id"],
                    "file": filename,
                    "raw_sha256": asset["digest"]["dandi:sha2-256"],
                    "metadata_sha256": fingerprint,
                    "mapping_basis": units.mapping_basis.iloc[0],
                }
            )
            bins, counts, trial_audit = prepare_bins(arrays, trials)
            trial_audits.append(trial_audit.assign(animal=animal, file=filename))
            expected_epochs = sorted(trials.epoch_index.unique())
            for epoch in expected_epochs:
                train, test = epoch_masks(bins, epoch)
                for region in ("CA1", "PFC"):
                    unit_mask = units.region.to_numpy() == region
                    regional = counts[:, unit_mask]
                    key = (filename, region, int(epoch))
                    base = {
                        "animal": animal,
                        "file": filename,
                        "region": region,
                        "heldout_epoch": int(epoch),
                        "status": "failed",
                        "failure_reason": "",
                        "source_units": int(unit_mask.sum()),
                    }
                    dest = out / asset["asset_id"] / region / f"epoch_{epoch}"
                    dest.mkdir(parents=True)
                    for task, kinds in (("position", ("poisson", "composition")), ("route", READOUTS)):
                        try:
                            if task == "position":
                                metrics, predictions, audit = spatial_scores(bins, regional, train, test, key)
                                np.savez_compressed(dest / "position_audit.npz", **audit, source_unit_ids=arrays["unit_ids"][unit_mask])
                            else:
                                metrics, predictions, nulls = route_scores(bins, regional, epoch, key)
                                nulls.to_csv(dest / "route_nulls.csv", index=False)
                            predictions.to_csv(dest / f"{task}_predictions.csv", index=False)
                            rows.extend({**base, **m, "status": "scored"} for m in metrics)
                        except ValueError as exc:
                            rows.extend({**base, "task": task, "readout": kind, "failure_reason": str(exc)} for kind in kinds)
                    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
                    print(json.dumps({**base, "status": "fold_finished", "rows_written": len(rows)}), flush=True)
            item["status"] = "processed"
        except (ValueError, KeyError, OSError, TypeError) as exc:
            item.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        item["runtime_s"] = time.monotonic() - started
        eligibility.append(item)
        pd.DataFrame(eligibility).to_csv(out / "cohort_eligibility.csv", index=False)
        if all_units:
            pd.concat(all_units).to_csv(out / "verified_unit_mappings.csv", index=False)
        if trial_audits:
            pd.concat(trial_audits).to_csv(out / "trial_eligibility.csv", index=False)
        write_json(out / "manifest.json", record)
        print(json.dumps(item), flush=True)
    if not rows:
        raise ValueError("No readouts attempted; inspect cohort_eligibility.csv")
    complete = summarize(out, rows, eligibility)
    write_json(
        out / "terminal_status.json",
        {
            "status": "complete" if complete else "complete_with_failed_gates",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "technical_complete": complete,
            "replay_scored": False,
        },
    )
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--author-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir / "terminal_status.json").exists():
            write_json(args.output_dir / "terminal_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "completed_at_utc": datetime.now(UTC).isoformat()})
        raise


if __name__ == "__main__":
    sys.exit(main())
