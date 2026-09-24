#!/usr/bin/env python3
"""Frozen native-SDE content model adequacy, without reward-effect contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from calibrate_kleinman_integrated_extent import fit_events, template_library
from validate_kleinman_run_decoder import align_behavior, matrix, split_units

PARAMETERS = {
    "seed": 20260924,
    "replicas": 99,
    "alarm_tail": 0.05,
    "negative_alarm_limit": 0.1,
    "controls_per_session": 40,
    "fit_bin_s": 0.01,
    "minimum_fit_bins": 4,
    "maximum_duration_s": 2.0,
    "baseline_pseudocount": 0.5,
}


def random_state(animal, session, event_id, namespace):
    identity = int.from_bytes(hashlib.sha256((animal + "/" + session).encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([PARAMETERS["seed"], identity, int(event_id), namespace]))


def complete_edges(start, end):
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("invalid_native_interval")
    n = int(np.floor((end - start) / PARAMETERS["fit_bin_s"] + 1e-9))
    return start + PARAMETERS["fit_bin_s"] * np.arange(n + 1)


def bin_counts(trains, edges):
    return np.column_stack([np.diff(np.searchsorted(s, edges, side="left")) for s in trains])


def multinomial_draws(totals, probability, n, rng):
    if np.any(totals != np.floor(totals)) or np.any(totals < 0):
        raise ValueError("noninteger or negative totals")
    result = np.empty((n, len(totals), probability.shape[1]), dtype=np.int64)
    for t, total in enumerate(totals):
        result[:, t] = rng.multinomial(int(total), probability[t], size=n)
    return result


def saturated_log_likelihood(counts):
    n = counts.sum(axis=-1, keepdims=True)
    return xlogy(counts, counts / np.maximum(n, 1)).sum(axis=(-1, -2))


def crossfit_against_free_composition(counts, logq):
    total_score, baseline_score = 0.0, 0.0
    even = np.arange(len(counts)) % 2 == 0
    for train in [even, ~even]:
        train_ll = logq[:, train].reshape(len(logq), -1) @ counts[train].reshape(-1)
        ties = train_ll >= train_ll.max() - 1e-10
        test_ll = logq[:, ~train].reshape(len(logq), -1) @ counts[~train].reshape(-1)
        total_score += float(test_ll[ties].mean())
        q = counts[train].sum(axis=0) + PARAMETERS["baseline_pseudocount"]
        q = q / q.sum()
        baseline_score += float(counts[~train].sum(axis=0) @ np.log(q))
    return total_score, baseline_score


def diagnose(counts, logq, templates, rng):
    counts = np.asarray(counts)
    if counts.shape != logq.shape[1:] or counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("invalid counts")
    totals = counts.sum(axis=1)
    ll = logq.reshape(len(logq), -1) @ counts.reshape(-1)
    best = int(np.argmax(ll))
    deviance = float(max(0.0, 2 * (saturated_log_likelihood(counts) - ll[best])))
    replicas = multinomial_draws(totals, np.exp(logq[best]), PARAMETERS["replicas"], rng)
    replica_ll = replicas.reshape(len(replicas), -1) @ logq.reshape(len(logq), -1).T
    dev = np.maximum(0, 2 * (saturated_log_likelihood(replicas) - replica_ll.max(axis=1)))
    tail = float((1 + np.count_nonzero(dev >= deviance - 1e-10)) / (1 + len(dev)))
    score, baseline = crossfit_against_free_composition(counts, logq)
    result = fit_events(counts[None], logq, templates).iloc[0].to_dict()
    result.update(
        best_template_index=best,
        best_template_profile=templates.iloc[best].profile,
        best_template_direction=int(templates.iloc[best].direction),
        conditional_deviance=deviance,
        replica_deviance_median=float(np.median(dev)),
        replica_deviance_p95=float(np.quantile(dev, 0.95)),
        deviance_tail_probability=tail,
        deviance_flag=tail <= PARAMETERS["alarm_tail"],
        crossfit_template_log_score=score,
        crossfit_free_composition_log_score=baseline,
        crossfit_template_minus_free=score - baseline,
        crossfit_template_minus_free_per_spike=(score - baseline) / max(counts.sum(), 1),
    )
    return result


def controls(model, animal, session):
    logq, templates = template_library(model, 0.2)
    rows = []
    for i in range(PARAMETERS["controls_per_session"]):
        rng = random_state(animal, session, i, 1)
        static = i % 2 == 0
        choices = np.flatnonzero((templates.extent == 0).to_numpy() == static)
        template = int(rng.choice(choices))
        total = 48 if (i // 2) % 2 == 0 else 96
        totals = rng.multinomial(total, np.full(20, 1 / 20))
        true_q = np.exp(logq[template])
        biased = true_q.copy()
        cells = rng.choice(true_q.shape[1], size=max(1, true_q.shape[1] // 2), replace=False)
        biased[:, cells] *= 4
        biased /= biased.sum(axis=1, keepdims=True)
        for condition, q, namespace in [("matched", true_q, 2), ("cell_gain_mismatch", biased, 3)]:
            counts = multinomial_draws(totals, q, 1, random_state(animal, session, i, namespace + 10))[0]
            fit = diagnose(counts, logq, templates, random_state(animal, session, i, namespace))
            rows.append(
                dict(
                    animal=animal,
                    session=session,
                    control_id=i,
                    condition=condition,
                    generated_profile=templates.iloc[template].profile,
                    generated_extent=float(templates.iloc[template].extent),
                    n_spikes=int(counts.sum()),
                    **fit,
                )
            )
    return pd.DataFrame(rows)


def inventory_session(folder, model, row):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, _, _, _, _, epochs = align_behavior(info)
    keys, all_trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    lookup = {tuple(key): train for key, train in zip(keys, all_trains, strict=True)}
    trains = [lookup[tuple(k)] for k in model["units"]]
    events = matrix(loadmat(folder / "sdes.mat", simplify_cells=True)["sdes"], 4, "sdes")
    records = []
    max_end = -np.inf
    for event_id, (start, end, peak, position) in enumerate(events):
        base = {
            "animal": row.animal,
            "session": row.session,
            "event_id": event_id,
            "extent_eligible_session": bool(row.extent_eligible),
            "start_s": start,
            "end_s": end,
            "peak_s": peak,
            "animal_position_at_onset_cm": position,
            "native_duration_s": end - start,
            "overlaps_earlier_native_event": bool(start < max_end),
            "n_encoding_units": len(trains),
        }
        max_end = max(max_end, end)
        try:
            edges = complete_edges(start, end)
        except ValueError as exc:
            records.append(dict(**base, fit_status=str(exc)))
            continue
        counts = bin_counts(trains, edges)
        full = np.array([np.searchsorted(s, end, side="left") - np.searchsorted(s, start, side="left") for s in trains])
        raw_total = sum(np.searchsorted(s, end, side="left") - np.searchsorted(s, start, side="left") for s in all_trains)
        n = len(edges) - 1
        duration = n * PARAMETERS["fit_bin_s"]
        even_spikes, odd_spikes = int(counts[::2].sum()), int(counts[1::2].sum())
        epoch = [i + 1 for i, (lo, hi) in enumerate(epochs) if start >= lo and end <= hi]
        base.update(
            n_fit_bins=n,
            fit_end_s=float(edges[-1]),
            fit_duration_s=duration,
            omitted_tail_s=float(end - edges[-1]),
            n_raw_spikes=int(raw_total),
            n_native_encoding_spikes=int(full.sum()),
            n_fit_spikes=int(counts.sum()),
            n_omitted_tail_spikes=int(full.sum() - counts.sum()),
            n_active_encoding_units=int((counts.sum(axis=0) > 0).sum()),
            even_bin_spikes=even_spikes,
            odd_bin_spikes=odd_spikes,
            native_epoch=epoch[0] if len(epoch) == 1 else np.nan,
            bank_range_descriptor=bool(0.1 <= duration <= 0.4 and 48 <= counts.sum() <= 96 and (counts.sum(axis=0) > 0).sum() >= 5),
        )
        if start < t[0] or end > t[-1] or not start <= peak <= end:
            status = "native_timing_outside_recording_or_peak"
        elif n < PARAMETERS["minimum_fit_bins"] or duration > PARAMETERS["maximum_duration_s"]:
            status = "outside_numerical_duration_limits"
        elif min(even_spikes, odd_spikes) <= 0:
            status = "empty_crossfit_half"
        elif not row.extent_eligible:
            status = "session_not_extent_eligible"
        else:
            status = "pending"
        records.append(dict(**base, fit_status=status))
    return pd.DataFrame(records), trains


def fit_session(job):
    model_path, folder, records, output = job
    model = dict(np.load(model_path))
    data = pd.DataFrame(records)
    animal, session = data.animal.iloc[0], data.session.iloc[0]
    keys, all_trains, _ = split_units(loadmat(Path(folder) / "spike_data.mat", simplify_cells=True)["spike_data"])
    lookup = {tuple(key): train for key, train in zip(keys, all_trains, strict=True)}
    trains = [lookup[tuple(k)] for k in model["units"]]
    calibration = controls(model, animal, session)
    output = Path(output)
    output.mkdir(parents=True)
    calibration.to_csv(output / "controls.csv", index=False)
    cache, fits = {}, []
    for row in data.loc[data.fit_status.eq("pending")].itertuples():
        if row.n_fit_bins not in cache:
            cache[row.n_fit_bins] = template_library(model, row.fit_duration_s)
        logq, templates = cache[row.n_fit_bins]
        edges = row.start_s + np.arange(row.n_fit_bins + 1) * PARAMETERS["fit_bin_s"]
        counts = bin_counts(trains, edges)
        if counts.sum() != row.n_fit_spikes:
            raise ValueError("inventory spike mismatch")
        result = diagnose(counts, logq, templates, random_state(animal, session, row.event_id, 0))
        fits.append(dict(animal=animal, session=session, event_id=row.event_id, fit_status="scored", **result))
    fitted = pd.DataFrame(fits)
    if fitted.empty:
        fitted = pd.DataFrame(columns=["animal", "session", "event_id", "fit_status"])
    fitted.to_csv(output / "fits.csv", index=False)
    return animal, session, len(fitted)


def summarize(frame):
    rows = []
    for (animal, session), group in frame.groupby(["animal", "session"]):
        fit = group.loc[group.fit_status.eq("scored")]
        rows.append(
            {
                "animal": animal,
                "session": session,
                "extent_eligible": bool(group.extent_eligible_session.iloc[0]),
                "native_events": len(group),
                "scored_events": len(fit),
                "bank_range_events": int(group.bank_range_descriptor.fillna(False).sum()),
                "median_native_duration_s": group.native_duration_s.median(),
                "median_encoding_spikes": group.n_fit_spikes.median(),
                "median_active_encoding_units": group.n_active_encoding_units.median(),
                "deviance_flags": int(fit.deviance_flag.sum()) if len(fit) else 0,
                "template_predictive_wins": int(fit.crossfit_template_minus_free.gt(0).sum()) if len(fit) else 0,
                "moving_predictive_wins": int(fit.crossfit_moving_minus_static.gt(0).sum()) if len(fit) else 0,
                "median_deviance_tail": fit.deviance_tail_probability.median() if len(fit) else np.nan,
                "median_template_minus_free_per_spike": fit.crossfit_template_minus_free_per_spike.median() if len(fit) else np.nan,
                "median_moving_minus_static": fit.crossfit_moving_minus_static.median() if len(fit) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(args):
    start = time.monotonic()
    source = pd.read_csv(args.cohort_dir / "kleinman_extent_cohort_sessions.csv")
    cohort_manifest = json.loads((args.cohort_dir / "manifest.json").read_text())
    if file_sha256(args.cohort_dir / "kleinman_extent_cohort_sessions.csv") != cohort_manifest["outputs"]["kleinman_extent_cohort_sessions.csv"]:
        raise ValueError("changed frozen session eligibility")
    source = source.loc[source.run_pass].sort_values(["animal", "session"])
    if len(source) != 127 or source.extent_eligible.sum() != 14:
        raise ValueError("frozen cohort mismatch")
    inputs = {
        "cohort_manifest": args.cohort_dir / "manifest.json",
        "cohort_sessions": args.cohort_dir / "kleinman_extent_cohort_sessions.csv",
        "producer": Path(__file__),
        "protocol": ROOT / "docs/kleinman_native_adequacy_protocol.md",
        "fitter": ROOT / "scripts/calibrate_kleinman_integrated_extent.py",
        "encoder": ROOT / "scripts/validate_kleinman_run_decoder.py",
    }
    for row in source.itertuples():
        prefix = row.animal + "/" + row.session
        folder = args.dataset_root / "Experiment_1" / row.animal / row.session
        for name in ["session_info.mat", "spike_data.mat", "sdes.mat"]:
            inputs[prefix + "/" + name] = folder / name
        model_path = args.cohort_dir / "sessions" / row.animal / row.session / "known_map.npz"
        if file_sha256(model_path) != cohort_manifest["outputs"][str(model_path.relative_to(args.cohort_dir))]:
            raise ValueError("changed frozen map")
        inputs[prefix + "/known_map.npz"] = model_path
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory, jobs = [], []
    for row in source.itertuples():
        folder = args.dataset_root / "Experiment_1" / row.animal / row.session
        model_path = inputs[row.animal + "/" + row.session + "/known_map.npz"]
        frame, _ = inventory_session(folder, dict(np.load(model_path)), row)
        inventory.append(frame)
        if row.extent_eligible:
            jobs.append((str(model_path), str(folder), frame.to_dict("records"), str(args.output_dir / "sessions" / row.animal / row.session)))
    inventory = pd.concat(inventory, ignore_index=True)
    inventory.to_csv(args.output_dir / "kleinman_native_inventory.csv", index=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for animal, session, n in pool.map(fit_session, jobs):
            print(json.dumps({"animal": animal, "session": session, "n_fits": n, "elapsed_s": time.monotonic() - start}), flush=True)
    fits = pd.concat([pd.read_csv(Path(j[3]) / "fits.csv") for j in jobs], ignore_index=True)
    controls_frame = pd.concat([pd.read_csv(Path(j[3]) / "controls.csv") for j in jobs], ignore_index=True)
    merged = inventory.merge(fits, on=["animal", "session", "event_id"], how="left", validate="one_to_one", suffixes=("", "_result"))
    pending = merged.fit_status.eq("pending")
    if not merged.loc[pending, "fit_status_result"].eq("scored").all():
        raise ValueError("missing fitted native events")
    merged.loc[pending, "fit_status"] = "scored"
    merged = merged.drop(columns="fit_status_result")
    summary = summarize(merged)
    cal_summary = (
        controls_frame.groupby(["animal", "condition"])
        .agg(
            n_events=("control_id", "size"),
            flags=("deviance_flag", "sum"),
            alarm_fraction=("deviance_flag", "mean"),
            median_predictive_delta=("crossfit_template_minus_free_per_spike", "median"),
        )
        .reset_index()
    )
    negative = cal_summary.loc[cal_summary.condition.eq("matched")]
    gates = pd.DataFrame(
        [
            {"gate": "all_127_session_identities", "passed": merged.groupby(["animal", "session"]).ngroups == 127},
            {"gate": "all_14_frozen_sessions_attempted", "passed": len(jobs) == 14 and summary.extent_eligible.sum() == 14},
            {"gate": "controls_complete", "passed": len(controls_frame) == 1120 and controls_frame.groupby(["animal", "session", "condition"]).size().eq(40).all()},
            {"gate": "native_ids_unique", "passed": not merged.duplicated(["animal", "session", "event_id"]).any()},
            {"gate": "all_pending_events_scored", "passed": len(fits) > 0 and len(fits) == int(pending.sum())},
            {"gate": "negative_control_alarm_not_above_10pct_per_animal", "passed": len(negative) == 4 and negative.alarm_fraction.le(0.1).all()},
            {
                "gate": "finite_native_fit_metrics",
                "passed": np.isfinite(fits[["conditional_deviance", "deviance_tail_probability", "crossfit_template_minus_free_per_spike", "crossfit_moving_minus_static"]])
                .all()
                .all(),
            },
        ]
    )
    for name, frame in [("event_adequacy", merged), ("session_summary", summary), ("controls", controls_frame), ("control_summary", cal_summary), ("gates", gates)]:
        frame.to_csv(args.output_dir / ("kleinman_native_" + name + ".csv"), index=False)
    for key, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][key]:
            raise ValueError("input changed during run " + key)
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "runtime_s": time.monotonic() - start,
        "parameters": PARAMETERS,
        "n_native_events_inventoried": len(inventory),
        "n_native_events_fitted": len(fits),
        "real_candidate_model_adequacy_scored": True,
        "reward_contrast_scored": False,
        "new_replay_biological_claim": False,
        "negative_calibration_passed": bool(negative.alarm_fraction.le(0.1).all()),
        "outputs": {str(p.relative_to(args.output_dir)): file_sha256(p) for p in args.output_dir.rglob("*") if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())
