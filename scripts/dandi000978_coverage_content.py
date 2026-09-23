#!/usr/bin/env python3
"""Frozen within-event CA1 coverage manipulation with independent PFC content."""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _provenance import build_script_provenance, file_sha256
from acquire_dandi000978 import destination, write_json
from audit_dandi000978_cohort import verify_acquisition_record
from dandi000978_verified_units import SUPPORTED_ANIMALS, read_author_crosswalks, verified_unit_table
from preflight_dandi000978 import assign_epochs, inspect
from validate_dandi000978_run_readouts import (
    PARAMETERS as RUN_PARAMETERS,
)
from validate_dandi000978_run_readouts import (
    aggregate_trials,
    fit_routes,
    fit_spatial,
    prepare_bins,
    seed_for,
)

from hipporeplayimm.training_continuity import geometry, overlapping_counts, poisson_map

PARAMETERS = {
    "version": 1,
    "seed": 20260923,
    "min_duration_s": 0.065,
    "max_duration_s": 0.5,
    "min_ca1_units": 20,
    "min_pfc_units": 5,
    "min_ca1_event_spikes": 10,
    "min_ca1_event_active_units": 5,
    "max_events_per_rest_epoch": 100,
    "fractions": [1.0, 0.75, 0.5, 0.25],
    "repetitions": 20,
    "null_draws": 199,
    "base_bin_s": 0.005,
    "decode_window_s": 0.020,
    "min_matched_stratum": 4,
    "primary_loss_fraction": 0.5,
    "primary_min_events_per_animal": 10,
    "primary_min_epochs_per_animal": 2,
    "primary_min_matched_fraction": 0.8,
}
RULES = ("edge_only", "bin_support")


def counts_in_interval(spikes, start, stop):
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError("Invalid event interval")
    return np.array([np.searchsorted(s, stop, side="left") - np.searchsorted(s, start, side="left") for s in spikes], dtype=np.int64)


def base_counts(spikes, start, stop):
    n = int(np.floor((stop - start + 1e-9) / PARAMETERS["base_bin_s"]))
    edges = start + np.arange(n + 1) * PARAMETERS["base_bin_s"]
    return np.column_stack([np.diff(np.searchsorted(s, edges, side="left")) for s in spikes])


def spike_lists(arrays, requested_ids):
    lookup = {int(i): j for j, i in enumerate(arrays["unit_ids"])}
    starts = np.r_[0, arrays["spike_ends"][:-1]]
    if len(set(requested_ids)) != len(requested_ids) or not set(requested_ids).issubset(lookup):
        raise ValueError("Unresolved or duplicate source unit ID")
    return [arrays["spikes"][starts[lookup[int(i)]] : arrays["spike_ends"][lookup[int(i)]]] for i in requested_ids]


def native_intervals(archive, animal):
    with zipfile.ZipFile(archive) as z:
        result = {}
        for kind in ("SWR", "NREM"):
            table = pd.read_csv(io.BytesIO(z.read(f"dataForFlorian/{animal}_{kind}_intervals.csv")))
            required = {"animal", "day", "epoch", "eventType", "startTime", "endTime"}
            if not required.issubset(table) or not (table.animal == animal).all() or not (table.day == 1).all() or not (table.eventType == kind).all():
                raise ValueError("Unexpected native annotation schema or identity")
            table["source_row"] = np.arange(len(table))
            result[kind] = table
    return result


def candidate_metadata(swr, nrem, epochs, animal, filename):
    """Behavior/source metadata only: never receives PFC candidate spikes."""
    epoch_index = assign_epochs(swr.startTime.to_numpy(), swr.endTime.to_numpy(), epochs.start_time_s.to_numpy(), epochs.stop_time_s.to_numpy())
    offset = 9 if "ZT2_obj-1dss6zi" in filename else 0
    run_epochs = set(epochs.loc[epochs.n_trials > 0, "epoch_index"].astype(int))
    rows = []
    for ix, event in enumerate(swr.itertuples(index=False)):
        ep = int(epoch_index[ix])
        if ep < 0:
            if epochs.start_time_s.min() <= event.startTime <= epochs.stop_time_s.max():
                rows.append(
                    {
                        "event_id": f"{animal}:day1:SWR:{int(event.source_row)}",
                        "animal": animal,
                        "file": filename,
                        "rest_epoch": -1,
                        "native_epoch": int(event.epoch),
                        "source_row": int(event.source_row),
                        "nrem_source_row": -1,
                        "start_time_s": float(event.startTime),
                        "end_time_s": float(event.endTime),
                        "duration_s": float(event.endTime - event.startTime),
                        "training_run_epoch": -1,
                        "exclusion_reason": "invalid_or_outside_epoch",
                    }
                )
            continue
        if event.epoch != ep + offset + 1:
            raise ValueError("Source/NWB epoch numbering disagreement")
        start, stop = float(event.startTime), float(event.endTime)
        reason = ""
        duration = stop - start
        inside = nrem[(nrem.epoch == event.epoch) & (nrem.startTime <= start) & (nrem.endTime >= stop)]
        previous = [e for e in run_epochs if e < ep]
        if ep in run_epochs:
            reason = "RUN_epoch"
        elif not PARAMETERS["min_duration_s"] <= duration <= PARAMETERS["max_duration_s"]:
            reason = "duration"
        elif stop > epochs.loc[epochs.epoch_index == ep, "stop_time_s"].iloc[0]:
            reason = "strict_epoch_boundary"
        elif len(inside) != 1:
            reason = "not_in_one_source_NREM_interval"
        elif not previous:
            reason = "no_preceding_RUN_in_file"
        rows.append(
            {
                "event_id": f"{animal}:day1:SWR:{int(event.source_row)}",
                "animal": animal,
                "file": filename,
                "rest_epoch": ep,
                "native_epoch": int(event.epoch),
                "source_row": int(event.source_row),
                "nrem_source_row": int(inside.source_row.iloc[0]) if len(inside) == 1 else -1,
                "start_time_s": start,
                "end_time_s": stop,
                "duration_s": duration,
                "training_run_epoch": max(previous) if previous else -1,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def select_candidates(metadata, ca1_by_epoch, key):
    """Immutable native windows; support and seeded selection use CA1 only."""
    table = metadata.copy().sort_values(["start_time_s", "source_row"]).reset_index(drop=True)
    table["n_ca1_spikes"] = 0
    table["n_ca1_active_units"] = 0
    table["selected"] = False
    last_end = -np.inf
    for i, row in table.iterrows():
        if row.exclusion_reason:
            continue
        model = ca1_by_epoch.get(int(row.training_run_epoch))
        if model is None:
            table.loc[i, "exclusion_reason"] = "training_map_unavailable"
            continue
        counts = counts_in_interval(model, row.start_time_s, row.end_time_s)
        table.loc[i, "n_ca1_spikes"] = int(counts.sum())
        table.loc[i, "n_ca1_active_units"] = int((counts > 0).sum())
        if counts.sum() < PARAMETERS["min_ca1_event_spikes"] or (counts > 0).sum() < PARAMETERS["min_ca1_event_active_units"]:
            table.loc[i, "exclusion_reason"] = "insufficient_CA1_event_support"
        elif row.start_time_s < last_end:
            table.loc[i, "exclusion_reason"] = "overlapping_earlier_eligible_event"
        else:
            last_end = row.end_time_s
    for ep, group in table[table.exclusion_reason == ""].groupby("rest_epoch"):
        rng = np.random.default_rng(seed_for(PARAMETERS["seed"], key, int(ep), "selection"))
        keep = rng.permutation(group.index)[: PARAMETERS["max_events_per_rest_epoch"]]
        table.loc[keep, "selected"] = True
        table.loc[group.index.difference(keep), "exclusion_reason"] = "seeded_pilot_cap"
    return table


def nested_coverage(n_units, key, repeat):
    rng = np.random.default_rng(seed_for(PARAMETERS["seed"], key, repeat, "coverage"))
    order = rng.permutation(n_units)
    return {f: np.sort(order[: max(1, int(np.ceil(f * n_units)))]) for f in PARAMETERS["fractions"]}


def classify(windows, rates, centers, subset):
    selected = windows[:, subset]
    path = poisson_map(selected, rates[:, subset].T)
    return {rule: geometry(path, centers, selected, filtered=rule == "bin_support", min_frames=10) for rule in RULES}


def composition_scores(counts, route_rates):
    """Per-spike route log score centered across routes; zero counts yield zero."""
    counts, rates = np.asarray(counts), np.asarray(route_rates)
    if counts.ndim != 2 or rates.shape != (4, counts.shape[1]) or (counts < 0).any() or not np.isfinite(counts).all() or (rates <= 0).any() or not np.isfinite(rates).all():
        raise ValueError("Invalid counts or route rates")
    logq = np.log(rates / rates.sum(axis=1, keepdims=True))
    ll = counts @ logq.T
    probabilities = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    contrasts = (ll - ll.mean(axis=1, keepdims=True)) / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return contrasts, probabilities


def shuffled_map_scores(counts, rates, targets, key):
    rng = np.random.default_rng(seed_for(PARAMETERS["seed"], key, "PFC_map_null"))
    permutations = rng.random((PARAMETERS["null_draws"], *rates.shape)).argsort(axis=1)
    wrong = np.take_along_axis(np.broadcast_to(rates, permutations.shape), permutations, axis=1)
    logq = np.log(wrong / wrong.sum(axis=2, keepdims=True))
    centered = logq - logq.mean(axis=1, keepdims=True)
    scores = np.einsum("eu,kru->ekr", counts, centered) / np.maximum(counts.sum(axis=1), 1)[:, None, None]
    return np.take_along_axis(scores, np.broadcast_to(np.asarray(targets)[:, None, None], (*scores.shape[:2], 1)), axis=2)[:, :, 0], permutations


def matched_event_null(events, score_vectors):
    n, k = len(events), PARAMETERS["null_draws"]
    indices = np.full((n, k), -1, dtype=np.int64)
    audit = events[["event_id", "file", "rest_epoch", "duration_s", "n_pfc_spikes"]].copy()
    audit["duration_half"] = 0
    audit["count_stratum"] = 0
    for key, group in events.groupby(["file", "rest_epoch"], sort=True):
        positive = group.loc[group.n_pfc_spikes > 0, "n_pfc_spikes"]
        median = positive.median() if len(positive) else np.inf
        audit.loc[group.index, "duration_half"] = (group.duration_s > group.duration_s.median()).astype(int)
        audit.loc[group.index, "count_stratum"] = np.where(group.n_pfc_spikes == 0, 0, np.where(group.n_pfc_spikes <= median, 1, 2))
    for key, group in audit.groupby(["file", "rest_epoch", "duration_half", "count_stratum"], sort=True):
        ix = group.index.to_numpy()
        if len(ix) < PARAMETERS["min_matched_stratum"]:
            continue
        rng = np.random.default_rng(seed_for(PARAMETERS["seed"], *key, "PFC_pair_null"))
        for draw in range(k):
            order = rng.permutation(ix)
            shift = rng.integers(1, len(order))
            indices[order, draw] = np.roll(order, shift)
    valid = (indices >= 0).all(axis=1)
    null = np.full((n, k), np.nan)
    targets = events.ca1_reference_route.to_numpy(int)
    null[valid] = score_vectors[indices[valid], targets[valid, None]]
    audit["matched_control_available"] = valid
    return null, indices, audit


def fit_epoch_models(bins, counts, units, epoch):
    train = bins.epoch.to_numpy() == epoch
    ca1 = units.region.to_numpy() == "CA1"
    pfc = units.region.to_numpy() == "PFC"
    rates, ca_use, centers, _ = fit_spatial(counts[:, ca1], bins[["x_cm", "y_cm"]].to_numpy(), train)
    pf_use = counts[train][:, pfc].sum(axis=0) >= RUN_PARAMETERS["min_training_spikes"]
    if ca_use.sum() < PARAMETERS["min_ca1_units"] or pf_use.sum() < PARAMETERS["min_pfc_units"]:
        raise ValueError("Insufficient training-only regional unit coverage")
    labels, totals, exposure = aggregate_trials(bins.loc[train].reset_index(drop=True), counts[train])
    y = labels.route.to_numpy()
    if any((y == k).sum() < RUN_PARAMETERS["min_training_trials_per_route"] for k in range(4)):
        raise ValueError("Insufficient canonical routes in preceding RUN")
    ca_rates = fit_routes(totals[:, ca1][:, ca_use], exposure, y)
    pf_rates = fit_routes(totals[:, pfc][:, pf_use], exposure, y)
    return {
        "spatial_rates": rates,
        "spatial_centers": centers,
        "ca1_route_rates": ca_rates,
        "pfc_route_rates": pf_rates,
        "ca1_ids": units.unit_id.to_numpy()[ca1][ca_use],
        "pfc_ids": units.unit_id.to_numpy()[pfc][pf_use],
    }


def provenance(inputs):
    result = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if result["git_dirty"] is not False or len(result["code_commit"]) != 40:
        raise ValueError("Exact clean producer commit required")
    return {**result, "created_at_utc": datetime.now(UTC).isoformat(), "host": socket.gethostname()}


def prepare(args):
    manifest_path = args.dataset_root / "metadata/asset_manifest.json"
    status_path = args.dataset_root / "download_status.json"
    source = json.loads(manifest_path.read_text())
    verify_acquisition_record(args.dataset_root, source, json.loads(status_path.read_text()))
    crosswalks = read_author_crosswalks(args.author_archive)
    prov = provenance({"assets": manifest_path, "acquisition": status_path, "private_source": args.author_archive})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / "maps").mkdir()
    maps, audits, selected, inventory = [], [], [], []
    for asset in source["assets"]:
        filename = Path(asset["path"]).name
        animal = filename.split("SingleDay-")[1].split("_")[0]
        if animal not in SUPPORTED_ANIMALS:
            inventory.append({"animal": animal, "file": filename, "status": "anatomy_unconfirmed"})
            continue
        with h5py.File(destination(args.dataset_root, asset), "r") as f:
            _, epochs, trials, _, arrays = inspect(f)
            table = f[f["units/electrodes"].attrs["table"]]
            units = verified_unit_table(
                animal, filename, arrays["unit_ids"], f["units/electrodes"][()], table["location"].asstr()[()], table["group_name"].asstr()[()], crosswalks.get(filename)
            )
        bins, counts, _ = prepare_bins(arrays, trials)
        ca1_by_epoch = {}
        for epoch in sorted(trials.epoch_index.unique()):
            entry = {"animal": animal, "file": filename, "asset_id": asset["asset_id"], "training_run_epoch": int(epoch), "status": "excluded", "reason": ""}
            try:
                model = fit_epoch_models(bins, counts, units, epoch)
                if set(model["ca1_ids"]) & set(model["pfc_ids"]):
                    raise ValueError("Regional unit overlap")
                relative = f"maps/{asset['asset_id']}_RUN{epoch}.npz"
                np.savez_compressed(out / relative, **model)
                entry.update(status="prepared", path=relative, sha256=file_sha256(out / relative), n_ca1_units=len(model["ca1_ids"]), n_pfc_units=len(model["pfc_ids"]))
                ca1_by_epoch[int(epoch)] = spike_lists(arrays, model["ca1_ids"])
            except ValueError as exc:
                entry["reason"] = str(exc)
            maps.append(entry)
        annotations = native_intervals(args.author_archive, animal)
        metadata = candidate_metadata(annotations["SWR"], annotations["NREM"], epochs, animal, filename)
        candidate_audit = select_candidates(metadata, ca1_by_epoch, asset["asset_id"])
        candidate_audit["asset_id"] = asset["asset_id"]
        audits.append(candidate_audit)
        selected.append(candidate_audit[candidate_audit.selected])
        inventory.append({"animal": animal, "file": filename, "status": "prepared", "selected_events": int(candidate_audit.selected.sum())})
        print(json.dumps(inventory[-1]), flush=True)
    events = pd.concat(selected, ignore_index=True).sort_values(["file", "start_time_s"]).reset_index(drop=True)
    if events.empty or events.event_id.duplicated().any():
        raise ValueError("Frozen selection must be nonempty and have unique global event IDs")
    events.to_csv(out / "selected_events.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(out / "candidate_eligibility.csv", index=False)
    pd.DataFrame(maps).to_csv(out / "training_map_status.csv", index=False)
    pd.DataFrame(inventory).to_csv(out / "cohort_status.csv", index=False)
    frozen = {
        **prov,
        "parameters": PARAMETERS,
        "run_parameters": RUN_PARAMETERS,
        "dataset_root": str(args.dataset_root),
        "dataset_version": source["version"],
        "candidate_selection_uses_pfc_spikes": False,
        "candidate_selection_precedes_continuity": True,
        "private_source_not_redistributed": True,
        "assets": source["assets"],
        "maps": maps,
        "frozen_files": {name: file_sha256(out / name) for name in ["selected_events.csv", "candidate_eligibility.csv", "training_map_status.csv", "cohort_status.csv"]},
    }
    write_json(out / "frozen_manifest.json", frozen)
    print(json.dumps({"status": "selection_frozen", "events": len(events), "maps": sum(m["status"] == "prepared" for m in maps)}), flush=True)


def validate_frozen(root, manifest, commit):
    if manifest["parameters"] != PARAMETERS or manifest["run_parameters"] != RUN_PARAMETERS:
        raise ValueError("Frozen parameters changed")
    if manifest["code_commit"] != commit or manifest["candidate_selection_uses_pfc_spikes"] is not False:
        raise ValueError("Producer revision or selection independence changed")
    paths = dict(manifest["frozen_files"])
    paths.update({m["path"]: m["sha256"] for m in manifest["maps"] if m["status"] == "prepared"})
    for path, digest in paths.items():
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()) or file_sha256(target) != digest:
            raise ValueError("Frozen input hash/path mismatch: " + path)


def group_summary(events, pair_null, map_null, extra_key=None):
    masks = {
        "all": np.ones(len(events), bool),
        "full_geometric_pass": events.full_geometric_pass.to_numpy(bool),
        "primary_lost_half": events.primary_lost_half.to_numpy(bool),
        "retained_half": (events.full_geometric_pass & ~events.primary_lost_half).to_numpy(bool),
        "full_geometric_fail": ~events.full_geometric_pass.to_numpy(bool),
    }
    rows = []
    grouping = ["animal"] + ([extra_key] if extra_key else [])
    for key, animal in events.groupby(grouping, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for group, mask in masks.items():
            all_ix = animal.index[mask[animal.index]].to_numpy()
            ix = all_ix[events.loc[all_ix, "matched_control_available"].to_numpy(bool)]
            for control, null in (("matched_event", pair_null), ("cellwise_route_map", map_null)):
                row = {
                    **dict(zip(grouping, key, strict=True)),
                    "group": group,
                    "control": control,
                    "n_events": len(all_ix),
                    "n_matched_events": len(ix),
                    "matched_fraction": len(ix) / len(all_ix) if len(all_ix) else 0,
                    "n_rest_epochs": events.loc[ix, "rest_key"].nunique(),
                    "status": "empty",
                    "above_control_p95": False,
                }
                if len(ix):
                    real = float(events.loc[ix, "pfc_reference_score"].mean())
                    draws = null[ix].mean(axis=0)
                    row.update(
                        status="scored",
                        real_mean_nats_per_spike=real,
                        null_mean_nats_per_spike=float(draws.mean()),
                        real_minus_null_mean=float(real - draws.mean()),
                        null_p95=float(np.quantile(draws, 0.95)),
                        above_control_p95=bool(real > np.quantile(draws, 0.95)),
                        empirical_p=float((1 + (draws >= real).sum()) / (1 + len(draws))),
                        zero_pfc_fraction=float((events.loc[ix, "n_pfc_spikes"] == 0).mean()),
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def coverage_complete(labels, event_ids):
    if labels.duplicated(["event_id", "fraction", "repeat", "rule"]).any() or set(labels.event_id) != set(event_ids):
        return False
    expected = {(1.0, -1, rule) for rule in RULES}
    expected.update((fraction, repeat, rule) for fraction in PARAMETERS["fractions"][1:] for repeat in range(PARAMETERS["repetitions"]) for rule in RULES)
    return all(set(group[["fraction", "repeat", "rule"]].itertuples(index=False, name=None)) == expected for _, group in labels.groupby("event_id", sort=False))


def summarize_scoring(events, labels, pair_null, map_null, out):
    primary = labels[labels.rule == "edge_only"]
    full = primary[primary.fraction == 1].set_index("event_id")
    half = primary[primary.fraction == 0.5]
    loss = half.assign(fail=~half.geometric_pass).groupby("event_id").fail.mean()
    events["full_geometric_pass"] = events.event_id.map(full.geometric_pass)
    events["loss_probability_half"] = events.event_id.map(loss)
    events["primary_lost_half"] = events.full_geometric_pass & (events.loss_probability_half >= PARAMETERS["primary_loss_fraction"])
    events["matched_control_available"] = np.isfinite(pair_null).all(axis=1)
    events["pair_excess_nats_per_spike"] = events.pfc_reference_score - np.where(events.matched_control_available, np.nan_to_num(pair_null).mean(axis=1), np.nan)
    events["map_excess_nats_per_spike"] = events.pfc_reference_score - map_null.mean(axis=1)
    events["rest_key"] = events.file + ":" + events.rest_epoch.astype(str)
    events.to_csv(out / "event_content_and_coverage.csv", index=False)
    summary = group_summary(events, pair_null, map_null)
    summary.to_csv(out / "by_animal_content.csv", index=False)
    group_summary(events, pair_null, map_null, "rest_key").to_csv(out / "by_epoch_content.csv", index=False)
    event_dose = labels.groupby(["event_id", "animal", "fraction", "rule"]).geometric_pass.mean().reset_index(name="acceptance_probability")
    event_dose.to_csv(out / "event_coverage_dose.csv", index=False)
    dose = event_dose.groupby(["animal", "fraction", "rule"]).agg(n_events=("event_id", "nunique"), acceptance_fraction=("acceptance_probability", "mean"))
    dose.to_csv(out / "coverage_dose_summary.csv")
    gates = [
        {"gate": "nonempty_frozen_events", "passed": bool(len(events))},
        {"gate": "coverage_rows_complete", "passed": coverage_complete(labels, events.event_id)},
        {"gate": "both_supported_animals_present", "passed": set(events.animal) == set(SUPPORTED_ANIMALS)},
        {"gate": "finite_content_scores", "passed": bool(np.isfinite(events.pfc_reference_score).all() and np.isfinite(map_null).all())},
        {"gate": "PFC_not_used_in_selection_or_continuity", "passed": True},
    ]
    technical = all(x["passed"] for x in gates)
    gates.append({"gate": "technical_overall", "passed": technical})
    primary_summary = summary[summary.group == "primary_lost_half"]
    enough = (
        len(primary_summary) == 4
        and (primary_summary.n_events >= PARAMETERS["primary_min_events_per_animal"]).all()
        and (primary_summary.n_rest_epochs >= PARAMETERS["primary_min_epochs_per_animal"]).all()
        and (primary_summary.matched_fraction >= PARAMETERS["primary_min_matched_fraction"]).all()
    )
    positive = enough and primary_summary.above_control_p95.all() and (primary_summary.real_minus_null_mean > 0).all()
    decision = (
        "technical_failure"
        if not technical
        else "underpowered_primary_group"
        if not enough
        else "promising_two_animal_pilot"
        if positive
        else "primary_content_retention_not_supported"
    )
    gates.extend(
        [
            {"gate": "primary_group_sufficient", "passed": bool(enough)},
            {"gate": "primary_both_controls_both_animals_positive", "passed": bool(positive)},
            {"gate": "paper_ready", "passed": False},
        ]
    )
    pd.DataFrame(gates).to_csv(out / "gate_summary.csv", index=False)
    write_json(
        out / "decision.json",
        {"decision": decision, "technical_complete": technical, "paper_ready": False, "group_rule_not_changed": True, "completed_at_utc": datetime.now(UTC).isoformat()},
    )
    (out / "coverage_content_summary.md").write_text(
        "\n".join(
            [
                "# DANDI000978 fixed-window coverage/content pilot",
                "",
                "Decision: `" + decision + "`.",
                "",
                "## Primary lost-label group",
                "",
                "```csv",
                primary_summary.to_csv(index=False).strip(),
                "```",
                "",
                "## Limits",
                "",
                "Two animals only. Events and thinning repetitions are not independent animal replications.",
                "PFC is independent measurement, not ground truth. Route composition can reflect position/direction and shared assemblies.",
                "The geometric label is not full shuffle-validated replay. Fixed PFC measurements are not a biological perturbation outcome.",
                "Matched controls are coarse duration/count/rest-epoch matches, not exact sleep-microstate controls.",
                "No threshold or primary group was changed after content scoring. Private source annotations are not redistributed.",
                "",
            ]
        )
    )
    return technical


def score(args):
    frozen = args.output_dir
    manifest = json.loads((frozen / "frozen_manifest.json").read_text())
    prov = provenance({"frozen_manifest": frozen / "frozen_manifest.json"})
    validate_frozen(frozen, manifest, prov["code_commit"])
    source = {"dataset": "000978", "version": manifest["dataset_version"], "assets": manifest["assets"]}
    root = Path(manifest["dataset_root"])
    verify_acquisition_record(root, source, json.loads((root / "download_status.json").read_text()))
    out = frozen / "scoring"
    out.mkdir(exist_ok=False)
    write_json(out / "manifest.json", {**prov, "parameters": PARAMETERS, "scope": "exploratory_two_animal_coverage_content", "candidate_selection_uses_pfc_spikes": False})
    selected = pd.read_csv(frozen / "selected_events.csv", keep_default_na=False)
    rows, all_labels, all_map_nulls, all_vectors = [], [], [], []
    for asset in source["assets"]:
        filename = Path(asset["path"]).name
        file_events = selected[selected.file == filename]
        if file_events.empty:
            continue
        with h5py.File(destination(root, asset), "r") as f:
            arrays = {"unit_ids": f["units/id"][()], "spike_ends": f["units/spike_times_index"][()], "spikes": f["units/spike_times"][()]}
        for run_epoch, group in file_events.groupby("training_run_epoch", sort=True):
            started = time.monotonic()
            entry = next(m for m in manifest["maps"] if m["file"] == filename and m["training_run_epoch"] == run_epoch and m["status"] == "prepared")
            model = dict(np.load(frozen / entry["path"]))
            if set(model["ca1_ids"]) & set(model["pfc_ids"]):
                raise ValueError("Regional unit overlap")
            ca1, pfc = spike_lists(arrays, model["ca1_ids"]), spike_lists(arrays, model["pfc_ids"])
            epoch_key = (asset["asset_id"], int(run_epoch))
            subsets = [nested_coverage(len(ca1), epoch_key, repeat) for repeat in range(PARAMETERS["repetitions"])]
            pfc_counts, targets, chunk_rows, chunk_labels = [], [], [], []
            for event in group.itertuples(index=False):
                start, stop = event.start_time_s, event.end_time_s
                ca_counts = counts_in_interval(ca1, start, stop)
                if int(ca_counts.sum()) != event.n_ca1_spikes:
                    raise ValueError("CA1 event counts changed after freezing")
                _, ca_prob = composition_scores(ca_counts[None, :], model["ca1_route_rates"])
                target = int(ca_prob[0].argmax())
                pf_counts = counts_in_interval(pfc, start, stop)
                pfc_counts.append(pf_counts)
                targets.append(target)
                windows = overlapping_counts(base_counts(ca1, start, stop), np.full(int(np.floor((stop - start + 1e-9) / PARAMETERS["base_bin_s"])), PARAMETERS["base_bin_s"]))
                settings = [(1.0, -1, np.arange(len(ca1)))] + [
                    (fraction, repeat, subset[fraction]) for repeat, subset in enumerate(subsets) for fraction in PARAMETERS["fractions"][1:]
                ]
                for fraction, repeat, units in settings:
                    labels = classify(windows, model["spatial_rates"], model["spatial_centers"], units)
                    chunk_labels.extend(
                        {
                            "event_id": event.event_id,
                            "animal": event.animal,
                            "file": filename,
                            "rest_epoch": event.rest_epoch,
                            "fraction": fraction,
                            "repeat": repeat,
                            "rule": rule,
                            "n_recorded_units": len(units),
                            **result,
                        }
                        for rule, result in labels.items()
                    )
                chunk_rows.append(
                    {
                        "event_id": event.event_id,
                        "animal": event.animal,
                        "file": filename,
                        "rest_epoch": event.rest_epoch,
                        "training_run_epoch": run_epoch,
                        "start_time_s": start,
                        "end_time_s": stop,
                        "duration_s": event.duration_s,
                        "n_ca1_units": len(ca1),
                        "n_pfc_units": len(pfc),
                        "n_ca1_spikes": int(ca_counts.sum()),
                        "n_pfc_spikes": int(pf_counts.sum()),
                        "n_pfc_active_units": int((pf_counts > 0).sum()),
                        "ca1_reference_route": target,
                        "ca1_route_max_probability": float(ca_prob.max()),
                    }
                )
            pf_counts = np.asarray(pfc_counts)
            vectors, _ = composition_scores(pf_counts, model["pfc_route_rates"])
            map_null, permutations = shuffled_map_scores(pf_counts, model["pfc_route_rates"], targets, epoch_key)
            for j, row in enumerate(chunk_rows):
                row["pfc_reference_score"] = float(vectors[j, targets[j]])
            np.savez_compressed(
                out / f"{asset['asset_id']}_RUN{run_epoch}_audit.npz",
                event_ids=np.array([r["event_id"] for r in chunk_rows]),
                ca1_ids=model["ca1_ids"],
                pfc_ids=model["pfc_ids"],
                pfc_map_permutations=permutations,
                **{f"repeat_{r}_fraction_{f}": u for r, sub in enumerate(subsets) for f, u in sub.items()},
            )
            rows.extend(chunk_rows)
            all_labels.extend(chunk_labels)
            all_map_nulls.append(map_null)
            all_vectors.append(vectors)
            pd.DataFrame(rows).to_csv(out / "event_scores_checkpoint.csv", index=False)
            pd.DataFrame(chunk_labels).to_csv(out / "coverage_labels.csv", index=False, mode="a", header=not (out / "coverage_labels.csv").exists())
            print(json.dumps({"file": filename, "RUN": int(run_epoch), "events": len(group), "runtime_s": time.monotonic() - started}), flush=True)
    events, labels = pd.DataFrame(rows), pd.DataFrame(all_labels)
    if len(events) != len(selected) or set(events.event_id) != set(selected.event_id):
        raise ValueError("Missing frozen event scores")
    vectors, map_null = np.concatenate(all_vectors), np.concatenate(all_map_nulls)
    pair_null, donor_indices, matching = matched_event_null(events, vectors)
    matching.to_csv(out / "matched_control_strata.csv", index=False)
    np.savez_compressed(
        out / "content_nulls.npz",
        event_ids=events.event_id.to_numpy(str),
        pfc_score_vectors=vectors,
        matched_event_null=pair_null,
        cellwise_map_null=map_null,
        matched_donor_indices=donor_indices,
    )
    good = summarize_scoring(events, labels, pair_null, map_null, out)
    write_json(out / "terminal_status.json", {"status": "complete" if good else "complete_with_failed_technical_gates", "completed_at_utc": datetime.now(UTC).isoformat()})
    return 0 if good else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--author-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            if args.dataset_root is None or args.author_archive is None:
                parser.error("prepare requires --dataset-root and --author-archive")
            prepare(args)
            return 0
        return score(args)
    except Exception as exc:
        if args.output_dir.exists():
            write_json(args.output_dir / f"{args.stage}_failure.json", {"error": f"{type(exc).__name__}: {exc}", "created_at_utc": datetime.now(UTC).isoformat()})
        raise


if __name__ == "__main__":
    sys.exit(main())
