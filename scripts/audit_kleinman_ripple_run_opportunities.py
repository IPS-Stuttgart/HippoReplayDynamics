#!/usr/bin/env python3
"""Chronological RUN/ripple/RUN opportunity audit; no outcome or drug-effect score."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from validate_kleinman_run_decoder import (
    PARAMETERS as RUN_PARAMETERS,
)
from validate_kleinman_run_decoder import (
    align_behavior,
    fit_maps,
    interval_counts,
    interval_data,
    make_traversals,
    matrix,
    split_units,
)

PARAMETERS = {
    "reference_traversals": 3,
    "visit_cap_s": 10.0,
    "immobile_max_cm_s": 8.0,
    "minimum_encoding_units_descriptor": 5,
    "readout_min_cm_s": 20.0,
    "readout_end_margin_cm": 20.0,
}


def merge_intervals(intervals):
    array = np.asarray(intervals, float).reshape(-1, 2)
    if not np.isfinite(array).all() or np.any(array[:, 1] <= array[:, 0]):
        raise ValueError("invalid_intervals")
    rows = []
    for lo, hi in array[np.argsort(array[:, 0], kind="stable")]:
        if rows and lo <= rows[-1][1]:
            rows[-1][1] = max(rows[-1][1], hi)
        else:
            rows.append([float(lo), float(hi)])
    return np.asarray(rows, float).reshape(-1, 2)


def intersect_intervals(first, second):
    a, b = merge_intervals(first), merge_intervals(second)
    rows, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            rows.append([lo, hi])
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(rows, float).reshape(-1, 2)


def duration(intervals):
    return float(np.diff(np.asarray(intervals).reshape(-1, 2), axis=1).sum())


def counts_in_intervals(trains, intervals):
    a = merge_intervals(intervals)
    return np.asarray([np.sum(np.searchsorted(s, a[:, 1], side="left") - np.searchsorted(s, a[:, 0], side="left")) for s in trains], dtype=int)


def opportunity_pairs(runs):
    rows = []
    for epoch in sorted({r["epoch"] for r in runs}):
        for direction in (0, 1):
            selected = sorted(
                (r for r in runs if r["epoch"] == epoch and r["direction"] == direction),
                key=lambda r: r["start_s"],
            )
            for i in range(1, len(selected)):
                baseline, target = selected[i - 1], selected[i]
                refs = selected[max(0, i - 1 - PARAMETERS["reference_traversals"]) : i - 1]
                if baseline["end_s"] >= target["start_s"]:
                    raise ValueError("overlapping_same_direction_runs")
                rows.append(
                    {
                        "opportunity_id": f"e{epoch}_d{direction}_t{target['traversal']}",
                        "epoch": epoch,
                        "direction": direction,
                        "baseline": baseline,
                        "target": target,
                        "reference": refs,
                        "status": "audited" if len(refs) == PARAMETERS["reference_traversals"] else "insufficient_prior_reference_runs",
                    }
                )
    return rows


def interval_run_mask(t, runs):
    keep = np.zeros(len(t) - 1, bool)
    for r in runs:
        keep |= (t[:-1] >= r["start_s"]) & (t[1:] <= r["end_s"])
    return keep


def session_audit(folder):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    keys, trains, excluded = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    ripples = matrix(
        loadmat(folder / "ripple_events.mat", simplify_cells=True)["ripple_events"],
        4,
        "ripple_events",
    )
    if np.any(ripples[:, 1] <= ripples[:, 0]):
        raise ValueError("nonpositive_native_ripple")
    if np.any((ripples[:, 2] < ripples[:, 0]) | (ripples[:, 2] > ripples[:, 1])):
        raise ValueError("peak_outside_native_ripple")
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    counts = interval_counts(trains, t)
    dt, valid, trainable, _, direction, spatial = interval_data(t, x, speed, runs, edges)
    n_bins = len(edges) - 1
    visit_intervals = [[v["start_s"], min(v["end_s"], v["start_s"] + PARAMETERS["visit_cap_s"])] for v in visits]
    immobile = valid & (np.maximum(speed[:-1], speed[1:]) <= PARAMETERS["immobile_max_cm_s"])
    allowed = intersect_intervals(np.column_stack((t[:-1][immobile], t[1:][immobile])), visit_intervals)
    readout = (
        trainable
        & ((speed[:-1] + speed[1:]) / 2 > PARAMETERS["readout_min_cm_s"])
        & ((x[:-1] + x[1:]) / 2 > ends[0] + PARAMETERS["readout_end_margin_cm"])
        & ((x[:-1] + x[1:]) / 2 < ends[1] - PARAMETERS["readout_end_margin_cm"])
    )
    rows = []
    for p in opportunity_pairs(runs):
        baseline, target, refs = p["baseline"], p["target"], p["reference"]
        lo, hi = baseline["end_s"], target["start_s"]
        gap_allowed = intersect_intervals(allowed, [[lo, hi]])
        ripple_exposure = intersect_intervals(ripples[:, :2], gap_allowed)
        eligible_counts = counts_in_intervals(trains, gap_allowed)
        ripple_counts = counts_in_intervals(trains, ripple_exposure)
        n_intersecting = sum(bool(len(intersect_intervals([[a, b]], gap_allowed))) for a, b in ripples[:, :2] if b > lo and a < hi)
        row = {k: p[k] for k in ("opportunity_id", "epoch", "direction", "status")}
        row.update(
            baseline_traversal=baseline["traversal"],
            target_traversal=target["traversal"],
            baseline_start_s=baseline["start_s"],
            baseline_end_s=lo,
            target_start_s=hi,
            target_end_s=target["end_s"],
            reference_traversals=json.dumps([r["traversal"] for r in refs]),
            reference_end_s=max((r["end_s"] for r in refs), default=np.nan),
            gap_duration_s=hi - lo,
            eligible_immobile_s=duration(gap_allowed),
            eligible_ripple_s=duration(ripple_exposure),
            eligible_background_s=duration(gap_allowed) - duration(ripple_exposure),
            n_native_ripples_intersecting=n_intersecting,
            n_units_raw=len(keys),
            n_encoding_units=0,
            n_encoding_units_baseline_active=0,
            n_encoding_units_target_active=0,
            n_encoding_units_both_runs_active=0,
            n_encoding_units_ripple_active=0,
            n_encoding_units_both_runs_and_ripple_active=0,
        )
        if p["status"] == "audited":
            ref_mask = trainable & interval_run_mask(t, refs)
            _, supported, units, _ = fit_maps(counts, dt, direction, spatial, ref_mask, n_bins)
            masks = {name: readout & interval_run_mask(t, [run]) for name, run in (("baseline", baseline), ("target", target))}
            active = {}
            for name, mask in masks.items():
                c = counts[mask][:, units].sum(axis=0)
                active[name] = c > 0
                row[name + "_readout_s"] = float(dt[mask].sum())
                row[name + "_encoding_spikes"] = int(c.sum())
                row["n_encoding_units_" + name + "_active"] = int((c > 0).sum())
                truth = direction[mask] * n_bins + spatial[mask]
                row[name + "_reference_support_fraction"] = float(dt[mask][supported[truth]].sum() / dt[mask].sum()) if dt[mask].sum() > 0 else np.nan
            both = active["baseline"] & active["target"]
            ripple_active = ripple_counts[units] > 0
            row.update(
                n_encoding_units=int(units.sum()),
                n_encoding_units_both_runs_active=int(both.sum()),
                n_encoding_units_ripple_active=int(ripple_active.sum()),
                n_encoding_units_both_runs_and_ripple_active=int((both & ripple_active).sum()),
                reference_encoding_spikes=int(counts[ref_mask][:, units].sum()),
                ripple_encoding_spikes=int(ripple_counts[units].sum()),
                background_encoding_spikes=int((eligible_counts - ripple_counts)[units].sum()),
            )
            row["minimum_units_descriptor"] = bool(units.sum() >= PARAMETERS["minimum_encoding_units_descriptor"])
        else:
            row["minimum_units_descriptor"] = False
        rows.append(row)
    summary = {
        "status": "audited",
        "n_traversals": len(runs),
        "n_units": len(keys),
        "n_native_ripples": len(ripples),
        "n_opportunities": len(rows),
        "n_prior_reference_available": sum(r["status"] == "audited" for r in rows),
        "n_reference_min_units": sum(r["minimum_units_descriptor"] for r in rows),
        "n_reference_min_units_with_ripple": sum(r["minimum_units_descriptor"] and r["eligible_ripple_s"] > 0 for r in rows),
        "n_reference_min_units_without_ripple": sum(r["minimum_units_descriptor"] and r["eligible_ripple_s"] == 0 for r in rows),
        "excluded_nonpositive_id_spikes": excluded,
    }
    return summary, rows


def run(args):
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    paths = sorted(root.glob("Experiment_1/*/*/spike_data.mat"))
    frozen = pd.read_csv(args.run_qc, float_precision="round_trip")
    if frozen.duplicated(["animal", "session"]).any():
        raise ValueError("duplicate_RUN_session")
    expected = {(p.parent.parent.name, p.parent.name) for p in paths}
    if set(zip(frozen.animal, frozen.session, strict=True)) != expected:
        raise ValueError("RUN_inventory_mismatch")
    run_manifest = json.loads(args.run_qc.with_name("manifest.json").read_text())
    if file_sha256(args.run_qc) != run_manifest["outputs"][args.run_qc.name]:
        raise ValueError("RUN_table_hash_mismatch")
    frozen = frozen.set_index(["animal", "session"])
    inputs = {str(p.relative_to(root)): p for spike in paths for p in (spike, spike.with_name("session_info.mat"), spike.with_name("ripple_events.mat"))}
    inputs.update(
        producer=Path(__file__),
        protocol=ROOT / "docs/kleinman_ripple_run_protocol.md",
        run_qc=args.run_qc,
        run_manifest=args.run_qc.with_name("manifest.json"),
        run_helper=ROOT / "scripts/validate_kleinman_run_decoder.py",
    )
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean_committed_producer_required")
    if any(v is None for v in provenance["input_file_sha256"].values()):
        raise ValueError("missing_input")
    output.mkdir(parents=True, exist_ok=False)
    sessions, opportunities = [], []
    for spike in paths:
        folder = spike.parent
        original = frozen.loc[(folder.parent.name, folder.name)]
        key = {
            "animal": folder.parent.name,
            "session": folder.name,
            "drug": int(original.drug),
            "novel": int(original.novel),
            "incr_end": int(original.incr_end),
            "original_run_pass": bool(original.decoder_pass),
        }
        try:
            summary, rows = session_audit(folder)
            sessions.append({**key, **summary})
            opportunities.extend({**key, **r} for r in rows)
        except (ValueError, KeyError, IndexError) as exc:
            sessions.append({**key, "status": "failed", "failure_reason": str(exc)})
        print(json.dumps(sessions[-1]), flush=True)
    sf, of = pd.DataFrame(sessions), pd.DataFrame(opportunities)
    sf.to_csv(output / "sessions.csv", index=False)
    of.to_csv(output / "opportunities.csv", index=False)
    primary = of.loc[of.original_run_pass].copy()
    primary["has_reference"] = primary.status.eq("audited")
    primary["min_units_with_ripple"] = primary.minimum_units_descriptor & primary.eligible_ripple_s.gt(0)
    primary["min_units_without_ripple"] = primary.minimum_units_descriptor & primary.eligible_ripple_s.eq(0)
    primary.groupby(["animal", "drug", "novel"]).agg(
        n_sessions=("session", "nunique"),
        n_opportunities=("opportunity_id", "size"),
        n_reference_available=("has_reference", "sum"),
        n_reference_min_units=("minimum_units_descriptor", "sum"),
        n_reference_min_units_with_ripple=("min_units_with_ripple", "sum"),
        n_reference_min_units_without_ripple=("min_units_without_ripple", "sum"),
        n_units_both_runs_and_ripple=("n_encoding_units_both_runs_and_ripple_active", "sum"),
    ).to_csv(output / "coverage.csv")
    for key, p in inputs.items():
        if file_sha256(p) != provenance["input_file_sha256"][key]:
            raise ValueError("input_changed_during_run")
    manifest = {
        **provenance,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "parameters": PARAMETERS,
        "run_parameters": RUN_PARAMETERS,
        "future_spatial_outcome_scored": False,
        "recruitment_outcome_correlation_scored": False,
        "drug_effect_scored": False,
        "confirmed_replay_required_or_claimed": False,
        "n_source_sessions": len(paths),
        "outputs": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
