#!/usr/bin/env python3
"""Frozen 48-packet native temporal asymmetry pilot; no pharmacological contrast."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from audit_kleinman_ripple_run_opportunities import interval_run_mask
from calibrate_kleinman_conditional_coupling import recruitment
from calibrate_kleinman_joint_temporal_specificity import infer
from calibrate_kleinman_spatial_expression import estimate_reference
from kleinman_conditional_spatial_score import anchor_score, reference_feature, spatial_score
from validate_kleinman_run_decoder import align_behavior, interval_counts, interval_data, make_traversals, split_units


def packet_data(folder, a):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    lookup = {r["traversal"]: r for r in runs}
    refs = [lookup[i] for i in json.loads(a.reference_traversals)]
    rs = [lookup[getattr(a, n + "_traversal")] for n in ("past", "baseline", "target")]
    assert max(r["end_s"] for r in refs) < rs[0]["start_s"]
    keys, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    dt, _, train, _, _, bins = interval_data(t, x, speed, runs, edges)
    counts = interval_counts(trains, t)
    readout = train & ((speed[:-1] + speed[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > ends[0] + 20) & ((x[:-1] + x[1:]) / 2 < ends[1] - 20)
    masks = [train & interval_run_mask(t, refs), *[readout & interval_run_mask(t, [r]) for r in rs]]
    occ = np.array([np.bincount(bins[m], weights=dt[m], minlength=len(edges) - 1) for m in masks])
    hist = np.array([[np.bincount(bins[m], weights=counts[m, u], minlength=len(edges) - 1) for u in range(len(keys))] for m in masks])
    rec = recruitment(folder, a, keys)
    return keys, occ, hist, rec


def score_packet(folder, a):
    keys, occ, hist, rec = packet_data(folder, a)
    common = np.all(occ[1:] > 0, axis=0)
    cells = []
    for i, key in enumerate(keys):
        model, keep = estimate_reference(hist[0, i], occ[0])
        feature = reference_feature(model, common)[common]
        row = {
            "unit_id": "_".join(map(str, key.astype(int))),
            "reference_included": keep,
            "reference_spikes": int(hist[0, i].sum()),
            "past_spikes": int(hist[1, i].sum()),
            "baseline_spikes": int(hist[2, i].sum()),
            "target_spikes": int(hist[3, i].sum()),
            "log_rate_enrichment": rec["predictor"][i],
            "ripple_spikes": int(rec["ripple_counts"][i]),
            "background_spikes": int(rec["background_counts"][i]),
        }
        for lag, b, y in (("preceding", 1, 2), ("prospective", 2, 3)):
            s = (
                spatial_score(hist[b, i, common], hist[y, i, common], occ[b, common], occ[y, common], feature)
                if keep
                else {"score": np.nan, "information": 0.0, "informative": False}
            )
            for k in ("score", "information", "informative"):
                row[lag + "_" + k] = s[k]
            row[lag + "_excluded_spikes"] = int((hist[b, i] + hist[y, i])[~common].sum())
        cells.append(row)
    frame = pd.DataFrame(cells)
    selected = frame.loc[frame.reference_included]
    assert len(selected) == a.n_reference_units
    for name in ("reference", "past", "baseline", "target"):
        assert selected[name + "_spikes"].sum() == getattr(a, name + "_spikes")
    for name in ("ripple", "background"):
        assert selected[name + "_spikes"].sum() == getattr(a, name + "_reference_spikes")
    x = selected.log_rate_enrichment.to_numpy()
    ready = len(x) >= 2 and np.isfinite(x).all() and x.std() > 1e-10
    if ready:
        x = (x - x.mean()) / x.std()
    result = {"n_raw_cells": len(frame), "n_reference_cells": len(selected), "common_bins": int(common.sum())}
    for lag in ("preceding", "prospective"):
        u, v = anchor_score(x, selected[lag + "_score"], selected[lag + "_information"]) if ready else (0.0, 0.0)
        result[lag + "_score"], result[lag + "_information"] = u, v
        result[lag + "_informative_cells"] = int(selected[lag + "_informative"].sum())
    result["status"] = "paired_information" if min(result["preceding_information"], result["prospective_information"]) > 1e-10 else "missing_paired_information"
    return frame, result


def run(args):
    source, out = args.calibration_dir, args.output_dir
    m = json.loads((source / "manifest.json").read_text())
    v = json.loads((source / "verification.json").read_text())
    if not m["engineering_screen_passed"] or v["status"] != "passed" or v["manifest_sha256"] != file_sha256(source / "manifest.json"):
        raise ValueError("verified_passing_calibration_required")
    for name, digest in m["outputs"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed_calibration")
    selection = pd.read_csv(source / "selected_packets.csv", float_precision="round_trip")
    previous = pd.read_csv(args.previous_native_selection)
    prior = set(zip(previous.animal, previous.session, previous.target_traversal, strict=True))
    inputs = {name: source / name for name in ("manifest.json", "verification.json", "selected_packets.csv")}
    inputs.update(producer=Path(__file__), protocol=ROOT / "docs/kleinman_native_joint_temporal_protocol.md", previous_native_selection=args.previous_native_selection)
    for name in (
        "_provenance.py",
        "audit_kleinman_ripple_run_opportunities.py",
        "calibrate_kleinman_conditional_coupling.py",
        "calibrate_kleinman_joint_temporal_specificity.py",
        "calibrate_kleinman_spatial_expression.py",
        "kleinman_conditional_spatial_score.py",
        "validate_kleinman_run_decoder.py",
    ):
        inputs[name] = ROOT / "scripts" / name
    for a in selection.itertuples():
        for name in ("session_info.mat", "spike_data.mat", "ripple_events.mat"):
            inputs[f"{a.animal}/{a.session}/{name}"] = args.dataset_root / "Experiment_1" / a.animal / a.session / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean_committed_producer_required")
    out.mkdir(parents=True, exist_ok=False)
    rows, frames = [], []
    for a in selection.itertuples():
        frame, row = score_packet(args.dataset_root / "Experiment_1" / a.animal / a.session, a)
        meta = {k: getattr(a, k) for k in ("animal", "session", "packet_id", "drug", "novel", "direction")}
        meta["overlaps_previous_native_target"] = (a.animal, a.session, a.target_traversal) in prior
        rows.append({**meta, **row})
        for k, value in meta.items():
            frame[k] = value
        frames.append(frame)
    animals, estimate = infer(rows)
    pd.DataFrame(rows).to_csv(out / "packet_scores.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "cell_scores.csv.gz", index=False)
    pd.DataFrame(animals).to_csv(out / "by_animal.csv", index=False)
    pd.DataFrame([estimate]).to_csv(out / "primary_temporal_contrast.csv", index=False)
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("changed_inputs")
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "native_outcomes_scored": True,
        "drug_effect_scored": False,
        "causal_plasticity_claim": False,
        "independent_replication": False,
        "overlapping_previous_targets": sum(r["overlaps_previous_native_target"] for r in rows),
        "outputs": {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(estimate))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--previous-native-selection", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
