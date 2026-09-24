#!/usr/bin/env python3
"""Frozen-anchor native ripple/spatial-firing association pilot; no drug contrast."""

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
from audit_kleinman_ripple_run_opportunities import interval_run_mask, opportunity_pairs
from calibrate_kleinman_conditional_coupling import animal_inference, recruitment
from calibrate_kleinman_spatial_expression import estimate_reference
from kleinman_conditional_spatial_score import anchor_score, reference_feature, spatial_score
from validate_kleinman_run_decoder import align_behavior, interval_counts, interval_data, make_traversals, split_units


def score_anchor(folder, anchor):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    pair = next(p for p in opportunity_pairs(runs) if p["opportunity_id"] == anchor.opportunity_id)
    assert pair["status"] == "audited"
    assert max(r["end_s"] for r in pair["reference"]) < pair["baseline"]["start_s"]
    keys, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    edges = np.arange(np.floor(x.min() / 2) * 2, np.ceil(x.max() / 2) * 2 + 2, 2)
    dt, _, train, _, _, bins = interval_data(t, x, speed, runs, edges)
    counts = interval_counts(trains, t)
    nb = len(edges) - 1
    reference = train & interval_run_mask(t, pair["reference"])
    readout = train & ((speed[:-1] + speed[1:]) / 2 > 20) & ((x[:-1] + x[1:]) / 2 > ends[0] + 20) & ((x[:-1] + x[1:]) / 2 < ends[1] - 20)
    masks = [reference, readout & interval_run_mask(t, [pair["baseline"]]), readout & interval_run_mask(t, [pair["target"]])]
    occupancy = [np.bincount(bins[mask], weights=dt[mask], minlength=nb) for mask in masks]
    hist = [np.array([np.bincount(bins[mask], weights=counts[mask, u], minlength=nb) for u in range(len(keys))]) for mask in masks]
    rec = recruitment(folder, anchor, keys)
    common = (occupancy[1] > 0) & (occupancy[2] > 0)
    cells = []
    for u, key in enumerate(keys):
        model, keep = estimate_reference(hist[0][u], occupancy[0])
        s = (
            spatial_score(hist[1][u], hist[2][u], occupancy[1], occupancy[2], reference_feature(model, common))
            if keep
            else {
                "score": np.nan,
                "information": 0.0,
                "informative": False,
                "before_common_spikes": int(hist[1][u][common].sum()),
                "after_common_spikes": int(hist[2][u][common].sum()),
                "excluded_spikes": int((hist[1][u] + hist[2][u])[~common].sum()),
            }
        )
        cells.append(
            {
                "unit_id": "_".join(map(str, key.astype(int))),
                "reference_included": keep,
                "reference_spikes": int(hist[0][u].sum()),
                "reference_peak_hz": model.max(),
                "baseline_spikes": int(hist[1][u].sum()),
                "target_spikes": int(hist[2][u].sum()),
                "ripple_spikes": int(rec["ripple_counts"][u]),
                "background_spikes": int(rec["background_counts"][u]),
                "log_rate_enrichment": rec["predictor"][u],
                **s,
            }
        )
    c = pd.DataFrame(cells)
    included = c.loc[c.reference_included]
    assert len(included) == anchor.n_encoding_units
    for name, source in (("reference", "reference"), ("baseline", "baseline"), ("target", "target"), ("ripple", "ripple"), ("background", "background")):
        assert included[source + "_spikes"].sum() == anchor[name + "_encoding_spikes"]
    ready = len(included) >= 2 and np.isfinite(included.log_rate_enrichment).all() and included.log_rate_enrichment.std(ddof=0) > 1e-10
    u, v = 0.0, 0.0
    if ready:
        z = included.log_rate_enrichment.to_numpy()
        z = (z - z.mean()) / z.std()
        u, v = anchor_score(z, included.score, included.information)
    summary = {
        "n_raw_units": len(c),
        "n_reference_units": len(included),
        "n_informative_cells": int(included.informative.sum()),
        "ripple_s": rec["ripple_seconds"],
        "background_s": rec["background_seconds"],
        "score": u,
        "information": v,
        "effect": u / v if v > 1e-10 else np.nan,
        "status": "scored" if v > 1e-10 else "no_predictor_or_information",
    }
    return c, summary


def run(args):
    cm = json.loads((args.calibration_dir / "manifest.json").read_text())
    cv = json.loads((args.calibration_dir / "verification.json").read_text())
    if not cm["engineering_screen_passed"] or cv["status"] != "passed" or cv["manifest_sha256"] != file_sha256(args.calibration_dir / "manifest.json"):
        raise ValueError("verified passing calibration required")
    for name, digest in cm["outputs"].items():
        if file_sha256(args.calibration_dir / name) != digest:
            raise ValueError("changed calibration")
    anchor_path = Path(cm["input_file_paths"]["selected_anchors.csv"])
    if file_sha256(anchor_path) != cm["input_file_sha256"]["selected_anchors.csv"]:
        raise ValueError("changed frozen anchors")
    anchors = pd.read_csv(anchor_path, float_precision="round_trip")
    inputs = {
        "calibration_manifest": args.calibration_dir / "manifest.json",
        "calibration_verification": args.calibration_dir / "verification.json",
        "selected_anchors": anchor_path,
        "producer": Path(__file__),
        "protocol": ROOT / "docs/kleinman_native_coupling_pilot_protocol.md",
    }
    for name in (
        "audit_kleinman_ripple_run_opportunities.py",
        "calibrate_kleinman_conditional_coupling.py",
        "calibrate_kleinman_spatial_expression.py",
        "kleinman_conditional_spatial_score.py",
        "validate_kleinman_run_decoder.py",
    ):
        inputs[name] = ROOT / "scripts" / name
    for a in anchors.itertuples():
        for name in ("session_info.mat", "spike_data.mat", "ripple_events.mat"):
            inputs[f"{a.animal}/{a.session}/{name}"] = args.dataset_root / "Experiment_1" / a.animal / a.session / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed pilot required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries, frames = [], []
    for a in anchors.itertuples():
        cells, summary = score_anchor(args.dataset_root / "Experiment_1" / a.animal / a.session, a)
        metadata = {k: getattr(a, k) for k in ("animal", "session", "opportunity_id", "drug", "novel", "direction")}
        for k, v in metadata.items():
            cells[k] = v
        frames.append(cells)
        summaries.append({**metadata, **summary})
    af = pd.DataFrame(summaries)
    af.to_csv(args.output_dir / "anchor_scores.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "cell_scores.csv.gz", index=False)
    animals, estimate = animal_inference(summaries)
    pd.DataFrame(animals).to_csv(args.output_dir / "by_animal.csv", index=False)
    pd.DataFrame([estimate]).to_csv(args.output_dir / "primary_association.csv", index=False)
    af.assign(available=af.status.eq("scored")).groupby(["animal", "drug", "novel"]).agg(
        selected_anchors=("opportunity_id", "size"),
        available_anchors=("available", "sum"),
        reference_cells=("n_reference_units", "sum"),
        informative_cells=("n_informative_cells", "sum"),
    ).to_csv(args.output_dir / "coverage.csv")
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("source changed during pilot")
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "real_future_outcome_scored": True,
        "biological_association_scored": True,
        "drug_effect_scored": False,
        "causal_plasticity_claim": False,
        "novelty_confirmed": False,
        "n_selected_anchors": len(anchors),
        "outputs": {p.name: file_sha256(p) for p in args.output_dir.iterdir()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(estimate))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
