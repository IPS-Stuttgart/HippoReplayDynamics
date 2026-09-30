"""Frozen 20-ms readout gaps on the shared bank, not prevalence calibration."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.selection_matched_regional import read_population
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import DELTAS, read_csv, write_json


def endpoint_counts(identities, templates, cells, window_starts):
    counts = []
    for i, end in enumerate(templates["endpoints"]):
        a, b = templates["offsets"][i:i+2]
        times = templates["times"][a:b]
        # Use frozen absolute endpoints, including the original floating-point convention.
        keep = (times >= window_starts[i]) & (times < end)
        counts.append(np.bincount(identities[a:b][keep], minlength=cells))
    return np.asarray(counts)


def worker(row):
    folder = Path(row["folder"])
    source, templates = load_npz(folder.parent / "source_inputs.npz"), load_npz(folder.parent / "templates.npz")
    definitions = json.loads((folder.parent / "population_definitions.json").read_text())
    family_sides = {}
    for j, pop in enumerate(definitions):
        if pop["family"] in ("targeted", "whole_tetrode"):
            family_sides.setdefault(pop["family"], {})[pop["side"]] = (j, pop)
    if not family_sides:
        return []
    meta = read_csv(folder / "panels.csv")
    out = []
    for job in meta.loc[meta.phase.eq("validation")].itertuples():
        path = folder / f"panel_{job.panel:03}.npz"
        expected = json.loads((folder / "manifest.json").read_text())["sha256"][path.name]
        assert file_sha256(path) == expected
        p = load_npz(path)
        counts = endpoint_counts(p["identities"], templates, len(source["cell_ids"]), source["windows"][templates["source_indices"], 0])
        truth = p["label"][:, DELTAS.index(row["delta_ms"]/1000.)]
        endpoint = p["endpoint_home"][:, 0]
        for family, sides in family_sides.items():
            masses = {}
            for side in ("high", "low"):
                j, pop = sides[side]
                population = {key: source[f"pop{j}_{key}"] for key in ("ids", "rates", "region")}
                _, _, masses[side] = read_population(counts, source["cell_ids"], population)
            for name, mask in (("all", np.ones(len(truth), bool)), ("segment_positive", truth),
                               ("segment_negative", ~truth), ("endpoint_positive", endpoint),
                               ("endpoint_negative", ~endpoint)):
                h, l = masses["high"][mask], masses["low"][mask]
                out.append({"animal": row["session"].split("/")[0], "session": row["session"],
                    "generator": row["generator"], "scale": row["scale"], "conditioning_delta_ms": row["delta_ms"],
                    "readout_window_ms": 20, "readout_estimand": "frozen_endpoint_regional_posterior_mass",
                    "family": family, "label_stratum": name, "prevalence": job.prevalence, "replica": job.replica,
                    "events": int(mask.sum()), "status": "available" if mask.any() else "empty_conditional_cell",
                    "high_mass": float(h.mean()) if h.size else np.nan, "low_mass": float(l.mean()) if l.size else np.nan,
                    "signed_gap_pp": 100*float((h-l).mean()) if h.size else np.nan,
                    "absolute_event_gap_pp": 100*float(np.abs(h-l).mean()) if h.size else np.nan,
                    "endpoint_prevalence": float(endpoint[mask].mean()) if mask.any() else np.nan,
                    "segment_prevalence": float(truth[mask].mean()) if mask.any() else np.nan,
                    "shared_cells": sides["high"][1]["shared_cells"], "legacy_eligibility": True})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--bank-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bank, audit, out = Path(args.bank), Path(args.bank_audit), Path(args.output_dir)
    quality = json.loads((audit / "manifest.json").read_text())
    if quality["status"] != "technical_pass":
        raise ValueError("resolve the bank audit before reading populations")
    if quality["input_file_sha256"]["bank_manifest"] != file_sha256(bank / "manifest.json"):
        raise ValueError("audit belongs to a different bank")
    out.mkdir(parents=True, exist_ok=False)
    strata = read_csv(bank / "strata.csv")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for records in executor.map(worker, strata.to_dict("records")):
            rows.extend(records)
    df = pd.DataFrame(rows)
    df.to_csv(out / "population_gap_panels.csv", index=False)
    keys = ["family", "generator", "scale", "conditioning_delta_ms", "label_stratum"]
    metrics = ["signed_gap_pp", "absolute_event_gap_pp", "high_mass", "low_mass", "endpoint_prevalence", "segment_prevalence"]
    # Equal weight for prevalence/replica within pair, then session within animal.
    pairs = df.groupby(["animal", "session", *keys], observed=True)[metrics].mean().reset_index()
    pairs.to_csv(out / "population_gap_by_pair.csv", index=False)
    animals = pairs.groupby(["animal", *keys], observed=True)[metrics].mean().reset_index()
    animals.to_csv(out / "population_gap_by_animal.csv", index=False)
    summary = animals.groupby(keys, observed=True)[metrics].mean().reset_index()
    summary = summary.merge(animals.groupby(keys, observed=True).animal.nunique().rename("animals").reset_index(), on=keys, validate="one_to_one")
    summary.to_csv(out / "population_gap_summary.csv", index=False)
    lines = ["# Generator-stratified population gaps on the replacement bank", "",
             "Descriptive 20-ms endpoint posterior mass, not a terminal-segment estimator or calibrated truth probability.",
             "Conditioning Delta controls the geometry label used to sample panels. Readout duration remains 20 ms throughout.",
             "A positive >=20-ms dwell label differs from the old forced late-jump endpoint label. Do not read old/new gaps as a like-for-like treatment effect.",
             "The common >=100-ms cohort is shorter than the original 1600-event bank; simulated samples are not independent animals.",
             "Pairs are confirmed legacy coverage-selected populations, can overlap, and retain the original eligibility limitation.",
             "Equal prevalence/replica weighting within pair, then equal sessions within animal, then equal animals. Empty conditional cells are retained with zero events and NaN metrics; averages omit their undefined metrics.",
             "No bound coverage, false-flag rate, real-data calibration or content reconciliation is established by a population gap.", ""]
    (out / "report.md").write_text("\n".join(lines))
    manifest = build_script_provenance(input_paths={"bank": bank / "manifest.json", "audit": audit / "manifest.json", "reporter": __file__})
    manifest.update(status="complete_descriptive_only", outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out / "manifest.json", manifest)
    print("descriptive population gaps complete", len(df), flush=True)


if __name__ == "__main__":
    main()
