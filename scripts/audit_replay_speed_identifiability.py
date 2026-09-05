#!/usr/bin/env python3
"""Reproduce every interval and a declared sample of simulation panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage_data import CoverageInputConfig
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_replay_coverage_geometry import compare_tables
from scripts.calibrate_replay_speed_identifiability import (
    IDENTITY,
    PANEL_KEY,
    READOUT,
    draw_schedule,
    evaluate_panels,
    simulate_panel,
    summarize_decisions,
    training_generator,
    validate_panel_contract,
)
from scripts.validate_replay_coverage_run_decoder import session_from_cache


def audit(root):
    manifest_path = root/"speed_identifiability_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("complete scoring required")
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(root/name) != expected:
            raise AssertionError(f"output changed: {name}")
    for name, path in manifest["input_file_paths"].items():
        expected = manifest["input_file_sha256"][name]
        if file_sha256(path) != expected or file_sha256(root/"inputs"/f"{name}{Path(path).suffix}") != expected:
            raise AssertionError(f"input changed: {name}")
    for path, expected in manifest["model_cache_sha256"].items():
        if file_sha256(path) != expected:
            raise AssertionError("model/cache changed")
    files, args = manifest["input_file_paths"], manifest["parameters"]
    parent = json.loads(Path(files["reference_manifest"]).read_text())
    config = CoverageInputConfig(**parent["encoding_settings"])
    sessions = pd.read_csv(files["sessions"])
    units = pd.read_csv(files["source_units"], dtype={"source_cell_type_allowed": "boolean"})
    reference = pd.read_csv(files["reference_batches"])
    profiles = pd.read_csv(files["profiles"], usecols=IDENTITY+["source_event_index", "n_base_bins"]).drop_duplicates()
    batches = pd.read_csv(root/"speed_identifiability_batches.csv")
    summaries = pd.read_csv(root/"speed_identifiability_session_summary.csv")
    rows = []
    for batch in batches.itertuples(index=False):
        identity = {k: getattr(batch, k) for k in IDENTITY}
        def choose(frame, identity=identity):
            return np.logical_and.reduce([frame[k].eq(v) for k, v in identity.items()])
        key = ":".join(identity.values())
        schedule = draw_schedule(args["seed"], key, args["fit_draws"], args["calibration_draws"], args["test_draws"], args["fixed_draws"])
        compare_tables(pd.read_csv(root/f"schedule__{batch.label}.csv"), schedule, ["phase", "draw_id"])
        with np.load(root/f"model__{batch.label}.npz", allow_pickle=False) as handle:
            model = dict(handle)
        ref = reference[choose(reference) & reference.direction.eq(0)].iloc[0]
        with np.load(ref.model_path, allow_pickle=False) as handle:
            for name in handle.files:
                np.testing.assert_array_equal(model[name], handle[name])
        record = next(sessions[choose(sessions)].itertuples(index=False))
        full = session_from_cache(record, units[choose(units)])
        np.testing.assert_array_equal(model["training_full_rates_hz"], training_generator(full, model, config))
        panels = pd.read_csv(root/f"panels__{batch.label}.csv.gz", float_precision="round_trip")
        validate_panel_contract(panels, schedule, identity)
        decisions = pd.read_csv(root/f"decisions__{batch.label}.csv.gz")
        expected_decisions, fits = evaluate_panels(panels)
        compare_tables(decisions, expected_decisions, PANEL_KEY+["method"])
        compare_tables(pd.read_csv(root/f"fits__{batch.label}.csv.gz"), fits, READOUT)
        compare_tables(summaries[choose(summaries)], summarize_decisions(decisions), IDENTITY+READOUT+["generator", "observation", "stratum", "method"])
        local = profiles[choose(profiles)].sort_values("source_event_index")
        if args["max_profiles"]:
            local = local.head(args["max_profiles"])
        sampled = schedule.groupby(["phase", "stratum"], sort=True).head(1)
        count = 0
        for draw in sampled.to_dict("records"):
            expected = simulate_panel(model, local, draw)
            for k, value in identity.items():
                expected[k] = value
            observed = panels[panels.phase.eq(draw["phase"]) & panels.draw_id.eq(draw["draw_id"])]
            compare_tables(observed, expected, PANEL_KEY)
            count += len(expected)
        row = {**identity, "status": "pass", "interval_rows_reproduced": len(decisions),
               "sampled_draws_reconstructed": len(sampled), "sampled_panel_rows_reproduced": count}
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(root/"speed_identifiability_reconstruction_audit.csv", index=False)
    meta = build_script_provenance(input_paths={"scoring_manifest": manifest_path, "auditor": Path(__file__)}, cwd=ROOT)
    meta.update(status="pass", sessions=len(rows), interval_rows_reproduced=sum(r["interval_rows_reproduced"] for r in rows),
        sampled_draws_reconstructed=sum(r["sampled_draws_reconstructed"] for r in rows),
        sampled_panel_rows_reproduced=sum(r["sampled_panel_rows_reproduced"] for r in rows),
        audit_scope="all input/output hashes and statistical intervals; full model-A refit; sampled path/count/decoder reconstruction, not all simulations or an independent decoder implementation")
    (root/"speed_identifiability_reconstruction_audit.json").write_text(json.dumps(meta, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    audit(parser.parse_args().input_dir.resolve())
