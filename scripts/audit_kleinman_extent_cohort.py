#!/usr/bin/env python3
"""Apply frozen synthetic extent eligibility to every RUN-qualified session."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from calibrate_kleinman_integrated_extent import condition_summary, fit_events, replay_counts, screens, template_library
from calibrate_kleinman_replay_content import full_map, trajectory


def new_bank(model, animal, session, offset):
    rows, observations = [], {}
    shapes = [(0.0, "linear")] + list(product([0.25, 0.5, 0.75], ["linear", "cosine", "pause_step"]))
    for i, (side, (extent, profile), duration, target, repeat) in enumerate(product(range(2), shapes, [0.1, 0.2, 0.4], [24, 48, 96], range(16))):
        x = trajectory(duration, extent, side, profile, model["ends"])
        states = np.digitize(x, model["edges"]) - 1 + side * (len(model["edges"]) - 1)
        intensity = 0.001 * model["rates"][states]
        event_index = offset + i
        rng = np.random.default_rng(np.random.SeedSequence([20260923, event_index]))
        fine = rng.poisson(intensity * (target / intensity.sum()))
        rows.append(
            {
                "animal": animal,
                "session": session,
                "event_index": event_index,
                "side": side,
                "extent": extent,
                "profile": profile,
                "duration_s": duration,
                "expected_spikes": target,
                "replicate": repeat,
                "n_spikes": int(fine.sum()),
                "n_active_units": int((fine.sum(axis=0) > 0).sum()),
            }
        )
        observations[event_index] = fine.reshape(-1, 10, fine.shape[1]).sum(axis=1)
    return pd.DataFrame(rows), observations


def session_eligible(gates):
    return bool(
        len(gates) == 6
        and set(gates.side) == {0, 1}
        and set(gates.screen) == {"matched_timing", "unseen_timing", "duration_only"}
        and not gates.duplicated(["side", "screen"]).any()
        and gates.passed.all()
    )


def cohort_tables(frame):
    animals = frame.groupby("animal").agg(
        released_sessions=("session", "size"), run_pass_sessions=("run_pass", "sum"), attempted_sessions=("attempted", "sum"), eligible_sessions=("extent_eligible", "sum")
    )
    conditions = frame.groupby(["animal", "drug", "novel"]).agg(
        released_sessions=("session", "size"), run_pass_sessions=("run_pass", "sum"), eligible_sessions=("extent_eligible", "sum")
    )
    checks = [
        {"gate": "all_source_sessions_accounted", "passed": len(frame) == 135 and not frame.duplicated(["animal", "session"]).any()},
        {"gate": "all_run_pass_sessions_attempted", "passed": frame.run_pass.sum() == 127 and frame.attempted.sum() == 127},
        {"gate": "no_technical_failures", "passed": not frame.status.eq("technical_failure").any()},
        {"gate": "reward_content_all_six_animals_retained", "passed": len(animals) == 6 and animals.eligible_sessions.gt(0).all()},
        {"gate": "drug_context_all_24_cells_retained", "passed": len(conditions) == 24 and conditions.eligible_sessions.gt(0).all()},
    ]
    return animals.reset_index(), conditions.reset_index(), pd.DataFrame(checks)


def run(args):
    started = time.monotonic()
    run_qc = pd.read_csv(args.run_qc)
    run_qc["run_pass"] = run_qc.decoder_pass.astype(str).str.lower().eq("true")
    if len(run_qc) != 135 or run_qc.run_pass.sum() != 127 or run_qc.duplicated(["animal", "session"]).any():
        raise ValueError("frozen RUN cohort mismatch")
    inputs = {
        "run_qc": args.run_qc,
        "producer": Path(__file__),
        "protocol": ROOT / "docs/kleinman_extent_cohort_protocol.md",
        "integrated_protocol": ROOT / "docs/kleinman_integrated_extent_protocol.md",
        "integrated_fitter": ROOT / "scripts/calibrate_kleinman_integrated_extent.py",
        "old_generator": ROOT / "scripts/calibrate_kleinman_replay_content.py",
        "run_encoder": ROOT / "scripts/validate_kleinman_run_decoder.py",
        "bank_manifest": args.bank_dir / "manifest.json",
        "old_fit_manifest": args.old_fit_dir / "manifest.json",
    }
    for directory in [args.bank_dir, args.old_fit_dir]:
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["outputs"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("changed historical bank " + name)
    old = {}
    for path in sorted(args.bank_dir.glob("*_events.csv")):
        data = pd.read_csv(path)
        data = data.loc[data.arm == "matched_gain_poisson"]
        old[(data.animal.iloc[0], data.session.iloc[0])] = data
    folders = {}
    for row in run_qc.loc[run_qc.run_pass].itertuples():
        matches = list(args.dataset_root.glob(f"**/{row.animal}/{row.session}/spike_data.mat"))
        if len(matches) != 1:
            raise ValueError("missing/duplicate source " + row.animal + "/" + row.session)
        folders[(row.animal, row.session)] = matches[0].parent
        for name in ["session_info.mat", "spike_data.mat"]:
            inputs[row.animal + "/" + row.session + "/" + name] = matches[0].parent / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries, all_gates = [], []
    next_index = 17280
    for row in run_qc.sort_values(["animal", "session"]).itertuples():
        base = {
            "animal": row.animal,
            "session": row.session,
            "drug": row.drug,
            "novel": row.novel,
            "run_pass": row.run_pass,
            "attempted": row.run_pass,
            "extent_eligible": False,
            "run_mean_error_cm": row.mean_posterior_mean_error_cm,
            "run_minimum_fold_units": row.minimum_fold_units,
        }
        if not row.run_pass:
            summaries.append(dict(**base, status="run_qc_ineligible", failure_reason=str(row.failure_reason)))
            continue
        out = args.output_dir / "sessions" / row.animal / row.session
        out.mkdir(parents=True)
        try:
            model = full_map(folders[(row.animal, row.session)])
            np.savez_compressed(out / "known_map.npz", **model)
            if (row.animal, row.session) in old:
                reused = True
                source = old[(row.animal, row.session)]
                previous = dict(np.load(args.bank_dir / (row.animal + "_known_map.npz")))
                for key in model:
                    np.testing.assert_array_equal(model[key], previous[key])
                counts = {r.event_index: replay_counts(model, r) for r in source.itertuples()}
            else:
                reused = False
                source, counts = new_bank(model, row.animal, row.session, next_index)
                next_index += len(source)
            pieces = []
            for duration, sub in source.groupby("duration_s"):
                q, templates = template_library(model, duration)
                fit = fit_events(np.array([counts[i] for i in sub.event_index]), q, templates)
                keys = ["animal", "session", "event_index", "side", "profile", "extent", "duration_s", "expected_spikes", "replicate", "n_spikes", "n_active_units"]
                pieces.append(pd.concat([sub[keys].reset_index(drop=True), fit], axis=1))
            estimates = pd.concat(pieces).sort_values("event_index").reset_index(drop=True)
            if reused:
                pd.testing.assert_frame_equal(estimates, pd.read_csv(args.old_fit_dir / (row.animal + "_estimates.csv")), check_dtype=False, rtol=1e-9, atol=1e-9)
            estimates.to_csv(out / "estimates.csv", index=False)
            condition_summary(estimates).to_csv(out / "condition_summary.csv", index=False)
            gates = screens(estimates)
            gates.insert(1, "session", row.session)
            gates.to_csv(out / "screens.csv", index=False)
            base["extent_eligible"] = session_eligible(gates)
            failed = [f"side{r.side}:{r.screen}" for r in gates.itertuples() if not r.passed]
            summaries.append(
                dict(
                    **base,
                    status="scored",
                    failure_reason=";".join(failed),
                    reused_bank=reused,
                    n_synthetic_events=len(estimates),
                    n_units=len(model["units"]),
                    n_screen_pass=int(gates.passed.sum()),
                )
            )
            all_gates.append(gates)
        except (ValueError, AssertionError, KeyError, IndexError) as exc:
            summaries.append(dict(**base, status="technical_failure", failure_reason=str(exc)))
        pd.DataFrame(summaries).to_csv(args.output_dir / "session_progress.csv", index=False)
        print(
            json.dumps(
                {"animal": row.animal, "session": row.session, "status": summaries[-1]["status"], "eligible": base["extent_eligible"], "elapsed_s": time.monotonic() - started}
            ),
            flush=True,
        )
    frame = pd.DataFrame(summaries)
    frame.to_csv(args.output_dir / "kleinman_extent_cohort_sessions.csv", index=False)
    pd.concat(all_gates, ignore_index=True).to_csv(args.output_dir / "kleinman_extent_cohort_screens.csv", index=False)
    animals, conditions, gates = cohort_tables(frame)
    for label, data in [("by_animal", animals), ("by_condition", conditions), ("gates", gates)]:
        data.to_csv(args.output_dir / ("kleinman_extent_cohort_" + label + ".csv"), index=False)
    for key, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][key]:
            raise ValueError("changed input " + key)
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "runtime_s": time.monotonic() - started,
        "n_source_sessions": len(frame),
        "n_attempted_sessions": int(frame.attempted.sum()),
        "n_eligible_sessions": int(frame.extent_eligible.sum()),
        "real_replay_scored": False,
        "reward_contrast_scored": False,
        "biological_result": False,
        "outputs": {str(p.relative_to(args.output_dir)): file_sha256(p) for p in args.output_dir.rglob("*") if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-qc", required=True, type=Path)
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--old-fit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
