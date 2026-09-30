"""Selection-matched synthetic regional calibration; no replay-model scoring."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd


def read_csv(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


from hipporeplayimm.data import load_mat_variable
from hipporeplayimm.regional_content_mua_null import load_detector, stable_seed
from hipporeplayimm.selection_matched_regional import (
    GENERATORS,
    calibration_hist,
    draw_panel,
    endpoint_interval,
    evaluate,
    read_population,
    threshold,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz, recount
from scripts.measure_edge_support_content import occupied_graph

PREVALENCES = (.05, .15, .30, .50)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def detector_run(detector, spikes, speed_times, speed, intervals, cells):
    result = []
    for start, end in intervals:
        result.extend(detector.detect_high_mua_in_interval(spikes, speed_times, speed, start, end,
            bin_s=.001, gaussian_sd_s=.01, z_threshold=3., maximum_speed_cm_s=5.,
            minimum_duration_s=.05, maximum_duration_s=2., minimum_active_cells=int(np.ceil(.1*cells))))
    return result


def worker(task):
    session, animal, args = task
    output = Path(args["output_dir"])/session.replace("/", "_")
    output.mkdir()
    source_manifest = json.loads((Path(args["frontier_root"])/"manifest.json").read_text())
    encoder_path = Path(source_manifest["input_file_paths"][session+":encoder"])
    if file_sha256(encoder_path) != source_manifest["input_file_sha256"][session+":encoder"]:
        raise ValueError("encoder changed")
    e = load_npz(encoder_path)
    metadata_path = encoder_path.parent/"encoding_manifest.json"
    emeta = json.loads(metadata_path.read_text())
    if not emeta["training_only"] or emeta["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("first-half training encoder required")
    prior_path = Path(args["frontier_root"])/session.replace("/", "_")/"run_calibration_audit.npz"
    prior = load_npz(prior_path)
    region, grid = prior["region"], prior["grid"]
    valid = e["valid_spatial_bins"]
    rates = np.maximum(e["rates_hz"][:, valid], 1e-4)
    np.testing.assert_array_equal(grid, e["bin_centers_cm"][valid])
    graph = occupied_graph(grid)
    ids = e["cell_ids"]
    inputs = {"encoder": encoder_path, "encoder_metadata": metadata_path, "frontier_calibration": prior_path}
    populations = [{"name": "full", "family": "full", "side": "full", "ids": ids[e["unit_qc_mask"]],
        "rates": rates[e["unit_qc_mask"]], "region": region, "legacy_eligibility": False, "shared_cells": 0}]
    matched = Path(args["matched_root"])/session.replace("/", "_")
    cp_path = matched/"freeze_checkpoint.json"
    cp = json.loads(cp_path.read_text())
    archive = matched/"run_validation.npz"
    if file_sha256(archive) != cp["run_validation_sha256"]:
        raise ValueError("matched population archive changed")
    old = load_npz(archive)
    home_path = Path(args["home_csv"])
    home = read_csv(home_path).set_index("session").loc[session]
    old_region = np.linalg.norm(old["grid_cm"]-[home.home_x_cm, home.home_y_cm], axis=1) <= 20
    source_grid = {tuple(x): i for i, x in enumerate(e["bin_centers_cm"])}
    source_ids = {int(c): i for i, c in enumerate(ids)}
    old_grid_ix = [source_grid[tuple(x)] for x in old["grid_cm"]]
    old_ids_ix = [source_ids[int(c)] for c in old["cell_ids"]]
    np.testing.assert_allclose(old["early_rates"], e["rates_hz"][old_ids_ix][:, old_grid_ix], atol=0, rtol=0)
    for pair in cp["selected"]:
        if not pair["confirmation_pass"] or pair["family"] not in ("targeted", "whole_tetrode"):
            continue
        overlap = len(set(pair["high"]["indices"]) & set(pair["low"]["indices"]))
        for side in ("high", "low"):
            ix = pair[side]["indices"]
            populations.append({"name": pair["family"]+"_"+side, "family": pair["family"], "side": side,
                "ids": old["cell_ids"][ix], "rates": np.maximum(old["early_rates"][ix], 1e-4), "region": old_region,
                "legacy_eligibility": True, "shared_cells": overlap})
    inputs.update(matched_checkpoint=cp_path, matched_archive=archive, home=home_path)
    raw = Path(args["dataset_root"])/session
    spikes = np.asarray(load_mat_variable(raw/"Spike_Data.mat", "Spike_Data"), float).reshape(-1, 2)
    spikes = spikes[np.argsort(spikes[:, 0], kind="stable")]
    position = np.asarray(load_mat_variable(raw/"Position_Data.mat", "Position_Data"), float).reshape(-1, 4)
    intervals = np.asarray(load_mat_variable(raw/"Epochs.mat", "Run_Times"), float).reshape(-1, 2)
    np.testing.assert_array_equal(np.unique(spikes[:, 1]), ids)
    detector = load_detector(Path(args["detector_script"]))
    speed_times, speed = detector.position_speed(position, .10)
    original = detector_run(detector, spikes, speed_times, speed, intervals, len(ids))
    lookup = {round(float(x["event_start_s"]), 6): x for x in original}
    templates, selections, windows = [], [], []
    for event_id, start, end in zip(e["candidate_event_indices"], e["candidate_start_s"], e["candidate_end_s"], strict=True):
        actual = lookup[round(float(start), 6)]
        np.testing.assert_allclose([start, end], [actual["event_start_s"], actual["event_end_s"]], atol=1e-8, rtol=0)
        left, right = np.searchsorted(spikes[:, 0], [start, end], side="left")
        templates.append({"event_id": int(event_id), "start": float(start), "end": float(end), "times": spikes[left:right, 0]})
        selections.append((left, right))
        windows.append(endpoint_interval(start, end))
    windows = np.array(windows)
    real_counts = recount(spikes, ids, windows[:, 0], windows[:, 1])
    np.testing.assert_array_equal(real_counts[:, e["unit_qc_mask"]], prior["real_counts"])
    for name in ("Spike_Data.mat", "Position_Data.mat", "Epochs.mat"):
        inputs[name] = raw/name
    freeze_arrays = {"rates": rates, "grid": grid, "region": region, "cell_ids": ids, "real_counts": real_counts,
        "windows": windows, "candidate_start": e["candidate_start_s"], "candidate_end": e["candidate_end_s"],
        "candidate_ids": e["candidate_event_indices"], "template_times": np.concatenate([x["times"] for x in templates]),
        "template_offsets": np.r_[0, np.cumsum([len(x["times"]) for x in templates])]}
    for j, pop in enumerate(populations):
        for key in ("ids", "rates", "region"):
            freeze_arrays[f"pop{j}_{key}"] = pop[key]
    np.savez_compressed(output/"frozen_inputs.npz", **freeze_arrays)
    descriptors = [{k: v for k, v in pop.items() if k not in ("ids", "rates", "region")} | {
        "cells": len(pop["ids"]), "area_fraction": float(pop["region"].mean())} for pop in populations]
    json_write(output/"population_definitions.json", descriptors)
    jobs = []
    for kind in GENERATORS:
        jobs.extend(("calibration", .5, kind, "none", r) for r in range(args["calibration_replicas"]))
    for prevalence in PREVALENCES:
        jobs.extend(("null", prevalence, "mix", "none", r) for r in range(args["null_replicas"]))
        for kind in (*GENERATORS, "mix"):
            jobs.extend(("validation", prevalence, kind, "none", r) for r in range(args["validation_replicas"]))
    jobs.extend(("validation", .38, "mix", "none", r) for r in range(args["validation_replicas"]))
    for perturbation in ("home_participation_half", "home_nospatial_2hz", "within_region_shift", "global_gain_x2"):
        jobs.extend(("perturbation", .30, "mix", perturbation, r) for r in range(args["validation_replicas"]))
    all_counts, all_targets, all_labels, all_accepted, all_active, all_kinds, job_rows = [], [], [], [], [], [], []
    audit_rows = []
    for j, (phase, prevalence, kind, perturbation, replica) in enumerate(jobs):
        seed_phase, seed_perturbation = ("validation", "none") if perturbation == "global_gain_x2" else (phase, perturbation)
        seed = stable_seed(args["seed"], session, seed_phase, prevalence, kind, seed_perturbation, replica)
        panel = draw_panel(templates, prevalence, kind, grid, rates, graph, region, np.random.default_rng(seed), perturbation)
        accepted = panel["active"] >= int(np.ceil(.1*len(ids)))
        np.testing.assert_array_equal(panel["counts"].sum(axis=1), real_counts.sum(axis=1))
        all_counts.append(panel["counts"])
        all_targets.append(panel["targets"])
        all_labels.append(panel["labels"])
        all_accepted.append(accepted)
        all_active.append(panel["active"])
        all_kinds.append(panel["kinds"])
        job_rows.append({"job": j, "phase": phase, "requested_prevalence": prevalence, "generator": kind,
            "perturbation": perturbation, "replica": replica, "seed": str(seed), "retained": int(accepted.sum())})
        if phase == "calibration" and replica == 0:
            altered = spikes.copy()
            cursor = 0
            for left, right in selections:
                altered[left:right, 1] = ids[panel["identities"][cursor:cursor+right-left]]
                cursor += right-left
            np.testing.assert_array_equal(altered[:, 0], spikes[:, 0])
            redetected = detector_run(detector, altered, speed_times, speed, intervals, len(ids))
            starts = {round(float(x["event_start_s"]), 6): x for x in redetected}
            for event, ok in zip(templates, accepted, strict=True):
                assert (round(event["start"], 6) in starts) == bool(ok)
                if ok:
                    assert abs(starts[round(event["start"], 6)]["event_end_s"]-event["end"]) < 1e-8
            np.savez_compressed(output/f"detector_audit_{kind}.npz", identities=panel["identities"], job=j)
            audit_rows.append({"generator": kind, "original_candidates": len(original), "redetected": len(redetected),
                "templates": len(templates), "accepted": int(accepted.sum()), "exact_timestamp_preservation": True, "status": "pass"})
        if j % 100 == 0:
            print(session, j, "/", len(jobs), flush=True)
    counts, labels, accepted = np.asarray(all_counts), np.asarray(all_labels), np.asarray(all_accepted)
    meta = pd.DataFrame(job_rows)
    meta.to_csv(output/"simulation_jobs.csv", index=False)
    np.savez_compressed(output/"simulation_counts.npz", counts=counts, labels=labels, accepted=accepted,
        targets=np.asarray(all_targets), active=np.asarray(all_active), generators=np.asarray(all_kinds))
    fit_rows, real_rows, calibration_arrays, real_bank = [], [], {}, []
    for j, pop in enumerate(populations):
        bf, call, mass = read_population(counts.reshape(-1, len(ids)), ids, pop)
        bf, call, mass = [x.reshape(counts.shape[:2]) for x in (bf, call, mass)]
        selected_cal = meta.phase.eq("calibration").to_numpy()
        calibration = calibration_hist(call[selected_cal], labels[selected_cal], accepted[selected_cal])
        calibration_arrays[f"pop{j}_calibration"] = calibration
        calibration_arrays[f"pop{j}_bf"] = bf
        calibration_arrays[f"pop{j}_calls"] = call
        for job in meta.loc[~meta.phase.eq("calibration")].itertuples():
            fit = evaluate(call[job.job], labels[job.job], accepted[job.job], calibration)
            fit_rows.append(dict(animal=animal, session=session, population=pop["name"], family=pop["family"],
                side=pop["side"], **job._asdict(), **fit))
        _, real_call, real_mass = read_population(real_counts, ids, pop)
        real_bank.append((pop, real_call, real_mass, calibration))
    np.savez_compressed(output/"calibration_and_readouts.npz", **calibration_arrays)
    fits = pd.DataFrame(fit_rows).drop(columns="Index")
    threshold_rows = []
    for population, group in fits.loc[fits.phase.eq("null")].groupby("population"):
        cutoff = max(threshold(g.fit_tv.to_numpy()) for _, g in group.groupby("requested_prevalence"))
        threshold_rows.append({"population": population, "tv_cutoff": cutoff, "alpha": .05,
            "null_replicas_per_prevalence": args["null_replicas"], "threshold_policy": "max_prevalence_finite_bank_95th"})
    cutoffs = pd.DataFrame(threshold_rows)
    fits = fits.merge(cutoffs, on="population", validate="many_to_one")
    fits["flagged"] = fits.fit_tv > fits.tv_cutoff
    for pop, real_call, real_mass, calibration in real_bank:
        validation = fits.loc[fits.population.eq(pop["name"]) & fits.phase.eq("validation") & ~fits.requested_prevalence.eq(.38)]
        budget = bool(len(validation)) and all(
            g.absolute_error.le(.05).mean() >= .90 and g.absolute_error.mean() <= .05
            for _, g in validation.groupby(["generator", "requested_prevalence"]))
        mixed = validation.loc[validation.generator.eq("mix")]
        false_flag_ok = bool(len(mixed)) and all(g.flagged.mean() <= .10 for _, g in mixed.groupby("requested_prevalence"))
        usable = budget and false_flag_ok
        real_fit = evaluate(real_call, np.zeros(len(real_call)), np.ones(len(real_call), bool), calibration)
        for k in ("true_prevalence", "signed_error", "absolute_error"):
            real_fit.pop(k)
        real_fit["descriptive_mixture_prevalence_not_validated"] = real_fit.pop("prevalence")
        real_fit["calibrated_prevalence"] = real_fit["descriptive_mixture_prevalence_not_validated"] if usable else np.nan
        real_rows.append(dict(animal=animal, session=session, population=pop["name"], family=pop["family"],
            side=pop["side"], legacy_eligibility=pop["legacy_eligibility"], shared_cells=pop["shared_cells"],
            naive_mass=float(real_mass.mean()), area_fraction=float(pop["region"].mean()),
            synthetic_budget_pass=budget, false_flag_validation_pass=false_flag_ok,
            calibrated_estimate_available=usable, **real_fit))
    fits.to_csv(output/"validation_and_null.csv", index=False)
    pd.DataFrame(real_rows).merge(cutoffs, on="population", validate="one_to_one").to_csv(output/"real_descriptive.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output/"detector_equivalence.csv", index=False)
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="complete", session=session, parameters=args,
        simulation_support_fraction=len(graph[0])/len(grid),
        conditioning="frozen 200 candidate whole-population spike time profiles and surrounding actual count trace",
        outputs={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()})
    json_write(output/"manifest.json", provenance)
    print("DONE", session, flush=True)
    return {"session": session, "animal": animal, "status": "complete", "folder": str(output), "populations": len(populations)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frontier-root", default="/mnt/seagate10tb/florianpfaff/regional-content-frontier-pf-20260915")
    p.add_argument("--matched-root", default="/mnt/seagate10tb/florianpfaff/pf-matched-population-content-20260913")
    p.add_argument("--home-csv", default="/mnt/seagate10tb/florianpfaff/pf-recording-goal-content-20260913/home_metadata_qc.csv")
    p.add_argument("--dataset-root", default="/mnt/lexar4tb/datasets/pfeiffer-foster")
    p.add_argument("--detector-script", default="/home/florianpfaff/HippoReplayIMM-replay-geometry-hypotheses/scripts/select_pfeiffer_foster_speed_candidates.py")
    p.add_argument("--seed", type=int, default=2026091531)
    p.add_argument("--calibration-replicas", type=int, default=20)
    p.add_argument("--null-replicas", type=int, default=40)
    p.add_argument("--validation-replicas", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--session-limit", type=int)
    args = p.parse_args()
    if min(args.calibration_replicas, args.validation_replicas, args.workers) < 1 or args.null_replicas < 20:
        p.error("positive replica/worker counts; at least 20 null replicas")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {"frontier": Path(args.frontier_root)/"manifest.json", "matched": Path(args.matched_root)/"frozen_populations.json",
        "script": Path(__file__), "core": ROOT/"src/hipporeplayimm/selection_matched_regional.py", "detector": Path(args.detector_script),
        "protocol": ROOT/"docs/selection_matched_regional_calibration_protocol.md", "readout": ROOT/"src/hipporeplayimm/regional_content_frontier.py"}
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    config = {k:str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(status="running", parameters=config)
    json_write(args.output_dir/"manifest.json", manifest)
    sessions = read_csv(Path(args.frontier_root)/"sessions.csv")
    if args.session_limit:
        sessions = sessions.iloc[:args.session_limit]
    tasks = [(s.session, s.animal, config) for s in sessions.itertuples()]
    if args.workers == 1:
        completed = [worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed = list(pool.map(worker, tasks))
    pd.DataFrame(completed).to_csv(args.output_dir/"sessions.csv", index=False)
    for name in ("validation_and_null", "real_descriptive", "detector_equivalence"):
        rows = []
        for s in completed:
            frame = read_csv(Path(s["folder"])/(name+".csv"))
            if "session" not in frame:
                frame["session"] = s["session"]
            rows.append(frame)
        pd.concat(rows, ignore_index=True).to_csv(args.output_dir/(name+".csv"), index=False)
    final = build_script_provenance(input_paths=inputs, cwd=ROOT)
    assert final["input_file_sha256"] == manifest["input_file_sha256"]
    manifest.update(final, status="complete")
    json_write(args.output_dir/"manifest.json", manifest)


if __name__ == "__main__":
    main()
