#!/usr/bin/env python3
"""Regional content identification without conditional-independent subset errors."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.regional_content_bounds import (
    calibration_intervals,
    coverage_groups,
    decode_region,
    fit_latent_class,
    identification_set,
    observed_intervals,
    pattern_counts,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.measure_encoding_uncertainty_content import endpoint_arrays

SLACKS = (0., .05, .10, .20, 1.)
CONDITIONS = ("native", "eightfold", "map_drift_eightfold", "shared_assembly_eightfold")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def losses(q, truth):
    q = np.clip(q, 1e-12, 1-1e-12)
    return {"brier": float(np.mean((q-truth)**2)),
                "log_loss": float(-np.mean(truth*np.log(q)+(1-truth)*np.log1p(-q)))}


def simulate(rates, region, count_pool, n, rng, prevalence=.30, balanced=False,
             multiplier=1, assembly_fraction=0.):
    truth = np.repeat([0, 1], n//2) if balanced else rng.binomial(1, prevalence, n)
    if len(truth) != n:
        raise ValueError("balanced calibration needs even sample count")
    states = np.array([rng.choice(np.flatnonzero(region == bool(z))) for z in truth])
    totals = rng.choice(count_pool, n) * multiplier
    weight = rates[:, states].T.copy()
    weight /= weight.sum(axis=1, keepdims=True)
    assembly_truth = rng.binomial(1, .70, n)
    assembly_states = np.array([rng.choice(np.flatnonzero(region == bool(z))) for z in assembly_truth])
    assembly = rates[:, assembly_states].T.copy()
    assembly /= assembly.sum(axis=1, keepdims=True)
    weight = (1-assembly_fraction)*weight + assembly_fraction*assembly
    counts = np.array([rng.multinomial(int(total), p) for total, p in zip(totals, weight, strict=True)])
    np.testing.assert_array_equal(counts.sum(axis=1), totals)
    return {"counts": counts, "truth": truth, "states": states, "assembly_truth": assembly_truth,
                "assembly_states": assembly_states, "totals": totals}


def bound_rows(calls, calibration, meta, truth_prevalence=None, conditional=False):
    obs = observed_intervals(calls)
    hist = pattern_counts(calls)
    rows, event_rows = [], []
    for slack in SLACKS:
        model = identification_set(obs, calibration, slack)
        bound = model.prevalence()
        width = bound["upper"]-bound["lower"]
        rows.append(dict(**meta, transfer_slack=slack, **bound, width=width,
                         useful=bool(width <= .20),
                         covers_truth=(bound["lower"]-1e-7 <= truth_prevalence <= bound["upper"]+1e-7)
                         if truth_prevalence is not None else np.nan))
        if conditional and bound["status"] == "feasible":
            for j in np.flatnonzero(hist):
                interval = model.conditional(j)
                event_rows.append(dict(**meta, transfer_slack=slack, pattern=int(j),
                    events=int(hist[j]), **interval, width=interval["upper"]-interval["lower"],
                    excludes_half=bool(interval["lower"] > .5 or interval["upper"] < .5)))
    return rows, event_rows


def stage_zero(args, home, output, inputs):
    root = args.matched_root
    manifest_path = root/"manifest.json"
    old = json.loads(manifest_path.read_text())
    if file_sha256(args.home_csv) != old["input_file_sha256"]["home"]:
        raise ValueError("Home annotation source changed")
    inputs["matched_manifest"] = manifest_path
    source = root/"matched_event_content.csv.gz"
    inputs["matched_events"] = source
    df = pd.read_csv(source)
    rows = []
    for session, group in df.loc[df.family.eq("targeted")].groupby("session"):
        path = root/session.replace("/", "_")/"run_validation.npz"
        checkpoint = json.loads((path.parent/"freeze_checkpoint.json").read_text())
        if file_sha256(path) != checkpoint["run_validation_sha256"]:
            raise ValueError("matched grid archive changed")
        inputs["matched_grid_"+session] = path
        grid = load_npz(path)["grid_cm"]
        h = home.loc[session]
        prior = float((np.linalg.norm(grid-[h.home_x_cm, h.home_y_cm], axis=1) <= 20).mean())
        if not 0 < prior < 1:
            raise ValueError("Home region or complement empty")
        for side in ("high", "low"):
            g = group.copy()
            q = g[f"home_mass_{side}"].to_numpy()
            if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                raise ValueError("invalid saved posterior masses")
            clipped = np.clip(q, 1e-15, 1-1e-15)
            g["population"] = side
            g["prior_home_mass"], g["posterior_home_mass"] = prior, q
            g["log_bayes_factor"] = np.log(clipped)-np.log1p(-clipped)-np.log(prior)+np.log1p(-prior)
            g["bf_clipped"] = clipped != q
            g["silent"] = g[f"{side}_endpoint_spikes"].eq(0)
            g["positive_evidence"] = g.log_bayes_factor > np.log(3)
            g["negative_evidence"] = g.log_bayes_factor < -np.log(3)
            g["near_prior"] = g.log_bayes_factor.abs() <= np.log(3)
            rows.append(g[["animal", "session", "window_uid", "encoding", "cohort", "population",
                           "prior_home_mass", "posterior_home_mass", "log_bayes_factor", "bf_clipped",
                           "silent", "positive_evidence", "negative_evidence", "near_prior"]])
    events = pd.concat(rows, ignore_index=True)
    events.to_csv(output/"matched_home_bayes_factor_events.csv.gz", index=False)
    keys = ["animal", "session", "encoding", "cohort", "population"]
    summary = events.groupby(keys).agg(events=("window_uid", "size"), prior_home_mass=("prior_home_mass", "first"),
        mean_home_mass=("posterior_home_mass", "mean"), median_log_bf=("log_bayes_factor", "median"),
        positive_fraction=("positive_evidence", "mean"), negative_fraction=("negative_evidence", "mean"),
        near_prior_fraction=("near_prior", "mean"), silent_fraction=("silent", "mean"), clipped=("bf_clipped", "sum")).reset_index()
    summary.to_csv(output/"matched_home_bayes_factor_summary.csv", index=False)


def session_experiment(row, h, args, output, inputs, number):
    target = output/(row.animal+"__"+row.session.replace("/", "_"))
    target.mkdir()
    frozen_path = Path(row.artifact_dir)/"frozen_input.json"
    inputs["input_"+row.session] = frozen_path
    frozen = json.loads(frozen_path.read_text())
    folder = Path(frozen["edge_source"])
    for name, sha in frozen["source_outputs"].items():
        if file_sha256(folder/name) != sha:
            raise ValueError("edge-support input changed: "+str(folder/name))
    encoding = Path(frozen["freeze"]["encoding_path"])
    if file_sha256(encoding) != frozen["freeze"]["encoding_sha256"]:
        raise ValueError("encoder changed")
    enc = load_npz(encoding)
    enc_meta = json.loads((encoding.parent/"encoding_manifest.json").read_text())
    if not enc_meta["training_only"] or enc_meta["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("training-only encoder required")
    inputs["encoding_"+row.session] = encoding
    real, run = [load_npz(folder/f"{s}_audit.npz") for s in ("real", "run_q4")]
    rates, grid, ids = real["rates_hz"], real["grid_cm"], real["cell_ids"]
    for field in ("rates_hz", "grid_cm", "cell_ids"):
        np.testing.assert_array_equal(real[field], run[field])
    support, keep = enc["valid_spatial_bins"].astype(bool), enc["unit_qc_mask"].astype(bool)
    np.testing.assert_array_equal(ids, enc["cell_ids"][keep])
    np.testing.assert_allclose(rates, np.maximum(enc["rates_hz"][keep][:, support], 1e-4), rtol=0, atol=0)
    drift = np.maximum(enc["rates_second_half_hz"][keep][:, support], 1e-4)
    region = np.linalg.norm(grid-[h.home_x_cm, h.home_y_cm], axis=1) <= 20
    groups, coverage = coverage_groups(rates, region, ids)
    real_counts, _, real_starts = endpoint_arrays(real)
    run_counts, run_truth, run_starts = endpoint_arrays(run)
    run_labels = (np.linalg.norm(run_truth-[h.home_x_cm, h.home_y_cm], axis=1) <= 20).astype(int)
    rd, ud = [decode_region(n, rates, region, groups) for n in (real_counts, run_counts)]
    run_order = np.argsort(run_starts, kind="stable")
    cut = len(run_order)//2
    cal_idx, val_idx = run_order[:cut], run_order[cut:]
    run_ci, run_hits, run_totals = calibration_intervals(ud["calls"][cal_idx], run_labels[cal_idx])
    meta = {"dataset": row.dataset, "animal": row.animal, "session": row.session}
    arrays = {"rates": rates, "drift_rates": drift, "grid": grid, "cell_ids": ids, "region": region,
                  "coverage": coverage, "group_assignment": np.zeros(len(ids), int),
                  "real_counts": real_counts, "real_starts": real_starts, "real_event_ids": real["event_ids"],
                  "run_counts": run_counts, "run_starts": run_starts, "run_truth": run_truth, "run_labels": run_labels,
                  "run_cal_indices": cal_idx, "run_val_indices": val_idx, "run_ci": run_ci,
                  "run_cal_hits": run_hits, "run_cal_totals": run_totals}
    for k, g in enumerate(groups):
        arrays["group_assignment"][g] = k
    for name, readout in (("real", rd), ("run", ud)):
        arrays.update({f"{name}_{k}": v for k, v in readout.items()})
    rows, events, score_rows, cal_rows = [], [], [], []
    for source, calls, truth_pi in (("real", rd["calls"], None),
                                    ("heldout_run", ud["calls"][val_idx], float(run_labels[val_idx].mean()))):
        r, e = bound_rows(calls, run_ci, dict(**meta, source=source, calibration="run_first_half_q4", replicate=0),
                          truth_pi, conditional=(source == "real"))
        rows.extend(r)
        events.extend(e)
    rng = np.random.default_rng(args.seed + number*100000)
    calibration = {}
    for multiplier in (1, 8):
        generated = simulate(rates, region, real_counts.sum(axis=1), 2*args.calibration_per_class,
                             rng, balanced=True, multiplier=multiplier)
        readout = decode_region(generated["counts"], rates, region, groups)
        ci, hits, totals = calibration_intervals(readout["calls"], generated["truth"])
        calibration[multiplier] = ci
        arrays.update({f"cal{multiplier}_{k}": v for k, v in generated.items()})
        arrays.update({f"cal{multiplier}_{k}": v for k, v in readout.items()})
        arrays[f"cal{multiplier}_ci"] = ci
        for z, k, response in np.ndindex(hits.shape):
            cal_rows.append(dict(**meta, multiplier=multiplier, truth=z, group=k, response=response,
                count=int(hits[z, k, response]), total=int(totals[z]),
                lower=ci[z, k, response, 0], upper=ci[z, k, response, 1],
                mean_coverage=float(coverage[groups[k]].mean()), cells=len(groups[k])))
    r, e = bound_rows(rd["calls"], calibration[1], dict(**meta, source="real", calibration="synthetic_native", replicate=0),
                      conditional=True)
    rows.extend(r)
    events.extend(e)
    real_latent = fit_latent_class(rd["calls"], args.seed+number)
    arrays["real_latent_probability"] = real_latent["event_probability"]
    arrays["real_latent_theta"] = real_latent["theta"]
    score_rows.append(dict(**meta, source="real", replicate=0, truth_prevalence=np.nan,
        latent_prevalence=real_latent["pi"], naive_mean_mass=float(rd["mass"].mean()),
        converged=real_latent["converged"], multistart_range=real_latent["multistart_prevalence_range"]))
    np.savez_compressed(target/"real_and_calibration.npz", **arrays)
    inputs["edge_real_"+row.session] = folder/"real_audit.npz"
    inputs["edge_run_"+row.session] = folder/"run_q4_audit.npz"
    write_json(target/"source.json", dict(**meta, frozen_input=str(frozen_path),
        source_sha256=file_sha256(frozen_path), region="inferred Home, radius 20 cm", home_xy=[h.home_x_cm, h.home_y_cm],
        source_outputs=frozen["source_outputs"], seed=args.seed+number*100000))
    for condition in CONDITIONS:
        multiplier = 1 if condition == "native" else 8
        for replicate in range(args.replicates):
            generated = simulate(drift if condition == "map_drift_eightfold" else rates,
                region, real_counts.sum(axis=1), args.target_events, rng, multiplier=multiplier,
                assembly_fraction=.9 if condition == "shared_assembly_eightfold" else 0.)
            readout = decode_region(generated["counts"], rates, region, groups)
            r, _ = bound_rows(readout["calls"], calibration[multiplier],
                dict(**meta, source=condition, calibration=f"synthetic_{multiplier}x", replicate=replicate), .30)
            rows.extend(r)
            latent = fit_latent_class(readout["calls"], args.seed+number*1000+replicate)
            naive = readout["mass"].mean(axis=1)
            target_z = generated["truth"]
            score_rows.append(dict(**meta, source=condition, replicate=replicate, truth_prevalence=.30,
                empirical_truth_prevalence=float(target_z.mean()), latent_prevalence=latent["pi"],
                naive_mean_mass=float(naive.mean()), converged=latent["converged"],
                multistart_range=latent["multistart_prevalence_range"],
                **{"latent_"+k: v for k, v in losses(latent["event_probability"], target_z).items()},
                **{"naive_"+k: v for k, v in losses(naive, target_z).items()},
                **{"oracle_prevalence_"+k: v for k, v in losses(np.full(len(target_z), .30), target_z).items()}))
            np.savez_compressed(target/f"{condition}_{replicate}.npz", **generated, **readout,
                                latent_probability=latent["event_probability"], latent_theta=latent["theta"])
        print(row.session, condition, "complete", flush=True)
    for name, data in (("bounds", rows), ("conditional_bounds", events), ("scores", score_rows), ("calibration", cal_rows)):
        pd.DataFrame(data).to_csv(target/f"{name}.csv", index=False)
    write_json(target/"outputs.json", {p.name: file_sha256(p) for p in target.iterdir() if p.is_file()})
    return dict(**meta, status="complete", folder=str(target), real_events=len(real_counts), run_events=len(run_counts),
        calibration_run_home=int(run_totals[1]), calibration_run_elsewhere=int(run_totals[0]),
        cells=len(ids), prior_home_mass=float(region.mean()),
        all_four_silent_fraction=float(np.all(rd["spikes"] == 0, axis=1).mean()),
        all_four_neutral_fraction=float(np.all(rd["calls"] == 1, axis=1).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--uncertainty-root", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/encoding-uncertainty-content-20260914/pf"))
    p.add_argument("--matched-root", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/pf-matched-population-content-20260913"))
    p.add_argument("--home-csv", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/pf-recording-goal-content-20260913/home_metadata_qc.csv"))
    p.add_argument("--seed", type=int, default=2026091451)
    p.add_argument("--calibration-per-class", type=int, default=2000)
    p.add_argument("--target-events", type=int, default=512)
    p.add_argument("--replicates", type=int, default=12)
    p.add_argument("--session-limit", type=int)
    args = p.parse_args()
    if min(args.calibration_per_class, args.target_events, args.replicates) <= 0:
        p.error("positive sample counts required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sessions_path = args.uncertainty_root/"measurement_sessions.csv"
    sessions = pd.read_csv(sessions_path)
    if not sessions.status.eq("complete").all() or sessions.session.duplicated().any():
        raise ValueError("incomplete or duplicated source sessions")
    if args.session_limit:
        sessions = sessions.iloc[:args.session_limit]
    home = pd.read_csv(args.home_csv).set_index("session", verify_integrity=True)
    inputs = {"sessions": sessions_path, "home": args.home_csv, "script": Path(__file__),
                  "kernel": ROOT/"src/hipporeplayimm/regional_content_bounds.py",
                  "protocol": ROOT/"docs/regional_content_bounds_protocol.md",
                  "endpoint_helper": ROOT/"scripts/measure_encoding_uncertainty_content.py"}
    provenance = build_script_provenance(input_paths=inputs)
    write_json(args.output_dir/"manifest.json", dict(**provenance, status="running", created_at_utc=datetime.now(UTC).isoformat(),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        independent_view_errors_assumed=False, calibration_transfer_required=True,
        real_bounds_have_certified_biological_coverage=False))
    stage_zero(args, home, args.output_dir, inputs)
    results = []
    for number, row in enumerate(sessions.itertuples()):
        results.append(session_experiment(row, home.loc[row.session], args, args.output_dir, inputs, number))
        pd.DataFrame(results).to_csv(args.output_dir/"sessions.csv", index=False)
    for name in ("bounds", "conditional_bounds", "scores", "calibration"):
        pd.concat([pd.read_csv(Path(r["folder"])/f"{name}.csv") for r in results], ignore_index=True).to_csv(args.output_dir/f"{name}.csv", index=False)
    final = json.loads((args.output_dir/"manifest.json").read_text())
    final.update(build_script_provenance(input_paths=inputs))
    for key, digest in provenance["input_file_sha256"].items():
        if final["input_file_sha256"][key] != digest:
            raise ValueError("input changed during experiment")
    final.update(status="complete", finished_at_utc=datetime.now(UTC).isoformat(),
                 session_count=len(results), animal_count=len({r["animal"] for r in results}))
    write_json(args.output_dir/"manifest.json", final)
    print("complete", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
