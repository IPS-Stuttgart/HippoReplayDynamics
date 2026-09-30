#!/usr/bin/env python3
"""Expanded held-out RUN calibration and full-population regional discrimination."""
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

from hipporeplayimm.encoding import _speed_cm_s
from hipporeplayimm.regional_content_frontier import (
    block_histograms,
    bootstrap_compatibility,
    calls_from_bf,
    discrimination,
    generate_counts,
    regional_log_bf,
    thin_counts,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz, recount, seed


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def expanded_run(data, ids, home):
    pos = data["position"]
    t, xy = pos[:, 0], pos[:, 1:3]
    speed = _speed_cm_s(t, xy)
    begin, end = float(data["training_end_s"]), float(data["run_bounds_s"][1])
    starts = begin+.02*np.arange(int(np.floor((end-begin)/.02)))
    rows, boundary = [], 0
    for a in starts:
        b = a+.02
        left, right = np.searchsorted(t, [a, b])
        left, right = max(0, left-1), min(len(t), right+1)
        if right-left < 2 or t[left] > a or t[right-1] < b:
            continue
        if np.max(np.diff(t[left:right])) > .1 or not np.isfinite(xy[left:right]).all():
            continue
        if not ((speed[left:right] >= 10) & (speed[left:right] <= 200)).all():
            continue
        if not np.any((a >= data["supported_run_intervals"][:, 0]) & (b <= data["supported_run_intervals"][:, 1])):
            continue
        times = a+(np.arange(20)+.5)*.001
        trace = np.column_stack([np.interp(times, t, xy[:, d]) for d in range(2)])
        inside = np.linalg.norm(trace-home, axis=1) <= 20
        if inside.any() and not inside.all():
            boundary += 1
            continue
        rows.append([a, b, *trace.mean(axis=0), int(inside[0])])
    if len(rows) < 100:
        raise ValueError("too few held-out RUN windows")
    windows = np.asarray(rows)
    bouts = np.cumsum(np.r_[True, np.diff(windows[:, 0]) > 1.0])-1
    n = recount(data["spikes"], ids, windows[:, 0], windows[:, 1])
    return windows, n, bouts, boundary


def terminal_counts(data, ids, ms):
    mask = np.isin(data["cell_ids"], ids)
    np.testing.assert_array_equal(data["cell_ids"][mask], ids)
    counts, keys = [], []
    width = ms//5
    for j, key in enumerate(data["candidate_event_indices"]):
        a, b = data["candidate_offsets"][j:j+2]
        good = np.flatnonzero(np.isclose(data["candidate_base_durations_s"][a:b], .005, rtol=0, atol=1e-9))
        if not np.array_equal(good, np.arange(len(good))):
            raise ValueError("nonterminal partial base bin")
        if len(good) < width:
            continue
        counts.append(data["candidate_base_counts"][a+good[-width:]][:, mask].sum(axis=0))
        keys.append(int(key))
    return np.asarray(counts), np.asarray(keys)


def populations(rates, region, ids, session, matched, rng):
    score = rates[:, region].mean(axis=1)/rates.mean(axis=1)
    order = np.lexsort((ids, score))
    peak_home = region[rates.argmax(axis=1)]
    home = np.flatnonzero(peak_home)
    variants = {"full": np.arange(len(ids)), "coverage_low_half": order[:len(ids)//2],
                "coverage_high_half": order[-(len(ids)//2):]}
    for fraction, tag in ((.5, "half"), (1., "all")):
        remove = home[np.argsort(score[home], kind="stable")][-int(np.ceil(len(home)*fraction)):] if len(home) else []
        variants["remove_"+tag+"_home_peak"] = np.setdiff1d(np.arange(len(ids)), remove)
        random = rng.choice(len(ids), len(remove), replace=False)
        variants["remove_random_size_matched_"+tag] = np.setdiff1d(np.arange(len(ids)), random)
    metadata = []
    item = next(x for x in matched["sessions"] if x["session"] == session)
    pair = next((x for x in item["selected"] if x["family"] == "targeted"), None)
    if pair:
        old = load_npz(Path(matched["root"])/session.replace("/", "_")/"run_validation.npz")["cell_ids"]
        for side in ("high", "low"):
            selected = old[np.asarray(pair[side]["indices"], int)]
            variants["frozen_targeted_"+side+"_qc_intersection"] = np.flatnonzero(np.isin(ids, selected))
            metadata.append({"population": "frozen_targeted_"+side+"_qc_intersection",
                                 "original_cells": len(selected), "lost_cell_ids": selected[~np.isin(selected, ids)].tolist()})
    return variants, peak_home, metadata


def measure_session(row, args, inputs):
    folder = args.output_dir/row.session.replace("/", "_")
    folder.mkdir()
    previous = Path(row.folder)
    source_path = previous/"source.json"
    source = json.loads(source_path.read_text())
    frozen_path = Path(source["frozen_input"])
    frozen = json.loads(frozen_path.read_text())
    path = Path(frozen["freeze"]["encoding_path"])
    if file_sha256(path) != frozen["freeze"]["encoding_sha256"]:
        raise ValueError("changed encoder")
    info_path = path.parent/"encoding_manifest.json"
    info = json.loads(info_path.read_text())
    if not info["training_only"] or info["holdout_spikes_used_for_rate_or_unit_selection"]:
        raise ValueError("encoder leakage")
    data = load_npz(path)
    prior = load_npz(previous/"real_and_calibration.npz")
    recorded = json.loads((previous/"outputs.json").read_text())
    if file_sha256(previous/"real_and_calibration.npz") != recorded["real_and_calibration.npz"]:
        raise ValueError("previous readout archive changed")
    ids, rates, region = prior["cell_ids"], prior["rates"], prior["region"]
    keep, support = data["unit_qc_mask"], data["valid_spatial_bins"]
    np.testing.assert_array_equal(ids, data["cell_ids"][keep])
    np.testing.assert_array_equal(rates, np.maximum(data["rates_hz"][keep][:, support], 1e-4))
    for key, value in {"source": source_path, "encoder": path, "encoder_manifest": info_path,
                       "previous": previous/"real_and_calibration.npz"}.items():
        inputs[row.session+":"+key] = value
    native, native_keys = terminal_counts(data, ids, 20)
    np.testing.assert_array_equal(native, prior["real_counts"])
    np.testing.assert_array_equal(native_keys, prior["real_event_ids"])
    windows, run_n, bouts, boundary = expanded_run(data, ids, np.array(source["home_xy"]))
    truth = windows[:, 4].astype(int)
    cal = bouts % 2 == 0
    rng = np.random.default_rng(seed(args.seed, row.session, "RUN"))
    desired = rng.choice(native.sum(axis=1), len(windows))
    thinned, retention, unattainable = thin_counts(run_n, desired, rng)
    native_bf = regional_log_bf(native, rates, region)
    banks = {"native_run": regional_log_bf(run_n, rates, region),
             "thinned_frozen_decoder": regional_log_bf(thinned, rates, region),
             "thinned_exposure_adjusted": regional_log_bf(thinned, rates, region, .02*retention)}
    meta = {"animal": row.animal, "session": row.session}
    audit = dict(run_windows=windows, run_counts=run_n, run_bouts=bouts, calibration=cal,
                 thinned_counts=thinned, retention_probability=retention, desired_counts=desired,
                 real_counts=native, real_log_bf=native_bf, real_event_ids=native_keys,
                 real_starts=prior["real_starts"], cell_ids=ids, rates=rates, region=region,
                 grid=prior["grid"], **banks)
    metrics, bounds, fits = [], [], []
    for name, bf in banks.items():
        total = run_n.sum(axis=1) if name == "native_run" else thinned.sum(axis=1)
        for subset, mask in (("calibration", cal), ("validation", ~cal)):
            metrics.append(dict(**meta, method=name, subset=subset, windows=int(mask.sum()),
                home_windows=int(truth[mask].sum()), bouts=len(np.unique(bouts[mask])),
                home_bouts=len(np.unique(bouts[mask & (truth == 1)])),
                mean_spikes=float(total[mask].mean()), silent_fraction=float((total[mask] == 0).mean()),
                **discrimination(bf[mask], truth[mask])))
        if name == "thinned_exposure_adjusted":
            continue  # These varied exposures are not the real endpoint decoder.
        calls = calls_from_bf(bf, total)
        cal_blocks = block_histograms(calls[cal], bouts[cal], truth[cal])
        for target in ("real", "heldout_run"):
            if target == "real":
                target_calls = calls_from_bf(native_bf, native.sum(axis=1))
                blocks = np.floor((prior["real_starts"]-prior["real_starts"].min())/30).astype(int)
                pi = np.nan
            else:
                target_calls, blocks = calls[~cal], bouts[~cal]
                pi = float(truth[~cal].mean())
            target_blocks = block_histograms(target_calls, blocks)
            fit, intervals, trace = bootstrap_compatibility(cal_blocks, target_blocks,
                np.random.default_rng(seed(args.seed, row.session, name, target)), args.bootstrap)
            fits.append(dict(**meta, calibration=name, target=target, empirical_truth_prevalence=pi,
                             calibration_blocks=len(cal_blocks), target_blocks=len(target_blocks), **fit))
            bounds.extend(dict(**meta, calibration=name, target=target, empirical_truth_prevalence=pi, **x) for x in intervals)
            audit.update({name+"_"+target+"_"+key: value for key, value in trace.items()})
    np.savez_compressed(folder/"run_calibration_audit.npz", **audit)
    matched = json.loads(args.matched_populations.read_text())
    matched["root"] = str(args.matched_populations.parent)
    matched_grid = args.matched_populations.parent/row.session.replace("/", "_")/"run_validation.npz"
    if matched_grid.exists():
        inputs[row.session+":matched_grid"] = matched_grid
    variants, peak_home, population_metadata = populations(rates, region, ids, row.session, matched,
        np.random.default_rng(seed(args.seed, row.session, "populations")))
    baseline_rate = .5*(rates[:, region].sum(axis=0).mean()+rates[:, ~region].sum(axis=0).mean())
    gain = float(native.sum(axis=1).mean()/(.02*baseline_rate))
    if gain <= 0:
        raise ValueError("zero native spike support")
    records, traces = [], {}
    population_table = []
    for name, indices in variants.items():
        population_table.append(dict(**meta, population=name, cells=len(indices), home_peak_cells=int(peak_home[indices].sum()),
                                     cell_ids=";".join(map(str, ids[indices]))))
        if not len(indices):
            continue
        for ms in ((20, 40, 100) if name == "full" else (20,)):
            counts, event_keys = terminal_counts(data, ids, ms)
            pool = counts[:, indices].sum(axis=1)
            for multiplier in ((1, 2, 4, 8) if ms == 20 else (1,)):
                for generator in ("poisson", "conditional_multinomial"):
                    key = f"{name}__{ms}__{multiplier}__{generator}"
                    current_seed = seed(args.seed, row.session, key)
                    n, z, states, requested = generate_counts(rates[indices], region, pool, args.per_class,
                        np.random.default_rng(current_seed), generator, multiplier, gain, ms/1000)
                    matched_bf = regional_log_bf(n, rates[indices], region, ms/1000*gain*multiplier,
                                                conditional=(generator == "conditional_multinomial"))
                    frozen_bf = regional_log_bf(n, rates[indices], region, ms/1000)
                    for readout, score in (("matched_likelihood", matched_bf), ("frozen_poisson", frozen_bf)):
                        records.append(dict(**meta, population=name, duration_ms=ms, multiplier=multiplier,
                            generator=generator, readout=readout, cells=len(indices),
                            home_peak_cells=int(peak_home[indices].sum()), source_events=len(event_keys),
                            source_events_too_short=len(native_keys)-len(event_keys), simulation_seed=current_seed,
                            generated_windows=len(n), mean_spikes=float(n.sum(axis=1).mean()),
                            mean_active_cells=float(np.count_nonzero(n, axis=1).mean()),
                            silent_fraction=float((n.sum(axis=1) == 0).mean()),
                            **discrimination(score, z)))
                    for field, value in {"counts": n, "truth": z, "states": states, "requested": requested,
                                             "indices": indices, "matched_log_bf": matched_bf, "frozen_log_bf": frozen_bf}.items():
                        traces[key+"___"+field] = value
    np.savez_compressed(folder/"synthetic_frontier_audit.npz", **traces)
    tables = {"run_calibration": metrics, "prevalence_compatibility": bounds, "prevalence_fits": fits,
              "discrimination_frontier": records, "populations": population_table}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(folder/(name+".csv"), index=False)
    summary = dict(**meta, real_events=len(native), run_windows=len(windows), home_windows=int(truth.sum()),
        running_bouts=len(np.unique(bouts)), home_bouts=len(np.unique(bouts[truth == 1])),
        boundary_crossing_run_windows_excluded=boundary, training_end_s=float(data["training_end_s"]),
        first_run_window_s=float(windows[0, 0]), last_run_window_s=float(windows[-1, 1]),
        mean_native_endpoint_spikes=float(native.sum(axis=1).mean()), mean_run_spikes=float(run_n.sum(axis=1).mean()),
        mean_desired_spikes=float(desired.mean()), mean_thinned_spikes=float(thinned.sum(axis=1).mean()),
        thinning_unattainable_fraction=float(unattainable.mean()), simulation_gain=gain,
        regional_grid_prior=float(region.mean()), status="complete", folder=str(folder))
    save_json(folder/"session.json", dict(**summary, matched_population_changes=population_metadata))
    save_json(folder/"outputs.json", {p.name: file_sha256(p) for p in folder.iterdir() if p.is_file()})
    print(row.session, "complete", "RUN", len(windows), "Home", int(truth.sum()), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-root", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/regional-content-bounds-pf-20260914"))
    p.add_argument("--matched-populations", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/pf-matched-population-content-20260913/frozen_populations.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026091501)
    p.add_argument("--per-class", type=int, default=512)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--session-limit", type=int)
    args = p.parse_args()
    if args.per_class < 20 or args.bootstrap < 20:
        p.error("at least 20 observations per class and bootstrap repetitions")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {"sessions": args.prior_root/"sessions.csv", "previous_manifest": args.prior_root/"manifest.json",
                  "matched_populations": args.matched_populations, "script": Path(__file__),
                  "core": ROOT/"src/hipporeplayimm/regional_content_frontier.py",
                  "bounds_core": ROOT/"src/hipporeplayimm/regional_content_bounds.py",
                  "source_helpers": ROOT/"scripts/audit_edge_support_content.py",
                  "protocol": ROOT/"docs/regional_content_frontier_protocol.md"}
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", started_at_utc=datetime.now(UTC).isoformat(),
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    calibrated_biological_prevalence=False, independent_windows_assumed_for_real_bounds=False)
    save_json(args.output_dir/"manifest.json", manifest)
    sessions = pd.read_csv(inputs["sessions"])
    if not sessions.status.eq("complete").all() or sessions.session.duplicated().any():
        raise ValueError("incomplete/duplicate source")
    if args.session_limit:
        sessions = sessions.iloc[:args.session_limit]
    rows = []
    for row in sessions.itertuples():
        rows.append(measure_session(row, args, inputs))
        pd.DataFrame(rows).to_csv(args.output_dir/"sessions.csv", index=False)
    for name in ("run_calibration", "prevalence_compatibility", "prevalence_fits", "discrimination_frontier", "populations"):
        pd.concat([pd.read_csv(Path(row["folder"])/(name+".csv")) for row in rows], ignore_index=True).to_csv(args.output_dir/(name+".csv"), index=False)
    final = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for key, value in manifest["input_file_sha256"].items():
        if final["input_file_sha256"][key] != value:
            raise ValueError("input changed during run: "+key)
    manifest.update(final, status="complete", finished_at_utc=datetime.now(UTC).isoformat())
    save_json(args.output_dir/"manifest.json", manifest)


if __name__ == "__main__":
    main()

