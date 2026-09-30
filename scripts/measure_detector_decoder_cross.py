"""Cross detector and decoder populations on unchanged real and simulated trains."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.data import load_replay_session
from hipporeplayimm.detector_decoder_cross import (
    count_windows,
    detect,
    endpoint_windows,
    event_matches,
    factorial,
    make_populations,
    posterior_readout,
    simulate,
    tile_ids,
    truth_mass,
)
from hipporeplayimm.regional_blind_bank import Geometry
from hipporeplayimm.regional_content_mua_null import load_detector, stable_seed
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.measure_edge_support_content import occupied_graph

LIKELIHOODS = ("poisson", "conditional_multinomial")
DETECTOR = Path("/home/florianpfaff/HippoReplayIMM-replay-geometry-hypotheses/scripts/select_pfeiffer_foster_speed_candidates.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tetrode_ids(row, ids, metadata):
    if row.dataset == "tanni2022":
        return ids.astype(int) // 1000, {"mapping": "verified_native_electrode_times_1000_plus_manual_cluster"}
    source = Path(row.source_path)
    for filename, record in metadata["consumed_files"].items():
        if file_sha256(filename) != record["sha256"]:
            raise ValueError("PF raw source changed: " + filename)
    raw = load_replay_session(source)
    mapping = {int(cell): int(tt) for tt, cell in raw.tetrode_cell_ids}
    if len(mapping) != len(raw.tetrode_cell_ids) or not set(ids.astype(int)).issubset(mapping):
        raise ValueError("PF tetrode map incomplete or duplicate")
    return np.array([mapping[int(i)] for i in ids]), {"mapping": "native_tetrode_cell_ids", "source_path": str(source)}


def source_measure(folder, spikes, truth_tiles, ids, rates, tiles, pops, detector, speed_times, speeds, intervals, meta, duration=None):
    folder.mkdir()
    by_name = {p["name"]: p for p in pops}
    events, windows, counts, truths = {}, {}, {}, {}
    for p in pops:
        name, ix = p["name"], p["indices"]
        events[name] = detect(detector, spikes, ids[ix], speed_times, speeds, intervals)
        windows[name] = endpoint_windows(events[name])
        counts[name] = count_windows(spikes, ids, windows[name])
        truths[name] = truth_mass(truth_tiles, windows[name]) if truth_tiles is not None else np.full((len(windows[name]), 9), np.nan)
    if truth_tiles is not None:
        fixed = np.arange(duration // 2) * 2 + 1.0
        windows["fixed"] = np.column_stack((fixed - 0.01, fixed + 0.01))
        counts["fixed"] = count_windows(spikes, ids, windows["fixed"])
        truths["fixed"] = truth_mass(truth_tiles, windows["fixed"])
    events_table = pd.DataFrame([dict(**e, detector=p, **meta) for p, ev in events.items() for e in ev])
    if events_table.empty:
        events_table = pd.DataFrame(columns=["event_id", "event_start_s", "event_end_s", "event_peak_s", "detector", *meta])
    events_table.to_csv(folder / "detector_events.csv", index=False)
    np.savez_compressed(
        folder / "endpoint_counts.npz",
        **{f"{name}__{field}": value for name in windows for field, value in (("windows", windows[name]), ("counts", counts[name]), ("truth", truths[name]))},
    )
    summaries, frames = [], []
    for detection, w in windows.items():
        decoders = list(by_name) if detection in ("full", "fixed") else ["full", detection]
        for decoding in decoders:
            ix = by_name[decoding]["indices"]
            n, r = counts[detection][:, ix], rates[ix]
            for likelihood in LIKELIHOODS:
                result = posterior_readout(n, r, tiles, likelihood)
                truth = truths[detection]
                base = dict(**meta, detector=detection, decoder=decoding, likelihood=likelihood)
                frame = pd.DataFrame(
                    dict(event_id=np.arange(len(w)), start_s=w[:, 0], end_s=w[:, 1], n_spikes=result["spikes"], n_active_cells=result["active"], entropy=result["entropy"], **base)
                )
                for region in range(9):
                    frame[f"mass_{region}"] = result["regional"][:, region]
                    frame[f"truth_{region}"] = truth[:, region]
                    value = float(result["regional"][:, region].mean()) if len(w) else np.nan
                    actual = float(truth[:, region].mean()) if len(w) and truth_tiles is not None else np.nan
                    summaries.append(
                        dict(
                            **base,
                            region=region,
                            n_windows=len(w),
                            n_decoder_cells=len(ix),
                            posterior_mass=value,
                            true_mass=actual,
                            decoded_minus_truth=value - actual,
                            prior_mass=float(np.mean(tiles == region)),
                            silent_fraction=float(np.mean(result["spikes"] == 0)) if len(w) else np.nan,
                            mean_spikes=float(np.mean(result["spikes"])) if len(w) else np.nan,
                            mean_active_cells=float(np.mean(result["active"])) if len(w) else np.nan,
                            status="complete" if len(w) else "no_events",
                        )
                    )
                frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(folder / "window_readouts.csv.gz", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(folder / "readout_summary.csv", index=False)
    contrasts = []
    for p in pops[1:]:
        name = p["name"]
        for likelihood in LIKELIHOODS:
            for region in range(9):
                lookup = summary.loc[summary.likelihood.eq(likelihood) & summary.region.eq(region)].set_index(["detector", "decoder"])
                keys = [("full", "full"), (name, "full"), ("full", name), (name, name)]
                qff, qsf, qfs, qss = [float(lookup.loc[k, "posterior_mass"]) for k in keys]
                true_f, true_s = [float(lookup.loc[(d, "full"), "true_mass"]) for d in ("full", name)]
                contrasts.append(
                    dict(
                        **meta,
                        population=name,
                        family=p["family"],
                        side=p["side"],
                        repeat=p["repeat"],
                        likelihood=likelihood,
                        region=region,
                        q_full_full=qff,
                        q_subset_full=qsf,
                        q_full_subset=qfs,
                        q_subset_subset=qss,
                        n_full=int(lookup.loc[("full", "full"), "n_windows"]),
                        n_subset=int(lookup.loc[(name, name), "n_windows"]),
                        true_full_selected=true_f,
                        true_subset_selected=true_s,
                        true_selection_shift=true_s - true_f,
                        readout_bias_shift=(qss - true_s) - (qff - true_f),
                        **factorial(qff, qsf, qfs, qss),
                    )
                )
    pd.DataFrame(contrasts).to_csv(folder / "factorial_contrasts.csv", index=False)
    reference = np.array([[e["event_start_s"], e["event_end_s"]] for e in events["full"]]).reshape(-1, 2)
    matching, epochs = [], []
    for p in pops:
        name = p["name"]
        alt = np.array([[e["event_start_s"], e["event_end_s"]] for e in events[name]]).reshape(-1, 2)
        match, overlap = event_matches(reference, alt)
        for j, ref in enumerate(match):
            matching.append(
                dict(
                    **meta,
                    population=name,
                    event_id=j,
                    full_event_id=ref,
                    overlap_s=overlap[j],
                    endpoint_shift_ms=1000 * (windows[name][j, 1] - windows["full"][ref, 1]) if ref >= 0 else np.nan,
                )
            )
        if truth_tiles is not None:
            peak_times = np.array([e["event_peak_s"] for e in events[name]])
            for epoch in range(duration // 2):
                hits = (peak_times >= epoch * 2) & (peak_times < (epoch + 1) * 2)
                near = hits & (np.abs(peak_times - (epoch * 2 + 1)) <= 0.15)
                epochs.append(
                    dict(
                        **meta,
                        population=name,
                        family=p["family"],
                        side=p["side"],
                        repeat=p["repeat"],
                        epoch=epoch,
                        n_detections=int(hits.sum()),
                        n_gain_peak_detections=int(near.sum()),
                        true_central_occupancy=truths["fixed"][epoch, 4],
                    )
                )
    pd.DataFrame(matching, columns=[*meta, "population", "event_id", "full_event_id", "overlap_s", "endpoint_shift_ms"]).to_csv(folder / "event_matches.csv", index=False)
    pd.DataFrame(epochs, columns=[*meta, "population", "family", "side", "repeat", "epoch", "n_detections", "n_gain_peak_detections", "true_central_occupancy"]).to_csv(
        folder / "epoch_detection.csv", index=False
    )
    write_json(folder / "outputs.json", {p.name: file_sha256(p) for p in folder.iterdir() if p.is_file()})
    return dict(**meta, folder=str(folder), full_events=len(events["full"]), minimum_population_events=min(map(len, events.values())), populations=len(pops), status="complete")


def run_session(record, args):
    row = pd.Series(record)
    target = args.output_dir / f"{row.dataset}__{row.animal}__{row.session.split('/')[-1]}"
    target.mkdir()
    inputs = {"cache": Path(row.artifact_path), "cache_metadata": Path(row.artifact_path).with_suffix(".json")}
    if file_sha256(inputs["cache"]) != row.artifact_sha256:
        raise ValueError("cache hash mismatch")
    data = load_npz(inputs["cache"])
    metadata = json.loads(inputs["cache_metadata"].read_text())
    if metadata["unit_selection_uses_replay_content"] or metadata["candidate_selection"] != "all_source_candidates_no_replay_content_filter":
        raise ValueError("outcome-selected encoding input")
    mask, support = data["unit_qc_mask"].astype(bool), data["valid_spatial_bins"].astype(bool)
    ids, grid = data["cell_ids"][mask].astype(int), data["bin_centers_cm"][support]
    rates = np.maximum(data["rates_hz"][mask][:, support], 1e-4)
    bounds = np.array([grid.min(axis=0), grid.max(axis=0)])
    tiles = tile_ids(grid, bounds)
    tetrodes, tetrode_provenance = tetrode_ids(row, ids, metadata)
    pops, coverage, missing = make_populations(ids, rates, tiles == 4, tetrodes, np.random.default_rng(stable_seed(args.seed, row.dataset, row.animal, row.session, "populations")))
    write_json(
        target / "populations.json",
        {
            "populations": pops,
            "missing": missing,
            "tetrode_provenance": tetrode_provenance,
            "cell_ids": ids.tolist(),
            "tetrode_ids": tetrodes.tolist(),
            "coverage_score": coverage.tolist(),
            "dataset": row.dataset,
            "animal": row.animal,
            "session": row.session,
            "region": "central_3x3_tile",
            "bounds_cm": bounds.tolist(),
        },
    )
    pd.DataFrame(
        [
            {
                "name": p["name"],
                "family": p["family"],
                "side": p["side"],
                "repeat": p["repeat"],
                "n_cells": len(p["indices"]),
                "mean_cell_coverage": float(coverage[p["indices"]].mean()),
                "total_inside_rate": float(rates[p["indices"]][:, tiles == 4].mean(axis=1).sum()),
                "total_grid_mean_rate": float(rates[p["indices"]].mean(axis=1).sum()),
            }
            for p in pops
        ]
    ).to_csv(target / "population_summary.csv", index=False)
    np.savez_compressed(target / "encoding.npz", ids=ids, rates=rates, grid=grid, bounds=bounds, tiles=tiles, tetrodes=tetrodes, source_unit_mask=mask, source_grid_mask=support)
    detector = load_detector(args.detector_script)
    times, speeds = detector.position_speed(data["position"], 0.1)
    meta = {"dataset": row.dataset, "animal": row.animal, "session": row.session}
    records = []
    if not args.skip_real:
        records.append(
            source_measure(
                target / "real",
                data["spikes"],
                None,
                ids,
                rates,
                tiles,
                pops,
                detector,
                times,
                speeds,
                data["supported_run_intervals"],
                dict(**meta, source="real", generator="real", peak_gain=0, replicate=0),
            )
        )
        print(row.dataset, row.session, "real", records[-1]["full_events"], "events", flush=True)
    geometry = Geometry.from_grid(grid, occupied_graph(grid))
    if not (tiles[geometry.component] == 4).any() or (tiles[geometry.component] == 4).all():
        raise ValueError("simulation component lacks two regional classes")
    for generator in ("stationary", "moving"):
        for replicate in range(args.replicates):
            for peak in (3, 6):
                source = f"{generator}_rep{replicate}_gain{peak}"
                # Save the unchanged full train before any population detection.
                spikes, positions, means = simulate(
                    geometry,
                    rates,
                    ids,
                    generator,
                    peak,
                    args.duration_s,
                    np.random.default_rng(stable_seed(args.seed, row.dataset, row.animal, row.session, generator, replicate, peak)),
                )
                truth = tile_ids(positions, bounds)
                raw = target / (source + "_train.npz")
                np.savez_compressed(raw, spikes=spikes, truth_xy_cm=positions, truth_tiles=truth, expected_counts_by_epoch_cell=means, grid_component=geometry.component)
                records.append(
                    source_measure(
                        target / source,
                        spikes,
                        truth,
                        ids,
                        rates,
                        tiles,
                        pops,
                        detector,
                        np.array([0.0, args.duration_s]),
                        np.zeros(2),
                        np.array([[0.0, args.duration_s]]),
                        dict(**meta, source=source, generator=generator, peak_gain=peak, replicate=replicate),
                        args.duration_s,
                    )
                )
                print(row.dataset, row.session, source, records[-1]["full_events"], "events", flush=True)
                pd.DataFrame(records).to_csv(target / "sources.csv", index=False)
    pd.DataFrame(records).to_csv(target / "sources.csv", index=False)
    frozen = build_script_provenance(input_paths=inputs, cwd=ROOT)
    frozen.update(
        status="complete",
        **meta,
        decoder_cells=len(ids),
        raw_recorded_cells=len(data["cell_ids"]),
        population_count=len(pops),
        missing_families=missing,
        outputs={p.name: file_sha256(p) for p in target.iterdir() if p.is_file()},
        source_manifests={p.name: file_sha256(p / "outputs.json") for p in target.iterdir() if p.is_dir()},
        true_replay_labels_available=False,
        reference_population="all_RUN_qualified_sorted_units",
    )
    write_json(target / "manifest.json", frozen)
    return dict(**meta, status="complete", folder=str(target), cells=len(ids), populations=len(pops), missing_families="; ".join(missing), sources=len(records), reason="")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/replay-coverage-real-inputs-all33-20260905"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--detector-script", type=Path, default=DETECTOR)
    p.add_argument("--seed", type=int, default=2026091701)
    p.add_argument("--duration-s", type=int, default=600)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--session", action="append", help="Exact dataset:animal:session key; only for runtime preflight")
    p.add_argument("--skip-real", action="store_true")
    args = p.parse_args()
    if args.duration_s < 60 or args.duration_s % 2 or min(args.replicates, args.workers) < 1:
        p.error("even duration >=60; positive replicas and workers")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "sessions": args.input_dir / "coverage_input_sessions.csv",
        "detector": args.detector_script,
        "script": Path(__file__),
        "core": ROOT / "src/hipporeplayimm/detector_decoder_cross.py",
        "protocol": ROOT / "docs/detector_decoder_cross_protocol.md",
        "paths": ROOT / "src/hipporeplayimm/regional_blind_bank.py",
        "graph": ROOT / "scripts/measure_edge_support_content.py",
    }
    manifest = build_script_provenance(input_paths=paths, cwd=ROOT)
    manifest.update(status="running", started_at_utc=datetime.now(UTC).isoformat(), parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(args.output_dir / "manifest.json", manifest)
    sessions = pd.read_csv(paths["sessions"])
    if sessions.duplicated(["dataset", "animal", "session"]).any() or not sessions.status.eq("cached").all():
        raise ValueError("invalid source catalog")
    if args.session:
        keys = sessions.dataset + ":" + sessions.animal + ":" + sessions.session
        if not set(args.session).issubset(keys):
            raise ValueError("requested session missing")
        sessions = sessions[keys.isin(args.session)]
    sessions.to_csv(args.output_dir / "frozen_sessions.csv", index=False)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_session, r, args): r for r in sessions.to_dict("records")}
        for future in as_completed(jobs):
            r = jobs[future]
            try:
                result = future.result()
            except (ValueError, OSError, RuntimeError, KeyError, AssertionError, TypeError) as exc:
                result = {k: r[k] for k in ("dataset", "animal", "session")}
                result.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            results.append(result)
            pd.DataFrame(results).to_csv(args.output_dir / "sessions.csv", index=False)
            print("SESSION", result["session"], result["status"], result.get("reason", ""), flush=True)
    final = build_script_provenance(input_paths=paths, cwd=ROOT)
    if final["input_file_sha256"] != manifest["input_file_sha256"]:
        raise ValueError("code or protocol changed during run")
    manifest.update(
        status="complete_pending_audit" if all(r["status"] == "complete" for r in results) else "incomplete",
        completed_at_utc=datetime.now(UTC).isoformat(),
        sessions_requested=len(sessions),
        sessions_finished=len(results),
        results_sha256=file_sha256(args.output_dir / "sessions.csv"),
    )
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
