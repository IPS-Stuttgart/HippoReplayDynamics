"""Shared fixed-count, region-blind geometry bank; no real-data calibration."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.data import load_mat_variable
from hipporeplayimm.regional_blind_bank import BlindPath, Geometry, path_content, sample_conditional_event
from hipporeplayimm.regional_content_mua_null import stable_seed
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.measure_edge_support_content import occupied_graph

DELTAS = (.02, .04, .06, .1)
STRATA = [("stationary", 1.)] + [(k, s) for k in ("moving", "jumping") for s in (.5, 1., 2.)]
PREVALENCES = (.05, .15, .30, .5)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fit_anchors(frame):
    """Use observed within-event intervals only; never bridge event boundaries."""
    rows, pools = [], {}
    for session, s in frame.groupby("session"):
        lengths, intervals, raw_lengths, raw_intervals, speeds, bins = [], [], [], [], [], []
        for _, e in s.groupby("event_index"):
            e = e.sort_values("transition_index")
            times = e.transition_time_s.to_numpy(float)
            if len(times) < 2 or not (np.diff(e.transition_index) == 1).all():
                raise ValueError("need contiguous within-event transition rows")
            dt = float(np.round(np.median(np.diff(times)), 9))
            if dt <= 0 or not np.allclose(np.diff(times)[:-1], dt, rtol=0, atol=1e-8) or not (.5*dt-1e-8 <= np.diff(times)[-1] <= dt+1e-8):
                raise ValueError("nonuniform decoded time steps")
            bins.append(dt)
            speed = e.posterior_mean_step_speed_cm_s.to_numpy(float)
            if not np.isfinite(speed).all() or (speed < 0).any():
                raise ValueError("invalid empirical speed")
            step = speed * dt
            speeds.extend(speed[(step >= .1) & (step < 20)])
            ix = np.flatnonzero(step >= 20)
            raw_lengths.extend(step[ix])
            raw_intervals.extend(np.diff(times[ix]))
            starts = []
            for chunk in np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1):
                if len(chunk):
                    lengths.append(float(step[chunk].max()))
                    starts.append(float(times[chunk[0]]))
            intervals.extend(np.diff(starts))
        if not lengths or not intervals or not speeds:
            raise ValueError(f"missing within-session empirical pool: {session}")
        pools[session] = {"lengths": lengths, "intervals": intervals, "raw_lengths": raw_lengths,
                          "raw_intervals": raw_intervals, "moving_speed": float(np.median(speeds))}
        rows.append({"session": session, "events": s.event_index.nunique(), "transitions": len(s),
                     "time_bin_s": float(np.median(bins)), "large_step_episodes": len(lengths),
                     "renewal_intervals": len(intervals), "jump_length_median_cm": float(np.median(lengths)),
                     "interval_median_ms": 1000 * float(np.median(intervals)),
                     "raw_large_step_count": len(raw_lengths),
                     "raw_interval_median_ms": 1000 * float(np.median(raw_intervals)),
                     "moving_speed_cm_s": float(np.median(speeds))})
    return pd.DataFrame(rows), pools


def jobs(replicas):
    return [(phase, p, r) for phase in ("calibration", "null", "validation")
            for p in ((.5,) if phase == "calibration" else PREVALENCES) for r in range(replicas)]


def quota_labels(n, prevalence, rng):
    label = np.arange(n) < round(n * prevalence)
    rng.shuffle(label)
    return label


def numeric_dwell(path, center, duration):
    """Independent fine-grid integration, used on saved-path audit samples."""
    step = .00005
    t = np.arange(step / 2, duration, step)
    ages, nodes = np.asarray(path.ages), np.asarray(path.nodes)
    if path.kind == "moving":
        xy = np.column_stack([np.interp(t, ages, path.geometry.grid[nodes, j]) for j in (0, 1)])
    else:
        xy = path.geometry.grid[nodes[np.searchsorted(ages, t, side="right") - 1]]
    return float((np.linalg.norm(xy - center, axis=1) <= 20.).mean() * duration)


def prepare(args):
    output, source = Path(args.output_dir), Path(args.source_bank)
    if output.exists():
        raise FileExistsError("use a new output directory; existing banks are immutable")
    output.mkdir(parents=True)
    frame = read_csv(args.transitions)
    anchors, pools = fit_anchors(frame)
    anchors.to_csv(output / "empirical_anchors.csv", index=False)
    write_json(output / "empirical_pools.json", pools)
    inputs = {"transitions": args.transitions, "source_manifest": source / "manifest.json",
              "protocol": ROOT / "docs/regional_shared_bank_protocol.md",
              "generator": ROOT / "src/hipporeplayimm/regional_blind_bank.py", "builder": __file__}
    fitroot = Path(args.transitions).parent
    inputs["transition_manifest"] = fitroot / "first_order_imm_switch_location_manifest.json"
    for path in sorted(fitroot.parent.glob("first-order-imm-switch-location-shard-*/first_order_imm_switch_location_manifest.json")):
        inputs[path.parent.name] = path
    cohort, sessions = [], []
    selected = read_csv(source / "sessions.csv")
    if args.session_limit:
        selected = selected.iloc[:args.session_limit]
    for row in selected.itertuples():
        session = row.session
        folder = source / session.replace("/", "_")
        sm = json.loads((folder / "manifest.json").read_text())
        frozen = folder / "frozen_inputs.npz"
        # Source manifests hash all outputs; do not silently accept edited templates.
        source_hashes = sm["outputs"]
        if not source_hashes:
            raise ValueError("source output hashes missing")
        expected = source_hashes.get("frozen_inputs.npz", source_hashes.get(str(frozen)))
        if expected != file_sha256(frozen):
            raise ValueError(f"frozen source changed: {session}")
        data = load_npz(frozen)
        homepath = sm["input_file_paths"]["home"]
        if file_sha256(homepath) != sm["input_file_sha256"]["home"]:
            raise ValueError("Home geometry changed")
        h = read_csv(homepath).set_index("session").loc[session]
        center = [float(h.home_x_cm), float(h.home_y_cm)]
        np.testing.assert_array_equal(data["region"], np.linalg.norm(data["grid"] - center, axis=1) <= 20.)
        available = data["windows"][:, 1] - data["candidate_start"]
        chosen = np.flatnonzero(available + 1e-9 >= .1)
        for i, event in enumerate(data["candidate_ids"]):
            cohort.append({"session": session, "animal": row.animal, "source_index": i, "event_id": int(event),
                           "available_ms": available[i] * 1000, "included": bool(i in chosen),
                           "exclusion_reason": "" if i in chosen else "less_than_100ms_before_native_endpoint"})
        if args.event_limit:
            chosen = chosen[:args.event_limit]
        if not len(chosen):
            raise ValueError("empty common cohort")
        dest = output / session.replace("/", "_")
        dest.mkdir()
        # Frozen source arrays preserve all raw cells and confirmed legacy populations.
        np.savez_compressed(dest / "source_inputs.npz", **data)
        definitions = folder / "population_definitions.json"
        (dest / definitions.name).write_bytes(definitions.read_bytes())
        descriptor = Path(sm["input_file_paths"]["matched_checkpoint"]).parent / "early_unit_descriptors.csv"
        units = read_csv(descriptor).set_index("cell_id").reindex(data["cell_ids"])
        raw_path = sm["input_file_paths"]["Spike_Data.mat"]
        if file_sha256(raw_path) != sm["input_file_sha256"]["Spike_Data.mat"]:
            raise ValueError("raw unit metadata changed")
        native = np.asarray(load_mat_variable(raw_path, "Tetrode_Cell_IDs"))
        mapping = {int(cell): int(tt) for tt, cell in native}
        if len(mapping) != len(native) or not set(data["cell_ids"]).issubset(mapping):
            raise ValueError("missing native cell/tetrode mapping")
        known = units.tetrode.notna()
        np.testing.assert_array_equal(units.loc[known, "tetrode"], [mapping[int(c)] for c in units.index[known]])
        units["tetrode"] = [mapping[int(c)] for c in units.index]
        units["in_legacy_qc_table"] = known
        inputs[session + ":raw_cell_metadata"] = raw_path
        units.reset_index().to_csv(dest / "unit_metadata.csv", index=False)
        starts, ends, offsets, times = [], [], [0], []
        for i in chosen:
            a, b = data["template_offsets"][i:i+2]
            t = data["template_times"][a:b]
            times.append(t)
            offsets.append(offsets[-1] + len(t))
            starts.append(data["candidate_start"][i])
            ends.append(data["windows"][i, 1])
        np.savez_compressed(dest / "templates.npz", source_indices=chosen,
                            event_ids=data["candidate_ids"][chosen], starts=starts, endpoints=ends,
                            candidate_ends=data["candidate_end"][chosen], times=np.concatenate(times), offsets=offsets)
        config = {"session": session, "animal": row.animal, "center": center, "pool": pools[session],
                  "source_manifest": str(folder / "manifest.json"), "source_bank": str(folder),
                  "seed": args.seed, "replicas": args.replicas, "reference_proposals": args.reference_proposals,
                  "max_attempts": args.max_attempts}
        write_json(dest / "config.json", config)
        sessions.append({"session": session, "animal": row.animal, "templates": len(chosen), "folder": str(dest)})
        inputs[session + ":source"] = frozen
        inputs[session + ":manifest"] = folder / "manifest.json"
        inputs[session + ":units"] = descriptor
        inputs[session + ":home"] = homepath
        inputs[session + ":populations"] = definitions
    pd.DataFrame(cohort).to_csv(output / "cohort_manifest.csv", index=False)
    pd.DataFrame(sessions).to_csv(output / "sessions.csv", index=False)
    manifest = build_script_provenance(input_paths=inputs)
    manifest.update(created_at_utc=datetime.now(UTC).isoformat(), parameters=vars(args),
                    status="running", initial_development_only=True,
                    limitations=["decoder-dependent anchors from 108 selected clean events", "fixed total counts",
                                 "native windows frozen, not redetected from unconstrained synthetic activity",
                                 "too few replicas to certify coverage or false-flag calibration"])
    write_json(output / "manifest.json", manifest)
    return sessions


def run_task(task):
    folder, kind, scale, delta = task
    started = time.monotonic()
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    source, templates = load_npz(folder / "source_inputs.npz"), load_npz(folder / "templates.npz")
    geo = Geometry.from_grid(source["grid"], occupied_graph(source["grid"]))
    pool, center = config["pool"], np.array(config["center"])
    n = len(templates["event_ids"])
    out = folder / f"{kind}_scale{scale:g}_delta{round(delta*1000)}"
    out.mkdir()
    hist_edges = np.linspace(0, 1, 21)
    pre_hist, post_hist = np.zeros((2, 20), int), np.zeros((2, 20), int)
    reference = []
    if delta == .1:
        rng = np.random.default_rng(stable_seed(config["seed"], config["session"], kind, scale, "reference"))
        for i in range(config["reference_proposals"]):
            path = BlindPath(geo, kind, pool["moving_speed"], pool["lengths"], pool["intervals"], scale, rng).extend(.1)
            for d in DELTAS:
                c = path_content(path, center, 20., d)
                reference.append({"proposal": i, "delta_ms": round(d * 1000), "label": c["label"],
                                  "occupancy_fraction": c["occupancy_fraction"],
                                  "nonhome_nonhome": int(c["jump_crossings"][0, 0]),
                                  "nonhome_home": int(c["jump_crossings"][0, 1]),
                                  "home_nonhome": int(c["jump_crossings"][1, 0]),
                                  "home_home": int(c["jump_crossings"][1, 1]), "late_crossing": c["late_crossing"]})
        pd.DataFrame(reference).to_csv(out / "unconditioned_reference.csv", index=False)
    metadata, checks = [], []
    for j, (phase, prevalence, replica) in enumerate(jobs(config["replicas"])):
        seed = stable_seed(config["seed"], config["session"], kind, scale, delta, phase, prevalence, replica)
        wanted = quota_labels(n, prevalence, np.random.default_rng(stable_seed(seed, "labels")))
        contents, ids, ages, nodes, path_offsets, seeds, active = [], [], [], [], [0], [], []
        attempts, geo_reject, spike_reject, requested, realized = [], [], [], [], []
        for i, event in enumerate(templates["event_ids"]):
            event_seed = stable_seed(seed, "event", int(event))
            seeds.append(event_seed)
            a, b = templates["offsets"][i:i+2]
            template = {"start": templates["starts"][i], "end": templates["endpoints"][i],
                        "times": templates["times"][a:b]}
            result = sample_conditional_event(geo, kind, pool["moving_speed"], pool["lengths"],
                pool["intervals"], scale, center, delta, wanted[i], template, source["rates"],
                np.random.default_rng(event_seed), max_attempts=config["max_attempts"])
            p = result["path"]
            contents.append(result["contents"])
            ids.extend(result["identities"])
            active.append(result["active"])
            ages.extend(p.ages)
            nodes.extend(p.nodes)
            path_offsets.append(len(ages))
            requested.extend(p.requested_lengths)
            realized.extend(p.realized_lengths)
            attempts.append(result["attempts"])
            geo_reject.append(result["geometry_rejections"])
            spike_reject.append(result["spike_rejections"])
            pre_hist[int(wanted[i])] += np.histogram(result["pre_support_occupancy"], bins=hist_edges)[0]
            post_hist[int(wanted[i])] += np.histogram([result["contents"][DELTAS.index(delta)]["occupancy_fraction"]], bins=hist_edges)[0]
            if j == 0 and i < 8:
                error = max(abs(numeric_dwell(p, center, d) - result["contents"][k]["dwell_s"]) for k, d in enumerate(DELTAS))
                tolerance = .00005 * max(1, len(p.nodes))
                if error > tolerance:
                    raise AssertionError("independent occupancy audit failed")
                checks.append({"event_id": int(event), "max_dwell_error_s": error, "tolerance_s": tolerance, "status": "pass"})
        arrays = {key: np.asarray([[c[key] for c in e] for e in contents]) for key in contents[0][0]}
        np.testing.assert_array_equal(arrays["label"][:, DELTAS.index(delta)], wanted)
        if len(ids) != len(templates["times"]) or min(active) < np.ceil(.1 * len(source["cell_ids"])):
            raise AssertionError("fixed-count or native support invariant failed")
        np.savez_compressed(out / f"panel_{j:03}.npz", **arrays, identities=np.asarray(ids, np.uint16),
            path_ages=np.asarray(ages), path_nodes=np.asarray(nodes, np.int32), path_offsets=path_offsets,
            event_seeds=np.asarray(seeds, np.uint64), desired_label=wanted, active=active, attempts=attempts,
            geometry_rejections=geo_reject, spike_rejections=spike_reject,
            requested_jump_cm=requested, realized_jump_cm=realized)
        metadata.append({"panel": j, "phase": phase, "prevalence": prevalence, "replica": replica,
            "seed": str(seed), "events": n, "positive": int(wanted.sum()), "attempts": sum(attempts),
            "geometry_rejections": sum(geo_reject), "spike_rejections": sum(spike_reject)})
    pd.DataFrame(metadata).to_csv(out / "panels.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "numeric_truth_audit.csv", index=False)
    rows = [{"stage": stage, "label": label, "left": hist_edges[b], "right": hist_edges[b+1], "count": int(hist[label, b])}
            for stage, hist in (("label_conditioned_before_support", pre_hist), ("retained", post_hist))
            for label in (0, 1) for b in range(20)]
    pd.DataFrame(rows).to_csv(out / "occupancy_histogram.csv", index=False)
    write_json(out / "manifest.json", {"session": config["session"], "generator": kind, "scale": scale,
        "delta_ms": round(delta * 1000), "panels": len(metadata), "events_per_panel": n,
        "runtime_s": time.monotonic() - started, "status": "complete",
        "sha256": {p.name: file_sha256(p) for p in sorted(out.iterdir())}})
    return {"session": config["session"], "generator": kind, "scale": scale, "delta_ms": round(delta * 1000),
            "folder": str(out), "panels": len(metadata), "events_per_panel": n, "runtime_s": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", default="/mnt/seagate10tb/florianpfaff/selection-matched-regional-pf-v3-20260915")
    parser.add_argument("--transitions", default="/home/florianpfaff/HippoReplayIMM-replay-geometry-hypotheses/results/first-order-imm-switch-location-full/first_order_imm_switch_location_transition_table.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026091607)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reference-proposals", type=int, default=2000)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--session-limit", type=int)
    parser.add_argument("--event-limit", type=int)
    args = parser.parse_args()
    if min(args.replicas, args.workers, args.reference_proposals, args.max_attempts) <= 0:
        parser.error("positive sizes required")
    sessions = prepare(args)
    tasks = [(s["folder"], kind, scale, delta) for s in sessions for kind, scale in STRATA for delta in DELTAS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(len(rows), "/", len(tasks), row, flush=True)
    out = Path(args.output_dir)
    pd.DataFrame(rows).sort_values(["session", "generator", "scale", "delta_ms"]).to_csv(out / "strata.csv", index=False)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(status="generated_pending_independent_audit", strata=len(rows),
                    simulated_event_samples=sum(r["panels"] * r["events_per_panel"] for r in rows),
                    finished_at_utc=datetime.now(UTC).isoformat())
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
