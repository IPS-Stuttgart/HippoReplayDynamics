#!/usr/bin/env python3
"""Frozen recording-intervention diagnostic of represented destinations."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import softmax
from scipy.spatial import cKDTree

from hipporeplayimm.data import load_mat_variable
from hipporeplayimm.replay_coverage_shuffle_baseline import edge_support, screen_maps
from scripts._provenance import build_script_provenance, file_sha256

RADII = (10, 20, 30)


def infer_home(sequence, position):
    seq, pos = np.asarray(sequence, float), np.asarray(position, float)
    if (seq.ndim != 2 or seq.shape[1] != 2 or len(seq) < 3
            or not np.isfinite(seq).all() or np.any(np.diff(seq[:, 0]) <= 0)
            or np.any(seq[:, 1] != np.floor(seq[:, 1]))):
        raise ValueError("invalid well sequence")
    pairs = [{int(a), int(b)} for a, b in zip(seq[:-1, 1], seq[1:, 1]) if a != b]
    candidates = set.intersection(*pairs) if pairs else set()
    if len(candidates) != 1:
        raise ValueError("Home is not uniquely identifiable")
    home = candidates.pop()
    visits = []
    for i in range(len(seq) - 1):
        if seq[i, 1] != home:
            continue
        t = seq[i + 1, 0]
        xy = pos[(pos[:, 0] >= t - 1) & (pos[:, 0] <= t), 1:3]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) >= 3:
            x, y = np.median(xy, axis=0)
            visits.append(dict(next_fill_s=t, x_cm=x, y_cm=y, samples=len(xy)))
    if len(visits) < 5:
        raise ValueError("insufficient Home coordinate support")
    table = pd.DataFrame(visits)
    center = table[["x_cm", "y_cm"]].median().to_numpy()
    table["radial_error_cm"] = np.linalg.norm(table[["x_cm", "y_cm"]] - center, axis=1)
    p75 = table.radial_error_cm.quantile(.75)
    if p75 > 15:
        raise ValueError("inconsistent Home coordinates")
    meta = dict(home_id=home, home_x_cm=center[0], home_y_cm=center[1],
                visits=len(table), visit_p75_error_cm=p75,
                visit_max_error_cm=table.radial_error_cm.max(),
                annotation_status="inferred_from_fill_sequence_and_position_not_author_label")
    return meta, table


def decode(counts, rates):
    counts, rates = np.asarray(counts), np.asarray(rates, float)
    if (counts.ndim != 2 or rates.ndim != 2 or counts.shape[1] != rates.shape[0]
            or min(rates.shape) < 1 or not np.isfinite(rates).all() or (rates <= 0).any()
            or not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()):
        raise ValueError("finite nonnegative integer counts and positive rates required")
    ll = csr_matrix(counts, dtype=np.float64) @ np.log(rates) - .02 * rates.sum(axis=0)
    return softmax(ll, axis=1)


def longest_segment(path, grid, counts):
    valid = edge_support(counts, False)
    best, current = [], []
    for i in range(len(path)):
        connected = bool(current) and np.linalg.norm(grid[path[i]] - grid[path[i-1]]) < 20 - 1e-9
        if not valid[i]:
            current = []
        elif connected:
            current.append(i)
        else:
            current = [i]
        if len(current) > len(best):
            best = current.copy()
    return np.asarray(best, int)


def home_features(p, grid, home):
    distances = np.linalg.norm(grid - home, axis=1)
    return {**{f"home_mass_r{r}": float(p[distances <= r].sum()) for r in RADII},
            "home_map": bool(distances[p.argmax()] <= 20)}


def decompose(frame):
    f, h = frame.accepted_full.astype(bool), frame.accepted_half.astype(bool)
    if not f.any() or not h.any():
        return dict(status="missing_accepted_group", decoding=np.nan, composition=np.nan,
                    timing=np.nan, total=np.nan, reference=np.nan, reselected=np.nan)
    reference = frame.loc[f, "trajectory_home_mass_full"].mean()
    same = frame.loc[f, "trajectory_home_mass_half"].mean()
    selected = frame.loc[h, "trajectory_home_mass_half"].mean()
    reselected = frame.loc[h, "own_home_mass_half"].mean()
    return dict(status="complete", decoding=same-reference, composition=selected-same,
                timing=reselected-selected, total=reselected-reference,
                reference=reference, reselected=reselected)


def verify(path, expected, inputs):
    path = Path(path)
    measured = file_sha256(path)
    if expected is not None and measured != expected:
        raise ValueError(f"input hash changed: {path}")
    inputs[str(path)] = measured
    return path


def simulate(rates, grid, home, specs, seed, n=200):
    rng = np.random.default_rng(seed)
    tree = cKDTree(grid)
    distance = np.linalg.norm(grid-home, axis=1)
    pools = [np.flatnonzero(distance > 30), np.flatnonzero(distance <= 10)]
    if any(len(x) == 0 for x in pools):
        raise ValueError("simulation endpoint pools unavailable")
    rows = []
    for i in range(n):
        target = i % 2
        for _ in range(10000):
            end = grid[rng.choice(pools[target])]
            angle = rng.uniform(0, 2*np.pi)
            start = end - 80*np.array([np.cos(angle), np.sin(angle)])
            base_xy = start + ((np.arange(40)+.5)/40)[:, None]*(end-start)
            error, state = tree.query(base_xy)
            if np.max(error) <= 10:
                break
        else:
            raise ValueError("cannot generate supported fixed-speed path")
        base = rng.poisson(rates[:, state].T*.005)
        counts = np.stack([base[j:j+4].sum(axis=0) for j in range(37)])
        truth = base_xy[-4:].mean(axis=0)
        for spec in specs:
            ix = spec["indices"]
            p = decode(counts[-1:, ix], rates[ix])[0]
            feat = home_features(p, grid, home)
            rows.append(dict(simulation_event=i, true_home=bool(target),
                cell_fraction=spec["cell_fraction"], population_replicate=spec["population_replicate"],
                endpoint_error_cm=float(np.linalg.norm(p@grid-truth)),
                true_endpoint_x_cm=truth[0], true_endpoint_y_cm=truth[1],
                full_population_spikes=int(base.sum()), **feat))
    return pd.DataFrame(rows)


def process(record, args, index):
    session, animal = record["session"], record["animal"]
    label = session.replace("/", "_")
    out = args.output_dir / label
    out.mkdir()
    inputs = {}
    source = record["source"]
    with np.load(verify(record["input_arrays_path"], record["input_arrays_sha256"], inputs)) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(verify(source["source_cache_path"], source["source_cache_sha256"], inputs)) as z:
        position = z["position"]
    sequence_path = verify(args.dataset_root / session / "Well_Sequence.mat", None, inputs)
    seq = np.atleast_2d(load_mat_variable(sequence_path, "Well_Sequence"))
    meta, visits = infer_home(seq, position)
    home = np.array([meta["home_x_cm"], meta["home_y_cm"]])
    grid = arrays["grid_cm"]
    rates = arrays["rates_hz"][:, arrays["support"]]
    windows = pd.read_csv(verify(source["windows_path"], source["windows_sha256"], inputs), float_precision="round_trip")
    windows = windows[windows.eligible & windows.window_variant.eq("detected_core")].sort_values("window_uid").reset_index(drop=True)
    if not np.array_equal(windows.window_uid.to_numpy(str), arrays["window_uids"]):
        raise ValueError("window order differs from frozen input")
    metrics = pd.read_csv(verify(record["metrics_path"], record["metrics_sha256"], inputs))
    decisions = metrics[metrics.observation.eq("original_order") & metrics.bin_filter.eq("edge_only") & metrics.min_frames.eq(10)]
    specs = record["population_specs"]
    reference, rows = {}, []
    if specs[0]["cell_fraction"] != 1 or len(specs) != 4:
        raise ValueError("expected full population and three frozen halves")
    for spec, artifact in zip(specs, record["shuffle_files"], strict=True):
        ix = np.asarray(spec["indices"], int)
        if not np.array_equal(arrays["cell_ids"][ix], spec["cell_ids"]):
            raise ValueError("frozen cell identities changed")
        with np.load(verify(artifact["path"], artifact["sha256"], inputs)) as z:
            saved_path = z["original_path"]
        dec = decisions[decisions.cell_fraction.eq(spec["cell_fraction"]) & decisions.population_replicate.eq(spec["population_replicate"])].set_index("window_uid", verify_integrity=True)
        for w in windows.itertuples():
            if w.detector != "source_high_mua":
                continue
            a, b = arrays["frame_offsets"][2*w.Index:2*w.Index+2]
            counts = arrays["frame_counts"][a:b][:, ix]
            p = decode(counts, rates[ix])
            path = p.argmax(axis=1)
            if not np.array_equal(path, saved_path[a:b]):
                raise ValueError(f"original MAP reconstruction failed {session} {w.window_uid}")
            d = dec.loc[w.window_uid]
            if bool(screen_maps(path[None], grid, counts)[0, 0]) != bool(d.geometric_pass):
                raise ValueError("geometry reconstruction failed")
            segment = longest_segment(path, grid, counts)
            accepted = bool(d["accepted_alpha_0.02"])
            if spec["cell_fraction"] == 1:
                support = np.flatnonzero(counts.sum(axis=1) >= 2)
                reference[w.window_uid] = (p, segment, int(support[-1]) if len(support) else None, accepted)
                continue
            full, full_segment, end, accepted_full = reference[w.window_uid]
            identity = dict(animal=animal, session=session, window_uid=w.window_uid,
                start_s=w.start_s, end_s=w.end_s, population_replicate=spec["population_replicate"])
            common = dict(accepted_full=accepted_full, accepted_half=accepted,
                n_cells_full=rates.shape[0], n_cells_half=len(ix),
                endpoint_status="available" if end is not None else "no_full_cell_supported_frame")
            if end is None:
                rows.append({**identity, **common})
                continue
            anchor = int(full_segment[-1])
            own = int(segment[-1]) if len(segment) else None
            target = int(np.searchsorted(seq[:, 0], w.start_s, side="right") - 1)
            bracketed = 0 <= target < len(seq)-1 and w.end_s < seq[target+1, 0]
            pf, ph = full[end], p[end]
            ff, fh = home_features(pf, grid, home), home_features(ph, grid, home)
            rows.append({**identity, **common, "active_fill_is_home": bool(seq[target, 1] == meta["home_id"]) if bracketed else None,
                "fixed_endpoint_frame": end, "full_segment_endpoint_frame": anchor,
                "half_segment_endpoint_frame": own,
                "endpoint_mean_shift_cm": float(np.linalg.norm(ph@grid-pf@grid)),
                "endpoint_map_shift_cm": float(np.linalg.norm(grid[path[end]]-grid[full[end].argmax()])),
                "endpoint_posterior_tv": float(np.abs(ph-pf).sum()/2),
                "path_mean_shift_cm": float(np.linalg.norm(p[full_segment]@grid-full[full_segment]@grid, axis=1).mean()),
                "home_map_disagreement": ff["home_map"] != fh["home_map"],
                **{f"{k}_full": v for k, v in ff.items()}, **{f"{k}_half": v for k, v in fh.items()},
                "trajectory_home_mass_full": home_features(full[anchor], grid, home)["home_mass_r20"],
                "trajectory_home_mass_half": home_features(p[anchor], grid, home)["home_mass_r20"],
                "own_home_mass_half": home_features(p[own], grid, home)["home_mass_r20"] if own is not None else np.nan})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "event_content.csv.gz", index=False)
    sim = simulate(rates, grid, home, specs, args.seed+index, args.simulations)
    sim.insert(0, "session", session)
    sim.insert(0, "animal", animal)
    sim.to_csv(out / "simulation_content.csv.gz", index=False)
    visits.to_csv(out / "home_visit_audit.csv", index=False)
    meta.update(animal=animal, session=session, full_cells=len(rates),
                candidate_events=windows.detector.eq("source_high_mua").sum().item(),
                prior_home_mass_r20=float((np.linalg.norm(grid-home, axis=1) <= 20).mean()))
    (out / "checkpoint.json").write_text(json.dumps(dict(metadata=meta, inputs=inputs,
        population_specs=specs, status="complete", original_map_and_geometry_reconstruction="pass"), indent=2))
    print(f"complete {session}: {meta['candidate_events']} real, {args.simulations} simulated", flush=True)
    return frame, sim, meta


def summarize(events, simulations, bootstraps=5000, seed=20260913):
    summaries = []
    for (animal, session, rep), group in events.groupby(["animal", "session", "population_replicate"]):
        selected = group.endpoint_status.eq("available")
        for cohort, mask in [("all_fixed_candidates", selected),
                             ("full_accepted_fixed", selected & group.accepted_full),
                             ("both_accepted", selected & group.accepted_full & group.accepted_half)]:
            sub = group[mask]
            row = dict(animal=animal, session=session, population_replicate=rep, cohort=cohort,
                       events=len(sub), accepted_full=int(group.accepted_full.sum()),
                       accepted_half=int(group.accepted_half.sum()))
            for col in ["endpoint_mean_shift_cm", "endpoint_map_shift_cm", "endpoint_posterior_tv", "path_mean_shift_cm"]:
                row[col] = sub[col].median()
            row["home_map_disagreement"] = sub.home_map_disagreement.mean()
            for radius in RADII:
                row[f"home_mass_r{radius}_delta"] = (sub[f"home_mass_r{radius}_half"]-sub[f"home_mass_r{radius}_full"]).mean()
            row["home_mass_full"] = sub.home_mass_r20_full.mean()
            row["home_mass_half"] = sub.home_mass_r20_half.mean()
            summaries.append(row)
        decomposition = decompose(group[selected])
        summaries.append(dict(animal=animal, session=session, population_replicate=rep,
                             cohort="reselection_decomposition", events=int(selected.sum()), **decomposition))
    simfull = simulations[simulations.cell_fraction.eq(1)].drop(columns=["cell_fraction", "population_replicate"])
    simhalf = simulations[simulations.cell_fraction.eq(.5)]
    paired = simhalf.merge(simfull, on=["animal", "session", "simulation_event"], suffixes=("_half", "_full"), validate="many_to_one")
    for (animal, session, rep), g in paired.groupby(["animal", "session", "population_replicate"]):
        truth = g.true_home_full
        row = dict(animal=animal, session=session, population_replicate=rep, cohort="known_content_simulation", events=len(g))
        for pop in ["full", "half"]:
            pred = g[f"home_map_{pop}"]
            row[f"home_sensitivity_{pop}"] = pred[truth].mean()
            row[f"home_specificity_{pop}"] = (~pred[~truth]).mean()
            row[f"home_fraction_{pop}"] = pred.mean()
            row[f"endpoint_error_{pop}"] = g[f"endpoint_error_cm_{pop}"].median()
        row["home_fraction_delta"] = row["home_fraction_half"] - row["home_fraction_full"]
        row["endpoint_error_delta"] = row["endpoint_error_half"] - row["endpoint_error_full"]
        summaries.append(row)
    population = pd.DataFrame(summaries)
    numerical = [c for c in population.select_dtypes(include="number") if c != "population_replicate"]
    session = population.groupby(["animal", "session", "cohort"])[numerical].mean().reset_index()
    animal = session.groupby(["animal", "cohort"])[numerical].mean().reset_index()
    rng, rows = np.random.default_rng(seed), []
    for cohort, g in animal.groupby("cohort"):
        for col in numerical:
            v = g[col].dropna().to_numpy()
            if not len(v):
                continue
            means = v[rng.integers(0, len(v), size=(bootstraps, len(v)))].mean(axis=1)
            lo, hi = np.quantile(means, [.025, .975])
            rows.append(dict(cohort=cohort, metric=col, equal_rat_mean=v.mean(), ci_low=lo, ci_high=hi,
                             rats=len(v), rats_positive=int((v > 0).sum()), rats_negative=int((v < 0).sum())))
    return population, session, animal, pd.DataFrame(rows)


def report(output, events, metadata, summary, gates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = summary.set_index(["cohort", "metric"]).equal_rat_mean
    def value(cohort, metric):
        return float(means.loc[(cohort, metric)])
    text = ["# PF Recording Coverage and Represented Destinations", "",
        "Exploratory recording-intervention diagnostic; no biological goal-planning claim.", "",
        f"{len(metadata)} sessions, {metadata.animal.nunique()} rats; {events.window_uid.nunique()} fixed MUA candidates.",
        "Three frozen half-cell subsets, independently decoded with the same flat spatial prior and RUN maps.",
        "Home labels/coordinates are inferred, not verified author annotations.", "",
        "## All Fixed Candidates", "",
        f"Equal-rat mean of session/replicate median endpoint shifts: {value('all_fixed_candidates', 'endpoint_mean_shift_cm'):.2f} cm (posterior mean), {value('all_fixed_candidates', 'endpoint_map_shift_cm'):.2f} cm (MAP).",
        f"MAP Home/non-Home label disagreement: {100*value('all_fixed_candidates','home_map_disagreement'):.2f}%.",
        f"Mean posterior Home-mass change (half minus full): {100*value('all_fixed_candidates','home_mass_r20_delta'):+.2f} percentage points.", "",
        "## Fixed Full-Accepted Events", "",
        f"Endpoint mean shift: {value('full_accepted_fixed','endpoint_mean_shift_cm'):.2f} cm; Home mass change: {100*value('full_accepted_fixed','home_mass_r20_delta'):+.2f} pp.", "",
        "## Selection Decomposition", "",
        "Home posterior-mass changes (percentage points):",
        *[f"- {k}: {100*value('reselection_decomposition', k):+.2f}" for k in ["decoding", "composition", "timing", "total"]], "",
        "The identity is algebraic; it is not causal mediation. Composition and timing must not be mistaken for same-event decoding changes.", "",
        "## Known-Content Recovery", "",
        f"Endpoint error: {value('known_content_simulation','endpoint_error_full'):.2f} cm full versus {value('known_content_simulation','endpoint_error_half'):.2f} cm half.",
        f"Known 50% Home endpoint prevalence decoded as {100*value('known_content_simulation','home_fraction_full'):.2f}% full and {100*value('known_content_simulation','home_fraction_half'):.2f}% half.",
        "Matched Poisson/map simulation tests recovery, not the biological replay mechanism. No simulation trajectory is removed by a continuity gate.", "",
        "## Boundaries", "",
        "- Cell deletion can test robustness; the full-cell reference is not ground truth.",
        "- Home mass alone does not test prospective planning or bias versus behavior-matched alternative wells.",
        "- Four rats limit population inference; bootstrap intervals are descriptive.",
        "- Full-selected analyses condition on the full population; all-candidate analyses are reported separately.",
        "- Correlated, overlapping decoding windows do not supply independent samples.",
        "- Transferred common-grid criterion, not exact reproduction of the original PF analysis.", "",
        f"Technical gates: {'pass' if gates.passed.all() else 'FAIL; do not interpret results'}."]
    (output / "report.md").write_text("\n".join(text)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    g = events.groupby("window_uid").agg(shift=("endpoint_mean_shift_cm", "median"),
        full=("home_mass_r20_full", "first"), half=("home_mass_r20_half", "mean"))
    axes[0].hist(g["shift"], bins=40, color="#187f8e")
    axes[0].set(xlabel="Endpoint shift after cell removal (cm)", ylabel="Events", title="Fixed event/time; posterior mean")
    axes[1].scatter(g.full, g.half, s=6, alpha=.2, color="#187f8e")
    axes[1].plot([0, 1], [0, 1], color="black", lw=1)
    axes[1].set(xlabel="Full-cell Home posterior mass", ylabel="Mean half-cell Home posterior mass", title="Home radius 20 cm")
    selected = summary[summary.cohort.eq("reselection_decomposition")].set_index("metric")
    cols = ["decoding", "composition", "timing", "total"]
    y = selected.loc[cols, "equal_rat_mean"].to_numpy()*100
    lo = selected.loc[cols, "ci_low"].to_numpy()*100
    hi = selected.loc[cols, "ci_high"].to_numpy()*100
    axes[2].errorbar(np.arange(4), y, yerr=[y-lo, hi-y], fmt="o", color="#a4315d", capsize=3)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set(xticks=np.arange(4), xticklabels=cols, ylabel="Home-mass change (percentage points)", title="Equal-rat decomposition; 95% CI")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "content_robustness.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.simulations < 2 or args.simulations % 2:
        parser.error("simulations must be positive and even")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "docs/pf_recording_coverage_goal_content_protocol.md"
    provenance = build_script_provenance(input_paths={"benchmark": args.benchmark_manifest,
        "protocol": protocol, "script": Path(__file__), "data_loader": ROOT/"src/hipporeplayimm/data.py",
        "screening": ROOT/"src/hipporeplayimm/replay_coverage_shuffle_baseline.py"})
    provenance.update(status="running", parameters=vars(args).copy(),
                      created_at_utc=datetime.now(UTC).isoformat(),
                      versions={name: version(name) for name in ["numpy", "scipy", "pandas"]})
    provenance["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(provenance, indent=2))
    (args.output_dir / "protocol.md").write_text(protocol.read_text())
    start = time.monotonic()
    try:
        benchmark = json.loads(args.benchmark_manifest.read_text())
        records = [r for r in benchmark["results"] if r["dataset"] == "pfeiffer_foster"]
        if len(records) != 8 or len({r["animal"] for r in records}) != 4:
            raise ValueError("expected frozen eight-session/four-rat PF benchmark")
        frames, simulations, metas = [], [], []
        for i, rec in enumerate(records):
            frame, sim, meta = process(rec, args, i)
            frames.append(frame)
            simulations.append(sim)
            metas.append(meta)
        events, sim, metadata = pd.concat(frames, ignore_index=True), pd.concat(simulations, ignore_index=True), pd.DataFrame(metas)
        events.to_csv(args.output_dir/"event_content.csv.gz", index=False)
        sim.to_csv(args.output_dir/"simulation_content.csv.gz", index=False)
        metadata.to_csv(args.output_dir/"home_metadata_qc.csv", index=False)
        population, sessions, animals, summary = summarize(events, sim, seed=args.seed)
        for name, frame in [("population_summary", population), ("session_summary", sessions), ("animal_summary", animals), ("summary", summary)]:
            frame.to_csv(args.output_dir/f"{name}.csv", index=False)
        d = population[population.cohort.eq("reselection_decomposition")]
        gates = pd.DataFrame([dict(gate=k, passed=bool(v)) for k, v in dict(
            eight_sessions_four_rats=metadata.session.nunique()==8 and metadata.animal.nunique()==4,
            fixed_candidate_coverage=events.window_uid.nunique()==4001 and len(events)==3*4001,
            endpoint_support=events.endpoint_status.eq("available").all(),
            original_map_geometry_reconstruction=True, metadata_qc=True,
            all_decompositions_available=d.status.eq("complete").all(),
            decomposition_identity=np.allclose(d.decoding+d.composition+d.timing, d.total),
            simulation_complete=len(sim)==8*args.simulations*4,
            git_commit_recorded=provenance.get("code_commit") not in [None, "unavailable", "unknown_without_git"]
        ).items()])
        gates.to_csv(args.output_dir/"gate_summary.csv", index=False)
        report(args.output_dir, events, metadata, summary, gates)
        provenance.update(status="complete" if gates.passed.all() else "technical_gate_failed",
                          runtime_s=time.monotonic()-start,
                          completed_at_utc=datetime.now(UTC).isoformat())
    except Exception as exc:
        provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
