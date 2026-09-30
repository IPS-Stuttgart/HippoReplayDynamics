"""Known-truth readout/generator/window factorial; no real-data scoring."""
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

from hipporeplayimm.regional_content_frontier import calls_from_bf, discrimination, regional_log_bf
from hipporeplayimm.regional_readout_endpoint import class_likelihoods, count_terminal, fit_prevalence, oracle_prevalence
from hipporeplayimm.selection_matched_regional import draw_panel
from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_edge_support_content import occupied_graph

READOUTS = ("ternary", "continuous_neutral", "continuous_native")


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"seed": str})


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def validation_mask(meta):
    return meta.phase.eq("validation").to_numpy() & meta.requested_prevalence.isin([.05, .15, .30, .50]).to_numpy()


def analyze_bank(meta, labels, accepted, kinds, bf, calls, totals, identity, calibration_modes):
    rows = []
    cm = meta.phase.eq("calibration").to_numpy()[:, None] & accepted
    vm = validation_mask(meta)
    if not cm.any() or not vm.any():
        raise ValueError("calibration and independent validation required")
    assert set(meta.loc[meta.phase.eq("calibration"), "job"]).isdisjoint(set(meta.loc[vm, "job"]))
    for readout in READOUTS:
        score = calls if readout == "ternary" else np.where(totals == 0, 0., bf) if readout == "continuous_neutral" else bf
        for calibration_mode in calibration_modes:
            f = np.full((*bf.shape, 2), np.nan)
            if calibration_mode == "pooled":
                f[:] = class_likelihoods(score[cm], labels[cm], score.ravel(), readout).reshape(*bf.shape, 2)
            else:
                for kind in np.unique(kinds):
                    use, query = cm & (kinds == kind), kinds == kind
                    f[query] = class_likelihoods(score[use], labels[use], score[query], readout)
            for i in np.flatnonzero(vm):
                ok = accepted[i]
                if not ok.any():
                    raise ValueError("empty validation panel")
                value = (fit_prevalence(f[i, ok]) if calibration_mode == "pooled"
                         else oracle_prevalence(f[i, ok], kinds[i, ok]))
                truth = float(labels[i, ok].mean())
                llr = np.log(f[i, ok, 1])-np.log(f[i, ok, 0])
                d = discrimination(llr, labels[i, ok])
                raw = discrimination(bf[i, ok], labels[i, ok])
                error = abs(value-truth)
                job = meta.iloc[i]
                rows.append(dict(**identity, job=int(job.job), generator=job.generator,
                    requested_prevalence=float(job.requested_prevalence), replica=int(job.replica),
                    readout=readout, calibration_mode=calibration_mode, estimate=value, truth=truth,
                    absolute_error=error, within_5pp=bool(np.isfinite(error) and error <= .05),
                    status="fit" if np.isfinite(value) else "unidentified", events=int(ok.sum()),
                    raw_bf_auc=raw["auc"], calibrated_readout_auc=d["auc"],
                    calibrated_balanced_brier=d["balanced_brier"], mean_spikes=float(totals[i, ok].mean()),
                    silent_fraction=float((totals[i, ok] == 0).mean())))
    return rows


def worker(task):
    session, animal, source, output = task
    folder = Path(source)/session.replace("/", "_")
    dest = Path(output)/session.replace("/", "_")
    dest.mkdir()
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if file_sha256(folder/name) != expected:
            raise ValueError(f"source hash mismatch: {session}/{name}")
    e = load_npz(folder/"frozen_inputs.npz")
    bank = load_npz(folder/"simulation_counts.npz")
    stored = load_npz(folder/"calibration_and_readouts.npz")
    populations = json.loads((folder/"population_definitions.json").read_text())
    meta = read_csv(folder/"simulation_jobs.csv")
    np.testing.assert_array_equal(meta.job, np.arange(len(meta)))
    assert set(meta.loc[meta.phase.eq("calibration"), "seed"]).isdisjoint(set(meta.loc[validation_mask(meta), "seed"]))
    rows = []
    lookup = {int(cell): i for i, cell in enumerate(e["cell_ids"])}
    for j, pop in enumerate(populations):
        ix = [lookup[int(c)] for c in e[f"pop{j}_ids"]]
        totals = bank["counts"][:, :, ix].sum(axis=2)
        identity = {"session": session, "animal": animal, "population": pop["name"], "family": pop["family"], "window_ms": 20}
        rows.extend(analyze_bank(meta, bank["labels"], bank["accepted"], bank["generators"],
            stored[f"pop{j}_bf"], stored[f"pop{j}_calls"], totals, identity, ("pooled", "oracle_generator")))
        print(session, "20ms", pop["name"], "complete", flush=True)
    baseline = pd.DataFrame(rows)
    original = read_csv(folder/"validation_and_null.csv")
    check = baseline.loc[baseline.readout.eq("ternary") & baseline.calibration_mode.eq("pooled")]
    paired = check.merge(original, on=["job", "population"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(paired) == len(check)
    np.testing.assert_allclose(paired.estimate, paired.prevalence, atol=1e-5, rtol=0)
    baseline.to_csv(dest/"readout_panels.csv", index=False)

    # Restore identical latent draws and spikes; only the readout duration changes.
    select = (meta.phase.eq("calibration").to_numpy() | validation_mask(meta)) & meta.generator.isin(["stationary", "late_jump"]).to_numpy()
    temporal_meta = meta.loc[select].copy().reset_index(drop=True)
    offsets = e["template_offsets"]
    templates = [{"start": a, "end": b, "times": e["template_times"][offsets[k]:offsets[k+1]]}
                 for k, (a, b) in enumerate(zip(e["candidate_start"], e["candidate_end"], strict=True))]
    graph = occupied_graph(e["grid"])
    short_counts = []
    for row in temporal_meta.itertuples():
        redraw = draw_panel(templates, row.requested_prevalence, row.generator, e["grid"], e["rates"],
            graph, e["region"], np.random.default_rng(int(row.seed)), row.perturbation)
        for key in ("counts", "targets", "labels", "active"):
            np.testing.assert_array_equal(redraw[key], bank[key][row.job])
        np.testing.assert_array_equal(count_terminal(redraw["identities"], templates, len(e["cell_ids"]), .020), bank["counts"][row.job])
        c = count_terminal(redraw["identities"], templates, len(e["cell_ids"]), .005)
        assert np.all(c <= redraw["counts"])
        short_counts.append(c)
        if len(short_counts) % 40 == 0:
            print(session, "exact 5ms redraw", len(short_counts), "/", len(temporal_meta), flush=True)
    c5 = np.asarray(short_counts)
    ix = [lookup[int(c)] for c in e["pop0_ids"]]
    bf5 = regional_log_bf(c5[:, :, ix].reshape(-1, len(ix)), e["pop0_rates"], e["pop0_region"], exposure=.005).reshape(c5.shape[:2])
    totals5 = c5[:, :, ix].sum(axis=2)
    calls5 = calls_from_bf(bf5, totals5)
    positions = temporal_meta.job.to_numpy()
    np.savez_compressed(dest/"terminal_5ms_audit.npz", jobs=positions, counts=c5, bf=bf5, calls=calls5,
                        labels=bank["labels"][positions], accepted=bank["accepted"][positions])
    temporal_meta.to_csv(dest/"terminal_jobs.csv", index=False)
    rows.extend(analyze_bank(temporal_meta, bank["labels"][positions], bank["accepted"][positions], bank["generators"][positions],
        bf5, calls5, totals5, {"session": session, "animal": animal, "population": "full", "family": "full", "window_ms": 5}, ("oracle_generator",)))
    frame = pd.DataFrame(rows)
    frame.to_csv(dest/"validation_panels.csv", index=False)
    support = []
    for local, row in enumerate(temporal_meta.itertuples()):
        if row.phase != "validation":
            continue
        ok = bank["accepted"][row.job]
        n20 = bank["counts"][row.job][:, ix].sum(axis=1)[ok]
        n5 = totals5[local, ok]
        support.append({"session": session, "animal": animal, "job": row.job, "generator": row.generator,
            "requested_prevalence": row.requested_prevalence, "events": int(ok.sum()),
            "mean_spikes_20ms": float(n20.mean()), "mean_spikes_5ms": float(n5.mean()),
            "silent_20ms": float((n20 == 0).mean()), "silent_5ms": float((n5 == 0).mean()),
            "fraction_spikes_final_5ms": float(n5.sum()/max(1, n20.sum()))})
    pd.DataFrame(support).to_csv(dest/"endpoint_support.csv", index=False)
    summary = {"session": session, "animal": animal, "status": "pass", "populations": len(populations),
        "source_hashes_verified": len(manifest["outputs"]), "baseline_rows_verified": len(paired),
        "panels_redrawn": len(temporal_meta), "detector_selection_unchanged": True, "source": str(folder/"manifest.json"),
        "source_sha256": file_sha256(folder/"manifest.json"),
        "outputs": {p.name:file_sha256(p) for p in dest.iterdir() if p.is_file()}}
    write_json(dest/"audit.json", summary)
    print("DONE", session, flush=True)
    return summary


def summarize(frame):
    keys = ["animal", "session", "population", "family", "window_ms", "readout", "calibration_mode", "generator", "requested_prevalence"]
    result = frame.groupby(keys, dropna=False).agg(
        replicas=("job", "size"), finite_fits=("estimate", "count"), mean_absolute_error=("absolute_error", "mean"),
        within_5pp_fraction=("within_5pp", "mean"), raw_bf_auc=("raw_bf_auc", "mean"),
        calibrated_readout_auc=("calibrated_readout_auc", "mean"), mean_spikes=("mean_spikes", "mean"),
        silent_fraction=("silent_fraction", "mean")).reset_index()
    result["budget_pass"] = (result.finite_fits.eq(result.replicas) & result.mean_absolute_error.le(.05)
                              & result.within_5pp_fraction.ge(.9))
    return result


def contrasts(frame):
    key = ["session", "animal", "population", "family", "window_ms", "generator", "requested_prevalence", "replica", "job", "calibration_mode"]
    a = frame.loc[frame.readout.eq("ternary")]
    b = frame.loc[frame.readout.eq("continuous_neutral")]
    paired = a.merge(b, on=key, suffixes=("_ternary", "_continuous"), validate="one_to_one")
    paired["continuous_error_reduction"] = paired.absolute_error_ternary-paired.absolute_error_continuous
    return paired[key+["absolute_error_ternary", "absolute_error_continuous", "continuous_error_reduction"]]


def make_report(root, frame, summary, sessions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = summary.loc[summary.family.eq("full")]
    keys = ["window_ms", "readout", "calibration_mode", "generator"]
    metrics = ["mean_absolute_error", "within_5pp_fraction", "raw_bf_auc", "calibrated_readout_auc", "mean_spikes", "silent_fraction"]
    animal = full.groupby(["animal"]+keys)[metrics].mean().reset_index()
    animal.to_csv(root/"by_animal.csv", index=False)
    pooled = animal.groupby(keys)[metrics].mean().reset_index()
    pooled.to_csv(root/"descriptive_summary.csv", index=False)
    mixed = pooled.loc[pooled.window_ms.eq(20) & pooled.generator.eq("mix")]
    temporal = pooled.loc[pooled.readout.eq("continuous_neutral") & pooled.calibration_mode.eq("oracle_generator") & pooled.generator.isin(["stationary", "late_jump"])]
    lines = ["# Readout compression and terminal-content diagnostic", "", "## Scope", "",
        (f"{len(sessions)} PF sessions, {len(frame.animal.unique())} rats. Known-truth simulations only. "
        "No real-content estimates, new replay scoring, or hc-11 transfer. Development on the existing simulation bank, not a fresh confirmatory holdout."), "",
        "## Matched mixture, full population, 20 ms", "",
        "| Readout | Calibration | Mean absolute error (pp) | Within 5 pp | Calibrated readout AUC |",
        "|---|---|---:|---:|---:|"]
    for r in mixed.itertuples():
        lines.append(f"| {r.readout} | {r.calibration_mode} | {100*r.mean_absolute_error:.2f} | {100*r.within_5pp_fraction:.1f}% | {r.calibrated_readout_auc:.3f} |")
    lines += ["", ("Continuous-neutral preserves neutral silence; continuous-native retains the Poisson silent-window readout. "
        "Oracle calibration knows the generator for every simulated event and is not available for real replay."), "",
        "## Temporal arm, continuous-neutral and generator-matched", "",
        "| Generator | Window | Error (pp) | Within 5 pp | Raw BF AUC | Calibrated readout AUC | Silent windows |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in temporal.itertuples():
        lines.append(f"| {r.generator} | {r.window_ms} ms | {100*r.mean_absolute_error:.2f} | {100*r.within_5pp_fraction:.1f}% | {r.raw_bf_auc:.3f} | {r.calibrated_readout_auc:.3f} | {100*r.silent_fraction:.1f}% |")
    lines += ["", ("The same spikes, labels and accepted events are used at both durations. "
        "Late jumps occur in the final 5 ms. Empty short windows remain in the analysis. "
        "Calibrated AUC can exploit inverted raw readouts; raw AUC below .5 is not absence of information."), "",
        "## Validation boundaries", "",
        ("Tables give equal-rat descriptive averages, not confidence intervals from independent simulated animals. "
        "The 90%/five-point criterion applies to every population/generator/prevalence combination; pooled averages do not establish it. "
        "Generator-specific calibration is an oracle. Density estimation and finite calibration error remain possible explanations. "
        "This diagnostic does not establish calibrated real endpoint content or test perturbation robustness of an improved method."), "",
        "## Technical checks", "",
        (f"All {len(sessions)} session audits passed. Source hashes checked, frozen ternary estimates reproduced, and "
        f"{sum(s['panels_redrawn'] for s in sessions)} panels reconstructed exactly before changing the readout window."), "",
        "![Diagnostic](regional_readout_endpoint.png)"]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = {"ternary": "#6b6b6b", "continuous_neutral": "#007f83", "continuous_native": "#ae4256"}
    for i, mode in enumerate(("pooled", "oracle_generator")):
        for j, readout in enumerate(READOUTS):
            row = mixed.loc[mixed.calibration_mode.eq(mode) & mixed.readout.eq(readout)].iloc[0]
            axes[0].bar(i+(j-1)*.22, row.mean_absolute_error*100, width=.2, color=colors[readout], label=readout.replace("continuous_", "") if i == 0 else None)
    axes[0].axhline(5, color="black", ls=":")
    axes[0].set(xticks=[0, 1], xticklabels=["Pooled", "Generator oracle"], ylabel="Mean absolute error (percentage points)", title="20 ms: readout and generator")
    axes[0].legend(fontsize=8)
    for kind, color in (("stationary", "#ae4256"), ("late_jump", "#007f83")):
        p = temporal.loc[temporal.generator.eq(kind)].sort_values("window_ms")
        axes[1].plot(p.window_ms, p.calibrated_readout_auc, "o-", color=color, label=kind)
        axes[2].plot(p.window_ms, p.silent_fraction*100, "o-", color=color, label=kind)
    axes[1].axhline(.5, color="gray", ls=":")
    axes[1].set(xticks=[5, 20], xlabel="Readout duration (ms)", ylabel="Calibrated readout AUC", title="Known-generator discrimination", ylim=(.45, 1))
    axes[2].set(xticks=[5, 20], xlabel="Readout duration (ms)", ylabel="Spike-free windows (%)", title="Short-window information loss")
    axes[1].legend()
    fig.savefig(root/"regional_readout_endpoint.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/selection-matched-regional-pf-v3-20260915"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--session-limit", type=int)
    args = p.parse_args()
    if args.workers < 1:
        p.error("positive workers required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.input_dir/"manifest.json").read_text())
    if source["status"] != "complete":
        raise ValueError("complete source bank required")
    inputs = {"source_manifest": args.input_dir/"manifest.json", "sessions": args.input_dir/"sessions.csv",
              "script": Path(__file__), "core": ROOT/"src/hipporeplayimm/regional_readout_endpoint.py",
              "protocol": ROOT/"docs/regional_readout_endpoint_protocol.md",
              "generator": ROOT/"src/hipporeplayimm/selection_matched_regional.py",
              "likelihood": ROOT/"src/hipporeplayimm/regional_content_frontier.py"}
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", parameters={k:str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(args.output_dir/"manifest.json", manifest)
    sessions = read_csv(args.input_dir/"sessions.csv")
    if args.session_limit:
        sessions = sessions.iloc[:args.session_limit]
    tasks = [(s.session, s.animal, str(args.input_dir), str(args.output_dir)) for s in sessions.itertuples()]
    if args.workers == 1:
        completed = [worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            completed = list(pool.map(worker, tasks))
    frame = pd.concat([read_csv(args.output_dir/s["session"].replace("/", "_")/"validation_panels.csv") for s in completed], ignore_index=True)
    support = pd.concat([read_csv(args.output_dir/s["session"].replace("/", "_")/"endpoint_support.csv") for s in completed], ignore_index=True)
    frame.to_csv(args.output_dir/"validation_panels.csv", index=False)
    support.to_csv(args.output_dir/"endpoint_support.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output_dir/"validation_summary.csv", index=False)
    contrasts(frame).to_csv(args.output_dir/"readout_contrasts.csv", index=False)
    make_report(args.output_dir, frame, summary, completed)
    pd.DataFrame([{"gate": "source_and_redraw_audits", "status": "pass"},
                  {"gate": "no_real_content_claims", "status": "pass"},
                  {"gate": "ready_for_real_calibration", "status": "not_tested"}]).to_csv(args.output_dir/"gate_summary.csv", index=False)
    manifest.update(status="complete", sessions=completed,
        outputs={f.name:file_sha256(f) for f in args.output_dir.iterdir() if f.is_file() and f.name != "manifest.json"})
    write_json(args.output_dir/"manifest.json", manifest)
    print((args.output_dir/"report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
