"""Independent bank checks, conditional-occupancy report and native redetection."""
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

from hipporeplayimm.data import load_mat_variable
from hipporeplayimm.regional_content_mua_null import load_detector
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import DELTAS, read_csv, write_json
from scripts.measure_selection_matched_regional_calibration import detector_run


def check_templates(folder):
    f, t = load_npz(folder / "source_inputs.npz"), load_npz(folder / "templates.npz")
    for j, i in enumerate(t["source_indices"]):
        a, b = t["offsets"][j:j+2]
        c, d = f["template_offsets"][i:i+2]
        np.testing.assert_array_equal(t["times"][a:b], f["template_times"][c:d])
        np.testing.assert_array_equal([t["starts"][j], t["endpoints"][j]], [f["candidate_start"][i], f["windows"][i, 1]])
        assert t["endpoints"][j] - t["starts"][j] + 1e-9 >= .1
    return f, t


def native_audit(session_row):
    folder = Path(session_row["folder"])
    config = json.loads((folder / "config.json").read_text())
    sm = json.loads(Path(config["source_manifest"]).read_text())
    f, t = check_templates(folder)
    inputs = sm["input_file_paths"]
    for key in ("Spike_Data.mat", "Position_Data.mat", "Epochs.mat"):
        if file_sha256(inputs[key]) != sm["input_file_sha256"][key]:
            raise ValueError("raw native data changed")
    spikes = np.asarray(load_mat_variable(inputs["Spike_Data.mat"], "Spike_Data"), float).reshape(-1, 2)
    spikes = spikes[np.argsort(spikes[:, 0], kind="stable")]
    position = np.asarray(load_mat_variable(inputs["Position_Data.mat"], "Position_Data"), float).reshape(-1, 4)
    intervals = np.asarray(load_mat_variable(inputs["Epochs.mat"], "Run_Times"), float).reshape(-1, 2)
    detector = load_detector(Path(sm["parameters"]["detector_script"]))
    st, speed = detector.position_speed(position, .1)
    original = detector_run(detector, spikes, st, speed, intervals, len(f["cell_ids"]))
    rows = []
    for sub in sorted(folder.glob("*_delta100")):
        panel = load_npz(sub / "panel_000.npz")
        altered = spikes.copy()
        for j, (start, end) in enumerate(zip(t["starts"], t["candidate_ends"], strict=True)):
            a, b = np.searchsorted(spikes[:, 0], [start, end], side="left")
            c, d = t["offsets"][j:j+2]
            np.testing.assert_array_equal(spikes[a:b, 0], t["times"][c:d])
            altered[a:b, 1] = f["cell_ids"][panel["identities"][c:d]]
        np.testing.assert_array_equal(altered[:, 0], spikes[:, 0])
        detected = detector_run(detector, altered, st, speed, intervals, len(f["cell_ids"]))
        actual = {round(float(x["event_start_s"]), 6): x for x in detected}
        for start, end in zip(t["starts"], t["candidate_ends"], strict=True):
            entry = actual[round(float(start), 6)]
            np.testing.assert_allclose([entry["event_start_s"], entry["event_end_s"]], [start, end], rtol=0, atol=1e-8)
        rows.append({"session": config["session"], "stratum": sub.name, "templates": len(t["event_ids"]),
                     "original_candidates": len(original), "redetected_candidates": len(detected),
                     "selected_boundaries_preserved": True, "status": "pass"})
    return rows


def independent_saved_path_dwell(ages, nodes, grid, kind, center, duration):
    step = .00005
    t = np.arange(step / 2, duration, step)
    if kind == "moving":
        xy = np.stack([np.interp(t, ages, grid[nodes, axis]) for axis in (0, 1)], axis=1)
    else:
        xy = grid[nodes[np.searchsorted(ages, t, side="right")-1]]
    value = float(np.count_nonzero(np.sum((xy-center)**2, axis=1) <= 400.) * step)
    tolerance = step * max(1, np.count_nonzero(ages < duration))
    return value, tolerance


def audit_stratum(row):
    folder = Path(row["folder"])
    sm = json.loads((folder / "manifest.json").read_text())
    for name, sha in sm["sha256"].items():
        assert file_sha256(folder / name) == sha, (folder, name)
    source, templates = check_templates(folder.parent)
    config = json.loads((folder.parent / "config.json").read_text())
    meta = read_csv(folder / "panels.csv")
    index = DELTAS.index(row["delta_ms"] / 1000.)
    all_occ, all_labels, endpoints, origin, late, jump_types = [], [], [], [], [], []
    seeds, requested, realized = [], [], []
    max_numeric_error, numeric_checks = 0., 0
    for job in meta.itertuples():
        p = load_npz(folder / f"panel_{job.panel:03}.npz")
        assert p["identities"].shape == templates["times"].shape
        assert (p["identities"] < len(source["cell_ids"])).all()
        np.testing.assert_array_equal(p["desired_label"], p["label"][:, index])
        assert int(p["desired_label"].sum()) == round(len(templates["event_ids"]) * job.prevalence)
        assert np.isfinite(p["occupancy_fraction"]).all()
        assert (p["occupancy_fraction"] >= 0).all() and (p["occupancy_fraction"] <= 1).all()
        np.testing.assert_allclose(p["dwell_s"], p["occupancy_fraction"] * np.asarray(DELTAS), rtol=0, atol=1e-10)
        np.testing.assert_array_equal(p["label"], p["dwell_s"] >= .020 - 1e-9)
        for i in range(len(templates["event_ids"])):
            a, b = templates["offsets"][i:i+2]
            assert len(np.unique(p["identities"][a:b])) == p["active"][i]
            assert p["active"][i] >= np.ceil(.1 * len(source["cell_ids"]))
            c, d = p["path_offsets"][i:i+2]
            assert p["path_ages"][c] == 0 and (np.diff(p["path_ages"][c:d]) > 0).all()
            assert p["path_ages"][d-1] + 1e-9 >= max(.1, templates["endpoints"][i] - templates["starts"][i])
        a, b = p["path_offsets"][:2]
        for k, duration in enumerate(DELTAS):
            observed, tolerance = independent_saved_path_dwell(
                p["path_ages"][a:b], p["path_nodes"][a:b], source["grid"], row["generator"],
                np.asarray(config["center"]), duration)
            error = abs(observed-p["dwell_s"][0, k])
            assert error <= tolerance+1e-9, "independent saved-path occupancy failed"
            max_numeric_error = max(max_numeric_error, error)
            numeric_checks += 1
        all_occ.extend(p["occupancy_fraction"][:, index])
        all_labels.extend(p["label"][:, index])
        endpoints.extend(p["endpoint_home"][:, index])
        origin.extend(p["origin_home"][:, index])
        late.extend(p["late_crossing"][:, index])
        jump_types.extend(p["jump_crossings"][:, index])
        seeds.extend(p["event_seeds"])
        requested.extend(p["requested_jump_cm"])
        realized.extend(p["realized_jump_cm"])
    occ, labels = np.asarray(all_occ), np.asarray(all_labels, bool)
    statistics = []
    for label in (False, True):
        x = occ[labels == label]
        if not len(x):
            raise ValueError("missing label in stratum")
        h = np.histogram(x, bins=np.linspace(0, 1, 21))[0]
        structural = bool(label and (row["generator"] == "stationary" or row["delta_ms"] == 20))
        concentration = float(h.max() / h.sum())
        near_degenerate = concentration >= .95
        statistics.append({k: row[k] for k in ("session", "generator", "scale", "delta_ms")} | {
            "label": label, "n": len(x), "mean_occupancy": float(x.mean()),
            "p10": float(np.quantile(x, .1)), "median": float(np.median(x)), "p90": float(np.quantile(x, .9)),
            "largest_5pct_bin_fraction": concentration, "structural_positive_degeneracy": structural,
            "unexpected_positive_degeneracy": bool(label and near_degenerate and not structural),
            "endpoint_home_fraction": float(np.asarray(endpoints)[labels == label].mean()),
            "origin_home_fraction": float(np.asarray(origin)[labels == label].mean()),
            "late_crossing_fraction": float(np.asarray(late)[labels == label].mean())})
    crossings = np.asarray(jump_types).sum(axis=0)
    stats = {k: row[k] for k in ("session", "generator", "scale", "delta_ms")} | {
        "events": len(labels), "independent_numeric_checks": numeric_checks, "max_numeric_dwell_error_s": max_numeric_error,
        "attempts": int(meta.attempts.sum()),
        "geometry_rejections": int(meta.geometry_rejections.sum()), "spike_rejections": int(meta.spike_rejections.sum()),
        "retention_per_proposal": len(labels) / meta.attempts.sum(),
        "nonhome_nonhome": int(crossings[0, 0]), "nonhome_home": int(crossings[0, 1]),
        "home_nonhome": int(crossings[1, 0]), "home_home": int(crossings[1, 1]),
        "requested_jump_median_cm": float(np.median(requested)) if len(requested) else np.nan,
        "realized_jump_median_cm": float(np.median(realized)) if len(realized) else np.nan,
        "jump_mismatch_over_4cm_fraction": float((np.abs(np.asarray(requested)-realized)>4.+1e-8).mean()) if len(requested) else np.nan}
    return statistics, stats, np.asarray(seeds, np.uint64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bank, out = Path(args.bank), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bank / "manifest.json").read_text())
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][key], key
    strata = read_csv(bank / "strata.csv")
    all_stats, proposals, all_seeds = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (statistics, stats, seeds) in enumerate(executor.map(audit_stratum, strata.to_dict("records"))):
            all_stats.extend(statistics)
            proposals.append(stats)
            all_seeds.extend(seeds)
            if i % 28 == 0:
                print("array audit", i+1, "/", len(strata), flush=True)
    all_seeds = np.asarray(all_seeds, np.uint64)
    assert len(all_seeds) == len(np.unique(all_seeds)), "phase/event seeds overlap"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        native = [r for rows in executor.map(native_audit, read_csv(bank / "sessions.csv").to_dict("records")) for r in rows]
    stats = pd.DataFrame(all_stats)
    stats.to_csv(out / "conditional_occupancy_summary.csv", index=False)
    pd.DataFrame(proposals).to_csv(out / "proposal_and_jump_summary.csv", index=False)
    pd.DataFrame(native).to_csv(out / "native_detector_audit.csv", index=False)
    histograms, references = [], []
    for row in strata.to_dict("records"):
        meta = {k: row[k] for k in ("session", "generator", "scale", "delta_ms")}
        h = read_csv(Path(row["folder"]) / "occupancy_histogram.csv").assign(**meta)
        histograms.append(h)
        p = Path(row["folder"]) / "unconditioned_reference.csv"
        if p.exists():
            references.append(read_csv(p).assign(session=row["session"], generator=row["generator"], scale=row["scale"]))
    hist = pd.concat(histograms, ignore_index=True)
    hist.to_csv(out / "conditional_occupancy_histogram.csv", index=False)
    ref = pd.concat(references, ignore_index=True)
    ref.groupby(["session", "generator", "scale", "delta_ms"], observed=True).agg(
        proposals=("label", "size"), natural_positive_fraction=("label", "mean"), mean_occupancy=("occupancy_fraction", "mean"),
        nonhome_nonhome=("nonhome_nonhome", "sum"), nonhome_home=("nonhome_home", "sum"),
        home_nonhome=("home_nonhome", "sum"), home_home=("home_home", "sum"), late_crossing=("late_crossing", "sum")
    ).reset_index().to_csv(out / "unconditioned_geometry_reference_summary.csv", index=False)
    unexpected = int(stats.unexpected_positive_degeneracy.sum())
    gates = [{"gate": k, "status": "pass", "detail": detail} for k, detail in (
        ("fixed_times_and_counts", "Every panel preserves original candidate timestamp templates"),
        ("native_active_support", "Every retained event passes original full-population active-cell gate"),
        ("native_detector_equivalence", f"{len(native)} representative full-session redetections"),
        ("label_quotas", "Each conditioning Delta/dynamics/phase/prevalence panel matches exact rounded quota"),
        ("seed_isolation", f"{len(all_seeds)} distinct event seeds"),
        ("independent_saved_path_truth", f"{sum(p['independent_numeric_checks'] for p in proposals)} fine-grid dwell checks"),
        ("source_and_output_hashes", "Verified pinned files and every stratum output"))]
    gates.append({"gate": "conditional_positive_occupancy", "status": "pass" if not unexpected else "review_required",
                  "detail": f"{unexpected} non-structural positive cells have >=95% mass in one 5%-wide occupancy bin"})
    gates.append({"gate": "real_data_calibration", "status": "not_tested", "detail": "Bank QC is not validation of a prevalence estimator"})
    pd.DataFrame(gates).to_csv(out / "gate_summary.csv", index=False)
    figure(hist, out)
    cohort = read_csv(bank / "cohort_manifest.csv")
    lines = ["# Shared regional-content bank: development audit", "",
             f"{manifest['simulated_event_samples']:,} synthetic event samples; {len(strata)} session/dynamics/Delta strata.",
             f"Original cohort {len(cohort)}; >=100-ms available {int(cohort.included.sum())}; shorter {int((~cohort.included).sum())}.",
             "Stationary positive occupancy=1 and Delta20 positive occupancy=1 are definition-imposed, not generator coupling.",
             f"Unexpected near-degenerate positive occupancy cells: {unexpected}. See the per-session table before downstream use.", "",
             "Dynamics are geometry-blind, then conditioned on geometric content and native active support. Conditioning changes path distributions deliberately.",
             "Home labels require >=20 ms cumulative dwell, not a brief disc touch. Endpoint, origin, occupancy and longest visit are separate saved quantities.",
             "Same-region jumps and crossings are counted separately; late crossings remain a labeled adversarial subset, not the jump generator.", "",
             "## Limits", "",
             "Empirical anchors are from the full-cell smoothed decoder on 108 selected clean-IMM events, not measured neural jumps or all MUA candidates.",
             "The 0.5/1/2 scales expose decoder-dependent choices. Geodesics and radial jumps are specified synthetic laws, not fitted true dynamics.",
             "Four replicas per phase/prevalence are development coverage only; they cannot certify 90% interval coverage or 5% false flags.",
             "No real prevalence calibration, terminal-length choice, cell-diagnostic specificity claim, or inferred-IMM validation follows yet.",
             "At fixed spike totals, common gain and absolute additive firing are not jointly identifiable.", ""]
    (out / "report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"bank_manifest": bank / "manifest.json", "auditor": __file__})
    provenance.update(status="technical_pass" if not unexpected else "technical_pass_occupancy_review_required",
                      unique_event_seeds=len(all_seeds), native_redetections=len(native), unexpected_occupancy_cells=unexpected,
                      outputs={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    write_json(out / "manifest.json", provenance)
    print(provenance["status"], flush=True)


def figure(hist, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["stationary", "moving", "jumping"]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
    selected = hist.loc[hist.label.eq(1) & hist.stage.eq("retained")]
    for row, kind in enumerate(names):
        for col, delta in enumerate((20, 40, 60, 100)):
            ax = axes[row, col]
            sub = selected.loc[selected.generator.eq(kind) & selected.delta_ms.eq(delta)]
            for scale, s in sub.groupby("scale"):
                # Equal-session weighting; Monte Carlo replicas are not animals.
                s = s.copy()
                s["fraction"] = s["count"] / s.groupby("session")["count"].transform("sum")
                curve = s.groupby("left").fraction.mean()
                ax.plot(curve.index+.025, curve.values, label=f"scale {scale:g}")
            ax.set_title(f"{kind}, {delta} ms", fontsize=10)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, 1)
            if col == 0:
                ax.set_ylabel("Fraction / 5% bin")
            if row == 2:
                ax.set_xlabel("Home occupancy fraction")
    axes[1, 3].legend(fontsize=8)
    fig.suptitle("Positive segment occupancy: shared geometry-blind bank\n20 ms minimum Home time; fixed-count synthetic controls, not real replay")
    fig.savefig(out / "positive_occupancy_distributions.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
