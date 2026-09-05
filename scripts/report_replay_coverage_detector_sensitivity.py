#!/usr/bin/env python3
"""Non-rescoring, animal-weighted detector/window sensitivity report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

GROUP = ["dataset", "detector", "window_variant", "likelihood", "estimator", "bin_filter", "cell_fraction"]
LABELS = ["MUA core", "MUA 200 ms", "Ripple core", "Ripple 200 ms"]


def primary(frame):
    return frame[frame.overlap_scope.eq("all_eligible") & ((frame.detector.eq("lfp_ripple_detected") & frame.peak_threshold_z.eq(3)) | (~frame.detector.eq("lfp_ripple_detected") & frame.peak_threshold_z.eq(0)))].copy()


def label(row):
    return ("MUA" if row.detector == "source_high_mua" else "Ripple") + (" core" if row.window_variant == "detected_core" else " 200 ms")


def paired_interactions(session, bootstraps=5000, seed=20260907):
    frame = primary(session)
    frame = frame[frame.cell_fraction.eq(.5)]
    keys = ["dataset", "animal", "session", "likelihood", "estimator", "bin_filter"]
    metric_names = ["paired_continuity_delta", "median_common_step_speed_delta_cm_s"]
    frame["cohort"] = [label(r) for r in frame.itertuples(index=False)]
    rows, summaries = [], []
    rng = np.random.default_rng(seed)
    contrasts = [("Ripple core", "MUA core"), ("Ripple 200 ms", "MUA 200 ms"), ("MUA 200 ms", "MUA core"), ("Ripple 200 ms", "Ripple core")]
    for a, b in contrasts:
        x = frame[frame.cohort.eq(a)][keys + metric_names]
        y = frame[frame.cohort.eq(b)][keys + metric_names]
        paired = x.merge(y, on=keys, validate="one_to_one", suffixes=("_a", "_b"))
        for metric in metric_names:
            for r in paired.to_dict("records"):
                rows.append({**{k: r[k] for k in keys}, "contrast": f"{a} minus {b}", "metric": metric,
                    "value": r[f"{metric}_a"] - r[f"{metric}_b"], "value_a": r[f"{metric}_a"], "value_b": r[f"{metric}_b"]})
    raw = pd.DataFrame(rows)
    group = ["dataset", "likelihood", "estimator", "bin_filter", "contrast", "metric"]
    animals = raw.groupby(group + ["animal"], as_index=False).agg(value=("value", "mean"), paired_sessions=("value", "count"))
    for g, local in animals.groupby(group, sort=True):
        v = local.value.dropna().to_numpy(float)
        ci = np.quantile(rng.choice(v, (bootstraps, len(v)), replace=True).mean(axis=1), [.025, .975]) if len(v) >= 2 else [np.nan, np.nan]
        summaries.append({**dict(zip(group, g, strict=True)), "animals_measurable": len(v), "animals_total": len(local),
            "paired_sessions": int(local.paired_sessions.sum()), "equal_animal_mean": v.mean() if len(v) else np.nan,
            "ci95_low": ci[0], "ci95_high": ci[1], "interpretation": "difference_of_paired_cell_removal_effects; different_event_sets_not_randomized_detector_effect"})
    return raw, animals, pd.DataFrame(summaries)


def run(root, out):
    mp = root / "coverage_detector_sensitivity_manifest.json"
    ap = root / "coverage_detector_sensitivity_reconstruction_audit.json"
    meta, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if meta["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["decoding_manifest"] != file_sha256(mp):
        raise ValueError("linked successful decoding and independent audit required")
    for name, digest in meta["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed source output: {name}")
    out.mkdir(parents=True, exist_ok=False)
    session = pd.read_csv(root / "coverage_detector_sensitivity_session_summary.csv")
    animal = pd.read_csv(root / "coverage_detector_sensitivity_animal_summary.csv")
    summary = pd.read_csv(root / "coverage_detector_sensitivity_summary.csv")
    denoms = primary(session).groupby(GROUP, as_index=False).agg(source_sessions=("session", "size"),
        available_sessions=("detector_available", "sum"), measurable_sessions=("continuity_fraction", "count"),
        eligible_windows=("eligible_events", lambda x: x.sum(min_count=1)))
    table = primary(summary).merge(denoms, on=GROUP, validate="many_to_one")
    table.to_csv(out / "coverage_detector_sensitivity_primary_table.csv", index=False)
    raw, interaction_animals, interactions = paired_interactions(session, meta["bootstraps"], meta["seed"] + 2)
    for name, df in [("session_interactions", raw), ("animal_interactions", interaction_animals), ("interactions", interactions)]:
        df.to_csv(out / f"coverage_detector_sensitivity_{name}.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = {1.: "#176b8e", .5: "#ba474d"}
    for i, dataset in enumerate(["pfeiffer_foster", "tanni2022"]):
        local = table[table.dataset.eq(dataset) & table.likelihood.eq("poisson") & table.bin_filter.eq("at_least_2cells_3spikes")]
        for fraction in [1., .5]:
            s = local[local.estimator.eq("map") & local.metric.eq("continuity_fraction") & local.cell_fraction.eq(fraction)]
            s = s.assign(cohort=[label(r) for r in s.itertuples(index=False)]).set_index("cohort").reindex(LABELS)
            axes[i, 0].plot(range(4), s.equal_animal_mean * 100, marker="o", color=colors[fraction], label="Full cells" if fraction == 1 else "Half cells")
            axes[i, 0].fill_between(range(4), s.ci95_low * 100, s.ci95_high * 100, alpha=.12, color=colors[fraction])
        a = primary(animal)
        a = a[a.dataset.eq(dataset) & a.likelihood.eq("poisson") & a.estimator.eq("map") & a.bin_filter.eq("at_least_2cells_3spikes") & a.cell_fraction.eq(.5)]
        for name, r in a.groupby("animal"):
            r = r.assign(cohort=[label(x) for x in r.itertuples(index=False)]).set_index("cohort").reindex(LABELS)
            axes[i, 1].plot(range(4), r.paired_continuity_delta * 100, marker="o", lw=1, label=name)
        s = local[local.estimator.eq("posterior_mean") & local.metric.eq("median_common_step_speed_delta_cm_s") & local.cell_fraction.eq(.5)]
        s = s.assign(cohort=[label(r) for r in s.itertuples(index=False)]).set_index("cohort").reindex(LABELS)
        axes[i, 2].plot(range(4), s.equal_animal_mean, marker="o", color="#297b54")
        axes[i, 2].fill_between(range(4), s.ci95_low, s.ci95_high, alpha=.15, color="#297b54")
        dataset_label = "Pfeiffer/Foster" if dataset == "pfeiffer_foster" else "Tanni (all arenas)"
        axes[i, 0].set_ylabel(f"{dataset_label}\nMAP continuity acceptance (%)")
        axes[i, 1].set_ylabel("Half minus full continuity (percentage points)")
        axes[i, 2].set_ylabel("Half minus full posterior-mean speed (cm/s)")
        axes[i, 0].legend(fontsize=9)
        axes[i, 1].legend(fontsize=8, ncol=2)
        for ax in axes[i]:
            ax.set_xticks(range(4), LABELS, rotation=20, ha="right")
            ax.axhline(0, lw=.6, color="black")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Geometric screening, not replay significance")
    axes[0, 1].set_title("Every animal's paired recording effect")
    axes[0, 2].set_title("Identical supported steps in both populations")
    fig.suptitle("Frozen event-definition sensitivity: independent flat-prior decoding\nPoisson; >=2 cells and >=3 spikes per frame; equal-animal means and animal-bootstrap intervals\nPF: native ripple tables. Tanni: new LFP envelope z >= 3, available in 23/25 sessions.", fontsize=12)
    fig.savefig(out / "coverage_detector_sensitivity_overview.png", dpi=160)
    plt.close(fig)
    lines = ["# Detector/Window Recording-Coverage Sensitivity", "", "Non-rescoring report. Geometric screening is not shuffle-significant replay.", "",
        f"Source: {meta['sessions']} sessions, {meta['eligible_windows']:,} eligible core/fixed windows, {meta['metric_rows']:,} metric rows. Windows and repeated analyses are not independent biological replicates.", "",
        "## Primary Paired Continuity Effect", "", "Poisson/MAP, >=2 cells and >=3 spikes. Effects are half minus full percentage points; intervals bootstrap animals after averaging subsets and sessions.", ""]
    rows = table[table.likelihood.eq("poisson") & table.estimator.eq("map") & table.bin_filter.eq("at_least_2cells_3spikes") & table.cell_fraction.eq(.5) & table.metric.eq("paired_continuity_delta")]
    for r in rows.itertuples(index=False):
        lines.append(f"- {r.dataset}, {label(r)}: {100*r.equal_animal_mean:.2f} [{100*r.ci95_low:.2f}, {100*r.ci95_high:.2f}] pp; {r.animals_measurable}/{r.animals_total} animals, {r.available_sessions}/{r.source_sessions} available sessions, {r.eligible_windows:g} windows.")
    lines += ["", "## Boundaries", "", "- MUA candidates, native PF ripple tables and newly detected Tanni ripple-like events are different assays; this is not precision/recall validation.",
        "- The two unavailable Tanni ripple sessions remain explicit, not zero-event sessions. No threshold was relaxed to fill them.",
        "- z4/z5 sensitivity and both/only/unknown strata remain in source summaries; overlap labels use primary z3.",
        "- Speed differences use the same supported non-overlapping steps, including all intervening frames. They are readout sensitivity, not known biological speed error.",
        "- Detector/window interactions compare different event sets within sessions, not randomized detector effects. No one setting is selected as optimal.",
        "- Published shuffle baselines, new-population transfer, and the integrated paper pack remain required. No biological uniformity claim."]
    (out / "coverage_detector_sensitivity_report.md").write_text("\n".join(lines) + "\n")
    result = build_script_provenance(input_paths={"decoding_manifest": mp, "audit": ap, "reporter": Path(__file__)}, cwd=ROOT)
    result.update(status="complete", rescoring=False, output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "coverage_detector_sensitivity_report_manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve())
