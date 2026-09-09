#!/usr/bin/env python3
"""Non-rescoring report of independently verified mixture recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

DATASETS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
CONDITIONS = {"exact": "Exact generating model", "train_geometry": "Training-cell geometry", "gain_drift": "Unmodeled rate drift"}
COLORS = {"exact": "#177e89", "train_geometry": "#a66c00", "gain_drift": "#ac4353"}


def figures(fits, summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    random = np.random.default_rng(521)
    for i, dataset in enumerate(DATASETS):
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i, j]
            group = fits[(fits.dataset == dataset) & (fits.condition == condition)]
            for n, color, offset in ((32, "#999999", -0.025), (128, COLORS[condition], 0.025)):
                sub = group[group.events_per_recording.eq(n)]
                ax.scatter(sub.scenario + offset + random.uniform(-0.012, 0.012, len(sub)), sub.phi_hat, s=12, alpha=0.35, color=color, label=f"{n} events / recording")
                mean = sub.groupby("scenario").phi_hat.mean()
                ax.scatter(mean.index + offset, mean, color=color, edgecolors="black", marker="D", s=36, zorder=3)
            ax.plot([0, 1], [0, 1], linestyle="--", color=".4", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([0.25, 0.5, 0.75])
            ax.set_title(f"{DATASETS[dataset]}\n{CONDITIONS[condition]}", fontsize=11)
            ax.set_xlabel("True neural fraction among moving events")
            if j == 0:
                ax.set_ylabel("Estimated neural fraction")
    fig.suptitle("Population mixture recovery: 50 independent simulation replicates\nGray: 32 events/recording; color: 128 events/recording; diamonds: means", fontsize=12)
    fig.savefig(output / "metric_population_recovery.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    primary = summary[summary.events_per_recording.eq(128)]
    for i, dataset in enumerate(DATASETS):
        for j, (metric, title) in enumerate((("covered", "Nominal 95% interval coverage"), ("claim", "Directional claims\nCorrect if enriched; false if equal"))):
            ax = axes[i, j]
            for c, condition in enumerate(CONDITIONS):
                group = primary[(primary.dataset == dataset) & (primary.condition == condition)].sort_values("scenario")
                field = ["covered"] * 3 if metric == "covered" else ["correct_direction", "directional_claim", "correct_direction"]
                y, lo, hi = [], [], []
                for (_, row), f in zip(group.iterrows(), field, strict=True):
                    y.append(row[f + "_fraction"])
                    lo.append(row[f + "_mc_low"])
                    hi.append(row[f + "_mc_high"])
                y = np.array(y)
                errors = np.array([y - np.array(lo), np.array(hi) - y])
                if np.min(errors) < -1e-12:
                    raise ValueError("Monte Carlo interval excludes its point estimate")
                ax.errorbar(group.scenario + (c - 1) * 0.025, y, yerr=np.maximum(errors, 0), fmt="o", capsize=3, color=COLORS[condition], label=CONDITIONS[condition])
            if metric == "covered":
                ax.axhline(0.95, color=".4", linestyle="--")
            else:
                ax.scatter([0.25, 0.5, 0.75], [0.8, 0.05, 0.8], marker="_", s=220, color="black", label="Operating target")
            ax.set_xticks([0.25, 0.5, 0.75], ["Physical-enriched", "Equal mixture", "Neural-enriched"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel("Fraction of simulated populations")
            ax.set_title(f"{DATASETS[dataset]}: {title}", fontsize=10)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("128 events per recording; bars show Monte Carlo Wilson intervals, not biological uncertainty", fontsize=12)
    fig.savefig(output / "metric_population_coverage.png", dpi=180)
    plt.close(fig)


def report(run, audit_path, output):
    mp = run / "metric_population_manifest.json"
    m, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("matching independently verified run required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed audited output")
    fits, summary, gates, decision = [pd.read_csv(run / f"metric_population_{name}.csv") for name in ("fits", "summary", "gates", "decision")]
    if (
        len(fits) != 1800
        or len(gates) != 12
        or len(decision) != 1
        or decision[["real_events_rescored", "biological_mechanism_established", "new_real_scoring_authorized"]].any().any()
    ):
        raise ValueError("invalid coverage or scientific boundary")
    output.mkdir(parents=True, exist_ok=False)
    figures(fits, summary, output)
    lines = [
        "# Population Metric Recovery",
        "",
        "Simulation only. No real replay population was assigned a mechanism.",
        "",
        f"Frozen producer: `{m['code_commit']}`. Recordings: 33; animals: 9; independent simulation replicates per scenario: 50.",
        f"Audited likelihoods: {audit['likelihoods_checked']:,}; regenerated paths: {audit['fresh_paths_regenerated']:,}; regenerated count arrays: {audit['count_arrays_regenerated']:,}; certified population fits: {audit['mixture_fits_checked']:,}.",
        "",
        "## Primary: 128 Events Per Recording",
        "",
        "1,024 events per simulated Pfeiffer/Foster population and 3,200 per simulated Tanni population.",
        "",
        "| Dataset | Condition | Worst absolute bias | Minimum coverage | Equal-mixture false direction | Minimum enriched power | Practical pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in gates[gates.events_per_recording.eq(128)].itertuples():
        lines.append(
            f"| {DATASETS[row.dataset]} | {CONDITIONS[row.condition]} | {row.max_absolute_bias:.3f} | {row.min_coverage:.2%} | {row.null_false_direction_fraction:.2%} | {row.min_direction_power:.2%} | {row.practical_pass} |"
        )
    lines += [
        "",
        f"Both-dataset exact-model practical pass: **{bool(decision.exact_population_recovery_pass.item())}**.",
        f"Both-dataset robustness practical pass: **{bool(decision.robust_population_recovery_pass.item())}**.",
        "",
        "## Interpretation Boundaries",
        "",
        "The target is the neural proportion among moving events, with true values .25, .50 and .75. Moving events make up .60 of the simulated population; stationary and independent-position events each make up .20. All four weights are fitted, so the decoder is not handed the true nuisance proportions. Every event contributes a likelihood mixture; no hard winner or continuity filter selects it.",
        "",
        "The physical/neural generators use stochastic physical-distance and conditional-identity Hellinger kernels. This does not test constant physical speed against constant anatomical neural-sheet speed. The exact arm knows the generating RUN geometry and emission model. Whole-event evidence uses all simulated spikes, not genuine held-out prediction.",
        "",
        "Fresh generator labels, paths and spike identities were sampled. Recorded count profiles are exogenous templates; repeated use of a template does not reuse its generated path/spikes. The 32-event cohort is nested in the 128-event cohort. Conditions and scenarios share aspects of the simulation design and are not independent biological replications.",
        "",
        "Profile intervals use an asymptotic 95% likelihood-ratio threshold; simulation coverage is measured explicitly. Monte Carlo Wilson intervals in the CSVs/figure describe uncertainty of the 50-replicate performance estimates. The practical screen uses coverage >=90%, absolute bias <=.10, null false direction <=5%, and power >=80% for both enriched mixtures. These are operating targets, not exact population guarantees.",
        "",
        "All recordings share a common generating composition in this first screen. Between-recording mixture heterogeneity, RUN map estimation uncertainty and other observation misspecification require further recovery before any real-data inference. The gain condition intentionally violates the decoder's stable-rate assumption. A failure there limits robustness, not the value of the true generating mixture.",
        "",
        "This experiment can establish or reject readiness of this specific population estimator. It cannot establish a biological propagation mechanism or a high-importance paper claim. Do not reinterpret a failed test by increasing its replicate count, pooling simulation replicates as events, or relaxing thresholds after seeing the results.",
        "",
        "![Mixture estimates](metric_population_recovery.png)",
        "",
        "![Coverage and directional claims](metric_population_coverage.png)",
        "",
    ]
    (output / "metric_population_report.md").write_text("\n".join(lines))
    for name, frame in (("summary", summary), ("gates", gates), ("decision", decision)):
        frame.to_csv(output / f"metric_population_{name}.csv", index=False)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(status="complete", non_rescoring=True, biological_mechanism_established=False)
    p["output_sha256"] = {f.name: file_sha256(f) for f in sorted(output.iterdir())}
    (output / "metric_population_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.run_dir, args.audit, args.output_dir)
