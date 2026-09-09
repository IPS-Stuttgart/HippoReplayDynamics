#!/usr/bin/env python3
"""Report verified synthetic recovery without rescoring real events."""

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

CONDITIONS = ["matched", "gain_drift", "map_error"]
LABELS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
COLORS = ["#177e89", "#c65e24", "#7b4b94"]


def make_figures(summary, out):
    datasets = sorted(summary.dataset.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False, constrained_layout=True)
    native = summary[(summary.support == 1) & (summary.origin == "decoded")]
    panels = [
        (["balanced_metric_accuracy"], ["Physical vs neural"], 0.5, "Balanced accuracy"),
        (["physical_structured_detection", "neural_structured_detection"], ["Physical", "Neural"], 0.5, "Moving-event detection"),
        (["stationary_structured_detection", "iid_structured_detection"], ["Stationary", "IID positions"], 0.05, "Null false-positive rate"),
    ]
    for row, dataset in enumerate(datasets):
        for col, (metrics, legends, line, title) in enumerate(panels):
            ax = axes[row, col]
            for j, (metric, legend) in enumerate(zip(metrics, legends, strict=True)):
                frame = native[(native.dataset == dataset) & (native.metric == metric)].set_index("condition").loc[CONDITIONS]
                x = np.arange(3) + (j - (len(metrics) - 1) / 2) * 0.14
                ax.errorbar(x, frame["mean"], yerr=np.vstack([frame["mean"] - frame.ci_low, frame.ci_high - frame["mean"]]), fmt="o", capsize=3, label=legend, color=COLORS[j])
            ax.axhline(line, color="0.4", linestyle="--", linewidth=1)
            ax.set_xticks(range(3), ["Matched", "Gain drift", "Map error"])
            high = native[(native.dataset == dataset) & native.metric.isin(metrics)].ci_high.max()
            ax.set_ylim(-0.025, 1.025 if col < 2 else min(1.025, max(0.12, high * 1.1)))
            ax.set_title(LABELS.get(dataset, dataset) + ": " + title, fontsize=11)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8, loc="upper right" if col == 2 else "best")
    fig.suptitle("Synthetic recovery: native spike counts, independently decoded origin", fontsize=13)
    fig.savefig(out / "metric_finite_recovery_primary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    arms = [(1, "decoded"), (4, "decoded"), (1, "known"), (4, "known")]
    for ax, dataset in zip(axes, datasets, strict=True):
        for j, condition in enumerate(CONDITIONS):
            frame = (
                summary[(summary.dataset == dataset) & (summary.condition == condition) & (summary.metric == "balanced_metric_accuracy")].set_index(["support", "origin"]).loc[arms]
            )
            x = np.arange(4) + (j - 1) * 0.15
            ax.errorbar(
                x,
                frame["mean"],
                yerr=np.vstack([frame["mean"] - frame.ci_low, frame.ci_high - frame["mean"]]),
                fmt="o",
                capsize=3,
                label=condition.replace("_", " "),
                color=COLORS[j],
            )
        ax.axhline(0.5, linestyle="--", color=".4", linewidth=1)
        ax.set_xticks(range(4), ["Native\ndecoded", "4x counts\ndecoded", "Native\nknown origin", "4x counts\nknown origin"])
        ax.set_ylim(-0.025, 1.025)
        ax.set_title(LABELS.get(dataset, dataset))
        ax.set_ylabel("Physical/neural balanced accuracy")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Information sensitivities, not replacements for the primary", fontsize=12)
    fig.savefig(out / "metric_finite_recovery_sensitivity.png", dpi=180)
    plt.close(fig)


def report(run, audit_path, out):
    mp = run / "metric_finite_recovery_manifest.json"
    m, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("independently verified completed simulation required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed audited artifact: " + name)
    summary, checks, decision, animals = [pd.read_csv(run / f"metric_finite_recovery_{name}.csv.gz") for name in ("summary", "checks", "decision", "animals")]
    if len(decision) != 1 or decision.biological_mechanism_established.item() or decision.real_events_rescored.item():
        raise ValueError("simulation-only claim boundary violated")
    if checks.dataset.nunique() != 2 or len(checks) != 6:
        raise ValueError("incomplete primary report")
    out.mkdir(parents=True, exist_ok=False)
    make_figures(summary, out)
    lines = [
        "# Finite-Spike Metric Recovery",
        "",
        "Synthetic feasibility experiment; no real replay event was rescored.",
        "",
        f"- Frozen producer commit: `{m['code_commit']}`.",
        f"- {len(m['completed'])} recordings; five partitions; 64 count profiles per generator/recording (32 calibration, 32 evaluation).",
        "- Four generators, three observation conditions, native/fourfold counts, decoded/known origins.",
        f"- {m['rows']:,} score rows; {audit['predictive_scores_checked']:,} predictive scores independently reconstructed.",
        f"- Maximum reconstruction error: {audit['max_absolute_score_error']:.3g} nats.",
        "",
        "## Primary Native-Count Results",
        "",
        "| Dataset | Observation | Accuracy (conditional MC 95% CI) | Lowest animal accuracy | Lowest moving power | Highest null FPR | Ready |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in checks.itertuples():
        hi = summary[
            (summary.dataset == row.dataset)
            & (summary.condition == row.condition)
            & (summary.support == 1)
            & (summary.origin == "decoded")
            & (summary.metric == "balanced_metric_accuracy")
        ].ci_high.item()
        lines.append(
            f"| {LABELS.get(row.dataset, row.dataset)} | {row.condition} | {row.accuracy:.3f} [{row.accuracy_ci_low:.3f}, {hi:.3f}] | {row.min_animal_accuracy:.3f} | {row.min_moving_power:.3f} | {row.max_null_fpr:.3f} | {bool(row.ready)} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Decision",
            "",
            f"Native matched readiness: **{bool(decision.native_matched_ready.item())}**.",
            f"Robust native readiness (including both observation stresses): **{bool(decision.robust_native_ready.item())}**.",
            "",
            "Readiness requires both datasets' accuracy CI above chance, every animal above chance, at least 50% power for each moving generator, finite calibration, and no more than 5% empirical false positives for either null. The 50% power floor is a predeclared practical target, not a theorem. Calibration thresholds use separate stationary/iid simulations.",
            "",
            "## Interpretation Boundary",
            "",
            "The neural metric is Hellinger distance between conditional cell-identity distributions, not anatomical distance along a neural sheet. Both moving kernels have matched equilibrium occupancy, dwell and average off-diagonal entropy. These are stochastic movements, not constant-speed trajectories.",
            "",
            "Matched maps are optimistic; gain and map perturbations are controlled stress tests, not measured RUN-map uncertainty. A high known-origin or fourfold-spike result cannot rescue failed native decoded-origin recovery. A failed readiness gate is a limitation of this experiment and forecast statistic, not proof that physical and neural mechanisms are intrinsically indistinguishable.",
            "",
            "Intervals resample synthetic trial/profile indices jointly across generators, with equal recordings within animals and equal animals. They are conditional Monte Carlo intervals, not biological population intervals. Splits are collapsed before aggregation. The 40-ms scores sum proper conditional multinomial forecasts; they are not joint-sequence Bayes factors.",
            "",
            "Passing recovery would only justify a future preregistered real-data test. It would not establish a mechanism, speed uniformity, or a high-importance discovery.",
            "",
            "![Native-count recovery](metric_finite_recovery_primary.png)",
            "",
            "![Information sensitivities](metric_finite_recovery_sensitivity.png)",
            "",
        ]
    )
    (out / "metric_finite_recovery_report.md").write_text("\n".join(lines))
    checks.to_csv(out / "metric_finite_recovery_primary_table.csv", index=False)
    animals.to_csv(out / "metric_finite_recovery_animal_table.csv", index=False)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(status="complete", non_rescoring=True, biological_mechanism_established=False)
    p["output_sha256"] = {x.name: file_sha256(x) for x in sorted(out.iterdir())}
    (out / "metric_finite_recovery_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.run_dir, args.audit, args.output_dir)
