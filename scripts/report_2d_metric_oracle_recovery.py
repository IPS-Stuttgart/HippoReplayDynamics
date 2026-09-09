#!/usr/bin/env python3
"""Non-rescoring report of verified oracle classification diagnostics."""

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

METHODS = ["latent_path", "whole_exact", "whole_train_geometry", "lagged_full_geometry", "lagged_train_geometry"]
NAMES = ["True latent path", "Whole event: exact geometry", "Whole event: estimated geometry", "40-ms forecast: full RUN geometry", "40-ms forecast: training RUN geometry"]
DATASETS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
COLORS = ["#177e89", "#c65e24"]


def figures(summary, paired, output):
    datasets = sorted(summary.dataset.unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, dataset in enumerate(datasets):
        ax = axes[row, 0]
        sub = summary[summary.dataset.eq(dataset) & summary.metric.eq("balanced_accuracy")]
        for support, offset, color in ((1, -0.13, COLORS[0]), (4, 0.13, COLORS[1])):
            data = sub[sub.support.eq(support)].set_index("method").loc[METHODS[1:]]
            ax.errorbar(
                data["mean"],
                np.arange(1, 5) + offset,
                xerr=np.vstack([data["mean"] - data.ci_low, data.ci_high - data["mean"]]),
                fmt="o",
                capsize=3,
                color=color,
                label="Native counts" if support == 1 else "4x counts",
            )
        latent = sub[sub.method.eq("latent_path")].iloc[0]
        ax.errorbar(latent["mean"], 0, xerr=[[latent["mean"] - latent.ci_low], [latent.ci_high - latent["mean"]]], fmt="s", color="0.2", capsize=3, label="Latent path diagnostic")
        ax.set_yticks(range(5), NAMES, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color=".5", linestyle="--")
        ax.set_xlabel("Physical/neural balanced accuracy")
        ax.set_title(DATASETS.get(dataset, dataset))
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
        ax = axes[row, 1]
        for i, generator in enumerate(("physical", "neural")):
            points = []
            for method, support in (("latent_path", 0), ("whole_exact", 1), ("whole_exact", 4)):
                points.append(
                    summary[
                        (summary.dataset == dataset) & (summary.method == method) & (summary.support == support) & (summary.metric == generator + "_structured_detection")
                    ].iloc[0]
                )
            data = pd.DataFrame(points)
            ax.errorbar(
                np.arange(3) + (i - 0.5) * 0.15,
                data["mean"],
                yerr=np.vstack([data["mean"] - data.ci_low, data.ci_high - data["mean"]]),
                fmt="o",
                capsize=3,
                color=COLORS[i],
                label=generator.capitalize(),
            )
        ax.set_xticks(range(3), ["True path", "Whole event\nnative", "Whole event\n4x counts"])
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color=".5", linestyle="--")
        ax.set_ylabel("Moving-generator detection power")
        ax.set_title("Detection against calibrated static/IID controls")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic oracle recovery: geometry, observations and estimator", fontsize=14)
    fig.savefig(output / "metric_oracle_recovery.png", dpi=180)
    plt.close(fig)

    labels = {
        "whole_geometry_increment": "Full vs estimated geometry: whole event",
        "lagged_geometry_increment": "Full vs estimated geometry: forecast",
        "whole_information_inference_increment": "Whole event vs forecast: full geometry",
        "latent_information_gap": "True path vs whole-event observations",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, dataset in zip(axes, datasets, strict=True):
        for support, offset, color in ((1, -0.13, COLORS[0]), (4, 0.13, COLORS[1])):
            data = paired[(paired.dataset == dataset) & (paired.support == support)].set_index("contrast").loc[list(labels)]
            ax.errorbar(
                data["mean"] * 100,
                np.arange(4) + offset,
                xerr=np.vstack([data["mean"] - data.ci_low, data.ci_high - data["mean"]]) * 100,
                fmt="o",
                capsize=3,
                color=color,
                label="Native" if support == 1 else "4x counts",
            )
        ax.set_yticks(range(4), list(labels.values()), fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=".5", linestyle="--")
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(DATASETS.get(dataset, dataset))
        ax.set_xlabel("Paired accuracy difference (percentage points)")
        ax.legend(fontsize=8)
    fig.suptitle("Conditional simulation intervals; not biological effect estimates", fontsize=12)
    fig.savefig(output / "metric_oracle_accuracy_differences.png", dpi=180)
    plt.close(fig)


def report(run, audit_path, output):
    mp = run / "metric_oracle_manifest.json"
    m, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("verified completed run required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed audited input " + name)
    summary, paired, checks, decision = [pd.read_csv(run / f"metric_oracle_{name}.csv.gz") for name in ("summary", "paired", "checks", "decision")]
    if (
        len(checks) != 18
        or checks.dataset.nunique() != 2
        or len(decision) != 1
        or decision[["real_events_rescored", "biological_mechanism_established", "new_real_scoring_authorized"]].any().any()
    ):
        raise ValueError("invalid coverage or biological claim boundary")
    output.mkdir(parents=True, exist_ok=False)
    figures(summary, paired, output)
    lines = [
        "# Exact-Generator Metric Recovery",
        "",
        "Simulation diagnostic only. No real replay event was rescored or assigned a mechanism.",
        "",
        f"Producer commit: `{m['code_commit']}`. Recordings: {len(m['completed'])}. Score rows: {m['rows']:,}.",
        f"Independent reconstructions: {audit['independently_reconstructed_scores']:,}; copies checked against audited parent: {audit['verified_parent_copies']:,}; hmmlearn log-domain crosschecks: {audit['hmmlearn_log_crosschecks']:,}.",
        f"Maximum absolute score error: {audit['max_absolute_score_error']:.3g} nats.",
        "",
        "## Native-Count Classification",
        "",
        "| Dataset | Method | Accuracy (conditional MC 95% CI) | Lowest animal accuracy |",
        "|---|---|---:|---:|",
    ]
    for dataset in sorted(summary.dataset.unique()):
        for method, name in zip(METHODS, NAMES, strict=True):
            s = summary[
                (summary.dataset == dataset) & (summary.method == method) & (summary.support == (0 if method == "latent_path" else 1)) & (summary.metric == "balanced_accuracy")
            ].iloc[0]
            lines.append(f"| {DATASETS.get(dataset, dataset)} | {name} | {s['mean']:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] | {s.min_animal:.3f} |")
    lines.extend(["", "## Whole-Event Oracle Gates", "", "| Dataset | Support | Minimum moving power | Maximum empirical null FPR | Practical pass |", "|---|---:|---:|---:|---|"])
    for row in checks[checks.method.eq("whole_exact")].itertuples():
        lines.append(f"| {DATASETS.get(row.dataset, row.dataset)} | {row.support}x | {row.min_moving_power:.3f} | {row.max_null_fpr:.3f} | {row.operating_pass} |")
    lines.extend(
        [
            "",
            f"Both-dataset native whole-event operating pass: **{bool(decision.whole_event_oracle_native_operating_pass.item())}**.",
            f"Both-dataset latent-path operating pass: **{bool(decision.latent_path_operating_pass.item())}**.",
            "",
            "The frozen practical rule requires accuracy CI above chance, all animals above chance, at least 50% detection power for each moving generator, finite calibration, and empirical FPR no higher than 5% for either null. A practical failure is not an impossibility theorem. Fourfold support is diagnostic, not a replacement primary.",
            "",
            "## Interpretation",
            "",
            "Whole-exact uses the correct simulated transition and observation distributions. Under equal physical/neural priors its whole-event likelihood classifier minimizes expected error for those specified observations. It uses all time bins and both cell groups, so it is not held-out predictive validation. The true-path diagnostic receives strictly richer information. Finite simulation accuracy need not obey the expected information ordering exactly.",
            "",
            "Each simulated observation realization is classified separately, then accuracy and detection indicators are averaged within path/profile across five realizations. We never combine realization likelihoods to improve classification. The saved forecast baseline is reduced by this same rule; it is not numerically identical to the previous report's median-contrast classifier.",
            "",
            "The emission probability is a product of separate training/held conditional multinomials, respecting the generator's fixed group totals. These are stochastic physical-distance and conditional-identity-Hellinger kernels, not constant-speed paths or anatomical neural-sheet geometry. Full RUN geometry and held replay spikes are distinct information sources.",
            "",
            "Bootstrap intervals condition on the source maps and simulation/calibration design, with shared trial indices resampled jointly across generators, equal recordings within animals and equal animals. They are not biological population intervals. Null calibration uses separate simulated trials; empirical FPR is not a high-confidence population guarantee.",
            "",
            "Native whole-event success would motivate improved inference plus fresh observation-mismatch validation. It does not overturn the earlier forecast failure or authorize a real-data mechanism claim. Weak exact-generator recovery can only bound this particular comparison, not every possible physical/neural propagation theory.",
            "",
            "![Oracle comparison](metric_oracle_recovery.png)",
            "",
            "![Paired accuracy differences](metric_oracle_accuracy_differences.png)",
            "",
        ]
    )
    (output / "metric_oracle_report.md").write_text("\n".join(lines))
    for name, frame in (("summary", summary), ("paired", paired), ("checks", checks)):
        frame.to_csv(output / f"metric_oracle_{name}.csv", index=False)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(status="complete", non_rescoring=True, biological_mechanism_established=False)
    p["output_sha256"] = {x.name: file_sha256(x) for x in sorted(output.iterdir())}
    (output / "metric_oracle_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.run_dir, args.audit, args.output_dir)
