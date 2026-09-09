#!/usr/bin/env python3
"""Non-rescoring report of matched fine/coarse timing recovery."""

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
ORDER = ["coarse_identity", "fine_identity", "fine_timing", "fine_joint"]
LABELS = ["20 ms\nCell identity", "1 ms\nCell identity", "1 ms\nTiming only", "1 ms\nJoint"]


def report(run, audit_path, output):
    mp = run / "clock_timing_manifest.json"
    m = json.loads(mp.read_text())
    a = json.loads(audit_path.read_text())
    if m["status"] != "complete" or a["status"] != "pass" or a["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("matching audited complete run required")
    for f, h in m["output_sha256"].items():
        if file_sha256(run / f) != h:
            raise ValueError("changed audited artifact")
    summary = pd.read_csv(run / "clock_timing_summary.csv")
    animal = pd.read_csv(run / "clock_timing_animals.csv")
    paired = pd.read_csv(run / "clock_timing_paired_animals.csv")
    gains = paired.groupby(["dataset", "condition"], as_index=False)[["identity_timing_gain", "joint_timing_gain"]].mean()
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True, sharey=True)
    gen = np.random.default_rng(950)
    colors = ["#555555", "#177e89", "#a66c00", "#ac4353"]
    for row, (dataset, title) in enumerate(DATASETS.items()):
        for col, condition in enumerate(["exact", "gain_drift"]):
            ax = axes[row, col]
            for i, mode in enumerate(ORDER):
                values = animal[(animal.dataset == dataset) & (animal.condition == condition) & (animal.observation == mode)].oracle_correct
                ax.scatter(i + gen.uniform(-0.08, 0.08, len(values)), values, color=colors[i], s=45)
                ax.scatter(i, values.mean(), color="black", s=450, marker="_")
            ax.axhline(0.5, color=".5", linestyle="--", label="Chance")
            ax.axhline(0.8, color=".4", linestyle=":", label="Practical target")
            ax.set_xticks(range(4), LABELS)
            ax.set_ylim(0, 1)
            ax.set_title(f"{title}: " + ("exact emission" if condition == "exact" else "unmodeled rate drift"))
            if col == 0:
                ax.set_ylabel("Known-path oracle accuracy")
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Matched observations: does 20 ms binning discard clock information?\nSimulation only; points: source animals; bars: equal-animal means", fontsize=13)
    fig.savefig(output / "clock_timing_recovery.png", dpi=170)
    plt.close(fig)
    lines = [
        "# Paired Clock Timing Recovery",
        "",
        "Simulation only. The true geometric path is supplied to every scorer. This is not a real-replay classification or a blind decoder test.",
        "",
        f"Producer: `{m['code_commit']}`. {len(m['completed'])} source recordings. Independent audit checked {a['scores_checked']:,} score rows.",
        f"Numerical readiness: {a['numerical_readiness_pass']}; integration discrepancy {a['max_probability_error']:.3g}; parent-marginal discrepancy {a['max_parent_marginal_error']:.3g}.",
        "",
        "| Dataset | Emission | 20 ms identity | 1 ms identity | 1 ms timing only | 1 ms joint | Identity improvement | Joint improvement |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, title in DATASETS.items():
        for condition in ["exact", "gain_drift"]:
            row = summary[(summary.dataset == dataset) & (summary.condition == condition)].set_index("observation")
            gain = gains[(gains.dataset == dataset) & (gains.condition == condition)].iloc[0]
            values = " | ".join(f"{row.loc[mode, 'oracle_correct']:.2%}" for mode in ORDER)
            lines.append(f"| {title} | {condition} | {values} | {100 * gain.identity_timing_gain:+.2f} pp | {100 * gain.joint_timing_gain:+.2f} pp |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Fine joint scoring sees both cell identities and their 1-ms subbin timing. Fine identity conditions on the number of spikes in each 1-ms bin and discards its population-envelope likelihood. Timing-only discards the cell labels. Every arm uses the same simulated spikes; the coarse arm sums the fine subbins into the original 20-ms bins. Gains are unmodeled per-cell rate multipliers, not a change to the true clock.",
        "",
        "Expected fine-joint true-vs-wrong information is at least coarse information under exact emissions, as verified by the data-processing check. This does not guarantee higher empirical accuracy in every finite sample. Fine-identity alone need not satisfy that inequality because the conditioning changes. A joint-only improvement would not establish better spatial-sequence recovery.",
        "",
        "The rates are integrated over subbins; no HMM, momentum prior or temporal smoothing produces the result. The scorer nevertheless receives the true path, start and end locations and the original encoding map. This oracle privilege must be removed before any practical or biological mechanism classification. One millisecond here is a simulation observation discretization, not exact continuous-time spike likelihood.",
        "",
        "Repeated observations are averaged within path, then recordings within animal and animals within dataset. Animal markers are conditional simulation results using their source maps, not replicated biological findings. Practical target is >=80% accuracy in each dataset and above chance for every animal. The separately reported 5-percentage-point gain threshold is descriptive; there is no statistical-significance or biological-equivalence claim.",
        "",
        "These paths and native count profiles were fixed in the earlier clock experiment. Only fine count observations are new. Therefore this is a paired measurement diagnostic, not an independent data-set confirmation. Single-event classification failure cannot rule out population-level inference, longer trajectories, different code metrics, different spike counts or unknown mechanisms.",
        "",
        "No real events were scored; no continuity rule was tuned. This does not test fuzzy processing, prove uniform speed, or establish a high-importance paper result.",
        "",
        "![Matched timing recovery](clock_timing_recovery.png)",
        "",
    ]
    (output / "clock_timing_report.md").write_text("\n".join(lines))
    for name, df in [("summary", summary), ("animals", animal), ("paired_animals", paired), ("paired_summary", gains)]:
        df.to_csv(output / f"clock_timing_{name}.csv", index=False)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(status="complete", non_rescoring=True, biological_mechanism_established=False)
    p["output_sha256"] = {f.name: file_sha256(f) for f in output.iterdir()}
    (output / "clock_timing_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.run_dir, args.audit, args.output_dir)
