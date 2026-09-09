#!/usr/bin/env python3
"""Non-rescoring report of an independently audited known-path clock screen."""

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
COLORS = {"exact": "#177e89", "gain_drift": "#ac4353"}


def report(run, audit_path, output):
    mp = run / "literal_clock_manifest.json"
    m, a = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if m["status"] != "complete" or a["status"] != "pass" or a["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("matching complete independently audited run required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed audited output")
    table = pd.concat([pd.read_csv(run / f"{x['tag']}_scores.csv.gz") for x in m["completed"]], ignore_index=True)
    summary = pd.read_csv(run / "literal_clock_summary.csv")
    animals = pd.read_csv(run / "literal_clock_animals.csv")
    table["speed_ratio"] = table.decoded_mean_step_speed_cm_s / table.true_bin_mean_step_speed_cm_s
    fields = ["speed_ratio", "posterior_mean_error_cm", "posterior_rms_cm", "hard_continuous_step_fraction", "true_arc_speed_cv"]
    keys = ["dataset", "animal", "session", "condition", "generator", "decoder_grid_cm"]
    paths = table.groupby(keys + ["path_id"], as_index=False)[fields].mean()
    recordings = paths.groupby(keys, as_index=False)[fields].mean()
    speed_animals = recordings.groupby([k for k in keys if k != "session"], as_index=False)[fields].mean()
    speed_summary = speed_animals.groupby(["dataset", "condition", "generator", "decoder_grid_cm"], as_index=False)[fields].mean()
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    jitter = np.random.default_rng(331)
    for row, (dataset, title) in enumerate(DATASETS.items()):
        ax = axes[row, 0]
        for j, condition in enumerate(COLORS):
            sub = animals[(animals.dataset == dataset) & (animals.condition == condition)]
            ax.scatter(j + jitter.uniform(-0.08, 0.08, len(sub)), sub.oracle_correct, color=COLORS[condition], s=50)
            ax.scatter(j, sub.oracle_correct.mean(), marker="_", s=600, color="black")
        ax.axhline(0.5, color=".5", linestyle="--", label="Chance")
        ax.axhline(0.8, color=".3", linestyle=":", label="Practical target")
        ax.set_xticks([0, 1], ["Exact emission", "Unmodeled rate drift"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Known-path oracle accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower left")
        ax = axes[row, 1]
        sub = speed_animals[(speed_animals.dataset == dataset) & (speed_animals.condition == "exact")]
        for j, (truth, grid) in enumerate([("physical", 8), ("physical", 16), ("neural", 8), ("neural", 16)]):
            values = sub[(sub.generator == truth) & (sub.decoder_grid_cm == grid)].speed_ratio
            ax.scatter(j + jitter.uniform(-0.08, 0.08, len(values)), values, color="#177e89" if truth == "physical" else "#a66c00", s=45)
            ax.scatter(j, values.mean(), marker="_", s=500, color="black")
        ax.axhline(1, color=".4", linestyle="--")
        ax.set_xticks(range(4), ["Physical\n8 cm", "Physical\n16 cm", "Code\n8 cm", "Code\n16 cm"])
        ax.set_ylabel("Decoded / true bin-mean step speed")
        ax.set_title(f"{title}: independent flat-prior decoder")
    fig.suptitle("Simulated clocks, not biological replay results\nPoints: source animals; bars: equal-animal means", fontsize=13)
    fig.savefig(output / "literal_clock_recovery.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, (dataset, title) in enumerate(DATASETS.items()):
        item = min((x for x in m["completed"] if x["dataset"] == dataset), key=lambda x: x["tag"])
        with np.load(run / f"{item['tag']}_p000.npz") as z:
            points = z["points"]
            s = z["s"]
            n = len(z["totals"])
            ax = axes[row, 0]
            ax.plot(points[:, 0], points[:, 1], color="#333333")
            ax.scatter(points[[0, -1], 0], points[[0, -1], 1], c=["#177e89", "#ac4353"])
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            ax.set_title(f"{title}: same path for both clocks")
            ax = axes[row, 1]
            for model, color in [("physical", "#177e89"), ("neural", "#a66c00")]:
                ax.plot(z[f"clock_{model}"] * n * 20, s, label="Physical clock" if model == "physical" else "Population-code clock", color=color)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Distance along shared path (cm)")
            ax.legend(fontsize=8)
            ax.set_title(f"{title}: different traversal timing")
    fig.suptitle("Fixed examples: path 0 of the first recording in each dataset", fontsize=12)
    fig.savefig(output / "literal_clock_examples.png", dpi=170)
    plt.close(fig)
    lines = [
        "# Literal Replay Clock Recovery",
        "",
        "Simulation only. The scorer knows the true geometric path and its endpoints. No real replay mechanism is classified.",
        "",
        f"Frozen producer: `{m['code_commit']}`. {m['n_observations']:,} simulated observations; {len(m['completed'])} encoders; {len({x['animal'] for x in m['completed']})} source animals.",
        f"Independent audit: {a['score_rows_checked']:,} score/decoding rows. Maximum 128-vs-256 quadrature probability error: {a['quadrature_max_probability_error']:.3g}. Class-label flips: {a['quadrature_label_flips']}. Numerical readiness: {a['numerical_readiness_pass']}.",
        "",
        "| Dataset | Emission | Equal-animal accuracy | Least accurate animal | Practical screen |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {DATASETS[row.dataset]} | {row.condition} | {row.oracle_correct:.1%} | {row.minimum_animal_accuracy:.1%} | {row.oracle_practical_pass} |")
    lines += [
        "",
        "## Meaning And Limits",
        "",
        "This screen asks whether the two explicitly specified clocks can be distinguished when geometry is supplied. It does not test a blind estimator, event detection, or biological speed uniformity. A pass only motivates a blind recovery experiment; a failure blocks this proposed route at the tested operating target, not all possible mechanisms.",
        "",
        "Both clocks share each path, duration, and native total-spike profile. Rates are integrated over each 20 ms bin. The code metric is Hellinger arc length of conditional cell-identity probabilities from RUN maps, not anatomical distance between cells. The neural clock has constant code arc-length speed by construction; its physical speed need not be constant.",
        "",
        "The oracle uses conditional-multinomial identities, omitting the common coefficient. The grid decoder is independent in each time bin with a uniform spatial prior. Neither uses an HMM or momentum prior. This conditional likelihood is not the full Poisson observation model used in some earlier analyses. Gain drift perturbs cell emissions but is unknown to the scorer.",
        "",
        "The speed panel divides decoded posterior-mean step speed by step speed of the true within-bin time-mean positions, using the same temporal spacing. It is not a direct estimate of continuous instantaneous speed. The coarse grid averages fine-bin rates and centers, with partial edge groups. Source maps are treated as known and their finite sampling uncertainty is not modeled.",
        "",
        "Observation repeats are averaged within paths, then recordings within animals; animal points are not independent new biological findings. No continuity filter is used. The reported <20 cm step fraction is not the full Foster trajectory-event heuristic, and is not a validated fuzzy classifier. Stationary/fragmented nulls and unknown-path inference remain necessary before real-event classification.",
        "",
        "![Recovery and decoder speed](literal_clock_recovery.png)",
        "",
        "![Explicit clock examples](literal_clock_examples.png)",
        "",
    ]
    (output / "literal_clock_report.md").write_text("\n".join(lines))
    for name, frame in [("summary", summary), ("animals", animals), ("decoder_summary", speed_summary), ("decoder_animals", speed_animals)]:
        frame.to_csv(output / f"literal_clock_{name}.csv", index=False)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(status="complete", non_rescoring=True, biological_mechanism_established=False)
    p["output_sha256"] = {f.name: file_sha256(f) for f in sorted(output.iterdir())}
    (output / "literal_clock_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.run_dir, args.audit, args.output_dir)
