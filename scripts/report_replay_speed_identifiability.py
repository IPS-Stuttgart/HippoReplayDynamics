#!/usr/bin/env python3
"""Non-rescoring report of interval informativeness, coverage and transfer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts._provenance import build_script_provenance, file_sha256
from scripts.calibrate_replay_speed_identifiability import READOUT
from scripts.report_replay_coverage_map_mismatch import aggregate

METRICS = ["coverage", "finite_fraction", "finite_coverage", "median_finite_width", "nonzero_fraction",
           "equivalence_fraction"] + [f"{side}_equivalence_fraction_{bound:.2f}" for side in ["true", "false"] for bound in [.1, .25, .5]]


def monte_carlo_intervals(table):
    """Wilson intervals conditional on each fixed map and fitted calibration."""
    denominators = {name: "panels" for name in ["coverage", "finite_fraction", "nonzero_fraction", "equivalence_fraction"]}
    denominators["finite_coverage"] = "finite_panels"
    for bound in [.1, .25, .5]:
        denominators[f"true_equivalence_fraction_{bound:.2f}"] = f"inside_panels_{bound:.2f}"
        denominators[f"false_equivalence_fraction_{bound:.2f}"] = f"outside_panels_{bound:.2f}"
    keys = ["dataset", "animal", "session"]+READOUT+["generator", "observation", "stratum", "method"]
    z = float(norm.ppf(.975))
    frames = []
    for metric, denominator in denominators.items():
        frame = table[keys].copy()
        n = table[denominator].to_numpy(float)
        p = table[metric].to_numpy(float)
        available = n > 0
        if np.any((n < 0) | (n != np.floor(n))) or np.any(available & (~np.isfinite(p) | (p < 0) | (p > 1))):
            raise ValueError("invalid Monte Carlo counts/proportions")
        if np.any(~available & np.isfinite(p)):
            raise ValueError("zero denominator cannot have a proportion")
        count = np.rint(n*p)
        if not np.allclose((n*p)[available], count[available], atol=1e-7, rtol=0):
            raise ValueError("proportion does not represent integer Monte Carlo counts")
        lo, hi = np.full(len(n), np.nan), np.full(len(n), np.nan)
        nn, pp = n[available], p[available]
        center = (pp+z*z/(2*nn))/(1+z*z/nn)
        radius = z*np.sqrt(pp*(1-pp)/nn+z*z/(4*nn*nn))/(1+z*z/nn)
        lo[available], hi[available] = np.maximum(0, center-radius), np.minimum(1, center+radius)
        frame["metric"], frame["numerator"], frame["denominator"] = metric, count, n
        frame["estimate"], frame["mc95_low"], frame["mc95_high"] = p, lo, hi
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot(summary, selection, support, output):
    conditions = [("A", "poisson"), ("A", "shared_gain"), ("B", "poisson"), ("B", "shared_gain")]
    methods = [("raw_bootstrap", "Raw slope bootstrap", "#4773a3"), ("inverse_gaussian", "Inverse Gaussian", "#1d7b83"),
               ("inverse_conformal", "Inverse conformal", "#a94f62")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, (dataset, title) in enumerate([("pfeiffer_foster", "Pfeiffer/Foster maps"), ("tanni2022", "Tanni maps, all sizes")]):
        data = summary[summary.dataset.eq(dataset) & summary.estimator.eq("posterior_mean") & summary.selection.eq(selection)
                       & summary.bin_filter.eq(support) & summary.stratum.eq("uniform")]
        for row, (metric, label) in enumerate([("finite_fraction", "Finite intervals (%)"), ("coverage", "Coverage, including unbounded intervals (%)")]):
            for method, legend, color in methods:
                part = data[data.method.eq(method) & data.metric.eq(metric)].set_index(["generator", "observation"]).reindex(conditions)
                x = np.arange(4)
                axes[row, col].plot(x, part["mean"]*100, "o-", label=legend, color=color)
                axes[row, col].fill_between(x, part.ci95_low*100, part.ci95_high*100, color=color, alpha=.12)
            axes[row, col].set(xticks=np.arange(4), xticklabels=["A / Poisson", "A / gain", "B / Poisson", "B / gain"],
                               ylabel=label, ylim=(-3, 103))
            if row == 0:
                axes[row, col].set_title(title)
                axes[row, col].legend(fontsize=8)
            else:
                axes[row, col].axhline(95, ls=":", color=".35")
    fig.suptitle(f"{selection}; {support}; posterior-mean readout; independent simulated test panels\n"
                 "A = calibration-generator map; B = disjoint RUN map; both decoded with A\n"
                 "Uniform simulated g distribution; animal-bootstrap intervals; unbounded coverage is not useful inference", fontsize=11)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(root, out):
    files = {"scoring_manifest": root/"speed_identifiability_manifest.json", "audit": root/"speed_identifiability_reconstruction_audit.json",
        "summary": root/"speed_identifiability_session_summary.csv", "batches": root/"speed_identifiability_batches.csv",
        "reporter": Path(__file__), "aggregation_helper": ROOT/"scripts/report_replay_coverage_map_mismatch.py"}
    manifest = json.loads(files["scoring_manifest"].read_text())
    audit = json.loads(files["audit"].read_text())
    if manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["scoring_manifest"] != file_sha256(files["scoring_manifest"]):
        raise ValueError("matching complete scoring and passing reconstruction required")
    for name in ["summary", "batches"]:
        if file_sha256(files[name]) != manifest["output_sha256"][files[name].name]:
            raise ValueError("report input changed")
    out.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(files["summary"])
    monte_carlo_intervals(table).to_csv(out/"speed_identifiability_session_monte_carlo.csv", index=False)
    groups = ["dataset"]+READOUT+["generator", "observation", "stratum", "method"]
    sessions, animals, summary = aggregate(table, groups, METRICS, seed=20260915)
    for name, frame in [("session_endpoints", sessions), ("animal_endpoints", animals), ("summary", summary)]:
        frame.to_csv(out/f"speed_identifiability_{name}.csv", index=False)
    for selection, support in [("all", "unfiltered"), ("selected", "at_least_2cells_3spikes")]:
        plot(summary, selection, support, out/f"speed_identifiability_{selection}.png")
    batches = pd.read_csv(files["batches"])
    animal_counts = batches.groupby("dataset").animal.nunique().to_dict()
    text = ("# Speed-Interval Calibration and Transfer\n\nNon-rescoring report.\n\n"
        f"{len(batches)} sessions; {batches.panel_rows.sum()} readout rows; {batches.decision_rows.sum()} interval decisions. "
        "These are repeated synthetic panels, not independent biological events.\n\n"
        "Primary: posterior mean, 2-cell/3-spike window support, selected continuous run. All/unfiltered is a diagnostic. "
        "Report interval availability, width and all/finite-only coverage together. Finite intervals can still be too wide to inform the hypothesis. Infinite intervals cover the truth vacuously and are counted as abstentions.\n\n"
        "Inverse models and residual radii are fitted only to A/Poisson simulation panels. B maps and shared gain are transfer tests. "
        "Conformal coverage is marginal under the matching simulated gradient distribution, not guaranteed at fixed gradients or under transfer. "
        "Uniform-g panels test marginal coverage; fixed-g panels test conditional coverage/power and false equivalence at the declared boundaries.\n\n"
        "Equivalence means an interval is strictly within +/-0.25 (plus 0.10/0.50 sensitivity), not merely that zero is included. "
        "Boundary truths count as outside equivalence. Intervals are never clipped to the training range.\n\n"
        f"Sessions average within animals and animals equally; animal counts: {animal_counts}. Monte Carlo draws condition on their maps and source durations. "
        "The session Monte Carlo table gives Wilson 95% intervals with each metric's actual denominator; these condition on the learned inverse fit and calibration radius, and do not quantify uncertainty from refitting either. They are not biological or animal-level intervals. "
        "No new biological animals were held out and no real replay speed was used as known truth. "
        "This report cannot authorize biological equivalence; real event-definition sensitivity and baseline/transfer validation remain necessary.\n")
    (out/"speed_identifiability_report.md").write_text(text)
    provenance = build_script_provenance(input_paths=files, cwd=ROOT)
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}
    (out/"speed_identifiability_report_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve())
