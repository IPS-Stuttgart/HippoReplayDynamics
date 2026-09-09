#!/usr/bin/env python3
"""Non-rescoring report of independently audited dense-path recovery."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.dense_path_clocks import summarize
from scripts._provenance import build_script_provenance, file_sha256

LABELS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
NAMES = ("fits", "summary", "gates", "convergence")


def classify(gates, convergence):
    primary = gates[gates.support.eq(8192)]
    if len(primary) != 8 or len(convergence) != 8:
        raise ValueError("both datasets, teachers and scorer banks required")
    recovery = bool(primary.practical_pass.all())
    stable = bool(convergence.integration_stable.all())
    if not recovery:
        verdict = "dense_path_recovery_failed"
    elif not stable:
        verdict = "path_integration_not_stable"
    else:
        verdict = "simulation_recovery_pass_restricted_prior_only"
    return verdict, recovery, stable


def load_verified(run, audit_dir):
    mp, ap = run / "dense_path_clock_manifest.json", audit_dir / "dense_path_clock_audit.json"
    manifest, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("complete run and matching passing independent audit required")
    if manifest["oracle_knows_path"] or manifest["real_events_rescored"] or not manifest["all_scorer_libraries_independent"]:
        raise ValueError("unknown-path simulation with independent scorer banks required")
    if manifest["supports"] != [1024, 4096, 8192] or manifest["candidate_banks"] != [0, 1]:
        raise ValueError("frozen supports and banks required")
    if manifest["n_fits"] != 3600 or audit["population_fits_certified"] != 3600 or audit["paths_regenerated"] != 540672 or audit["likelihoods_recomputed"] != 594000:
        raise ValueError("complete frozen audit scope required")
    frames = {}
    for name in NAMES:
        p = run / f"dense_path_clock_{name}.csv"
        if file_sha256(p) != manifest["output_sha256"][p.name]:
            raise ValueError("table hash mismatch")
        frames[name] = pd.read_csv(p)
    recomputed = summarize(frames["fits"])
    for name, reference in zip(NAMES[1:], recomputed[1:], strict=True):
        pd.testing.assert_frame_equal(frames[name], reference, check_dtype=False, atol=1e-10, rtol=1e-8)
    return manifest, audit, frames


def plot_recovery(fits, output):
    colors = {1024: "#ac6727", 4096: "#247c85", 8192: "#923750"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True, sharey=True, layout="constrained")
    for i, (dataset, label) in enumerate(LABELS.items()):
        for j, teacher in enumerate(("original_bank_0", "original_bank_1")):
            ax = axes[i, j]
            group = fits[fits.dataset.eq(dataset) & fits.source_teacher.eq(teacher)]
            for support, color in colors.items():
                for bank, marker in ((0, "o"), (1, "s")):
                    points = group[group.support.eq(support) & group.candidate_bank.eq(bank)].groupby("scenario").phi_hat
                    mean = points.mean()
                    x = mean.index.to_numpy() + (-0.004 if bank == 0 else 0.004)
                    ax.plot(x, mean, marker=marker, linestyle="-" if bank == 0 else "--", markersize=4, color=color, label=f"{support:,} paths; bank {bank}")
                    ax.vlines(x, points.quantile(0.05), points.quantile(0.95), color=color, alpha=0.4, linewidth=1)
            ax.plot([0, 1], [0, 1], ":", color="0.45")
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0.25, 0.5, 0.75), yticks=np.arange(0, 1.01, 0.25), title=f"{label}: frozen teacher {j}" + (" (primary)" if j == 1 else ""))
            if i == 1:
                ax.set_xlabel("True neural-clock fraction among coherent events")
            if j == 0:
                ax.set_ylabel("Estimated neural-clock fraction")
    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Dense independent-path population recovery\nMean estimate and 5th-95th simulation percentiles; diagonal = exact recovery", fontsize=13)
    fig.savefig(output / "dense_path_clock_recovery.png", dpi=170)
    plt.close(fig)


def report(run, audit_dir, output):
    manifest, audit, frames = load_verified(run, audit_dir)
    fits, summary, gates, convergence = [frames[n] for n in NAMES]
    verdict, recovery, stable = classify(gates, convergence)
    output.mkdir(parents=True, exist_ok=False)
    for name in NAMES:
        shutil.copy2(run / f"dense_path_clock_{name}.csv", output)
    plot_recovery(fits, output)
    lines = [
        "# Dense-path clock recovery",
        "",
        f"Verdict: `{verdict}`. Recovery gates: {recovery}. Integration stability gates: {stable}.",
        "",
        "Non-rescoring simulation report. This is NOT a biological speed, uniformity, fuzzy-continuity or novelty result.",
        "",
        f"Producer commit: `{manifest['code_commit']}`.",
        f"Frozen observations reused: {manifest['observations']:,}; score rows: {manifest['score_rows']:,}; population fits: {manifest['n_fits']:,}.",
        f"Independent audit regenerated {audit['paths_regenerated']:,} scorer paths, recomputed {audit['likelihoods_recomputed']:,} likelihoods for {audit['observations_audited']:,} preselected observations, and certified all {audit['population_fits_certified']:,} mixture/profile optima.",
        "This audits every geometry path but a stratified subset of scores, not every production likelihood.",
        "",
        "Both scorer banks are independent of BOTH frozen teacher banks. Teacher 1 is the primary previously difficult corpus; teacher 0 is a second fixed corpus, not a matched scorer condition.",
        "",
        "## Recovery at 8,192 paths",
        "",
        "Targets unchanged: |bias| <=0.10, coverage >=0.90, directional power >=0.80, false direction at 50% <=0.05. These are measurement targets, not biological gates.",
        "",
        "| Dataset | Teacher | Scorer bank | Max absolute bias | Min coverage | Min power | False direction | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in gates[gates.support.eq(8192)].sort_values(["dataset", "source_teacher", "candidate_bank"]).itertuples():
        lines.append(
            f"| {LABELS[row.dataset]} | {row.source_teacher[-1]} | {row.candidate_bank} | {row.max_absolute_bias:.3f} | {row.min_coverage:.0%} | {row.min_direction_power:.0%} | {row.null_false_direction_fraction:.0%} | {row.practical_pass} |"
        )
    lines += [
        "",
        "## Primary corpus fraction estimates",
        "",
        "| Dataset | Scorer bank | True fraction | Mean estimate | Median profile interval width |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary[summary.support.eq(8192) & summary.source_teacher.eq("original_bank_1")].sort_values(["dataset", "candidate_bank", "scenario"]).itertuples():
        lines.append(f"| {LABELS[row.dataset]} | {row.candidate_bank} | {row.scenario:.2f} | {row.mean_estimate:.3f} | {row.median_interval_width:.3f} |")
    lines += [
        "",
        "## Numerical stability",
        "",
        "Median bank disagreement <=0.025, median 4,096-to-8,192 estimate shift <=0.025 and interval-endpoint shift <=0.05. CSVs also report p95 shifts. Stable integration alone does not establish accurate recovery.",
        "",
        "| Dataset | Teacher | Bank | Median bank disagreement | Median support shift | Median interval shift | Stable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in convergence.itertuples():
        lines.append(
            f"| {LABELS[row.dataset]} | {row.source_teacher[-1]} | {row.candidate_bank} | {row.median_cross_bank_phi_difference:.3f} | {row.median_support_phi_shift:.3f} | {row.median_support_interval_endpoint_shift:.3f} | {row.integration_stable} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- The target is the neural-clock fraction within the coherent mixture, not the fraction of all events or a speed in cm/s.",
        "- The true mixture has 60% coherent events, 20% stationary, and 10% each physical-reset/neural-reset. All five weights are fitted; reset events independently redraw a path in each bin.",
        "- These are fresh simulated spike identities from the parent study, with native 20 ms count profiles retained. This run reuses those observations unchanged; it does not decode new biological events.",
        "- There are 128 simulated events per recording, 50 repetitions and 33 fixed encoders from nine animals. Repetitions are not new animals; profile intervals condition on fixed encoders and finite teacher libraries.",
        "- Error-rate estimates from 50 repetitions are coarse. Wilson Monte Carlo intervals are in the summary CSV. A 3/50 false-direction rate misses the strict gate but is not by itself evidence of severe miscalibration.",
        "- The geometry prior is restricted to straight/sinusoidal paths with 40-120 cm endpoint distance. The neural clock uses conditional rate-code Hellinger arc length, not a complete model of neural dynamics.",
        "- Failure does not prove constant replay speed or absence of biological information. A pass would establish recovery only for this restricted simulation model.",
        "",
        "![Recovery](dense_path_clock_recovery.png)",
        "",
    ]
    (output / "dense_path_clock_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": run / "dense_path_clock_manifest.json", "audit": audit_dir / "dense_path_clock_audit.json"}, cwd=ROOT)
    provenance.update(
        verdict=verdict, recovery_pass=recovery, integration_stable=stable, real_events_rescored=False, output_sha256={p.name: file_sha256(p) for p in output.iterdir()}
    )
    (output / "dense_path_clock_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.run_dir, args.audit_dir, args.output_dir))
