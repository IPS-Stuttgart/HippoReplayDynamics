#!/usr/bin/env python3
"""Non-rescoring report of audited unknown-path clock population recovery."""

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

from scripts._provenance import build_script_provenance, file_sha256


def report(run, audit_dir, output):
    mp, ap = run / "unknown_path_clock_manifest.json", audit_dir / "unknown_path_clock_audit.json"
    manifest, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("complete run and matching passing independent audit required")
    if manifest["oracle_knows_path"] or manifest["real_events_rescored"] or audit["population_fits_certified"] != manifest["n_fits"]:
        raise ValueError("unknown-path simulation and complete numerical certification required")
    for name in ("fits", "summary", "gates"):
        p = run / f"unknown_path_clock_{name}.csv"
        if file_sha256(p) != manifest["output_sha256"][p.name]:
            raise ValueError("summary hash mismatch")
    fits = pd.read_csv(run / "unknown_path_clock_fits.csv")
    summary = pd.read_csv(run / "unknown_path_clock_summary.csv")
    gates = pd.read_csv(run / "unknown_path_clock_gates.csv")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("fits", "summary", "gates"):
        shutil.copy2(run / f"unknown_path_clock_{name}.csv", output)
    weights = [c for c in fits if c.startswith("weight_")]
    fits.groupby(["dataset", "teacher", "support", "scenario"], as_index=False)[weights].mean().to_csv(output / "unknown_path_clock_mean_mixture_weights.csv", index=False)
    labels = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
    colors = {128: "#ad6624", 256: "#187382"}
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.2), sharex=True, sharey=True, layout="constrained")
    for i, dataset in enumerate(labels):
        for j, teacher in enumerate(("matched", "independent")):
            ax = axes[i, j]
            group = fits[fits.dataset.eq(dataset) & fits.teacher.eq(teacher)]
            for support in (128, 256):
                points = group[group.support.eq(support)].groupby("scenario").phi_hat
                means = points.mean()
                lower, upper = points.quantile(0.05), points.quantile(0.95)
                x = means.index.to_numpy() + (-0.008 if support == 128 else 0.008)
                ax.plot(x, means, "o-", color=colors[support], label=f"{support} candidate paths")
                ax.vlines(x, lower, upper, color=colors[support], linewidth=2)
            ax.plot([0, 1], [0, 1], ":", color="0.45", label="Exact recovery")
            ax.axhline(0.5, color="0.8", linewidth=0.7)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0.25, 0.5, 0.75), yticks=np.arange(0, 1.01, 0.25), title=f"{labels[dataset]}: {teacher} teacher paths")
            if i == 1:
                ax.set_xlabel("True neural-clock fraction among coherent events")
            if j == 0:
                ax.set_ylabel("Estimated neural-clock fraction")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Population recovery without supplied true paths\nPoints: mean estimate; bars: 5th-95th Monte Carlo percentiles", fontsize=13)
    fig.savefig(output / "unknown_path_clock_recovery.png", dpi=170)
    plt.close(fig)

    primary = gates[gates.support.eq(256)]
    matched_pass = bool(primary[primary.teacher.eq("matched")].practical_pass.all())
    independent_pass = bool(primary[primary.teacher.eq("independent")].practical_pass.all())
    if not matched_pass:
        verdict = "population_clock_recovery_not_ready_even_with_matched_path_library"
    elif not independent_pass:
        verdict = "matched_library_recovery_only_not_robust_to_unknown_geometry"
    else:
        verdict = "simulation_recovery_pass_restricted_path_prior_only"
    lines = [
        "# Unknown-path population clock recovery",
        "",
        "Non-rescoring report. No real replay events were rescored. No biological mechanism or uniform-speed claim is established.",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"Producer commit: `{manifest['code_commit']}`.",
        f"Run manifest SHA256: `{file_sha256(mp)}`.",
        f"Fresh observations: {manifest['n_observations']:,}; score rows: {manifest['n_rows']:,}; population fits: {manifest['n_fits']:,}.",
        f"Independent audit: {audit['observations_regenerated']:,} observations regenerated, {audit['likelihoods_checked']:,} likelihoods checked, {audit['population_fits_certified']:,} fits certified.",
        "",
        "## Practical recovery gates",
        "",
        "Targets: |bias| <=0.10; coverage >=0.90; directional power >=0.80 at both 25% and 75%; false directional claims <=0.05 at 50%.",
        "These are measurement targets, not biological gates. Monte Carlo error bars are reported in the CSVs.",
        "",
        "| Dataset | Teacher | Scorer paths | Max absolute bias | Min coverage | Min power | False direction at 50% | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in gates.sort_values(["dataset", "teacher", "support"]).itertuples():
        lines.append(
            f"| {labels[row.dataset]} | {row.teacher} | {row.support} | {row.max_absolute_bias:.3f} | {row.min_coverage:.0%} | {row.min_direction_power:.0%} | {row.null_false_direction_fraction:.0%} | {bool(row.practical_pass)} |"
        )
    lines += [
        "",
        "## Fraction estimates with 256 candidate paths",
        "",
        "| Dataset | Teacher | True fraction | Mean estimate | Median profile interval width |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary[summary.support.eq(256)].sort_values(["dataset", "teacher", "scenario"]).itertuples():
        lines.append(f"| {labels[row.dataset]} | {row.teacher} | {row.scenario:.2f} | {row.mean_estimate:.3f} | {row.median_interval_width:.3f} |")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Both coherent clock models marginalize over the same geometric path library. The scorer receives neither true path indices nor generator labels.",
        "- Matched/256 is a favorable finite-library calibration. Independent/256 generates new geometric paths from the same prior. Matched/128 also has missing path support and is not exact-model recovery.",
        "- The true mixture is 60% coherent, 20% stationary, 10% physical-reset and 10% neural-reset. All five weights are fitted. Reset events independently redraw a path each bin; they are not literal time-bin shuffles.",
        "- Each population has 128 events per recording: 1024 PF and 3200 Tanni. Monte Carlo replicates are new simulated observations, not new animals. Intervals condition on fixed maps and geometry libraries.",
        "- Fifty repetitions give coarse error-rate estimates. This is not an animal bootstrap or evidence of biological equivalence.",
        "- Paths are restricted to straight/sinusoidal curves, 40-120 cm endpoint distances. The neural clock is a conditional rate-code Hellinger metric, not a complete model of neural dynamics.",
        "- A failure may reflect low information or insufficient path support. It is not evidence that replay speed is constant and does not validate or invalidate fuzzy continuity.",
        "- No claim of novelty or a high-importance paper follows from this recovery experiment alone.",
        "",
        "![Population recovery](unknown_path_clock_recovery.png)",
        "",
    ]
    (output / "unknown_path_clock_population_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "audit": ap}, cwd=ROOT)
    provenance.update(verdict=verdict, real_events_rescored=False, output_sha256={p.name: file_sha256(p) for p in output.iterdir()})
    (output / "unknown_path_clock_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.run_dir, args.audit_dir, args.output_dir))
