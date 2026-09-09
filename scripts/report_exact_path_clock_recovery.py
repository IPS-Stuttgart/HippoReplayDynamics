#!/usr/bin/env python3
"""Non-rescoring comparison of Monte Carlo and exhaustive finite-path inference."""

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
from scripts.run_exact_path_clock_recovery import MODELS, SUPPORTS, TEACHERS

LABELS = {"pfeiffer_foster": "Pfeiffer/Foster Rat1/Open1", "tanni2022": "Tanni R2470, largest arena"}
KEYS = ["dataset", "animal", "session", "teacher", "scenario", "repeat"]


def comparison_table(fits):
    if fits.empty or fits.duplicated([*KEYS, "support"]).any() or not fits.groupby(KEYS).support.agg(lambda s: set(s) == set(SUPPORTS)).all():
        raise ValueError("all distinct paired integration methods required")
    exact = fits[fits.support.eq("exhaustive")][KEYS + ["phi_hat", "phi_low", "phi_high"]].rename(columns={c: "exact_" + c for c in ("phi_hat", "phi_low", "phi_high")})
    result = fits[fits.support.ne("exhaustive")].merge(exact, on=KEYS, how="left", validate="many_to_one")
    result["estimate_error_vs_exact"] = result.phi_hat - result.exact_phi_hat
    result["absolute_error_vs_exact"] = abs(result.estimate_error_vs_exact)
    result["max_interval_endpoint_error_vs_exact"] = np.maximum(abs(result.phi_low - result.exact_phi_low), abs(result.phi_high - result.exact_phi_high))
    result["direction_agreement"] = np.sign(result.phi_low.gt(0.5).astype(int) - result.phi_high.lt(0.5).astype(int)) == np.sign(
        result.exact_phi_low.gt(0.5).astype(int) - result.exact_phi_high.lt(0.5).astype(int)
    )
    return result


def score_errors(metadata, exact, mc):
    if metadata.empty or exact.shape != (len(metadata), 5) or mc.shape != (2, 3, len(metadata), 5) or not np.isfinite(exact).all() or not np.isfinite(mc).all():
        raise ValueError("complete finite paired score arrays required")
    rows = []
    for (dataset, teacher, scenario, generator), group in metadata.groupby(["dataset", "source_teacher", "scenario", "generator"]):
        index = group.row_index.to_numpy()
        for bank in (0, 1):
            for hi, h in enumerate((1024, 4096, 8192)):
                residual = mc[bank, hi, index] - exact[index]
                columns = [*residual.T, residual[:, 1] - residual[:, 0]]
                for model, errors in zip((*MODELS, "neural_minus_physical"), columns, strict=True):
                    rows.append(
                        {
                            "dataset": dataset,
                            "teacher": teacher,
                            "scenario": scenario,
                            "generator": generator,
                            "support": f"mc_{h}_b{bank}",
                            "axis": model,
                            "n_events": len(index),
                            "mean_signed_score_error": errors.mean(),
                            "median_signed_score_error": np.median(errors),
                            "median_absolute_score_error": np.median(abs(errors)),
                            "p95_absolute_score_error": np.quantile(abs(errors), 0.95),
                        }
                    )
    return pd.DataFrame(rows)


def report(run, audit_dir, output):
    mp, ap = run / "exact_path_clock_manifest.json", audit_dir / "exact_path_clock_audit.json"
    m, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("complete run and matching passing audit required")
    if m["n_fits"] != 4200 or audit["fits_certified"] != 4200 or m["real_events_rescored"] or m["all_33_encoders"]:
        raise ValueError("fully certified two-recording simulation pilot required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("run output hash mismatch")
    fits = pd.read_csv(run / "exact_path_clock_fits.csv")
    summary = pd.read_csv(run / "exact_path_clock_summary.csv")
    gates = pd.read_csv(run / "exact_path_clock_gates.csv")
    comparisons = comparison_table(fits)
    dense = Path(m["dense_dir"])
    dense_manifest = json.loads((dense / "dense_path_clock_manifest.json").read_text())
    errors = []
    for tag in m["tags"]:
        p = dense / f"{tag}_scores.npy"
        if file_sha256(p) != dense_manifest["output_sha256"][p.name]:
            raise ValueError("Monte Carlo score hash mismatch")
        metadata = pd.read_csv(run / f"{tag}_observations.csv.gz")
        errors.append(score_errors(metadata, np.load(run / f"{tag}_exact_scores.npy", mmap_mode="r"), np.load(p, mmap_mode="r")))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("fits", "summary", "gates"):
        shutil.copy2(run / f"exact_path_clock_{name}.csv", output)
    comparisons.to_csv(output / "exact_path_clock_population_comparison.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(output / "exact_path_clock_score_error.csv", index=False)
    metrics = ["phi_hat", "exact_phi_hat", "estimate_error_vs_exact", "absolute_error_vs_exact", "max_interval_endpoint_error_vs_exact", "direction_agreement"]
    comparisons.groupby(["dataset", "teacher", "scenario", "support"], as_index=False)[metrics].mean().to_csv(output / "exact_path_clock_comparison_summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, layout="constrained")
    styles = (
        ("mc_8192_b0", "#247c85", "o", "8,192 paths, bank 0"),
        ("mc_8192_b1", "#ad6726", "s", "8,192 paths, bank 1"),
        ("exhaustive", "#923750", "D", "Exhaustive finite prior"),
    )
    for i, (dataset, label) in enumerate(LABELS.items()):
        for j, teacher in enumerate(TEACHERS):
            ax = axes[i, j]
            for method, color, marker, legend in styles:
                g = fits[fits.dataset.eq(dataset) & fits.teacher.eq(teacher) & fits.support.eq(method)].groupby("scenario").phi_hat
                mean = g.mean()
                ax.plot(mean.index, mean, marker=marker, color=color, label=legend, markersize=5)
                ax.vlines(mean.index, g.quantile(0.05), g.quantile(0.95), color=color, alpha=0.5)
            ax.plot([0, 1], [0, 1], ":", color="0.4")
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0.25, 0.5, 0.75), yticks=np.linspace(0, 1, 5), title=f"{label}\nFrozen source bank {j}")
            if i == 1:
                ax.set_xlabel("True neural-clock fraction among coherent events")
            if j == 0:
                ax.set_ylabel("Estimated neural-clock fraction")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Same-population clock recovery: Monte Carlo versus exhaustive integration\n128 simulated events per population; mean and 5th-95th simulation percentiles", fontsize=12
    )
    fig.savefig(output / "exact_path_clock_recovery.png", dpi=170)
    plt.close(fig)
    exact_gates = gates[gates.support.eq("exhaustive")]
    if len(exact_gates) != 4:
        raise ValueError("both datasets and teachers required")
    passed = bool(exact_gates.practical_pass.all())
    verdict = "restricted_two_recording_recovery_pass_needs_all_encoders" if passed else "exhaustive_prior_pilot_recovery_incomplete"
    lines = [
        "# Exhaustive path-clock calibration pilot",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "Non-rescoring report. Two fixed recording encoders only, including Tanni's largest arena. This is not a biological mechanism, uniform-speed, or publication-novelty claim.",
        "",
        f"Producer: `{m['code_commit']}`. Observations: {m['n_observations']:,}; mixture fits: {m['n_fits']:,}; runtime: {m['runtime_s']:.1f} s.",
        "Audit: first/middle/last start-index blocks independently recomputed in each recording; all output hashes, descriptor/proposal accounting, all weighted merges and all mixture/profile optima checked. Not every geometry likelihood was independently recomputed.",
        "",
        "## Exhaustive path counts",
        "",
        "| Recording | Straight paths | Signed curved paths |",
        "| --- | ---: | ---: |",
    ]
    for r in m["records"]:
        lines.append(f"| {r['tag']} | {r['accepted_straight']:,} | {r['accepted_curved']:,} |")
    lines += [
        "",
        "Each family has prior mass 0.5, normalized over its own valid paths. Reset models mix families within each bin, not once per event.",
        "",
        "## Fraction estimates",
        "",
        "| Dataset | Teacher | True fraction | MC 8,192 bank 0 | MC 8,192 bank 1 | Exhaustive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (dataset, teacher, scenario), g in summary.groupby(["dataset", "teacher", "scenario"]):
        x = g.set_index("support").mean_estimate
        lines.append(f"| {LABELS[dataset]} | {teacher[-1]} | {scenario:.2f} | {x['mc_8192_b0']:.3f} | {x['mc_8192_b1']:.3f} | {x['exhaustive']:.3f} |")
    lines += [
        "",
        "## Recovery gates",
        "",
        "Unchanged targets: absolute bias <=0.10, interval coverage >=0.90, directional power >=0.80, false direction at 50% <=0.05. Wilson simulation intervals remain in the CSV.",
        "",
        "| Dataset | Teacher | Worst bias | Min coverage | Min power | False direction | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in exact_gates.itertuples():
        lines.append(
            f"| {LABELS[row.dataset]} | {row.teacher[-1]} | {row.max_absolute_bias:.3f} | {row.min_coverage:.0%} | {row.min_direction_power:.0%} | {row.null_false_direction_fraction:.0%} | {row.practical_pass} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Comparisons use exactly the same 128-observation populations and both frozen teacher banks. Do not compare these one-recording values with older pooled-dataset estimates as if sample sizes were equal.",
        "- Exhaustive integration removes candidate-path sampling error for this finite endpoint/curve prior. It does not remove observation noise, finite-population estimation bias, finite teacher-library mismatch, or model misspecification.",
        "- All five mixture weights are fitted, including stationary and independent-per-bin reset controls. The target is neural / (neural + physical), not a fraction of all events or a speed in cm/s.",
        "- Likelihood residuals are Monte Carlo minus exhaustive. A positive neural-minus-physical residual means numerical approximation favors the neural clock relative to the exhaustive calculation. Source generator labels are used only for evaluation, never scoring or path selection.",
        "- Fifty repetitions are simulated observations, not independent animals. This pilot contains only two animals; it cannot establish generality across either dataset.",
        "- A failed recovery gate is not evidence that real replay has uniform speed. A pass would still require the full encoder set and broader geometry checks before a biological mechanism inference.",
        "",
        "![Same-population recovery](exact_path_clock_recovery.png)",
        "",
    ]
    (output / "exact_path_clock_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "audit": ap}, cwd=ROOT)
    provenance.update(verdict=verdict, real_events_rescored=False, all_33_encoders=False, output_sha256={p.name: file_sha256(p) for p in output.iterdir()})
    (output / "exact_path_clock_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.run_dir, args.audit_dir, args.output_dir))
