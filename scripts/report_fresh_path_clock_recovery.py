#!/usr/bin/env python3
"""Non-rescoring report of fresh-path versus reused-library calibration."""

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

DATASETS = ("pfeiffer_foster", "tanni2022")
LABELS = ("Pfeiffer/Foster Rat1/Open1", "Tanni R2470, largest arena")
TEACHERS = ("fresh_stream_0", "fresh_stream_1")
KEYS = ["dataset", "stream", "scenario"]


def comparison_table(fresh, previous):
    fresh, previous = fresh.copy(), previous[previous.support.eq("exhaustive")].copy()
    fresh["stream"] = fresh.teacher.map(dict(zip(TEACHERS, (0, 1), strict=True)))
    previous["stream"] = previous.teacher.map({"original_bank_0": 0, "original_bank_1": 1})
    for frame in (fresh, previous):
        expected = {(d, s, f) for d in DATASETS for s in (0, 1) for f in (0.25, 0.5, 0.75)}
        if len(frame) != 12 or set(frame[KEYS].itertuples(index=False, name=None)) != expected or frame.duplicated(KEYS).any():
            raise ValueError("complete distinct two-recording source conditions required")
        if not np.isfinite(frame[["mean_estimate", "bias", "covered_fraction", "correct_direction_fraction"]]).all().all():
            raise ValueError("finite summary metrics required")
    columns = [*KEYS, "mean_estimate", "bias", "rmse", "median_interval_width", "covered_fraction", "correct_direction_fraction"]
    result = fresh[columns].merge(previous[columns], on=KEYS, suffixes=("_fresh", "_library"), validate="one_to_one")
    result["absolute_bias_reduction"] = result.bias_library.abs() - result.bias_fresh.abs()
    return result


def report(args):
    run, audit_dir, out = args.run_dir, args.audit_dir, args.output_dir
    mp, ap = run / "fresh_path_clock_manifest.json", audit_dir / "fresh_path_clock_audit.json"
    m, audit = json.loads(mp.read_text()), json.loads(ap.read_text())
    if (
        m["status"] != "complete"
        or audit["status"] != "pass"
        or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp)
        or audit["observations_regenerated"] != 76800
        or audit["fits_certified"] != 612
    ):
        raise ValueError("complete matching independently verified fresh-path run required")
    if m["real_events_rescored"] or m["all_33_encoders"] or m["finite_teacher_bank"]:
        raise ValueError("expected fresh-path simulation pilot only")
    previous = Path(m["previous_dir"])
    pp = previous / "exact_path_clock_manifest.json"
    if file_sha256(pp) != m["input_file_sha256"]["previous"]:
        raise ValueError("previous run manifest changed")
    pm = json.loads(pp.read_text())
    for directory, manifest in ((run, m), (previous, pm)):
        for name, digest in manifest["output_sha256"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("source output hash mismatch")
    fits, summary, gates, pooled = [pd.read_csv(run / f"fresh_path_clock_{name}.csv") for name in ("fits", "summary", "gates", "pooled_fits")]
    old_fits = pd.read_csv(previous / "exact_path_clock_fits.csv")
    old_summary = pd.read_csv(previous / "exact_path_clock_summary.csv")
    comparison = comparison_table(summary, old_summary)
    if len(gates) != 4 or gates.duplicated(["dataset", "teacher"]).any():
        raise ValueError("four unique recovery gates required")
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (("fits", fits), ("summary", summary), ("gates", gates), ("pooled_fits", pooled), ("library_comparison", comparison)):
        frame.to_csv(out / f"fresh_path_clock_{name}.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, layout="constrained")
    for i, (dataset, label) in enumerate(zip(DATASETS, LABELS, strict=True)):
        for j, stream in enumerate(TEACHERS):
            ax = axes[i, j]
            for frame, teacher, color, marker, title in (
                (old_fits, f"original_bank_{j}", "#a26927", "s", "Reused 256-path library"),
                (fits, stream, "#176a82", "o", "Fresh independent paths"),
            ):
                group = frame[frame.dataset.eq(dataset) & frame.teacher.eq(teacher) & frame.support.eq("exhaustive")].groupby("scenario").phi_hat
                mean = group.mean()
                ax.plot(mean.index, mean, color=color, marker=marker, label=title)
                ax.vlines(mean.index, group.quantile(0.05), group.quantile(0.95), color=color, alpha=0.45)
            ax.plot([0, 1], [0, 1], ":", color=".4")
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0.25, 0.5, 0.75), title=f"{label}\nSource schedule {j}")
            if i == 1:
                ax.set_xlabel("True neural fraction among coherent events")
            if j == 0:
                ax.set_ylabel("Estimated neural fraction")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Exhaustive integration: fresh paths versus reused source libraries\n128 simulated events per population; mean and 5th-95th simulation percentiles", fontsize=12)
    fig.savefig(out / "fresh_path_clock_recovery.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True, layout="constrained")
    for ax, dataset, label in zip(axes, DATASETS, LABELS, strict=True):
        for j, teacher in enumerate(TEACHERS):
            g = pooled[pooled.dataset.eq(dataset) & pooled.teacher.eq(teacher)].sort_values("scenario")
            color = ("#176a82", "#983d5a")[j]
            ax.plot(g.scenario, g.phi_hat, "o-", color=color, label=f"Fresh stream {j}")
            ax.vlines(g.scenario, g.phi_low, g.phi_high, color=color)
        ax.plot([0, 1], [0, 1], ":", color=".4")
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0.25, 0.5, 0.75), title=label, xlabel="True neural fraction")
    axes[0].set_ylabel("Estimated neural fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle("Secondary pooled calibration: 6,400 simulated events per condition\nConditional likelihood-profile intervals; two recording encoders only", fontsize=12)
    fig.savefig(out / "fresh_path_clock_pooled.png", dpi=170)
    plt.close(fig)
    verdict = "restricted_fresh_path_recovery_pass_needs_broader_validation" if gates.practical_pass.all() else "fresh_path_primary_recovery_incomplete"
    lines = [
        "# Fresh-Path Clock Calibration",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "Non-rescoring simulation report. This is a matched-prior diagnostic on two recording encoders, not a biological or uniform-speed claim.",
        "",
        f"Producer: `{m['code_commit']}`. Runtime: {m['runtime_s']:.1f} s. All 76,800 count observations independently regenerated; 612 mixture/profile fits certified.",
        "The audit checks every merged likelihood and sampled first/middle/last path blocks. It does not independently rescore every path/event likelihood.",
        "",
        "Fresh event paths use the full finite prior, without a persistent small teacher library. All five models and native spike-count profiles are unchanged. The target fraction is neural / (neural + physical), not a speed or fraction of all events.",
        "",
        "## Recovery Comparison",
        "",
        "| Dataset | Stream | True fraction | Reused-library mean | Fresh-path mean | Fresh interval coverage | Fresh directional power |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples():
        power = "not applicable" if row.scenario == 0.5 else f"{row.correct_direction_fraction_fresh:.0%}"
        lines.append(
            f"| {row.dataset} | {row.stream} | {row.scenario:.2f} | {row.mean_estimate_library:.3f} | {row.mean_estimate_fresh:.3f} | {row.covered_fraction_fresh:.0%} | {power} |"
        )
    lines += [
        "",
        "## Primary Gates",
        "",
        "Targets unchanged: bias <=0.10, coverage >=0.90, power >=0.80, false direction at true 50% <=0.05. Simulation uncertainty intervals remain in the CSV.",
        "",
        "| Dataset | Stream | Worst bias | Min coverage | Min power | False direction | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in gates.itertuples():
        lines.append(
            f"| {row.dataset} | {row.teacher} | {row.max_absolute_bias:.3f} | {row.min_coverage:.0%} | {row.min_direction_power:.0%} | {row.null_false_direction_fraction:.0%} | {row.practical_pass} |"
        )
    lines += [
        "",
        "## Secondary Pooled Fits",
        "",
        "Each pools all 50 repetitions into 6,400 observations, without adding animals. These fits were predeclared but do not replace primary gates or estimate interval coverage. Intervals condition on the model, encoder and finite path prior.",
        "",
        "| Dataset | Stream | True fraction | Estimate | Profile interval |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in pooled.itertuples():
        lines.append(f"| {row.dataset} | {row.teacher} | {row.scenario:.2f} | {row.phi_hat:.3f} | [{row.phi_low:.3f}, {row.phi_high:.3f}] |")
    lines += [
        "",
        "## Boundaries",
        "",
        "- Generation and scoring share the specified finite prior and encoder. This is a best-case model calibration, not robustness to different RUN maps, gain fluctuations, unknown paths outside the family, or biological model error.",
        "- Reused-library and fresh-path simulations preserve source schedules and bin counts but use different paths and spikes. Their difference is not a paired biological effect.",
        "- A fresh path can coincidentally equal an earlier path; sampling with replacement is intentional. Neither path labels nor generator identities enter event scoring.",
        "- A recovered pooled fraction is not reliable classification of individual replay events, nor validation of fuzzy continuity criteria.",
        "- No conclusion that real replay maintains constant physical speed, constant code-space speed or a compensatory neural mechanism follows from this calibration alone.",
        "",
        "![Fresh-path recovery](fresh_path_clock_recovery.png)",
        "",
        "![Pooled diagnostic](fresh_path_clock_pooled.png)",
        "",
    ]
    (out / "fresh_path_clock_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "audit": ap, "previous_manifest": pp}, cwd=ROOT)
    provenance.update(verdict=verdict, real_events_rescored=False, all_33_encoders=False, output_sha256={p.name: file_sha256(p) for p in out.iterdir()})
    (out / "fresh_path_clock_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(report(parser.parse_args()))
