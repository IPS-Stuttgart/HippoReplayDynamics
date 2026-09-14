#!/usr/bin/env python3
"""Non-rescoring report for the independently audited spatial diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_md(frame):
    def text(x):
        return (f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x)).replace("|", "\\|")

    return "\n".join(
        ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        + ["| " + " | ".join(map(text, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    )


def report(root, audit_path, output):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit["manifest_sha256"] != sha(root / "manifest.json"):
        raise ValueError("independent audit absent/stale")
    for name, digest in manifest["output_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"changed artifact {name}")
    frames = {name: pd.read_csv(root / f"{name}.csv") for name in ("session_summary", "animal_summary", "summary", "correlations", "gates")}
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    summary = frames["summary"]
    real = summary[summary.source.isin(["all_fixed_candidates", "full_accepted_segment"])].copy()
    primary = real[(real.source == "all_fixed_candidates") & (real.encoding == "early_run")]
    truth = summary[~summary.source.isin(["all_fixed_candidates", "full_accepted_segment"])].copy()
    truth["balanced_mean_error"] = 0.5 * (truth.balanced_high_error + truth.balanced_low_error)
    truth["balanced_mean_brier"] = 0.5 * (truth.balanced_high_brier + truth.balanced_low_brier)
    passed = bool(frames["gates"].set_index("gate").loc["development_numerical_screen", "passed"])
    known = frames["session_summary"]
    known = known[known.source.isin(truth.source.unique()) & known.method.eq("predictive_half")]
    minimum_class = known[["class0_retention", "class1_retention"]].min().min()
    lines = [
        "# Within-bin spatial predictive diagnostic",
        "",
        "Independent technical reconstruction: PASS.",
        f"Frozen development screen: {'PASS' if passed else 'FAIL'}.",
        "External validation: NOT RUN. No validated remedy claim.",
        "",
        "The diagnostic tests shared-location consistency of spike identities in three internal cell folds. "
        "Both original populations contribute to ranking. It is not A-only prediction of an unused B, "
        "not a temporal sequence test, and not an estimate of biological replay truth.",
        "",
        "No posterior or event time is changed. Exactly half of each source is retained. This is a conditional subset, not corrected all-event prevalence.",
        "",
        "## Original Home-content contrasts",
        "",
        table_md(real[["source", "encoding", "method", "home_gap", "high_home", "low_home"]]),
        "",
        "## Primary real disagreement and controls",
        "",
        table_md(primary[["method", "separation", "regional_tv", "high_entropy", "low_entropy", "zero_score_fraction"]]),
        "",
        "## Class-balanced truth safeguards",
        "",
        "Each true class has equal weight. Each session is averaged within rat, then rats equally. "
        "Class balancing prevents removal of difficult Home observations from manufacturing an improvement.",
        "",
        table_md(truth[["source", "method", "balanced_mean_error", "balanced_mean_brier"]]),
        "",
        f"Minimum retained fraction over all native/synthetic sessions and both truth classes: {minimum_class:.6f}.",
        "",
        "## Predictive association",
        "",
        table_md(frames["correlations"]),
        "",
        "Correlations use within-session ranks and equal session means within rat. The three rats, not the number of decoded bins, limit replication.",
        "",
        "## Gates",
        "",
        table_md(frames["gates"][["gate", "passed"]]),
        "",
        f"Producer commit: `{manifest['code_commit']}`.",
        f"Source manifest SHA256: `{sha(root / 'manifest.json')}`.",
        f"Independent audit SHA256: `{sha(audit_path)}`.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    methods = [("all", "All events", "#657783"), ("predictive_half", "Predictive half", "#bb4c56")]
    labels = [(enc, src) for enc in ("early_run", "full_run") for src in ("all_fixed_candidates", "full_accepted_segment")]
    for j, (method, label, color) in enumerate(methods):
        r = real[real.method.eq(method)].set_index(["encoding", "source"])
        axes[0, 0].barh(np.arange(4) + (j - 0.5) * 0.36, [100 * r.loc[key, "home_gap"] for key in labels], height=0.34, color=color, label=label)
    axes[0, 0].set(
        yticks=np.arange(4),
        yticklabels=[f"{e.split('_')[0]} / {'candidates' if s == 'all_fixed_candidates' else 'accepted'}" for e, s in labels],
        xlabel="Absolute population Home-mass gap (pp)",
        title="Original content target",
    )
    axes[0, 0].legend(frameon=False)
    baseline = primary.set_index("method").loc["all"]
    p = primary.set_index("method").loc[["predictive_half", "spike_half", "entropy_half"]]
    for j, metric in enumerate(("separation", "regional_tv")):
        axes[0, 1].barh(np.arange(3) + (j - 0.5) * 0.36, 100 * (p[metric] / baseline[metric] - 1), height=0.34, label=metric)
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set(yticks=np.arange(3), yticklabels=["Predictive", "Spike count", "Entropy"], xlabel="Change versus all events (%)", title="Real disagreement: negative is better")
    axes[0, 1].legend(frameon=False)
    sources = ["run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly"]
    names = ["RUN-Q4", "Matched Poisson", "Gain x4", "Event-total counts", "Map drift", "Shared assembly"]
    for ax, metric, unit in ((axes[1, 0], "balanced_mean_error", "cm"), (axes[1, 1], "balanced_mean_brier", "Brier error")):
        for j, (method, label, color) in enumerate(methods):
            t = truth[truth.method.eq(method)].set_index("source").loc[sources]
            ax.barh(np.arange(6) + (j - 0.5) * 0.36, t[metric], height=0.34, color=color)
        ax.set(yticks=np.arange(6), yticklabels=names, xlabel=unit, title="Class-balanced known truth")
    fig.suptitle("Spatial predictive screening: development, not validated biological replay content")
    fig.savefig(output / "spatial_predictive_screen.png", dpi=160)
    plt.close(fig)
    meta = dict(
        source_manifest_sha256=sha(root / "manifest.json"),
        audit_sha256=sha(audit_path),
        report_script_sha256=sha(__file__),
        development_screen_passed=passed,
        external_validation=False,
        files={p.name: sha(p) for p in output.iterdir()},
    )
    (output / "report_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", required=True, type=Path)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    report(args.result_dir, args.audit, args.output_dir)


if __name__ == "__main__":
    main()
