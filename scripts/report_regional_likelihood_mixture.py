#!/usr/bin/env python3
"""Report an independently audited regional mixture screen without rescoring."""

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


def report(result_dir, audit_path, output):
    manifest = json.loads((result_dir / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit["manifest_sha256"] != sha(result_dir / "manifest.json"):
        raise ValueError("independent audit absent or not bound to these results")
    for name, digest in manifest["output_sha256"].items():
        if sha(result_dir / name) != digest:
            raise ValueError(f"changed result: {name}")
    output.mkdir(parents=True, exist_ok=False)
    tables = {name: pd.read_csv(result_dir / f"{name}.csv") for name in ("estimates", "truth_summary", "animal_truth", "content_summary", "pair_contrasts", "gate_summary")}
    gates = tables["gate_summary"]
    numerical_pass = bool(gates.set_index("gate").loc["development_numerical_screen", "passed"])
    truth = tables["truth_summary"]
    truth = truth[(truth.observation == "poisson") & (truth.panel_type == "fixed_prevalence")].copy()
    content = tables["content_summary"]
    content = content[content.observation == "poisson"].copy()
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    truth_view = truth[["source", "raw_error", "mixture_error", "relative_error_reduction"]]
    gap_view = content[["encoding", "source", "raw_gap", "mixture_gap", "relative_gap_reduction"]]
    fits = tables["estimates"]
    real = fits[(fits.observation == "poisson") & fits.panel.eq("real")]
    lines = [
        "# Regional likelihood-mixture development screen",
        "",
        f"Technical reconstruction: PASS ({audit['reconstructed_likelihoods']:,} regional likelihoods; {audit['reconstructed_estimates']} mixture estimates).",
        f"Frozen development screen: {'PASS' if numerical_pass else 'FAIL'}.",
        "Independent-recording validation: NOT RUN. Biological Home prevalence: NOT VALIDATED.",
        "",
        "## Primary known-truth results",
        "",
        "Errors are absolute prevalence errors (fractions, not percentage points). Equal rat weights; "
        "panels and populations averaged within session. Synthetic panels reuse observations.",
        "",
        truth_view.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Original-content discrepancy",
        "",
        "Absolute high/low discrepancy averaged within rat and then across rats. Both populations are fit separately; no agreement target enters fitting.",
        "",
        gap_view.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Boundary and failure accounting",
        "",
        real.fit_status.value_counts().to_string(),
        "",
        gates[["gate", "passed"]].to_markdown(index=False),
        "",
        "## Claim boundary",
        "",
        "The mixture proportion is a model-implied aggregate, not a count of verified Home replays. "
        "Correct within-region spatial mixtures and observation likelihoods are assumptions. "
        "Boundary fits are not evidence of reliable absence/presence. Profile-likelihood support "
        "is conditional, not a calibrated confidence interval for clustered RUN data.",
        "",
        "This applies established likelihood/prior adjustment (Saerens et al., 2002, doi:10.1162/089976602753284446); the estimator itself is not novel.",
        "",
        "Passing would authorize a frozen external test, not complete the remedy claim. "
        "Failure does not authorize threshold changes or promotion of the exploratory conditional likelihood.",
        "",
        f"Producer commit: `{manifest['code_commit']}`",
        f"Manifest SHA256: `{sha(result_dir / 'manifest.json')}`",
        f"Independent audit SHA256: `{sha(audit_path)}`",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    names = {
        "run_q4": "Held-out RUN",
        "test_poisson_gain1": "Matched Poisson",
        "test_poisson_gain4": "Gain x4",
        "test_conditional": "Event-total counts",
        "test_conditional_map_drift": "Map drift",
        "test_conditional_shared_assembly": "Shared assembly",
    }
    truth = truth.set_index("source").loc[list(names)].reset_index()
    x = np.arange(len(truth))
    axes[0].barh(x - 0.18, truth.raw_error * 100, height=0.34, color="#657783", label="Mean posterior")
    axes[0].barh(x + 0.18, truth.mixture_error * 100, height=0.34, color="#ba4c59", label="Mixture MLE")
    axes[0].set(yticks=x, yticklabels=[names[s] for s in truth.source], xlabel="Absolute prevalence error (percentage points)", title="Known-truth safeguards")
    axes[0].legend(frameon=False)
    g = content.reset_index(drop=True)
    x = np.arange(len(g))
    axes[1].barh(x - 0.18, g.raw_gap * 100, height=0.34, color="#657783")
    axes[1].barh(x + 0.18, g.mixture_gap * 100, height=0.34, color="#ba4c59")
    labels = [f"{'Early' if r.encoding == 'early_run' else 'Full'} map / {'candidates' if r.source == 'all_fixed_candidates' else 'accepted endpoints'}" for r in g.itertuples()]
    axes[1].set(yticks=x, yticklabels=labels, xlabel="Absolute high/low gap (percentage points)", title="Original regional-content instability")
    fig.suptitle("Regional mixture estimation: development screen, not biological replay truth")
    fig.savefig(output / "regional_mixture_screen.png", dpi=160)
    plt.close(fig)
    metadata = dict(
        source_manifest_sha256=sha(result_dir / "manifest.json"),
        audit_sha256=sha(audit_path),
        report_script_sha256=sha(__file__),
        development_screen_passed=numerical_pass,
        external_validation=False,
        files={p.name: sha(p) for p in output.iterdir()},
    )
    (output / "report_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.result_dir, args.audit, args.output_dir)


if __name__ == "__main__":
    main()
