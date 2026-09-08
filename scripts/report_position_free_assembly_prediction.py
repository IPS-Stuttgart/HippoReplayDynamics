#!/usr/bin/env python3
"""Non-rescoring, audit-gated report of spatial versus assembly prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256

PANELS = (
    ("PF", "RUN", "pooled", "Pfeiffer/Foster: RUN ripples"),
    ("hc11", "POST", "direction_mixture", "hc-11: POST ripples"),
    ("hc11", "PRE", "direction_mixture", "hc-11: PRE ripples"),
)
ROWS = (
    ("spatial_first_order_imm_minus_assembly_global", 3, "Spatial IMM - global"),
    ("spatial_first_order_imm_minus_assembly_persistent", 3, "Spatial IMM - assembly (3)"),
    ("spatial_first_order_imm_minus_assembly_persistent", 8, "Spatial IMM - assembly (8)"),
    ("assembly_persistent_minus_global", 3, "Assembly (3) - global"),
    ("assembly_persistent_minus_global", 8, "Assembly (8) - global"),
)


def validate_inputs(root, audit_path):
    manifest_path = root / "assembly_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest.get("status") != "complete" or audit.get("status") != "pass":
        raise ValueError("complete scoring and passing independent audit required")
    if audit.get("input_file_sha256", {}).get("run_manifest") != file_sha256(manifest_path):
        raise ValueError("audit does not refer to this run")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed artifact: {name}")
    required = {"raw_event_counts": 480, "calibration_fits": 96, "independent_scores": 19200, "aggregate_panels": 190}
    if any(audit.get(k) != v for k, v in required.items()):
        raise ValueError("incomplete audit coverage")
    return manifest, audit


def readout(summary):
    rows = []
    for dataset, phase, variant, _ in PANELS:
        part = summary[summary.dataset.eq(dataset) & summary.phase.eq(phase) & summary.encoding_variant.eq(variant)]
        for contrast, k, label in ROWS:
            match = part[part.contrast.eq(contrast) & part.n_components.eq(k)]
            if len(match) != 1:
                raise ValueError("missing primary readout contrast")
            row = match.iloc[0].to_dict()
            if not np.isfinite([row[v] for v in ("mean", "ci_low", "ci_high")]).all() or row["animals"] != 4:
                raise ValueError("invalid readout")
            rows.append(row | {"label": label})
    return pd.DataFrame(rows)


def decision(table):
    pf = table[table.dataset.eq("PF")]
    assembly = pf[pf.contrast.eq("assembly_persistent_minus_global")]
    spatial = pf[pf.contrast.eq("spatial_first_order_imm_minus_assembly_persistent")]
    bounded = len(spatial) == 2 and spatial.ci_low.gt(0).all() and spatial.positive_animals.eq(4).all()
    weak = len(assembly) == 2 and assembly.ci_high.lt(0).all() and assembly.positive_animals.eq(0).all()
    return {
        "bounded_pf_spatial_advantage": bool(bounded),
        "tested_assemblies_underperform_global_pf": bool(weak),
        "mechanism_identified": False,
        "external_replication_established": False,
        "recommendation": "retain_bounded_predictive_result_no_mechanistic_upgrade" if bounded else "spatial_advantage_not_established",
        "no_new_tuning_or_positive_subset": True,
    }


def figure(table, animals, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    for ax, (dataset, phase, variant, title) in zip(axes, PANELS, strict=True):
        for index, (contrast, k, label) in enumerate(ROWS):
            row = table[table.dataset.eq(dataset) & table.phase.eq(phase) & table.contrast.eq(contrast) & table.n_components.eq(k)].iloc[0]
            color = "#146f63" if contrast.startswith("spatial") else "#b13d58"
            points = animals[
                animals.dataset.eq(dataset) & animals.phase.eq(phase) & animals.encoding_variant.eq(variant) & animals.contrast.eq(contrast) & animals.n_components.eq(k)
            ].sort_values("rat")
            ax.scatter(points.delta, index + np.linspace(-0.12, 0.12, len(points)), color=color, alpha=0.35, s=22, zorder=2)
            ax.errorbar(row["mean"], index, xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]], fmt="D", color=color, capsize=4, markersize=5, zorder=3)
        ax.axvline(0, color="#444444", lw=1, ls="--")
        ax.set_yticks(range(len(ROWS)), [r[2] for r in ROWS])
        ax.invert_yaxis()
        ax.set_title(title + "\n160 events; 4 rats", fontsize=12, pad=12)
        ax.set_xlabel("Paired held-out log-score difference (nats)")
        ax.grid(axis="x", alpha=0.15)
        ax.set_ylim(4.6, -0.6)
    fig.suptitle("Spatial prediction beats the tested PF comparators, but assembly baselines are weak", fontsize=14)
    fig.supxlabel(
        "Diamonds: equal-animal means and 95% animal-only bootstrap intervals. Dots: animal means.\nCompare within panels only; training data, scoring bins and behavioral states differ. Global: cross-event mean cell composition.",
        fontsize=10,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    audit_path = Path(args.audit_manifest).resolve()
    manifest, audit = validate_inputs(root, audit_path)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite report")
    summary = pd.read_csv(root / "assembly_prediction_summary.csv")
    animals = pd.read_csv(root / "assembly_prediction_by_animal.csv")
    table = readout(summary)
    verdict = decision(table)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "assembly_prediction_readout.csv", index=False)
    (out / "assembly_prediction_decision.json").write_text(json.dumps(verdict, indent=2) + "\n")
    figure(table, animals, out / "assembly_prediction_report.png")
    lines = [
        "# Position-free assembly prediction control",
        "",
        "Non-rescoring report; independently audited frozen candidate cohorts.",
        "",
        "## Result",
        "",
        "PF retains a bounded predictive advantage over the tested three- and eight-component assembly models. However, both assembly models underperform the cross-event global composition baseline in all four PF rats. This materially limits the mechanistic interpretation of beating them.",
        "",
        "| Dataset/phase | Contrast | Mean nats | 95% CI | Positive animals |",
        "|---|---|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.dataset}/{row.phase} | {row.label} | {row.mean:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {row.positive_animals}/4 |")
    lines += [
        "",
        "hc-11 direction-mixture PRE and POST do not beat the global composition baseline. Pooled-direction sensitivities remain in the full source tables; do not promote a sensitivity or significance difference into external replication or a state interaction.",
        "",
        "## What Was Compared",
        "",
        "PF: 160 previously inspected native RUN ripple candidates, 8 sessions/4 rats, native 4 ms test bins. hc-11: 160 PRE plus 160 POST native ripple candidates, 8 sessions/4 rats, 20 ms test bins. Each event retains five fixed 70/30 cell splits. No new event selection or scoring of the spatial models.",
        "",
        "Assembly models learn only from the opposite chronological half of 20 candidate events per session/phase, with a one-second event guard. Calibration uses 20 ms spike bins; all cells may contribute in calibration events, including neurons held out in the separate scored event. No calibration event shares the scored interval.",
        "",
        "Three/eight multinomial co-firing components have independent, event-fixed or 60 ms persistent assignments. Training-cell spike identities alone infer the event assignments. Held-out scores are properly normalized multinomial probabilities conditioned on the held-out total in each bin; the held-out spikes never update the assignment posterior. These are sums of marginal predictive log scores, not joint event evidences or forecasts of future time bins.",
        "",
        "Comparisons have unequal training exposure and capacity: spatial models have RUN maps, assembly models have at most ten other short events per fold. Components passing synthetic checks does not establish that this calibration budget adequately learns real assemblies. Position-free models can learn spatial correlations indirectly; they are not guaranteed non-replay nulls.",
        "",
        "## Aggregation and Scope",
        "",
        "Pair model scores within splits, median across five splits per event, mean per session, equal-session mean per rat, then equal-rat mean. Per-held-out-spike sensitivities retain zero-held-out cases as undefined ratios rather than divide by zero. CIs resample only four animals (5,000 draws), keeping maps, calibration fits and within-animal observations fixed. They omit those uncertainty sources and do not certify a population-wide mechanism. Differences between separately aggregated contrasts need not add because medians are nonlinear.",
        "",
        "## Verification",
        "",
        f"Scoring commit: `{manifest['code_commit']}`. Audit commit: `{audit['code_commit']}`.",
        f"Raw counts independently reconstructed for {audit['raw_event_counts']} events; {audit['calibration_fits']} guarded calibration fits checked/refitted; {audit['independent_scores']:,} predictions recomputed with a separate dense solver (maximum absolute error {audit['max_prediction_error']:.3g}). All {audit['split_contrasts']:,} split contrasts, {audit['event_contrasts']:,} event contrasts and {audit['aggregate_panels']} aggregate/CI panels reconstructed.",
        "",
        "The deterministic EM refit shares the production fitter, with a separately evaluated calibration objective. Parent spatial scores are hash-pinned and checked against their source tables; this audit does not independently refit RUN maps or spatial posteriors. Synthetic known-assembly checks cover 1,080 events and are operating checks, not recovery guarantees for arbitrary real assemblies.",
        "",
        "## Paper Boundary",
        "",
        "This adds a useful comparator to the bounded PF predictive result, not a newly identified replay mechanism. Co-firing versus sequence structure and cross-validated HMMs were already studied by [Maboudi et al. (2018)](https://elifesciences.org/articles/34467). The present comparator does not reproduce that paper's fully learned HMM. Do not claim that every nonspatial explanation has been excluded, that hc-11 lacks replay, or that PF and hc-11 differ biologically on the basis of these unmatched cohorts.",
        "",
        "Do not retune component count, persistence or event selection to seek a desired outcome. Retain this diagnostic alongside the existing proper predictive and recording-coverage results. It does not by itself satisfy the high-importance-paper goal.",
        "",
    ]
    (out / "assembly_prediction_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": root / "assembly_prediction_manifest.json", "audit_manifest": audit_path})
    provenance.update(status="complete", non_rescoring=True, decision=verdict)
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "assembly_prediction_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
