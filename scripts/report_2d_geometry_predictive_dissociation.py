#!/usr/bin/env python3
"""Audit-gated, non-rescoring geometry/prediction readout."""

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

DATASETS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
STRATA = {"all": "All candidates", "geometric_accepted": "Geometry accepted", "geometric_rejected": "Geometry rejected"}
METRICS = {
    "imm_minus_iid": "IMM - independent positions",
    "imm_order_advantage": "IMM original - shuffled order",
    "imm_order_map_interaction": "IMM order x map interaction",
    "imm_minus_event_global": "IMM - other-event composition",
    "diffusion_minus_iid": "Diffusion - independent positions",
    "diffusion_order_advantage": "Diffusion original - shuffled order",
    "diffusion_order_map_interaction": "Diffusion order x map interaction",
}


def validate(root, audit_path):
    path = root / "geometry_predictive_manifest.json"
    manifest, audit = json.loads(path.read_text()), json.loads(audit_path.read_text())
    if manifest.get("status") != "complete" or audit.get("status") != "pass" or audit.get("input_file_sha256", {}).get("run_manifest") != file_sha256(path):
        raise ValueError("matching complete experiment and independent audit required")
    expected = {"sessions": 33, "events": 9225, "labels": 184500, "predictions": 46125, "summary_rows": 840, "primary_interval_rows": 42}
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError("incomplete independent audit")
    for record, folder in ((manifest, root), (audit, audit_path.parent)):
        if not record.get("output_sha256"):
            raise ValueError("missing artifact hashes")
        for name, digest in record["output_sha256"].items():
            if file_sha256(folder / name) != digest:
                raise ValueError("changed artifact: " + name)
    return manifest, audit


def plot(primary, animal, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, (dataset, label) in enumerate(DATASETS.items()):
        color = "#16746b" if row == 0 else "#ab3f60"
        for col, metric in enumerate(("imm_order_advantage", "imm_order_map_interaction")):
            ax = axes[row, col]
            ticks = []
            for y, (stratum, name) in enumerate(STRATA.items()):
                r = primary[primary.dataset.eq(dataset) & primary.stratum.eq(stratum) & primary.metric.eq(metric)].iloc[0]
                points = animal[animal.dataset.eq(dataset) & animal.stratum.eq(stratum)][metric].dropna()
                ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=2)
                ax.scatter(r["mean"], y, marker="D", color=color, s=35)
                ax.scatter(points, y + np.linspace(-0.1, 0.1, len(points)), color=color, alpha=0.4, s=24)
                ticks.append(f"{name} (n={r.events:,})")
            ax.axvline(0, color="#666666", ls="--", lw=1)
            ax.set_yticks(range(3), ticks)
            ax.set_ylim(2.5, -0.5)
            ax.set_title(f"{label}\n{METRICS[metric]}", fontsize=12, pad=12)
            ax.set_xlabel("Paired held-out predictive log-score difference (nats)")
            ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Does geometric rejection imply absence of independently predictive order?", fontsize=14)
    fig.supxlabel(
        "Primary frozen split 0; labels from training neurons only. Equal-animal means, 95% hierarchical intervals, animal dots.\nGeometric acceptance is not shuffle-validated replay. Strata are not quality-matched; composition baseline is reported separately.",
        fontsize=10,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args):
    root, audit_path, out = Path(args.run_dir).resolve(), Path(args.audit_manifest).resolve(), Path(args.output_dir).resolve()
    manifest, audit = validate(root, audit_path)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite report")
    summary = pd.read_csv(root / "geometry_predictive_summary.csv")
    primary = summary[summary.split.eq(0) & summary.criterion.eq("edge10")]
    animals = pd.read_csv(root / "geometry_predictive_by_animal.csv")
    animals = animals[animals.split.eq(0) & animals.criterion.eq("edge10")]
    labels = pd.read_csv(root / "geometry_predictive_labels.csv")
    labels = labels[labels.split.eq(0) & labels.criterion.eq("edge10")]
    reasons = labels.groupby(["dataset", "failure_reason"]).size().rename("events").reset_index()
    decisions = pd.read_csv(root / "geometry_predictive_decisions.csv").fillna("")
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("primary", primary), ("animal", animals), ("sensitivity", summary), ("rejection_reasons", reasons), ("decisions", decisions)):
        frame.to_csv(out / f"geometry_predictive_{name}_readout.csv", index=False)
    plot(primary, animals, out / "geometry_predictive_dissociation.png")
    lines = ["# Training-only geometry and held-out predictive structure", "", "Non-rescoring report of the independently verified frozen experiment.", "", "## Decisions", ""]
    for r in decisions.itertuples(index=False):
        lines.append(
            f"- {DATASETS[r.dataset]}: {r.rejected_events:,} geometry-rejected events; order/adjacency rule {'passes' if r.rejected_order_adjacency_supported else 'fails'}; additional other-event-composition rule {'passes' if r.rejected_beyond_other_event_composition_supported else 'fails'}. Failed primary metrics: {r.failed_primary_metrics or 'none'}."
        )
    lines += [
        "",
        "## Primary Readout",
        "",
        "| Dataset | Stratum | Metric | Events | Mean nats | 95% CI | Positive animals | Sessions contributing |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_label in DATASETS.items():
        for stratum, stratum_label in STRATA.items():
            for metric, metric_label in METRICS.items():
                r = primary[primary.dataset.eq(dataset) & primary.stratum.eq(stratum) & primary.metric.eq(metric)].iloc[0]
                lines.append(
                    f"| {dataset_label} | {stratum_label} | {metric_label} | {r.events} | {r['mean']:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.positive_animals}/{r.total_animals} | {r.contributing_sessions}/{r.total_sessions} |"
                )
    lines += ["", "## Geometry Rejection Reasons", ""]
    lines += [f"- {DATASETS[r.dataset]}: {r.failure_reason}, {r.events:,} events." for r in reasons.itertuples(index=False)]
    lines += [
        "",
        "## What Was Tested",
        "",
        "Use the same 9,225 common high-MUA candidates, 33 sessions/nine animals, RUN-only 8 cm maps and five frozen 70/30 cell partitions as the parent proper-prediction study. Geometry uses training neurons only: independent uniform-prior Poisson decoding in complete overlapping 20 ms frames every 5 ms, edge trim to >=2 spikes, earliest longest MAP run with jumps <20 cm, >=10 frames, >=40 cm displacement. No path prior or temporal smoothing in this label. It is the geometric component of a transferred PF-style screen, not an exact original-author implementation or a shuffle-significant replay decision.",
        "",
        "Primary labels and predictions both use split0. Other splits and the 11-frame/2-cells-3-spikes variants are separate sensitivities, not pooled criteria or a search for a better result. No cross-split labels, all-cell geometry or held-out event spikes determine the primary stratum. Candidate detection originally used all cells, so the analysis remains conditional on fixed ascertainment.",
        "",
        "Predictive scores are copied, not recomputed or tuned: proper multinomial held-out cell identities conditioned on each bin's total, evaluated against a posterior inferred only from training neurons. Five-split event medians from the parent are NOT used for this selection analysis. There is one primary score per event. Pair original/shuffled/map contrasts within the same split. Average events within session, sessions within animal, then animals equally. Bootstrap animals/sessions/events 5,000 times, keeping maps, partitions and surrogate draws fixed. Zero-stratum sessions stay explicit with missing means. Per-spike ratios are sensitivity endpoints; zero-count ratios stay missing.",
        "",
        "Accepted and rejected groups differ in duration, support and geometry by construction. Their contrast is not a causal effect or a quality-matched comparison. A positive result in rejected events establishes that this geometric screen does not exhaust the specific predictive-order endpoint. It does not show that rejected events were continuous replay, estimate missed replay prevalence, recover physical speed, or identify unique IMM circuitry. Keep the other-event composition comparison separate; order sensitivity cannot rescue its failure.",
        "",
        f"Verification: {audit['labels']:,} training labels, {audit['predictions']:,} predictive rows, {audit['summary_rows']} stratified summaries and {audit['primary_interval_rows']} primary interval rows independently reconstructed. {audit['raw_base_bins']:,} base bins recounted from cached native timestamps, yielding {audit['overlapping_frames']:,} overlapping frames. Parent native/predictive audits are reused; original MAT/NWB files and RUN-map fitting were not repeated. Producer commit `{manifest['code_commit']}`; auditor `{audit['code_commit']}`.",
        "",
        "## Novelty Boundary",
        "",
        "Conventional Bayesian and HMM replay labels can disagree, including conventional rejections with temporal model congruence, as already shown by [Maboudi et al. (2018)](https://elifesciences.org/articles/34467). The candidate added contribution is the quantitative link between recording-dependent geometry and independently predicted order in the same two-dataset cohort, integrated with known-path kinematic recovery. This result alone does not establish a new high-importance discovery. Do not redefine a failed geometry screen as positive biological replay merely because a different statistic is positive.",
        "",
    ]
    (out / "geometry_predictive_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": root / "geometry_predictive_manifest.json", "audit_manifest": audit_path})
    provenance.update(status="complete", non_rescoring=True, new_mechanism_established=False, decisions=decisions.to_dict("records"))
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "geometry_predictive_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(decisions.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
