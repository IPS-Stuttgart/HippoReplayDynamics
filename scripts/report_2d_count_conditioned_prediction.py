#!/usr/bin/env python3
"""Non-rescoring, independent-audit-gated common-pipeline predictive report."""

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

DATASETS = {"pfeiffer_foster": ("Pfeiffer/Foster", 4001, 8, 4), "tanni2022": ("Tanni", 5224, 25, 5)}
PRIMARY = {
    "imm_minus_iid_position": "IMM - independent positions",
    "imm_minus_static_location": "IMM - static location",
    "imm_minus_event_global": "IMM - other-event composition",
    "real_minus_wrong_imm": "IMM: real - permuted map",
}
SECONDARY = {
    "imm_minus_diffusion": "IMM - diffusion",
    "diffusion_minus_iid": "Diffusion - independent positions",
    "iid_minus_event_global": "Independent - other-event composition",
    "imm_minus_run_global": "IMM - RUN composition",
    "event_global_minus_run_global": "Other-event - RUN composition",
}


def validate(root, audit_path):
    path = root / "conditional_2d_manifest.json"
    manifest, audit = json.loads(path.read_text()), json.loads(audit_path.read_text())
    if manifest.get("status") != "complete" or audit.get("status") != "pass":
        raise ValueError("complete scoring and passing independent audit required")
    if audit.get("input_file_sha256", {}).get("run_manifest") != file_sha256(path):
        raise ValueError("audit belongs to a different scoring run")
    required = {
        "raw_events": 9225,
        "sessions": 33,
        "analytic_scores": 184500,
        "global_scores": 184500,
        "dynamic_scores": 792,
        "split_contrasts": 415125,
        "event_contrasts": 83025,
        "bootstrap_panels": 18,
    }
    if any(audit.get(k) != v for k, v in required.items()):
        raise ValueError("incomplete independent audit coverage")
    if not np.isfinite(audit.get("max_dynamic_error", np.nan)) or audit["max_dynamic_error"] > 1e-7:
        raise ValueError("independent predictive reconstruction failed")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed scoring artifact: {name}")
    for name, digest in audit["output_sha256"].items():
        if file_sha256(audit_path.parent / name) != digest:
            raise ValueError(f"changed audit artifact: {name}")
    gates = pd.read_csv(root / "conditional_2d_gates.csv")
    if gates.empty or gates.gate.duplicated().any() or not {"complete_nonleaking_proper_prediction", "overall"}.issubset(gates.gate) or not gates.passed.eq(True).all():
        raise ValueError("technical scoring gates failed")
    return manifest, audit


def readout(summary):
    expected = {(d, c) for d in DATASETS for c in (PRIMARY | SECONDARY)}
    if summary.duplicated(["dataset", "contrast"]).any() or set(summary[["dataset", "contrast"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("missing, duplicate or unexpected dataset/contrast")
    result = summary.copy()
    for d, (_, events, sessions, animals) in DATASETS.items():
        part = result[result.dataset.eq(d)]
        if not (part.events.eq(events).all() and part.sessions.eq(sessions).all() and part.animals.eq(animals).all()):
            raise ValueError("frozen cohort denominator mismatch")
    numeric = ["mean", "ci_low", "ci_high", "mean_per_heldout_spike", "per_spike_ci_low", "per_spike_ci_high", "positive_animals"]
    if (
        not np.isfinite(result[numeric]).all().all()
        or (result.ci_low > result.ci_high).any()
        or (result.per_spike_ci_low > result.per_spike_ci_high).any()
        or (result.positive_animals < 0).any()
        or (result.positive_animals > result.animals).any()
    ):
        raise ValueError("invalid predictive readout")
    result["primary"] = result.contrast.isin(PRIMARY)
    result["label"] = result.contrast.map(PRIMARY | SECONDARY)
    result["raw_contrast_gate_passed"] = result["mean"].gt(0) & result.ci_low.gt(0) & result.positive_animals.eq(result.animals)
    result["per_spike_interval_positive"] = result.per_spike_ci_low.gt(0)
    return result


def decision(table):
    table = readout(table)
    rows = []
    for dataset in DATASETS:
        p = table[table.dataset.eq(dataset) & table.primary]
        passed = bool(p.raw_contrast_gate_passed.all())
        failed = list(p.loc[~p.raw_contrast_gate_passed, "contrast"])
        temporal = bool(p.loc[p.contrast.eq("imm_minus_iid_position"), "raw_contrast_gate_passed"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "technical_status": "pass",
                "temporal_vs_independent_supported": temporal,
                "all_four_primary_gates_passed": passed,
                "failed_primary_contrasts": ",".join(failed),
                "bounded_external_replication_established": passed if dataset == "tanni2022" else False,
                "new_biological_mechanism_established": False,
                "recommendation": "bounded_common_pipeline_prediction_supported" if passed else "report_partial_predictive_result_without_full_replication",
                "retrospective_cohort": True,
                "no_favorable_subset_or_parameter_retuning": True,
            }
        )
    return pd.DataFrame(rows)


def figure(table, animals, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for col, (dataset, (label, events, sessions, rats)) in enumerate(DATASETS.items()):
        color = "#16746b" if dataset == "pfeiffer_foster" else "#ab3f60"
        for row_index, (estimate, lo, hi, individual, xlabel) in enumerate(
            [
                ("mean", "ci_low", "ci_high", "delta", "Paired predictive log-score difference (nats)"),
                ("mean_per_heldout_spike", "per_spike_ci_low", "per_spike_ci_high", "delta_per_heldout_spike", "Paired difference per held-out spike (nats/spike)"),
            ]
        ):
            ax = axes[row_index, col]
            for index, contrast in enumerate(PRIMARY):
                r = table[table.dataset.eq(dataset) & table.contrast.eq(contrast)].iloc[0]
                points = animals[animals.dataset.eq(dataset) & animals.contrast.eq(contrast)].sort_values("animal")
                ax.scatter(points[individual], index + np.linspace(-0.12, 0.12, len(points)), s=24, color=color, alpha=0.45)
                ax.plot([r[lo], r[hi]], [index, index], color=color, lw=2)
                ax.scatter(r[estimate], index, marker="D", s=35, color=color, zorder=3)
            ax.axvline(0, color="#555555", ls="--", lw=1)
            ax.set_yticks(range(len(PRIMARY)), PRIMARY.values())
            ax.set_ylim(3.5, -0.5)
            ax.grid(axis="x", alpha=0.15)
            ax.set_xlabel(xlabel)
            prefix = "Raw primary endpoint" if row_index == 0 else "Per-spike sensitivity (not a replacement gate)"
            ax.set_title(f"{label}: {events:,} events, {sessions} sessions, {rats} animals\n{prefix}", fontsize=12, pad=12)
    fig.suptitle("Held-out cell prediction under one common 2D pipeline", fontsize=15)
    fig.supxlabel(
        "Diamonds: equal-animal means; lines: 95% animal/session/event bootstrap intervals; dots: individual animals.\nFive split medians per event; fixed maps, partitions and calibration fits. Compare contrasts within each dataset.",
        fontsize=10,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def report_text(table, verdicts, manifest, audit):
    lines = ["# Common-pipeline 2D held-out-cell prediction", "", "Non-rescoring report of a frozen, independently audited retrospective experiment.", "", "## Decision", ""]
    for r in verdicts.itertuples(index=False):
        lines.append(
            f"- {DATASETS[r.dataset][0]}: `{r.recommendation}`. Four primary gates: {'pass' if r.all_four_primary_gates_passed else 'not all passed'}. Failed axes: {r.failed_primary_contrasts or 'none'}."
        )
    lines += [
        "",
        "A temporal-versus-independent advantage does not by itself establish spatial replay, a unique IMM mechanism, or superiority over every nonspatial explanation. The raw-score primary rule is not replaced by a favorable per-spike sensitivity.",
        "",
        "## Primary Contrasts",
        "",
        "| Dataset | Contrast | Mean nats | 95% CI | Positive animals | Per-spike mean [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for contrast in PRIMARY:
            r = table[table.dataset.eq(dataset) & table.contrast.eq(contrast)].iloc[0]
            lines.append(
                f"| {DATASETS[dataset][0]} | {r.label} | {r['mean']:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.positive_animals}/{r.animals} | {r.mean_per_heldout_spike:+.3f} [{r.per_spike_ci_low:+.3f}, {r.per_spike_ci_high:+.3f}] |"
            )
    lines += ["", "## Secondary Contrasts", "", "| Dataset | Contrast | Mean nats | 95% CI | Positive animals |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for contrast in SECONDARY:
            r = table[table.dataset.eq(dataset) & table.contrast.eq(contrast)].iloc[0]
            lines.append(f"| {DATASETS[dataset][0]} | {r.label} | {r['mean']:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.positive_animals}/{r.animals} |")
    lines += [
        "",
        "## What Is Being Predicted",
        "",
        "All 4,001 PF and 5,224 Tanni common-eligible high-MUA candidate windows are retained, including weak events, all 33 sessions and nine animals. There is no selection on trajectory shape, model scores or ripple overlap. This is not the old 160-event PF pilot or a claim that all candidates are replay.",
        "",
        "Both datasets use the existing common RUN-only rate maps and unit QC, 8 cm spatial bins, and nonoverlapping 20 ms test bins with final partial bins retained. Five deterministic 70/30 cell partitions are used. Cell-identity likelihoods are properly normalized multinomials conditional on the total number of spikes in each bin, separately for training and held-out cells. No likelihood temperature is used.",
        "",
        "Infer the entire event's latent posterior using only training cells, freeze it, and score held-out cell identities. Held-out event spikes never update that posterior. The endpoint is a sum of marginal predictive log scores, not joint event evidence, future-time forecasting or actual-position reconstruction during the event.",
        "",
        "Independent positions, a single static location, diffusion, and first-order IMM share the observation maps. Diffusion sigma is 60 cm/sqrt(s), stationary sigma 2 cm, kernel cutoff 3 sigma and IMM switching time constant 60 ms. Parameters were frozen before this scoring run. A shared permutation of population-code map columns tests spatial adjacency while preserving population snapshots; independent/static scores must remain invariant.",
        "",
        "The other-event composition baseline predicts relative cell participation without any event position or latent path. It is fit using four chronological event folds, excluding events within one second of the tested fold, with 100 pseudospikes toward the RUN composition. Held-out cells may contribute in other calibration events, never the scored event. Calibration can use later events, so this is cross-validation rather than a prospective predictor. Training exposure is unequal: composition adapts on other candidate events, whereas spatial maps remain RUN-trained. Its superiority can reflect encoding transfer, not necessarily absent spatial content.",
        "",
        "## Aggregation and Limits",
        "",
        "Compute each paired contrast within cell split; take five-split event medians; average events within session, sessions within animal, and animals within dataset. The 5,000-draw hierarchical bootstrap resamples animals, sessions and events. Maps, cell splits and calibration fits stay fixed, so their uncertainty is not included. Only four/five animals are independent biological units; bootstrap intervals do not overcome that limitation. Separately aggregated contrasts need not add because medians are nonlinear.",
        "",
        "Zero-held-out-spike events contribute zero to raw scores; their per-spike ratios are undefined. Normalizing changes the estimand and weighting, not merely units. The two estimates can differ and neither licenses selecting the more favorable one. Raw score magnitudes must not be used as a between-dataset biological contrast.",
        "",
        "Candidate detection used all recorded cells before the prediction split. The analysis is conditional on that frozen ascertainment; it is not fully training-only event detection. Source maps were fitted outside immobile candidates but are reused, not independently refitted here. RUN decoder uncertainty was previously imperfectly calibrated. A single fixed population-code permutation per session does not sample a wrong-map distribution or test a genuine alternate environment.",
        "",
        "## Verification",
        "",
        f"Producer commit: `{manifest['code_commit']}`. Audit commit: `{audit['code_commit']}`. {audit['raw_events']:,} native raw event-count matrices reconstructed across {audit['sessions']} sessions. All {audit['analytic_scores']:,} analytic and {audit['global_scores']:,} global scores reconstructed. A separate dynamic solver checked {audit['dynamic_scores']:,} predictions across all sessions, both maps and two fixed splits, with maximum absolute discrepancy {audit['max_dynamic_error']:.3g}.",
        "",
        f"All {audit['split_contrasts']:,} split contrasts, {audit['event_contrasts']:,} event contrasts and {audit['bootstrap_panels']} dataset/contrast bootstrap intervals were independently reconstructed. Source windows/maps, event-fold guards, native spike hashes and score artifacts were checked. RUN maps are source-checked, not independently refitted.",
        "",
        "## Publication Boundary",
        "",
        "This tests a specific cross-population predictive endpoint across two independent 2D datasets. It does not establish new switching biology, uniform replay speed, an absence of replay in weak datasets, or superiority over all co-firing models. Cross-validation and temporal-versus-co-firing distinctions already have direct precedent in [Maboudi et al. (2018)](https://elifesciences.org/articles/34467); replay validation without ground truth is studied by [Takigawa et al. (2024)](https://elifesciences.org/articles/85635). A predictive validation contribution must be positioned against those methods, not advertised as the discovery of sequential hippocampal activity.",
        "",
        "Retain the outcome alongside the recording-coverage and proper-prediction studies. Do not retune the failed gate, select favorable animals or relabel a sensitivity as confirmation. The current results do not by themselves establish a new high-importance biological finding.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    root, audit_path, out = Path(args.run_dir).resolve(), Path(args.audit_manifest).resolve(), Path(args.output_dir).resolve()
    manifest, audit = validate(root, audit_path)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite report")
    table = readout(pd.read_csv(root / "conditional_2d_summary.csv"))
    verdicts = decision(table)
    animals = pd.read_csv(root / "conditional_2d_by_animal.csv")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "conditional_2d_readout.csv", index=False)
    verdicts.to_csv(out / "conditional_2d_decisions.csv", index=False)
    animals.to_csv(out / "conditional_2d_animal_readout.csv", index=False)
    pd.read_csv(root / "conditional_2d_by_session.csv").to_csv(out / "conditional_2d_session_readout.csv", index=False)
    figure(table, animals, out / "conditional_2d_prediction.png")
    (out / "conditional_2d_prediction_report.md").write_text(report_text(table, verdicts, manifest, audit))
    provenance = build_script_provenance(input_paths={"run_manifest": root / "conditional_2d_manifest.json", "audit_manifest": audit_path})
    provenance.update(status="complete", non_rescoring=True, decisions=verdicts.to_dict("records"))
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "conditional_2d_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(verdicts.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
