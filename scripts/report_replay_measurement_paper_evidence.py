#!/usr/bin/env python3
"""Assemble a source-linked methods-paper overview without rescoring events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256

DATASETS = {"pfeiffer_foster": ("PF", 4), "tanni2022": ("Tanni", 5)}
SOURCE_NAMES = {
    "shuffle_baseline": "coverage_shuffle_baseline_primary_table.csv",
    "geometry": "geometry_endpoint_summary.csv",
    "map_mismatch": "coverage_map_mismatch_gradient_response_summary.csv",
}


def checked(path, digest):
    path = Path(path)
    if file_sha256(path) != digest:
        raise ValueError(f"source checksum mismatch: {path}")
    return path


def one(frame, filters):
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    result = frame.loc[mask]
    if len(result) != 1:
        raise ValueError(f"expected exactly one source row, got {len(result)}: {filters}")
    return result.iloc[0]


def collect_panels(tables):
    rows = []

    def add(panel, dataset, condition, source, filters, field, scale=1.0, ci=None, units="nats"):
        r = one(tables[source], filters)
        value = float(r[field]) * scale
        bounds = [float(r[c]) * scale for c in ci] if ci else [np.nan, np.nan]
        if not np.isfinite(value) or (ci and (not np.isfinite(bounds).all() or bounds[0] > bounds[1])):
            raise ValueError("nonfinite or inverted source estimate")
        rows.append(
            {
                "panel": panel,
                "dataset": dataset,
                "condition": condition,
                "value": value,
                "ci_low": bounds[0],
                "ci_high": bounds[1],
                "units": units,
                "source_table": source,
                "source_filter": json.dumps(filters, sort_keys=True),
                "source_field": field,
                "source_ci_fields": json.dumps(ci),
                "scale": scale,
            }
        )

    for dataset, (_, animals) in DATASETS.items():
        f = {"dataset": dataset, "detector": "source_high_mua", "observation": "original_order"}
        r = one(tables["shuffle_baseline"], f)
        if r.animals_with_events != animals or r.eligible_events <= 0:
            raise ValueError("missing source animals or candidates")
        for condition, field in [("full", "significant_full_percent"), ("half", "significant_half_percent")]:
            add("A", dataset, condition, "shuffle_baseline", f, field, units="percent accepted")
        add("A", dataset, "half_minus_full", "shuffle_baseline", f, "half_minus_full_pp", ci=("delta_ci_low_pp", "delta_ci_high_pp"), units="percentage points")

        f = {"dataset": dataset, "support": "parent", "bin_filter": "edge_only", "min_frames": 10, "group": "rejected_with_opportunity"}
        for contrast in ["imm_minus_iid", "imm_minus_static", "imm_minus_composition", "imm_order_advantage", "imm_order_map_interaction"]:
            selection = f | {"contrast": contrast}
            r = one(tables["prediction"], selection)
            if r.animals != animals or r.events <= 0:
                raise ValueError("incomplete rejected-group prediction")
            add("B", dataset, contrast, "prediction", selection, "mean", ci=("ci_low", "ci_high"))

        f = {
            "dataset": dataset,
            "observation": "poisson",
            "cell_fraction": 1.0,
            "likelihood": "poisson",
            "estimator": "posterior_mean",
            "bin_filter": "unfiltered",
            "selection": "all",
            "coordinate": "true_coordinate",
            "metric": "gradient_response",
        }
        conditions = [
            ("arclength_truth", "generator_known", "true_arclength"),
            ("window_truth", "generator_known", "true_chord"),
            ("known_map", "generator_known", "decoded"),
            ("independent_map", "independent_RUN_half", "decoded"),
        ]
        for condition, decoder_map, readout in conditions:
            selection = f | {"decoder_map": decoder_map, "readout": readout}
            r = one(tables["map_mismatch"], selection)
            if r.animals_available != animals or r.animals_expected != animals:
                raise ValueError("incomplete gradient recovery")
            add("D", dataset, condition, "map_mismatch", selection, "mean", ci=("ci95_low", "ci95_high"), units="response to injected contrast 1.0")

    f = {
        "area_m2": 8.75,
        "aspect": 1.4,
        "sigma_cm": 30.0,
        "n_cells": 128,
        "support_domain": "full_arena",
        "grid_cm": 8.0,
        "stride_ms": 5.0,
        "observation": "native_poisson",
        "likelihood": "poisson",
        "estimator": "map",
        "bin_filter": "unfiltered",
        "continuity_rule": "literal_20cm_10frames",
        "gradient": 0.0,
    }
    for window in [20.0, 40.0]:
        for kind, metric in [("continuous", "eligible_recovery_fraction"), ("shuffled_continuous", "acceptance_fraction")]:
            selection = f | {"window_ms": window, "truth_kind": kind, "metric": metric}
            r = one(tables["geometry"], selection)
            if r.populations_available != 8 or r.populations_expected != 8:
                raise ValueError("incomplete synthetic populations")
            add("C", "synthetic_large_arena", f"{kind}_{int(window)}ms", "geometry", selection, "mean", 100.0, ("ci95_low", "ci95_high"), units="percent accepted")
    return pd.DataFrame(rows)


def draw(panels, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    colors = {"pfeiffer_foster": "#16748a", "tanni2022": "#aa4c53"}

    def value(panel, dataset, condition):
        return one(panels, {"panel": panel, "dataset": dataset, "condition": condition})

    ax = axes[0, 0]
    for dataset, (label, _) in DATASETS.items():
        ax.plot([0, 1], [value("A", dataset, c).value for c in ["full", "half"]], "o-", color=colors[dataset], label=label, linewidth=2)
    ax.set(
        xticks=[0, 1], xticklabels=["All included cells", "Half the cells"], ylabel="Accepted candidates (%)", ylim=(0, 26), title="A. Same real candidates; changed observation"
    )
    ax.legend(frameon=False)
    ax.text(0.03, 0.95, "Transferred continuity + two-shuffle criterion", transform=ax.transAxes, fontsize=8, va="top")

    ax = axes[0, 1]
    for i, (dataset, (label, _)) in enumerate(DATASETS.items()):
        r = value("B", dataset, "imm_minus_composition")
        ax.plot([r.ci_low, r.ci_high], [i, i], color=colors[dataset], linewidth=2)
        ax.scatter([r.value], [i], color=colors[dataset])
    ax.axvline(0, color="0.5", linestyle=":")
    ax.set(yticks=[0, 1], yticklabels=["PF", "Tanni"], ylim=(-0.7, 1.7), xlabel="Held-out IMM - nonspatial composition (nats)", title="B. Rejected does not mean information-free")
    ax.text(0.03, 0.95, "Training-only geometry; separate held-out neurons", transform=ax.transAxes, fontsize=8, va="top")

    ax = axes[1, 0]
    for kind, label, color in [("continuous", "Known continuous paths", "#338657"), ("shuffled_continuous", "Order-randomized paths", "#7c589a")]:
        r = [value("C", "synthetic_large_arena", f"{kind}_{w}ms") for w in [20, 40]]
        ax.plot([20, 40], [x.value for x in r], "o-", label=label, color=color, linewidth=2)
        for w, x in zip([20, 40], r, strict=True):
            ax.plot([w, w], [x.ci_low, x.ci_high], color=color)
    ax.set(xticks=[20, 40], xlabel="Decoding window (ms)", ylabel="Geometric acceptance (%)", ylim=(0, 85), title="C. Better recovery can also admit more nulls")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    conditions = ["arclength_truth", "window_truth", "known_map", "independent_map"]
    for dataset, (label, _) in DATASETS.items():
        rr = [value("D", dataset, c) for c in conditions]
        x = np.arange(4) + (-0.04 if dataset == "pfeiffer_foster" else 0.04)
        ax.plot(x, [r.value for r in rr], "o-", label=label, color=colors[dataset], linewidth=2)
        for xx, r in zip(x, rr, strict=True):
            ax.plot([xx, xx], [r.ci_low, r.ci_high], color=colors[dataset])
    ax.axhline(1, color="0.5", linestyle=":")
    ax.set(
        xticks=np.arange(4),
        xticklabels=["Latent\ntruth", "Window\ntruth", "Known\nmap", "Estimated\nmap"],
        ylabel="Recovered gradient contrast",
        ylim=(-0.05, 1.15),
        title="D. Strong speed variation can look much flatter",
    )
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle("Replay measurement: continuity, neural prediction and kinematics are distinct", fontsize=14)
    fig.text(
        0.5, 0.025, "A/B: real recordings. C/D: prescribed-path simulations. Different experiments and denominators; no biological uniform-speed claim.", ha="center", fontsize=8
    )
    fig.tight_layout(rect=(0.015, 0.055, 0.995, 0.95), h_pad=2.5, w_pad=2.5)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_sources(index_dir, prediction_dir, prediction_audit):
    index_manifest = index_dir / "replay_coverage_artifact_index_manifest.json"
    index = json.loads(index_manifest.read_text())
    if index.get("status") != "complete":
        raise ValueError("incomplete coverage registry")
    for name, digest in index["output_sha256"].items():
        checked(index_dir / name, digest)
    registry = pd.read_csv(index_dir / "replay_coverage_artifact_registry.csv")
    files = pd.read_csv(index_dir / "replay_coverage_artifact_files.csv")
    paths = {"coverage_index_manifest": index_manifest, "script": Path(__file__)}
    tables, scopes = {}, []
    for study, filename in SOURCE_NAMES.items():
        record = one(registry, {"study": study})
        if record.registry_status != "pass":
            raise ValueError("source registry stage did not pass")
        for role in ["manifest", "audit", "report"]:
            path = Path(record[role])
            registered = one(files, {"study": study, "path": str(path)})
            checked(path, registered.sha256)
            paths[f"{study}_{role}"] = path
        production = json.loads(Path(record.manifest).read_text())
        audit = json.loads(Path(record.audit).read_text())
        checked(Path(record.manifest), audit["input_file_sha256"]["scoring_manifest"])
        if production.get("status") != "complete" or audit.get("status") != "pass":
            raise ValueError("incomplete or failed source experiment")
        report = Path(record.report)
        meta = json.loads(report.read_text())
        path = report.parent / filename
        checked(path, meta["output_sha256"][filename])
        checked(path, one(files, {"study": study, "path": str(path)}).sha256)
        paths[f"table_{study}"] = path
        tables[study] = pd.read_csv(path)
        scopes.append({"study": study, "verification_scope": record.audit_scope, "report_manifest": str(report)})

    report_path = prediction_dir / "training_continuity_report_manifest.json"
    report = json.loads(report_path.read_text())
    audit = json.loads(prediction_audit.read_text())
    if report.get("status") != "complete" or audit.get("status") != "pass":
        raise ValueError("prediction report must be complete and audited")
    checked(report_path, audit["input_file_sha256"]["report_manifest"])
    paths.update(prediction_report_manifest=report_path, prediction_report_audit=prediction_audit)
    for key, digest in report["input_file_sha256"].items():
        path = checked(report["input_file_paths"][key], digest)
        paths[f"prediction_{key}"] = path
    for name, digest in report["output_sha256"].items():
        checked(prediction_dir / name, digest)
    paths["table_prediction"] = prediction_dir / "training_continuity_primary_summary.csv"
    tables["prediction"] = pd.read_csv(paths["table_prediction"])
    paths["prediction_decisions"] = prediction_dir / "training_continuity_decisions.csv"
    scopes.append({"study": "training_only_prediction", "verification_scope": audit["validation_scope"], "report_manifest": str(report_path)})
    return tables, paths, pd.DataFrame(scopes)


def run(args):
    tables, paths, scopes = load_sources(args.coverage_index.resolve(), args.prediction_report.resolve(), args.prediction_audit.resolve())
    panels = collect_panels(tables)
    before = {k: file_sha256(p) for k, p in paths.items()}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    panels.to_csv(out / "replay_measurement_paper_figure_values.csv", index=False)
    scopes.to_csv(out / "replay_measurement_paper_source_scopes.csv", index=False)
    pd.read_csv(paths["prediction_decisions"]).to_csv(out / "replay_measurement_prediction_decisions.csv", index=False)
    draw(panels, out / "replay_measurement_paper_overview.png")
    text = [
        "# Replay Measurement Evidence Overview",
        "",
        "Non-rescoring consolidation; not a new statistical discovery or publication-priority certificate.",
        "",
        "| Panel | Dataset | Condition | Estimate | 95% interval, where provided | Units |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for r in panels.itertuples(index=False):
        ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" if np.isfinite(r.ci_low) else "not supplied for this estimate"
        text.append(f"| {r.panel} | {r.dataset} | {r.condition} | {r.value:.3f} | {ci} | {r.units} |")
    text += [
        "",
        "## Reading The Figure",
        "",
        "A: real immobile high-MUA candidates, 4,001 PF and 5,224 Tanni; equal-animal means (4/5 animals). Transferred common-encoding PF-style continuity plus two 5,000-shuffle tests. Acceptance is not biological prevalence.",
        "",
        "B: training-only geometric rejection, not all-cell shuffle rejection. Predictions come from separate held-out cells. 3,542 PF and 5,055 Tanni events qualify in at least one split; one event median over its qualifying splits. The figure shows the nonspatial-composition contrast; all five primary contrasts and the original joint decisions remain in the tables. PF passes; Tanni does not. Conditional hierarchical intervals are not biological replay labels.",
        "",
        "C: one frozen synthetic configuration, not additional animal data: N128, Gaussian fields sigma30 cm, 8.75 square metres, aspect1.4, native Poisson counts, 8 cm grid, 5 ms stride, unfiltered MAP. Eight population seeds, 24 paths per seed. Continuous recovery is conditional on true geometric eligibility; shuffled acceptance uses all corresponding null observations. Intervals resample synthetic populations, not animals. Null acceptance is not a real-data false-positive rate.",
        "",
        "D: independent RUN-half map experiment, full training-selected cells and independent 20 ms posterior-mean decoding before continuity selection. Known horizontal synthetic coordinates, imposed positive-minus-negative gradient contrast 1.0; not a biological wall-distance correlation. Equal-animal summaries. Window truth shows finite-window attenuation; known and separately fitted maps show additional decoding loss.",
        "",
        "## Claim Boundary",
        "",
        "The evidence supports a measurement-methods contribution: continuity classification, independently predictive structure and recoverable kinematics need separate validation. It does not prove true replay in rejected events, uniform biological speed, a uniquely IMM mechanism, a validated fuzzy classifier, or complete explanation of PF/Tanni differences. Tanni's failed composition comparison is retained. These experiments were developed through sequential retrospective follow-ups, not one prospectively preregistered study.",
        "",
        "![Overview](replay_measurement_paper_overview.png)",
    ]
    (out / "replay_measurement_paper_overview.md").write_text("\n".join(text) + "\n")
    for key, path in paths.items():
        checked(path, before[key])
    manifest = build_script_provenance(input_paths=paths, cwd=ROOT)
    manifest.update(
        status="complete",
        non_rescoring=True,
        row_count=len(panels),
        verification_scope="source registry/report/audit linkage and used-table checksums; exact prespecified row extraction; no new underlying scientific reconstruction",
        output_sha256={p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    )
    (out / "replay_measurement_paper_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-index", required=True, type=Path)
    parser.add_argument("--prediction-report", required=True, type=Path)
    parser.add_argument("--prediction-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
