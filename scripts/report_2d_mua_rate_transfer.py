#!/usr/bin/env python3
"""Non-rescoring presentation of the audited observation-transfer diagnostic."""

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

PRIMARY = [
    "alpha100__imm_adaptation_gain",
    "alpha100__imm_minus_iid",
    "alpha100__imm_minus_event_global",
    "alpha100__imm_real_minus_wrong",
    "alpha100__first_order_imm__real_order_advantage",
    "alpha100__first_order_imm__order_map_interaction",
]
LABELS = [
    "Recalibrated - original IMM",
    "IMM - independent positions",
    "IMM - matched MUA rates",
    "Real map - permuted map",
    "Original order - shuffled order",
    "Order x map interaction",
]
DATASETS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
COLORS = ["#21818a", "#c04a45", "#718e36", "#9b65a8", "#ad832e"]


def report(root, audit_dir, out):
    root, audit_dir, out = (Path(p).resolve() for p in (root, audit_dir, out))
    manifest_path = root / "mua_rate_transfer_manifest.json"
    audit_path = audit_dir / "mua_rate_transfer_audit.json"
    manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    if manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(manifest_path):
        raise ValueError("matching completed scoring and passed audit required")
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("changed output: " + name)
    out.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(root / "mua_rate_transfer_summary.csv")
    animals = pd.read_csv(root / "mua_rate_transfer_by_animal.csv")
    decisions = pd.read_csv(root / "mua_rate_transfer_decisions.csv", keep_default_na=False).set_index("dataset")
    for dataset in DATASETS:
        part = summary[summary.dataset.eq(dataset) & summary.contrast.isin(PRIMARY)]
        if len(part) != 6 or set(part.contrast) != set(PRIMARY):
            raise ValueError("missing primary outcome")
    summary[summary.contrast.isin(PRIMARY)].to_csv(out / "mua_rate_transfer_primary_table.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), constrained_layout=True, sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items(), strict=True):
        part = summary[summary.dataset.eq(dataset)].set_index("contrast").loc[PRIMARY]
        local = animals[animals.dataset.eq(dataset)]
        names = sorted(local.animal.unique())
        y = np.arange(6)
        for i, animal in enumerate(names):
            values = local[local.animal.eq(animal)].set_index("contrast").loc[PRIMARY].delta
            ax.scatter(values, y + (i - (len(names) - 1) / 2) * 0.065, s=24, color=COLORS[i], label=animal, zorder=3)
        ax.errorbar(part["mean"], y, xerr=np.array([part["mean"] - part.ci_low, part.ci_high - part["mean"]]), fmt="ks", ms=4, capsize=4, label="Mean / 95% CI", zorder=4)
        ax.axvline(0, color="0.5", linewidth=1)
        ax.set(yticks=y, yticklabels=LABELS, xlabel="Paired held-out score difference (nats/event)", title=title, ylim=(5.5, -0.5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.11))
        ax.tick_params(axis="y", labelsize=9)
    fig.savefig(out / "mua_rate_transfer_primary.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Matched MUA observation-transfer diagnostic",
        "",
        "Non-rescoring exploratory report. No change to the frozen candidate events or dynamics.",
        "",
        "## Scope",
        "",
        "Pfeiffer/Foster: 4,001 MUA events, eight sessions/four animals. Tanni: 5,224 MUA events, 25 sessions/five animals. Five 70/30 cell splits, four spatial models, real and permuted maps. Alpha=100 is primary; alpha=1000 is sensitivity. Twenty whole-bin shuffles per event for the primary.",
        "",
        "Both spatial and nonspatial models receive the same other-event calibration exposure. Each cell's entire RUN map is multiplied by a constant derived from guarded calibration-event counts, preserving its within-cell map shape. The test event/fold never calibrates itself. No held-out event spike updates a gain, context or latent path.",
        "",
        "Target scores are normalized multinomial held-out cell-identity log probabilities conditional on each time bin's total. They are sums of frozen training-posterior marginal scores, not joint event log evidence. Candidate detection originally used all cells; validation is conditional on that fixed ascertainment. Calibration is retrospective and can use later events.",
        "",
        "## Primary results",
        "",
        "| Dataset | Contrast | Mean nats/event | 95% CI | Positive animals |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, title in DATASETS.items():
        table = summary[summary.dataset.eq(dataset)].set_index("contrast")
        for contrast, label in zip(PRIMARY, LABELS, strict=True):
            row = table.loc[contrast]
            lines.append(f"| {title} | {label} | {row['mean']:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {int(row.positive_animals)}/{int(row.animals)} |")
    lines += [
        "",
        "Differences are paired within event/split, then median-aggregated across splits per event. Events are averaged within session, sessions within animal, and animals equally within each dataset. Intervals use the fixed-seed 5,000-draw animal/session/event bootstrap. They remain limited by four/five animals and do not absorb all model-choice or exploratory-selection uncertainty. A cross-dataset comparison of absolute log scores is not made.",
        "",
        "## Decisions",
        "",
    ]
    for dataset, title in DATASETS.items():
        row = decisions.loc[dataset]
        lines.append(f"- {title}: full six-axis primary rule passed = **{row.full_observation_transfer_gate}**; failed axes: `{row.failed_primary_axes or 'none'}`.")
    lines += [
        "",
        "A likelihood improvement by itself is not a temporal replay result. A full pass only motivates independent confirmation and stronger temporal nonspatial controls; it does not demonstrate biological gain, unique IMM circuitry, physical replay speed, or a high-importance mechanism. If a primary comparison fails, a favorable shrinkage setting, normalized endpoint or animal subset cannot replace it.",
        "",
        "## Matched nonspatial comparisons and sensitivity",
        "",
        "| Dataset | Contrast | Mean | 95% CI | Positive animals |",
        "|---|---|---:|---:|---:|",
    ]
    extras = [
        "alpha100__global_adaptation_gain",
        "alpha100__iid_adaptation_gain",
        "alpha100__iid_minus_event_global",
        "alpha100__change_in_real_order_advantage",
        "alpha100__change_in_order_map_interaction",
        "alpha1000__imm_minus_iid",
        "alpha1000__imm_minus_event_global",
        "alpha1000__imm_real_minus_wrong",
    ]
    for dataset, title in DATASETS.items():
        table = summary[summary.dataset.eq(dataset)].set_index("contrast")
        for contrast in extras:
            row = table.loc[contrast]
            lines.append(f"| {title} | {contrast} | {row['mean']:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {int(row.positive_animals)}/{int(row.animals)} |")
    lines += [
        "",
        "## Scope of inference",
        "",
        "Estimated cell gains can reflect which locations/events are sampled as well as physiology; their values are not measured biological gain. The procedure does not alter a cell's map shape, but cross-cell normalization and inferred positions can change. The equality is of spatial-mean raw rates, not an assertion that every mixture of conditional likelihoods has the same cell marginal.",
        "",
        "State-dependent place-cell firing and reinstated rate codes have direct precedent. This diagnostic tests unequal observation-model training exposure in these particular cohorts; it is not the first discovery of rate transfer or replay modulation. The prior hc-11 recalibration failure is a separate retained result, not relabeled by a new 2D outcome.",
        "",
        "## Verification",
        "",
        f"Scoring commit `{manifest['code_commit']}`, clean at launch: `{not manifest['git_dirty']}`. Manifest SHA256 `{file_sha256(manifest_path)}`.",
        f"Audit reconstructed {audit['split_contrasts']:,} split contrasts, {audit['event_contrasts']:,} event contrasts and {audit['hierarchical_interval_panels']} hierarchical interval panels. Independently implemented dynamic inference checked {audit['independent_dynamic_predictions']:,} predictions, maximum absolute discrepancy {audit['max_prediction_error']:.3g}.",
        "All original global predictions, calibration folds/counts/gains, factorial coverage, invariants and source/output hashes were checked. Native event counts and RUN maps reuse the previously audited parent; they were not refitted or re-read from original recordings in this diagnostic.",
    ]
    (out / "mua_rate_transfer_report.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={"score_manifest": manifest_path, "audit": audit_path})
    provenance.update(rescored=False, outputs={p.name: file_sha256(p) for p in out.iterdir()})
    (out / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report(args.run_dir, args.audit_dir, args.output_dir)
