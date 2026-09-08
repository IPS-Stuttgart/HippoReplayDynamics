#!/usr/bin/env python3
"""Non-rescoring report of cross-event sleep relative-rate recalibration."""

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

AXES = ("imm_minus_iid", "imm_minus_static", "imm_minus_global", "real_minus_wrong_imm")
LABELS = ("IMM - independent positions", "IMM - static location", "IMM - nonspatial baseline", "IMM: real - permuted map")


def decision(summary):
    primary = summary[summary.phase.eq("POST") & summary.regime.eq("sleep_alpha100") & summary.encoding_variant.eq("direction_mixture")].set_index("contrast")
    needed = ["imm_minus_iid", "imm_minus_static", "real_minus_wrong_imm"]
    if primary.index.duplicated().any() or not set(needed).issubset(primary.index):
        raise ValueError("missing/duplicate primary contrasts")
    values = primary.loc[needed, ["mean", "ci_low", "ci_high", "animals", "positive_animals"]]
    if not np.isfinite(values).all().all() or not values.animals.eq(4).all():
        raise ValueError("incomplete primary animal support")
    passed = values.ci_low.gt(0).all() and values.positive_animals.eq(4).all()
    return "diagnostic_transfer_lead_requires_replication" if passed else "external_temporal_advantage_remains_unsupported"


def figure(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    settings = (
        ("run_original", "Original RUN maps", "#287c8e"),
        ("sleep_alpha100", "Sleep recalibration (100)", "#984d87"),
        ("sleep_alpha1000", "Shrinkage sensitivity (1000)", "#c4752f"),
    )
    for ax, (phase, variant) in zip(axes.ravel(), (("POST", "direction_mixture"), ("POST", "pooled"), ("PRE", "direction_mixture"), ("PRE", "pooled")), strict=True):
        for offset, (regime, label, color) in zip((-0.2, 0, 0.2), settings, strict=True):
            table = summary[summary.phase.eq(phase) & summary.encoding_variant.eq(variant) & summary.regime.eq(regime)].set_index("contrast").loc[list(AXES)]
            x = table["mean"].to_numpy()
            ax.errorbar(x, np.arange(4) + offset, xerr=np.stack([x - table.ci_low, table.ci_high - x]), fmt="o", color=color, label=label, capsize=3, markersize=4)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_yticks(np.arange(4), LABELS, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Equal-animal predictive difference (nats/event)")
        ax.set_title(f"{phase}: {'direction mixture' if variant == 'direction_mixture' else 'pooled direction sensitivity'}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("hc-11: cross-event sleep relative-rate recalibration\nExploratory animal-cluster 95% intervals; fitted maps/calibration counts held fixed", fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(args):
    root, audit, out = (Path(p).resolve() for p in (args.run_dir, args.audit_dir, args.output_dir))
    path = root / "sleep_rate_transfer_manifest.json"
    manifest = json.loads(path.read_text())
    verification = json.loads((audit / "audit_manifest.json").read_text())
    if manifest["status"] != "complete" or verification["status"] != "passed" or verification["input_file_sha256"]["run_manifest"] != file_sha256(path):
        raise ValueError("missing or mismatched independent audit")
    for name in ("summary", "by_animal", "by_session", "gates"):
        p = root / f"sleep_rate_transfer_{name}.csv"
        if file_sha256(p) != manifest["output_sha256"][p.name]:
            raise ValueError("changed result")
    gates = pd.read_csv(root / "sleep_rate_transfer_gates.csv")
    if gates.empty or not gates.passed.eq(True).all():
        raise ValueError("technical failure")
    summary = pd.read_csv(root / "sleep_rate_transfer_summary.csv")
    verdict = decision(summary)
    if verdict != manifest["decision"]:
        raise ValueError("decision differs from declared rule")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing report overwrite")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "sleep_rate_transfer_report_table.csv", index=False)
    figure(summary, out / "sleep_rate_transfer.png")
    primary = summary[summary.phase.eq("POST") & summary.encoding_variant.eq("direction_mixture")]
    lines = [
        "# Cross-event sleep relative-rate recalibration",
        "",
        f"Decision: `{verdict}`.",
        "",
        "All 320 frozen PRE/POST events, eight sessions, four rats and five neural splits were scored. Calibration uses the opposite chronological half of a session/phase's selected events with a one-second guard. No test-event spikes enter calibration or latent inference.",
        "",
        "All frozen units can contribute distinct calibration-event spikes, including neurons later held out for a test event. This is a cross-event calibrated observation model, not a RUN-only sleep encoder. Spatial fields, priors and selection remain frozen.",
        "",
        "## POST primary direction-mixture results",
        "",
        "Split-median event contrasts are averaged equally across sessions and animals. Intervals resample four animal clusters only, conditional on maps, calibration counts and within-animal data. These are not the original parent's full hierarchical intervals.",
        "",
        "| Regime | Contrast | Mean [95% CI] | Positive animals |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in primary.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.contrast} | {row.mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | {row.positive_animals}/{row.animals} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Improving a nonspatial rate baseline or beating only static location does not establish temporal replay. The required contrasts are IMM versus independent positions, IMM versus static location, and real-map versus permuted-map prediction. The declared decision above must not be replaced by whichever model comparison is positive.",
            "",
            "Recalibration changes relative cell frequencies, which can reflect represented occupancy, state, sampling or biological gain. It does not identify the cause. These candidates are not a ground-truth absence-of-replay dataset. Do not infer a formal RUN-by-sleep interaction or tune alpha/select animals after inspecting these outcomes.",
            "",
            "A positive outcome would remain a diagnostic lead requiring new-event confirmation. A failed outcome closes this frozen simple-recalibration route, not all possible observation models or the existence of sleep replay.",
            "",
            "## Audit",
            "",
            verification["scope"],
            "",
            f"All {verification['raw_events']} raw event counts, {verification['global_scores']} global predictions, {verification['original_score_regressions']} original-run predictions and {verification['aggregate_panels']} aggregate/interval panels passed. {verification['independent_scores']} separate dense-model predictions agree to {verification['max_predictive_error']:.3g}.",
            "",
            f"Producer commit `{manifest['code_commit']}`; audit `{verification['code_commit']}`.",
        ]
    )
    (out / "sleep_rate_transfer_report.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={"run_manifest": path, "audit_manifest": audit / "audit_manifest.json"})
    provenance.update(status="complete", non_rescoring=True, decision=verdict, output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
