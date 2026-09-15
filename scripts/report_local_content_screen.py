#!/usr/bin/env python3
"""Non-rescoring, region-resolved report of audited local screening."""

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
from scripts.report_content_screening_bound import markdown


def report(root, audit_path, output):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(root / "manifest.json"):
        raise ValueError("independently verified measurement required")
    for name, value in manifest["output_sha256"].items():
        if file_sha256(root / name) != value:
            raise ValueError(f"result changed: {name}")
    gates = pd.read_csv(root / "gates.csv")
    if dict(zip(gates.gate, gates.passed, strict=True)) != audit["gates"]:
        raise ValueError("audited gates disagree")
    models = json.loads((root / "frozen_models.json").read_text())["models"]
    training = pd.DataFrame(
        [
            dict(
                session=session,
                training_rows=model["training_rows"],
                leaves=len(model["leaf_nodes"]),
                maximum_training_progress=model["maximum_training_progress"],
                chosen_training_progress=model["chosen_training_progress"],
            )
            for session, model in models.items()
        ]
    )
    summary = pd.read_csv(root / "summary.csv")
    animals = pd.read_csv(root / "animal_summary.csv")
    sessions = pd.read_csv(root / "session_summary.csv")
    classes = pd.read_csv(root / "truth_by_class.csv")
    real = summary[summary.source.isin(("all_fixed_candidates", "full_accepted_segment"))].copy()
    real["home_gap_pp"] = real.home_gap * 100
    real = real[["source", "encoding", "method", "home_gap_pp", "separation", "regional_tv", "high_entropy", "low_entropy"]]
    class_delta = []
    for key, frame in classes.groupby(["animal", "session", "source", "true_home"]):
        values = frame.set_index("method")
        for side in ("high", "low"):
            row = dict(zip(("animal", "session", "source", "true_home"), key, strict=True))
            row.update(side=side, retained=int(values.loc["local_half", "retained_events"]), total=int(values.loc["all", "total_events"]))
            for loss in ("error", "brier", "home"):
                row[f"{loss}_before"] = values.loc["all", f"{side}_{loss}"]
                row[f"{loss}_after"] = values.loc["local_half", f"{side}_{loss}"]
                row[f"{loss}_change"] = row[f"{loss}_after"] - row[f"{loss}_before"]
            class_delta.append(row)
    differences = pd.DataFrame(class_delta)
    passed = audit["gates"]["development_numerical_screen"]
    counts = (
        sessions[sessions.source.isin(("all_fixed_candidates", "full_accepted_segment")) & sessions.encoding.eq("early_run")]
        .groupby(["source", "method"])[["total_events", "retained_events"]]
        .sum()
        .reset_index()
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, table in dict(
        training=training, real_summary=real, animal_summary=animals, session_summary=sessions, truth_by_class=classes, truth_class_changes=differences, gates=gates, counts=counts
    ).items():
        table.to_csv(output / f"{name}.csv", index=False)
    primary = real[real.source.eq("all_fixed_candidates") & real.encoding.eq("early_run")].set_index("method")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    methods = ("all", "local_half", "spike_half", "entropy_half")
    labels = ("All", "Local\nhalf", "Spike\nhalf", "Entropy\nhalf")
    for ax, metric, ylabel in ((axes[0, 0], "home_gap_pp", "Home gap (percentage points)"), (axes[0, 1], "separation", "Between-population separation (cm)")):
        ax.bar(range(4), [primary.loc[m, metric] for m in methods], color=("#777777", "#167b6c", "#b47c32", "#976b99"))
        ax.set_xticks(range(4), labels)
        ax.set_ylabel(ylabel)
        ax.set_title("Original candidate endpoints", fontsize=11)
    rat = animals[animals.source.eq("all_fixed_candidates") & animals.encoding.eq("early_run")].pivot(index="animal", columns="method", values="home_gap")
    for j, method in enumerate(("all", "local_half")):
        axes[1, 0].bar(np.arange(len(rat)) + (j - 0.5) * 0.34, 100 * rat[method], width=0.34, label=method.replace("_", " "))
    axes[1, 0].set_xticks(range(len(rat)), rat.index)
    axes[1, 0].set_ylabel("Home gap (percentage points)")
    axes[1, 0].legend(frameon=False)
    native = differences[differences.source.eq("run_q4") & differences.true_home.eq(1)].sort_values(["session", "side"])
    axes[1, 1].barh(range(len(native)), native.brier_change, color=np.where(native.brier_change <= 0, "#167b6c", "#bb4b46"))
    axes[1, 1].set_yticks(range(len(native)), native.session + " " + native.side, fontsize=8)
    axes[1, 1].axvline(0, color="black", linewidth=0.7)
    axes[1, 1].set_xlabel("Home Brier loss change (lower is better)")
    axes[1, 1].set_title("Known Home positions: held-out RUN", fontsize=11)
    fig.suptitle(f"Session-local region-protected screen: development {'PASS' if passed else 'FAIL'}\nExternal validation NOT RUN; no validated remedy", fontsize=13)
    fig.savefig(output / "local_content_screen.png", dpi=170)
    plt.close(fig)
    lines = [
        "# Session-local region-protected screening",
        "",
        f"Development numerical screen: {'PASS' if passed else 'FAIL'}. Independent technical audit: PASS. External validation: NOT RUN. Validated remedy: NOT ESTABLISHED.",
        "",
        "Each session's earlier RUN and separate calibration simulations fit its own rule. Both true spatial classes have separate calibration error/Brier constraints. No evaluation truth is used in selection; replay truth remains unknown.",
        "",
        "The original populations and 20-ms endpoints do not change. This is a paired-observation diagnostic, not a blind prediction of an unseen second population. Fixed top-half selection need not preserve fractional training constraints.",
        "",
        "## Training",
        "",
        markdown(training),
        "",
        "## Fixed denominators",
        "",
        markdown(counts),
        "",
        "## Real-data readouts",
        "",
        markdown(real),
        "",
        "## Held-out RUN: regional losses",
        "",
        markdown(differences[differences.source.eq("run_q4")][["session", "true_home", "side", "retained", "total", "error_before", "error_after", "brier_before", "brier_after"]]),
        "",
        "All six truth sources, both classes and both populations are included in truth_class_changes.csv. Pooled gains cannot override local accuracy failures.",
        "",
        "## Failed gates",
        "",
        markdown(gates[~gates.passed]),
        "",
        "No external experiment is authorized by an incomplete development pass. No parameter or threshold was tuned after evaluation. The prior failed leave-rat-out result remains part of the evidence.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    state = {
        **build_script_provenance(),
        "measurement_manifest_sha256": file_sha256(root / "manifest.json"),
        "audit_sha256": file_sha256(audit_path),
        "reporter_sha256": file_sha256(__file__),
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "report_manifest.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.result_dir, args.audit, args.output_dir)
