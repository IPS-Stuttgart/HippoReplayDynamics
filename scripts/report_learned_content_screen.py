#!/usr/bin/env python3
"""Non-rescoring report of audited leave-rat-out content screening."""

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

REAL = ("all_fixed_candidates", "full_accepted_segment")
METHODS = ("all", "learned_half", "spike_half", "entropy_half")
COLORS = ("#777777", "#15796d", "#bd7631", "#94658c")


def report(root, audit_path, output):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(root / "manifest.json"):
        raise ValueError("independently reconstructed measurement required")
    for name, value in manifest["output_sha256"].items():
        if file_sha256(root / name) != value:
            raise ValueError(f"changed result: {name}")
    gates = pd.read_csv(root / "gates.csv")
    if dict(zip(gates.gate, gates.passed, strict=True)) != audit["gates"]:
        raise ValueError("audited gates disagree")
    summary = pd.read_csv(root / "summary.csv")
    animal = pd.read_csv(root / "animal_summary.csv")
    session = pd.read_csv(root / "session_summary.csv")
    models = json.loads((root / "frozen_models.json").read_text())["models"]
    training = pd.DataFrame(
        [
            dict(
                held_rat=rat,
                training_sessions=";".join(m["training_sessions"]),
                training_rows=m["training_rows"],
                feature_leaves=len(m["leaf_nodes"]),
                maximum_training_progress=m["maximum_training_progress"],
                chosen_training_progress=m["chosen_training_progress"],
                minimum_leaf_probability=min(m["probabilities"]),
                maximum_leaf_probability=max(m["probabilities"]),
            )
            for rat, m in models.items()
        ]
    )
    real = summary[summary.source.isin(REAL)].copy()
    real["home_gap_pp"] = real.home_gap * 100
    real = real[["source", "encoding", "method", "home_gap_pp", "separation", "regional_tv", "high_entropy", "low_entropy"]]
    truth = summary[~summary.source.isin(REAL)].copy()
    truth["mean_population_error_cm"] = (truth.balanced_high_error + truth.balanced_low_error) / 2
    truth["mean_population_home_brier"] = (truth.balanced_high_brier + truth.balanced_low_brier) / 2
    truth = truth[["source", "method", "mean_population_error_cm", "mean_population_home_brier", "class0_retention", "class1_retention"]]
    events = pd.read_csv(root / "event_selection.csv.gz", dtype={"event_id": str})
    score_rows = []
    for key, group in events.groupby(["session", "source", "encoding"]):
        cutoff = group.loc[group.predictive_half, "pair_score"].min()
        score_rows.append(
            dict(zip(("session", "source", "encoding"), key, strict=True))
            | dict(
                observations=len(group),
                selected=int(group.predictive_half.sum()),
                unique_scores=group.pair_score.nunique(),
                mean_predicted_retention=group.pair_score.mean(),
                score_cutoff=cutoff,
                fraction_at_cutoff=np.mean(np.abs(group.pair_score - cutoff) < 1e-12),
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in dict(
        training=training, real_summary=real, truth_summary=truth, animal_summary=animal, session_summary=session, gates=gates, selection_scores=pd.DataFrame(score_rows)
    ).items():
        frame.to_csv(output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    primary = real[real.source.eq(REAL[0]) & real.encoding.eq("early_run")].set_index("method")
    for ax, metric, label in [(axes[0, 0], "home_gap_pp", "Home-content gap (percentage points)"), (axes[0, 1], "separation", "Between-population separation (cm)")]:
        ax.bar(range(4), [primary.loc[m, metric] for m in METHODS], color=COLORS)
        ax.set_xticks(range(4), ["All", "Learned\nhalf", "Spike\nhalf", "Entropy\nhalf"])
        ax.set_ylabel(label)
        ax.set_title("Fixed candidate endpoints, early maps", fontsize=11)
    rat = animal[animal.source.eq(REAL[0]) & animal.encoding.eq("early_run")].pivot(index="animal", columns="method", values="home_gap")
    for j, method in enumerate(("all", "learned_half")):
        axes[1, 0].bar(np.arange(len(rat)) + (j - 0.5) * 0.34, 100 * rat[method], width=0.34, label=method.replace("_", " "), color=COLORS[j])
    axes[1, 0].set_xticks(range(len(rat)), rat.index)
    axes[1, 0].set_ylabel("Home-content gap (percentage points)")
    axes[1, 0].legend(frameon=False)
    delta = truth.pivot(index="source", columns="method", values="mean_population_error_cm")
    labels = ["Native Q4" if s == "run_q4" else s.removeprefix("test_").replace("_", " ") for s in delta.index]
    change = delta.learned_half - delta["all"]
    axes[1, 1].barh(range(len(delta)), change, color=np.where(change <= 0, COLORS[1], "#b64a44"))
    axes[1, 1].set_yticks(range(len(delta)), labels, fontsize=9)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Known-position error change (cm; lower is better)")
    axes[1, 1].set_title("Pooled description; gates also check each rat/Brier", fontsize=10)
    passed = audit["gates"]["development_numerical_screen"]
    fig.suptitle(f"Learned paired-population screen: development {'PASS' if passed else 'FAIL'}\nNo external validation; no validated-remedy claim", fontsize=14)
    fig.savefig(output / "learned_content_screen.png", dpi=170)
    plt.close(fig)
    counts = session[session.source.isin(REAL) & session.encoding.eq("early_run")].groupby(["source", "method"])[["total_events", "retained_events"]].sum().reset_index()
    lines = [
        "# Leave-rat-out accuracy-aware content screen",
        "",
        f"Development numerical screen: {'PASS' if passed else 'FAIL'}. Independent technical reconstruction: PASS. External validation: NOT RUN. Validated remedy: NOT ESTABLISHED.",
        "",
        "The frozen screen uses both populations' observable features. It is not an independent A-only forecast, and score/disagreement correlations are not separate validation. All original endpoints and population identities remain fixed.",
        "",
        "## Training",
        "",
        markdown(training),
        "",
        "Each rat is excluded entirely from its model's calibration. Trees and constrained leaf probabilities are fit on other rats' earlier RUN and two simulation banks, then frozen before evaluation. Training probabilities are converted to a deterministic top-half cohort; calibration guarantees do not automatically survive that conversion or a new rat.",
        "",
        "## Original denominators",
        "",
        markdown(counts),
        "",
        "## Real-data readouts",
        "",
        markdown(real),
        "",
        "## Known-position controls",
        "",
        markdown(truth),
        "",
        "These summaries equally weight rats after averaging sessions within rat. The gates additionally forbid worsening either population's class-balanced position error and Home Brier within any rat/source. True replay content remains unknown.",
        "",
        "## Failed gates",
        "",
        markdown(gates[~gates.passed]) if not gates.passed.all() else "None in the development numerical screen. Independent-recording validation is still required.",
        "",
        "No change to leaf count/depth, training domains, retained fraction or scientific thresholds was made to rescue this result. Development banks have appeared in earlier experiments, so held-rat fitting separation does not make this a new blinded external confirmation.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    provenance = {
        **build_script_provenance(),
        "input_file_sha256": {
            str(root / "manifest.json"): file_sha256(root / "manifest.json"),
            str(audit_path): file_sha256(audit_path),
            str(Path(__file__)): file_sha256(__file__),
        },
        "non_rescoring": True,
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.result_dir, args.audit, args.output_dir)
