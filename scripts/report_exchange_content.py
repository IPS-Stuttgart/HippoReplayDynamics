#!/usr/bin/env python3
"""Non-rescoring report of fixed-budget population exchange."""

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
        raise ValueError("independent reconstruction required")
    for path, value in manifest["output_sha256"].items():
        if file_sha256(root / path) != value:
            raise ValueError(f"modified output: {path}")
    gates = pd.read_csv(root / "gates.csv")
    if dict(zip(gates.gate, gates.passed, strict=True)) != audit["gates"]:
        raise ValueError("audited gates disagree")
    assignments = json.loads((root / "pre_scoring.json").read_text())["assignments"]
    selection, risks = [], []
    for session, choice in assignments.items():
        selection.append(
            dict(
                session=session,
                cells_per_population=len(choice["methods"]["baseline"]["high"]),
                shared_cells=len(set(choice["methods"]["baseline"]["high"]) & set(choice["methods"]["baseline"]["low"])),
                accepted_steps=choice["accepted_steps"],
                net_exchanged=choice["net_exchanged"],
                stop_reason=choice["stop_reason"],
                calibration_objective_before=choice["baseline_objective"],
                calibration_objective_after=choice["final_objective"],
            )
        )
        index = 0
        for side in ("high", "low"):
            for label in (0, 1):
                for metric in ("error_cm", "home_brier"):
                    risks.append(dict(session=session, side=side, true_home=label, metric=metric, before=choice["baseline_risks"][index], after=choice["final_risks"][index]))
                    index += 1
    selection, risks = pd.DataFrame(selection), pd.DataFrame(risks)
    summary = pd.read_csv(root / "summary.csv")
    animal = pd.read_csv(root / "animal_summary.csv")
    real = summary[summary.source.isin(("all_fixed_candidates", "full_accepted_segment"))].copy()
    real["home_gap_pp"] = 100 * real.home_gap
    real = real[["source", "encoding", "method", "home_gap_pp", "separation", "regional_tv", "high_entropy", "low_entropy"]]
    classes = pd.read_csv(root / "truth_by_class.csv")
    changes = []
    for key, group in classes.groupby(["session", "source", "true_home"]):
        values = group.set_index("method")
        row = dict(zip(("session", "source", "true_home"), key, strict=True))
        for metric in ("high_error", "low_error", "high_brier", "low_brier"):
            row[f"{metric}_before"] = values.loc["baseline", metric]
            row[f"{metric}_after"] = values.loc["targeted", metric]
            row[f"{metric}_change"] = row[f"{metric}_after"] - row[f"{metric}_before"]
        changes.append(row)
    changes = pd.DataFrame(changes)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in dict(
        selection=selection,
        calibration_risks=risks,
        real_summary=real,
        animal_summary=animal,
        truth_class_changes=changes,
        gates=gates,
        session_draws=pd.read_csv(root / "session_draws.csv"),
    ).items():
        frame.to_csv(output / f"{name}.csv", index=False)
    primary = real[real.source.eq("all_fixed_candidates") & real.encoding.eq("early_run")].set_index("method")
    passed = audit["gates"]["development_numerical_screen"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, metric, ylabel in ((axes[0, 0], "home_gap_pp", "Home gap (percentage points)"), (axes[0, 1], "separation", "Population separation (cm)")):
        ax.bar(range(3), [primary.loc[m, metric] for m in ("baseline", "random_mean", "targeted")], color=("#777777", "#b58340", "#15796d"))
        ax.set_xticks(range(3), ("Original", "Random\nexchange", "RUN-guided\nexchange"))
        ax.set_ylabel(ylabel)
        ax.set_title("All original candidate endpoints", fontsize=11)
    rat = animal[animal.source.eq("all_fixed_candidates") & animal.encoding.eq("early_run")].pivot(index="animal", columns="method", values="home_gap")
    for j, method in enumerate(("baseline", "targeted")):
        axes[1, 0].bar(np.arange(len(rat)) + (j - 0.5) * 0.34, 100 * rat[method], width=0.34, label=method, color=("#777777", "#15796d")[j])
    axes[1, 0].set_xticks(range(len(rat)), rat.index)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_ylabel("Home gap (percentage points)")
    native = changes[changes.source.eq("run_q4") & changes.true_home.eq(1)].sort_values("session")
    delta, labels = [], []
    for row in native.itertuples():
        for side in ("high", "low"):
            delta.append(getattr(row, f"{side}_brier_change"))
            labels.append(f"{row.session} {side}")
    axes[1, 1].barh(range(len(delta)), delta, color=np.where(np.array(delta) <= 0, "#15796d", "#b64a44"))
    axes[1, 1].set_yticks(range(len(delta)), labels, fontsize=8)
    axes[1, 1].axvline(0, color="black", linewidth=0.7)
    axes[1, 1].set_xlabel("Home Brier change (lower is better)")
    axes[1, 1].set_title("Known Home: held-out RUN", fontsize=11)
    fig.suptitle(f"Fixed-budget population exchange: development {'PASS' if passed else 'FAIL'}\nExternal validation not run; no validated remedy", fontsize=13)
    fig.savefig(output / "exchange_content.png", dpi=170)
    plt.close(fig)
    lines = [
        "# Fixed-budget RUN-guided population exchange",
        "",
        f"Development numerical screen: {'PASS' if passed else 'FAIL'}. Independent technical reconstruction: PASS. External validation: NOT RUN. Validated remedy: NOT ESTABLISHED.",
        "",
        "All original endpoints are retained. Population sizes, union and intersection are exactly unchanged. The rule uses only earlier RUN and never observes replay outcomes during selection. High/low labels retain original ancestry after swapping.",
        "",
        "## Frozen selection",
        "",
        markdown(selection),
        "",
        "## Local calibration risks",
        "",
        markdown(risks),
        "",
        "The gradient search evaluates only32 proposals per iteration, for up to10 accepted swaps. A stop means no admissible proposal in this bounded search, not global impossibility. Random controls match the NET cell-exchange count, including zero.",
        "",
        "## Original candidate and accepted endpoints",
        "",
        markdown(real),
        "",
        "Home mass concerns an inferred region, not known replay destinations. Known-position controls and full regional-loss tables are required even if these gaps fall.",
        "",
        "## Known-Home RUN loss changes",
        "",
        markdown(native[["session", "high_error_change", "low_error_change", "high_brier_change", "low_brier_change"]]),
        "",
        "## Failed safeguards",
        "",
        markdown(gates[~gates.passed]),
        "",
        "This is development, not independent-recording replication. Calibration gains, green tests, agreement gains or equal-budget wins cannot override a failed truth or coverage gate. All previous failures remain part of the evidence.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    provenance = {
        **build_script_provenance(),
        "measurement_manifest_sha256": file_sha256(root / "manifest.json"),
        "audit_sha256": file_sha256(audit_path),
        "reporter_sha256": file_sha256(__file__),
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
