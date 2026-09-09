#!/usr/bin/env python3
"""Non-rescoring report of an optimistic synthetic geometry screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_2d_lagged_neural_prediction import table_markdown

VALUE = "neural_true_train_neural_minus_physical"
ORACLE = "neural_true_oracle_neural_minus_physical"


def report(run, audit_path, out):
    mp = run / "physical_neural_metric_manifest.json"
    m, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("matching passing audit required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed output " + name)
    summary = pd.read_csv(run / "physical_neural_metric_summary.csv")
    animals = pd.read_csv(run / "physical_neural_metric_animals.csv")
    decision = pd.read_csv(run / "physical_neural_metric_decision.csv")
    if (
        len(decision) != 1
        or decision.horizon_ms.iloc[0] != 40
        or decision[["real_event_scoring_performed", "real_replay_mechanism_established", "high_importance_discovery_established"]].any().any()
    ):
        raise ValueError("screen cannot declare real replay evidence")
    primary = summary[summary.horizon.eq(2)].copy()
    if len(primary) != 2 or primary.dataset.duplicated().any() or not np.isfinite(primary[[VALUE, ORACLE]].to_numpy()).all():
        raise ValueError("complete finite two-dataset primary table required")
    passed = bool(decision.necessary_oracle_screen_pass.iloc[0])
    conclusion = (
        "The necessary known-origin screen passes. Unknown-origin, finite-spike and observation-mismatch recovery is still required before real-event interpretation."
        if passed
        else "The necessary known-origin screen fails. A physical-model preference in real events would be ambiguous because the partial-population neural comparator is not adequate in every animal."
    )
    out.mkdir(parents=True, exist_ok=False)
    primary.to_csv(out / "physical_neural_metric_primary.csv", index=False)
    animals[animals.horizon.eq(2)].to_csv(out / "physical_neural_metric_primary_animals.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (dataset, label) in zip(axes, [("pfeiffer_foster", "Pfeiffer/Foster"), ("tanni2022", "Tanni")], strict=True):
        group = animals[animals.horizon.eq(2) & animals.dataset.eq(dataset)].sort_values("animal")
        for x, field in enumerate((VALUE, ORACLE)):
            ax.scatter(np.linspace(x - 0.12, x + 0.12, len(group)), group[field], color="#447b9c", s=28)
            ax.plot([x - 0.23, x + 0.23], [group[field].mean()] * 2, color="#182a33", linewidth=3)
        ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
        ax.set_xticks([0, 1], ["Training-cell neural metric", "Oracle full-population metric"])
        ax.set_ylabel("Expected advantage over physical metric\n(nats / future held-cell identity)")
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Known neural-geometry generator, known origin, 40-ms forecast\nDots: animals; bars: equal-animal means. Not real replay scores.", fontsize=12)
    fig.savefig(out / "physical_neural_metric_screen.png", dpi=180)
    plt.close(fig)
    display = primary[["dataset", VALUE, ORACLE, "positive_animals", "animals"]].copy()
    display = display.rename(columns={VALUE: "training_metric_advantage", ORACLE: "oracle_advantage"})
    for c in ("training_metric_advantage", "oracle_advantage"):
        display[c] = display[c].map(lambda x: f"{x:+.6f}")
    text = f"""# Physical Versus Neural Geometry: Identifiability Screen

{conclusion}

{table_markdown(display)}

![Known-origin screen](physical_neural_metric_screen.png)

## What This Tests

Use RUN maps from 33 recordings/nine animals, not candidate replay spikes.
Construct physical-distance and neural-code-similarity reversible random walks
with identical uniform equilibrium, state dwell, and mean transition entropy.
The full recorded neural population defines the known neural generator; only
training-cell RUN maps define its predictive approximation in five 70/30 splits.
Held-cell RUN maps define future identity likelihoods, with no held spikes
used to construct geometry or infer an origin.

The origin is known exactly. Exactly integrate all possible origins,
destinations and one future held-cell identity instead of drawing Monte Carlo
events. This is an optimistic model operating check, not Poisson spike-count
recovery or a real-event classification experiment. The oracle advantage is
a KL divergence and must be nonnegative mathematically; that is not a finding.
Split medians precede equal recording and animal means. These are exact
synthetic expectations conditional on fitted maps, not biological significance
tests. Panels may have different horizontal/vertical scales.

## Limits

This neither measures replay speed nor demonstrates that it is constant in
physical or neural space. Matching entropy is an explicit modeling convention;
individual physical jump sizes and individual row entropies are not matched.
Passing the screen is only necessary: finite spike support, unknown origins,
unrecorded neurons and independently estimated/perturbed maps remain untested.
Failing it does not refute neural-space propagation. Both outcomes require
keeping the previous failed forecasting criteria unchanged.
The necessary screen is a predeclared readiness rule, not a mathematical
impossibility bound for other estimators or conditioning populations.

## Audit And Provenance

Scoring commit `{m["code_commit"]}`; run SHA256 `{file_sha256(mp)}`.
{audit["kernels_checked"]} reconstructed kernels; {audit["expected_scores_checked"]}
independently checked expected scores. Maximum discrepancy
{audit["max_absolute_score_error"]:.3g} nats. {audit["scope"]}
No new biological mechanism or high-importance discovery is established.
"""
    (out / "physical_neural_metric_report.md").write_text(text)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(non_rescoring=True, real_event_scoring_performed=False, conclusion=conclusion)
    p["output_sha256"] = {f.name: file_sha256(f) for f in sorted(out.iterdir())}
    (out / "physical_neural_metric_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.run_dir, args.audit, args.output_dir)["conclusion"])
