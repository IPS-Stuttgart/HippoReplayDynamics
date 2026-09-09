#!/usr/bin/env python3
"""Non-rescoring report of the frozen occupancy-matched forecast control."""

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
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_2d_lagged_neural_prediction import DATASETS, MODELS, table_markdown


def interpret(decisions):
    if decisions.empty or decisions.model.duplicated().any() or set(decisions.model) != set(MODELS):
        raise ValueError("complete unique model decisions required")
    for column in ("independent_confirmation", "high_importance_discovery_established", "original_compound_verdict_changed"):
        if decisions[column].any():
            raise ValueError("exploratory control cannot promote discovery or change original verdict")
    if not decisions.all_updated_controls_pass.eq(decisions.matched_destination_structure_lead & decisions.other_original_controls_pass).all():
        raise ValueError("inconsistent compound decision")
    passing = decisions.loc[decisions.all_updated_controls_pass, "model"].map(MODELS).tolist()
    if passing:
        return "Exploratory lead requiring independent confirmation: " + ", ".join(passing) + ". No unique generative mechanism or high-importance discovery is established."
    return "No model passes the updated full two-dataset forecasting criterion. Positive individual contrasts remain bounded diagnostic evidence."


def figures(primary, animals, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for i, (dataset, label) in enumerate(DATASETS.items()):
        for j, (model, name) in enumerate(MODELS.items()):
            ax = axes[i, j]
            for y, condition, color in ((0, "Old dwell null", "#777777"), (1, "Occupancy + dwell null", "#19755b")):
                row = primary[primary.dataset.eq(dataset) & primary.model.eq(model) & primary.control.eq(condition)].iloc[0]
                ax.plot([row.ci_low, row.ci_high], [y, y], color=color, linewidth=2)
                ax.scatter(row["mean"], y, color=color, zorder=3)
                ax.annotate(f"{int(row.positive_animals)}/{int(row.animals)} animals positive", (0.03, 0.92 - y * 0.10), xycoords="axes fraction", fontsize=9)
            ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
            ax.set_yticks([0, 1], ["Old dwell null", "Occupancy + dwell"])
            ax.set_ylim(1.5, -0.5)
            ax.set_xlabel("Dynamic advantage (nats / held-out spike)")
            ax.set_title(label + " | " + name)
    fig.suptitle("40 ms forward prediction: does matching occupancy change the result?\nAnimal-balanced estimates; conditional animal-bootstrap 95% intervals", fontsize=14)
    fig.savefig(out / "occupancy_matched_primary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = ["#19755b", "#ab3c53", "#2364ad", "#986c00", "#666666"]
    for ax, (dataset, label) in zip(axes, DATASETS.items(), strict=True):
        sub = animals[animals.dataset.eq(dataset) & animals.horizon.eq(2)]
        names = sorted(sub.animal.unique())
        for k, animal in enumerate(names):
            values = []
            for model in MODELS:
                g = sub[sub.animal.eq(animal) & sub.contrast.eq(model + "__dynamic_minus_matched")]
                if len(g) != 1:
                    raise ValueError("missing animal/model contrast")
                values.append(float(g.delta_per_spike.iloc[0]))
            offset = (k - (len(names) - 1) / 2) * 0.07
            ax.scatter([j + offset for j in range(3)], values, color=colors[k % len(colors)], label=animal, s=45)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_xticks(range(3), ["Spatial IMM", "Spatial diffusion", "Neural HMM"])
        ax.set_ylabel("Dynamic minus occupancy/dwell null (nats / spike)")
        ax.set_title(label)
        ax.legend(fontsize=9)
    fig.suptitle("All animals retained at the fixed 40 ms horizon\nEach point equally averages recording means of event-median split contrasts", fontsize=13)
    fig.savefig(out / "occupancy_matched_animals.png", dpi=180)
    plt.close(fig)


def report(run, audit_path, out):
    mp = run / "occupancy_matched_manifest.json"
    manifest, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("complete run and matching passing audit required")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed scored output " + name)
    parent_path = Path(manifest["input_file_paths"]["lagged_manifest"])
    if file_sha256(parent_path) != manifest["input_file_sha256"]["lagged_manifest"]:
        raise ValueError("changed original manifest")
    old_manifest = json.loads(parent_path.read_text())
    for name in ("lagged_prediction_summary.csv.gz", "lagged_prediction_coverage.csv"):
        if file_sha256(parent_path.parent / name) != old_manifest["output_sha256"][name]:
            raise ValueError("changed original summary/coverage")
    summary = pd.read_csv(run / "occupancy_matched_summary.csv.gz")
    animals = pd.read_csv(run / "occupancy_matched_animals.csv.gz")
    decisions = pd.read_csv(run / "occupancy_matched_decisions.csv")
    old = pd.read_csv(parent_path.parent / "lagged_prediction_summary.csv.gz")
    coverage = pd.read_csv(parent_path.parent / "lagged_prediction_coverage.csv")
    events = pd.read_csv(run / "occupancy_matched_events.csv.gz")
    classification = interpret(decisions)
    counts = coverage.groupby(["dataset", "horizon"], as_index=False).agg(
        selected_events=("selected_events", "sum"),
        eligible_events=("eligible_events", "sum"),
        sessions=("session", "size"),
        animals=("animal", "nunique"),
    )
    support = events[events.contrast.eq("learned_hmm__dynamic_minus_matched")].groupby(["dataset", "horizon"]).valid_neural_splits.agg(lambda v: int(v.gt(0).sum()))
    counts = counts.merge(support.rename("per_spike_supported_events").reset_index(), validate="one_to_one")
    if not counts.per_spike_supported_events.le(counts.eligible_events).all():
        raise ValueError("per-spike support exceeds eligible events")
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            for name, frame, suffix in (("Old dwell null", old, "dynamic_minus_dwell_only"), ("Occupancy + dwell null", summary, "dynamic_minus_matched")):
                g = frame[frame.dataset.eq(dataset) & frame.horizon.eq(2) & frame.metric.eq("delta_per_spike") & frame.contrast.eq(model + "__" + suffix)]
                if len(g) != 1:
                    raise ValueError("complete unique primary contrasts required")
                records.append(g.iloc[0].to_dict() | {"model": model, "control": name})
    primary = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (("primary_table", primary), ("all_contrasts", summary), ("by_animal", animals), ("coverage", counts), ("decisions", decisions)):
        frame.to_csv(out / f"occupancy_matched_{name}.csv", index=False)
    figures(primary, animals, out)
    display = primary.assign(
        dataset=primary.dataset.map(DATASETS),
        model=primary.model.map(MODELS),
        estimate=primary["mean"].map(lambda v: f"{v:+.5f}"),
        interval=[f"[{a:+.5f}, {b:+.5f}]" for a, b in zip(primary.ci_low, primary.ci_high, strict=True)],
        positive=[f"{int(a)}/{int(b)}" for a, b in zip(primary.positive_animals, primary.animals, strict=True)],
    )[["dataset", "model", "control", "estimate", "interval", "positive"]]
    text = f"""# Occupancy-Matched Forward Forecast Control

Non-rescoring report. All original eligible forecast hashes were reproduced.

## Decision

{classification}

The new criterion requires the same model to have a positive lower CI and
positive estimates in every animal in BOTH datasets at the fixed 40 ms horizon.
The full updated criterion additionally retains the original frozen-state,
no-history and global-rate controls. No change to the earlier failed original
compound verdict; this is a subsequent exploratory control, not confirmation.

## Primary Comparison

Animal-balanced nats per held-out target spike; conditional 95% bootstrap CIs.

{table_markdown(display)}

![Old and new controls](occupancy_matched_primary.png)

![Animal estimates](occupancy_matched_animals.png)

## Coverage

{table_markdown(counts)}

All immobile MUA candidate events remain included, not just continuous or
model-winning events. These are not all validated replay. Events too short
for a horizon remain explicitly unscored. Zero-spike targets have score zero
and undefined per-spike contrast. Event medians across five neural partitions
precede equal session means and equal animal weights. Horizon coverage changes.

## What The Control Preserves

The maximum-entropy null preserves the original MODEL stationary state
frequencies, state self-transition probabilities, and, for spatial IMM, every
mode-transition probability and probability of unchanged position within and
across modes. It removes remaining preferences for destinations conditional
on those constraints. Model equilibrium is not empirical event occupancy.
Both forecasts start from the same original forward-filtered origin posterior.
Parameters are reused, not refitted on target events. Null factors are solved
from transition constraints, not from held-out predictive scores.

The target is held-out cell identity conditional on total held-out spikes.
At 40 ms center lag, a full unobserved 20 ms gap separates origin and target
windows. Neither intervening nor target-time training spikes update forecasts.
Held-out spikes enter only the final scoring likelihood.

## Boundaries

- Any positive effect is relative to this constrained null, not proof of
  directional replay, Bayesian smoothing, planning, or a unique IMM mechanism.
- Spatial kernels are local diffusion, not directional momentum. Learned
  neural states may encode space despite lacking explicit coordinates.
- This null is not optimized as an alternative encoding/filtering model.
  It does not preserve all pairwise flow, distance, or recurrence structure.
- No causal intervention or independent confirmation. Detection used all
  cells and calibration folds are retrospective. Conditioning on spike count
  does not predict the occurrence or amplitude of a burst.
- Four PF and five Tanni animals limit generalization. Bootstrap intervals
  condition on these data/fits and do not correct the wider hypothesis search.
- No fuzzy continuity threshold or speed-uniformity claim is introduced.

## Verification

Frozen scoring commit: `{manifest["code_commit"]}`.
Run manifest SHA256: `{file_sha256(mp)}`.
Independent audit: {audit["independent_scores"]} forecast scores,
{audit["parameter_sets"]} complete null parameter sets;
maximum score error {audit["max_score_error"]:.3g} nats;
maximum stationary-frequency error {audit["max_equilibrium_error"]:.3g}.
{audit["scope"]}

The failed first run is preserved. Its operation-order/hash mismatch was
corrected and tested before this full rerun; no scientific threshold changed.
"""
    (out / "occupancy_matched_report.md").write_text(text)
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(non_rescoring=True, classification=classification, high_importance_discovery_established=False)
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}
    (out / "occupancy_matched_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = report(a.run_dir, a.audit, a.output_dir)
    print(result["classification"])
