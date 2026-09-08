#!/usr/bin/env python3
"""Non-rescoring report for the frozen lagged neural prediction experiment."""

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

DATASETS = {"pfeiffer_foster": "Pfeiffer-Foster", "tanni2022": "Tanni"}
MODELS = {"spatial_imm_real": "Spatial IMM", "spatial_diffusion_real": "Spatial diffusion", "learned_hmm": "Learned neural HMM"}
BASELINES = {"dwell_only": "Dwell-preserving reset", "frozen": "Frozen origin state", "no_history": "No event history", "global": "Calibration cell rates"}
COLORS = {"dwell_only": "#19755b", "frozen": "#ab3c53", "no_history": "#555555", "global": "#2364ad"}


def table_markdown(frame):
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def interpret(decisions):
    if decisions.empty or set(decisions.model) != set(MODELS) or decisions.model.duplicated().any():
        raise ValueError("complete unique model decisions required")
    if decisions.independent_confirmation.any() or decisions.high_importance_discovery_established.any():
        raise ValueError("exploratory data cannot assert independent confirmation")
    names = decisions.loc[decisions.replicated_forecasting_lead, "model"].map(MODELS).tolist()
    if names:
        return "Forecasting lead requiring independent confirmation: " + ", ".join(names) + ". This is not proof of a unique generative mechanism."
    return "No model met the frozen two-dataset forecasting criterion. Individual positive contrasts are diagnostic results, not a replicated sequence-generation mechanism."


def make_figures(summary, out):
    primary = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for i, (dataset, label) in enumerate(DATASETS.items()):
        for j, (model, title) in enumerate(MODELS.items()):
            ax = axes[i, j]
            for y, (baseline, name) in enumerate(BASELINES.items()):
                row = primary[primary.dataset.eq(dataset) & primary.contrast.eq(model + "__dynamic_minus_" + baseline)].iloc[0]
                ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[baseline], linewidth=2)
                ax.scatter(row["mean"], y, color=COLORS[baseline], zorder=3)
            ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
            ax.set_yticks(range(4), list(BASELINES.values()))
            ax.invert_yaxis()
            ax.set_title(label + " | " + title)
            ax.set_xlabel("Forecast advantage (nats / held-out spike)")
    fig.suptitle("40 ms center lag: dynamics versus four controls\nAnimal-balanced estimates and conditional animal-bootstrap 95% intervals", fontsize=14)
    fig.savefig(out / "lagged_prediction_primary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for i, (dataset, label) in enumerate(DATASETS.items()):
        for j, (model, title) in enumerate(MODELS.items()):
            ax = axes[i, j]
            for baseline in ("dwell_only", "frozen", "global"):
                g = summary[summary.dataset.eq(dataset) & summary.metric.eq("delta_per_spike") & summary.contrast.eq(model + "__dynamic_minus_" + baseline)].sort_values("horizon")
                x = g.horizon.to_numpy() * 20
                ax.plot(x, g["mean"], "o-", color=COLORS[baseline], label=BASELINES[baseline])
                ax.fill_between(x, g.ci_low, g.ci_high, color=COLORS[baseline], alpha=0.12)
            ax.axhline(0, color="0.5", linewidth=1)
            ax.axvline(40, color="0.6", linestyle=":", linewidth=1)
            ax.set_xticks([20, 40, 80])
            ax.set_xlabel("Center-to-center lag (ms)")
            ax.set_ylabel("Nats / held-out spike")
            ax.set_title(label + " | " + title)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Horizon sensitivity; 40 ms was fixed as primary\nThe eligible event population can change with horizon", fontsize=14)
    fig.savefig(out / "lagged_prediction_horizons.png", dpi=180)
    plt.close(fig)


def report(run, audit_path, out):
    mp = run / "lagged_prediction_manifest.json"
    manifest, audit = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("complete run and matching passing independent audit required")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed scored output " + name)
    summary = pd.read_csv(run / "lagged_prediction_summary.csv.gz")
    animals = pd.read_csv(run / "lagged_prediction_animals.csv.gz")
    coverage = pd.read_csv(run / "lagged_prediction_coverage.csv")
    decisions = pd.read_csv(run / "lagged_prediction_decisions.csv")
    classification = interpret(decisions)
    primary = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike")].copy()
    counts = coverage.groupby(["dataset", "horizon"], as_index=False).agg(
        selected_events=("selected_events", "sum"),
        eligible_events=("eligible_events", "sum"),
        target_bins=("target_bins", "sum"),
        discarded_partial_spikes=("discarded_partial_spikes", "sum"),
        sessions=("session", "size"),
        animals=("animal", "nunique"),
    )
    events = pd.read_csv(run / "lagged_prediction_events.csv.gz")
    representative = events[events.contrast.eq("learned_hmm__dynamic_minus_global")]
    support = representative.groupby(["dataset", "horizon"]).valid_neural_splits.agg(lambda v: int(v.gt(0).sum())).rename("per_spike_supported_events").reset_index()
    counts = counts.merge(support, on=["dataset", "horizon"], validate="one_to_one")
    if not counts.per_spike_supported_events.le(counts.eligible_events).all():
        raise ValueError("per-spike support exceeds temporally eligible events")
    out.mkdir(parents=True, exist_ok=False)
    primary.to_csv(out / "lagged_prediction_primary_table.csv", index=False)
    summary.to_csv(out / "lagged_prediction_all_contrasts.csv", index=False)
    animals.to_csv(out / "lagged_prediction_by_animal.csv", index=False)
    counts.to_csv(out / "lagged_prediction_coverage.csv", index=False)
    decisions.to_csv(out / "lagged_prediction_decisions.csv", index=False)
    make_figures(summary, out)
    display = []
    for dataset, label in DATASETS.items():
        for model, title in MODELS.items():
            for baseline, control in BASELINES.items():
                row = primary[primary.dataset.eq(dataset) & primary.contrast.eq(model + "__dynamic_minus_" + baseline)].iloc[0]
                display.append(
                    {
                        "Dataset": label,
                        "Model": title,
                        "Control": control,
                        "Nats/spike": f"{row['mean']:+.4f}",
                        "95% CI": f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]",
                        "Positive animals": f"{int(row.positive_animals)}/{int(row.animals)}",
                    }
                )
    text = f"""# Lagged Neural Prediction: Dwell And Forecast Controls

Non-rescoring report. Technical scoring and independent reconstruction passed.

## Decision

{classification}

The frozen criterion requires the same model to beat all four controls at
40 ms center lag, with a positive CI lower bound and positive mean in every
animal in both datasets. Spatial-route claims additionally require a real-map
advantage and a positive map-by-route interaction. No endpoint was selected
after inspecting these scores.

## Coverage

{table_markdown(counts)}

These are immobile MUA candidate windows, not all validated continuous replay.
Each event has five neural partitions. Events too short for a horizon retain
explicit missing-score status; empty held-out target vectors score zero, with
undefined per-spike ratios. Final partial 20-ms bins are discarded and counted.

## Primary Results

{table_markdown(pd.DataFrame(display))}

![Primary forecast contrasts](lagged_prediction_primary.png)

![Fixed horizon sensitivity](lagged_prediction_horizons.png)

## What Was Predicted

At each origin, a forward filter sees training-cell spikes only up to that
origin. It predicts held-out cell identities in a later complete 20-ms bin,
conditional on that bin's total held-out spike count. At the primary lag there
is a 20-ms unobserved interval between source-window end and target-window
start. No held-out, intervening or contemporaneous training spikes update the
forecast. This is not prediction of total population firing rate.

The dwell control starts from the same origin posterior, preserves individual
stay probabilities, and replaces preferred destinations. The frozen control
retains the origin posterior. No-history propagation uses no event spikes.
The global control is a separate-event calibrated population composition.
The separately reported same-time filtered score is NOT a forecast.

## Interpretation Limits

- Spatial IMM and diffusion have symmetric local kernels, not directional
  momentum. An advantage over a spatial reset is not proof of a planned route.
- Dwell controls preserve specified self-transition probabilities, not every
  stationary occupancy or full-joint recurrence statistic. No alternative
  transition model was refitted to optimize this control.
- Forecasting beyond frozen state may reflect uncertainty growth, rather than
  a specific biological mechanism. All four contrasts matter together.
- The learned neural HMM is coordinate-free, but its latent states can still
  encode space. Neither model superiority nor a wrong-map difference uniquely
  identifies spatial computation.
- Within-event inference is forward-only; event detection used all cells,
  encoding maps may use later RUN, and calibration folds are retrospective.
  This is not an online causal or intervention study.
- Event medians across neural splits, then equal session and animal weights.
  CIs resample only four PF or five Tanni animals conditional on the fixed
  fits, folds and events. They do not correct the broader hypothesis search.
- A failed compound criterion does not prove absent temporal structure.
  A passing one is an exploratory lead requiring independent confirmation.

## Provenance And Verification

- Frozen scoring commit: `{manifest["code_commit"]}`.
- Run manifest SHA256: `{file_sha256(mp)}`.
- Independent reconstruction: {audit["independent_score_reconstructions"]}
  predictive scores from {audit["sampled_events"]} deterministically sampled
  events; maximum absolute error {audit["maximum_absolute_score_error"]:.3g} nats.
- {audit["scope"]}
- No source parameter refits or event rescoring are performed by this reporter.

## Novelty Boundary

Neural sequence prediction is established. Relevant precedents include
[associative versus predictive hippocampal codes](https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/),
[predictive sequence learning](https://www.sciencedirect.com/science/article/pii/S0896627324003714),
and [replay evaluation without ground truth](https://elifesciences.org/articles/85635).
These results alone do not establish a novel high-importance biological finding.
"""
    (out / "lagged_prediction_report.md").write_text(text)
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(status="complete", non_rescoring=True, classification=classification, independent_confirmation=False, high_importance_discovery_established=False)
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}
    (out / "lagged_prediction_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(classification)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    report(a.run_dir, a.audit, a.output_dir)
