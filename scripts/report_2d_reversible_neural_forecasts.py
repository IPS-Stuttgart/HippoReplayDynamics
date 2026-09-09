#!/usr/bin/env python3
"""Non-rescoring report separating directional from reversible prediction."""

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
from scripts.report_2d_lagged_neural_prediction import DATASETS, table_markdown

LABELS = {"dynamic_minus_reversible": "Original - reversible", "dynamic_minus_reverse": "Original - reversed", "reversible_minus_matched": "Reversible - occupancy/dwell"}


def interpret(d):
    if len(d) != 1 or d.primary_model.iloc[0] != "learned_hmm" or d.horizon_ms.iloc[0] != 40:
        raise ValueError("fixed primary neural model/horizon required")
    if d[["original_compound_verdict_changed", "independent_confirmation", "high_importance_discovery_established"]].any().any():
        raise ValueError("exploratory result cannot claim confirmation or alter prior verdict")
    if d.directional_predictive_lead.iloc[0]:
        return (
            "A directional predictive lead meets the frozen two-dataset criterion; it still requires independent confirmation and does not identify a unique biological mechanism."
        )
    return "The frozen two-dataset directional predictive criterion is not met. This does not establish absence of individual neural sequences."


def figures(summary, animals, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items(), strict=True):
        for y, (contrast, label) in enumerate(LABELS.items()):
            name = "learned_hmm__" + contrast
            row = summary[summary.dataset.eq(dataset) & summary.contrast.eq(name)].iloc[0]
            g = animals[animals.dataset.eq(dataset) & animals.horizon.eq(2) & animals.contrast.eq(name)].sort_values("animal")
            ax.plot([row.ci_low, row.ci_high], [y, y], color="#19755b", linewidth=3)
            ax.scatter(row["mean"], y, color="#19755b", s=55, zorder=3)
            for i, v in enumerate(g.delta_per_spike):
                ax.scatter(v, y + 0.12 + i * 0.04, color="#777777", s=18)
        ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_yticks(range(3), list(LABELS.values()))
        ax.set_ylim(2.6, -0.4)
        ax.set_title(title)
        ax.set_xlabel("Forecast advantage (nats / held-out spike)")
    fig.suptitle("Learned neural HMM: direction versus reversible connections\n40 ms center lag; green = animal mean/95% bootstrap CI, gray = individual animals", fontsize=13)
    fig.savefig(out / "reversible_forecasts_primary.png", dpi=180)
    plt.close(fig)


def report(run, audit_path, out):
    mp = run / "reversible_forecasts_manifest.json"
    m, a = json.loads(mp.read_text()), json.loads(audit_path.read_text())
    if m["status"] != "complete" or a["status"] != "pass" or a["input_file_sha256"]["run_manifest"] != file_sha256(mp):
        raise ValueError("matching passing audit and complete run required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError("changed scored output " + name)
    s = pd.read_csv(run / "reversible_forecasts_summary.csv.gz")
    animals = pd.read_csv(run / "reversible_forecasts_animals.csv.gz")
    d = pd.read_csv(run / "reversible_forecasts_decision.csv")
    conclusion = interpret(d)
    primary = s[s.horizon.eq(2) & s.metric.eq("delta_per_spike")].copy()
    g = primary[primary.contrast.isin(["learned_hmm__" + c for c in LABELS])]
    if len(g) != 6 or g.duplicated(["dataset", "contrast"]).any() or set(g.dataset) != set(DATASETS):
        raise ValueError("complete primary contrast table required")
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (("primary_table", primary), ("all_contrasts", s), ("by_animal", animals), ("decision", d)):
        frame.to_csv(out / f"reversible_forecasts_{name}.csv", index=False)
    figures(primary, animals, out)
    display = g.assign(
        dataset=g.dataset.map(DATASETS),
        contrast=g.contrast.str.replace("learned_hmm__", "", regex=False).map(LABELS),
        estimate=g["mean"].map(lambda v: f"{v:+.5f}"),
        interval=[f"[{v:+.5f}, {w:+.5f}]" for v, w in zip(g.ci_low, g.ci_high, strict=True)],
        positive=[f"{int(v)}/{int(w)}" for v, w in zip(g.positive_animals, g.animals, strict=True)],
    )[["dataset", "contrast", "estimate", "interval", "positive"]]
    text = f"""# Directional Versus Reversible Neural Forecasts

Non-rescoring report. {conclusion}

## Primary Results

{table_markdown(display)}

![Direction and reversible association contrasts](reversible_forecasts_primary.png)

Original-versus-reversible tests the predictive contribution of net directional
transition flow. Original-versus-reversed tests inversion of that flow.
Reversible-versus-occupancy/dwell tests the remaining unordered connectivity.
The nulls are propagation kernels, not reversed observations or time-bin shuffles.

Both models start from the same original, forward-filtered training-cell state.
No intervening or target-time training spikes, and no held-out spikes, update
forecasts. Target likelihoods are proper conditional cell-identity probabilities
given the total held-out spike count; they do not predict burst amplitude.

The primary model was the K50 learned neural HMM, fixed before these scores.
The criterion requires both direction contrasts to have positive lower CIs
and positive estimates in every animal in both datasets at 40 ms. Spatial
IMM/diffusion rows and other horizons are diagnostics, not substitutes for a
failed primary endpoint. Previous broad forecasting gates remain unchanged.

## Population And Uncertainty

{m["events"]:,} frozen immobile MUA candidates across {len(m["completed"])}
recordings; {m["rows"]:,} event/split/model/horizon rows include explicit
too-short statuses. These are not all validated replay. Five neural partitions
per event; event medians precede equal session and animal means. Intervals
are conditional animal bootstraps, with four PF and five Tanni animals.
They do not correct the extensive preceding hypothesis search. Per-spike
ratios exclude zero-spike targets; raw-score tables retain their zero scores.

## Limits And Novelty

- Reversible transitions can generate ordered individual paths in either
  direction. Absence of a population-level directional advantage is not
  absence of sequences or memory replay.
- The original inference rule is shared. This is not a comparison of fully
  refitted optimal reversible and irreversible models.
- Spatial diffusion is reversible by construction. Joint spatial-IMM
  asymmetry can arise from its mode/position prior and is not learned
  biological direction. It is a diagnostic model, not the primary endpoint.
- This does not estimate physical entropy production, planning, or Bayesian
  smoothing. Learned neural states can encode space implicitly.
- Neural irreversibility is established. A
  [2026 hippocampal preprint](https://arxiv.org/abs/2601.05284) relates flows
  during movement to behavior and encoding resolution. An immobile-event
  predictive contrast still needs independent evidence for a novel mechanism.

## Provenance And Verification

Scoring commit: `{m["code_commit"]}`.
Manifest SHA256: `{file_sha256(mp)}`.
Independent scores checked: {a["independent_scores"]:,}; maximum error
{a["max_score_error"]:.3g} nats. {a["scope"]}
No source models, event selections or thresholds were refitted by this report.
"""
    (out / "reversible_forecasts_report.md").write_text(text)
    p = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    p.update(non_rescoring=True, conclusion=conclusion, high_importance_discovery_established=False)
    p["output_sha256"] = {f.name: file_sha256(f) for f in out.iterdir() if f.is_file()}
    (out / "reversible_forecasts_report_manifest.json").write_text(json.dumps(p, indent=2) + "\n")
    return p


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = report(a.run_dir, a.audit, a.output_dir)
    print(result["conclusion"])
