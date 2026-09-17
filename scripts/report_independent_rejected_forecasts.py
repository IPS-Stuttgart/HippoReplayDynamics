#!/usr/bin/env python3
"""Non-rescoring compact report of independently detected rejected-event forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._provenance import build_script_provenance, file_sha256

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY = "dynamic_minus_matched_own"
ID = ["dataset", "animal", "session"]


def primary_rows(table, group="rejected_with_opportunity", level="full"):
    mask = table.horizon.eq(2) & table.model.eq("learned_hmm") & table.group.eq(group) & table.level.eq(level)
    if "metric" in table:
        mask &= table.metric.eq("delta_per_spike")
    return table[mask].copy()


def evidence_role(summary, expected_animals):
    needed = [PRIMARY, "dynamic_minus_global", "dynamic_minus_no_history"]
    x = summary[summary.contrast.isin(needed)]
    return bool(len(x) == 3 and set(x.contrast) == set(needed) and x.animals.eq(expected_animals).all() and x.positive_animals.eq(expected_animals).all() and x.ci_low.gt(0).all())


def run(root, audit_path, out):
    manifest_path = root / "independent_rejected_forecast_manifest.json"
    m = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if m["status"] != "complete" or audit["status"] != "pass" or m["subset_debug"]:
        raise ValueError("complete, audited full-cohort run required")
    if audit["input_file_sha256"]["run_manifest"] != file_sha256(manifest_path):
        raise ValueError("audit references another run")
    for name, digest in m["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("changed output " + name)
    summary = pd.read_csv(root / "rejected_forecast_summary.csv.gz")
    animals = pd.read_csv(root / "rejected_forecast_animals.csv.gz")
    sessions = pd.read_csv(root / "rejected_forecast_sessions.csv.gz")
    events = pd.read_csv(root / "rejected_forecast_events.csv.gz")
    status = pd.read_csv(root / "rejected_forecast_session_status.csv")
    primary = primary_rows(summary)
    supported = {d: evidence_role(primary[primary.dataset.eq(d)], n) for d, n in [("pfeiffer_foster", 4), ("tanni2022", 5)]}
    # This dated narrative is specific to the verified outcome, never a generic
    # positive-result template for an arbitrary subsequent scoring run.
    if not supported["pfeiffer_foster"] or supported["tanni2022"]:
        raise ValueError("outcome differs from this dated narrative; report explicitly")
    by_animal = primary_rows(animals)
    by_session = primary_rows(sessions)
    cohorts = events[events.horizon.eq(2) & events.level.eq("full") & events.model.eq("learned_hmm") & events.contrast.eq(PRIMARY)]
    denominators = cohorts.groupby(["dataset", "group"], as_index=False).agg(
        distinct_events=("event_id", "size"),
        events_with_target_spikes=("delta_per_spike", "count"),
        animal_count=("animal", "nunique"),
        session_count=("session", "nunique"),
    )
    labels = pd.concat([pd.read_csv(root / x["tag"] / "labels.csv.gz") for x in m["completed"]], ignore_index=True)
    acceptance = labels.groupby(ID, as_index=False).agg(
        full_acceptance=("full_geometric_pass", "mean"),
        half_acceptance=("half_geometric_pass", "mean"),
    )
    acceptance_animals = acceptance.groupby(["dataset", "animal"], as_index=False)[["full_acceptance", "half_acceptance"]].mean()
    thinning = summary[
        summary.horizon.eq(2) & summary.model.eq("learned_hmm") & summary.group.eq("lost_with_thinning") & summary.contrast.eq(PRIMARY) & summary.metric.eq("delta_per_spike")
    ].copy()
    gates = [
        ("complete_33_sessions", len(status) == 33 and status.status.eq("complete").all()),
        ("nonempty_event_population", status.events.sum() > 0),
        ("independent_reconstruction_pass", audit["status"] == "pass"),
        ("all_detections_and_geometry_reconstructed", audit["all_detections_reconstructed"] and audit["all_geometric_paths_reconstructed"]),
        ("all_primary_estimates_reconstructed", audit["all_primary_aggregates_reconstructed"]),
        ("evaluation_cells_excluded_from_detection", m["independent_event_detection"]),
    ]
    gates.append(("overall_technical", all(v for _, v in gates)))
    out.mkdir(parents=True, exist_ok=False)
    tables = dict(
        primary_summary=primary,
        primary_by_animal=by_animal,
        primary_by_session=by_session,
        event_denominators=denominators,
        thinning_summary=thinning,
        geometric_acceptance_by_session=acceptance,
        geometric_acceptance_by_animal=acceptance_animals,
        technical_gates=pd.DataFrame(gates, columns=["gate", "passed"]),
        session_status=status,
    )
    for name, table in tables.items():
        table.to_csv(out / (name + ".csv"), index=False)
    figure(out, primary, by_animal, thinning, acceptance_animals)
    lines = [
        "# Do trajectory-rejected events predict independent neurons?",
        "",
        "## Answer",
        "",
        "**Supported in Pfeiffer-Foster under this specified test; not replicated in Tanni.**",
        "This is group-level predictive temporal organization, not proof that each rejected event is continuous replay.",
        "",
        "## Primary Result",
        "",
        "Predict the identities of held-out spikes 40 ms ahead across a 20-ms unobserved gap.",
        "Compare a learned neural HMM with an occupancy/self-transition-matched null using its own causal filter.",
        "Units: nats per held-out target spike. Event medians over five partitions precede equal-session and equal-animal means.",
        "",
        "| Dataset | Detected candidates | Rejected events with informative targets | Mean advantage | Descriptive animal-bootstrap 95% interval | Positive animals |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for dataset in ["pfeiffer_foster", "tanni2022"]:
        p = primary[primary.dataset.eq(dataset) & primary.contrast.eq(PRIMARY)].iloc[0]
        n = int(status[status.dataset.eq(dataset)].events.sum())
        d = denominators[denominators.dataset.eq(dataset) & denominators.group.eq("rejected_with_opportunity")].iloc[0]
        lines.append(f"| {dataset} | {n} | {d.events_with_target_spikes} | {p['mean']:+.6f} | [{p.ci_low:+.6f}, {p.ci_high:+.6f}] | {p.positive_animals}/{p.animals} |")
    lines += [
        "",
        "A candidate may fail geometry in only some partitions. The event count is the union of qualifying",
        "partitions, not a count of significant individual events. Zero-target events remain in the denominator",
        "tables but cannot contribute a per-spike identity score. The two rejected/thinning groups can overlap",
        "across partitions and must not be added.",
        "",
        "## What Changed Since The Earlier Test",
        "",
        "- Detection now uses a fixed reserved 20% of RUN-QC neurons. Those cells do not enter evaluation.",
        "- The remaining population supplies five 70/30 inference/evaluation splits. Evaluation spikes never",
        "  enter test-event selection, MAP geometry, latent inference or forecasts.",
        "- Prediction is forward-only across a genuine observation gap, not same-time reconstruction.",
        "- The matched null now has its own origin filter, as well as the legacy shared-origin sensitivity.",
        "- All 33 recordings were retained. Candidates were redetected from full-session caches; old event",
        "  labels/scores were not reused. Counts therefore differ from the historical all-cell candidate set.",
        "",
        "## Interpretation",
        "",
        "PF's primary contrast and the global-composition/no-history adequacy contrasts are positive in all",
        "four rats. Tanni's matched-null primary mean is near zero and only one of five animal estimates is",
        "positive. Beating frozen origin or global rates alone is not sufficient to override that failure.",
        "The result supports retained temporal information in PF geometric failures, not a universal",
        "cross-dataset finding or a unique IMM mechanism.",
        "",
        "The prespecified lost-with-thinning sensitivity is also positive in PF when prediction actually",
        "uses the SMALLER inference population, with evaluation cells fixed. This closes the earlier gap",
        "where scores from the larger population had been used to describe events lost with fewer cells.",
        "",
        "## Inference Limits",
        "",
        "- The recordings and earlier aggregate results were already explored. This is a prospectively frozen",
        "  retrospective control, not independent biological confirmation.",
        "- Four PF and five Tanni animals remain small samples. PF's exact two-sided animal sign-test p is",
        "  0.125 despite all four estimates being positive. Bootstrap intervals are conditional/descriptive;",
        "  they do not account for the broader exploratory hypothesis search.",
        "- The geometric screen transfers the PF-style necessary continuity/displacement criterion. It does",
        "  not implement both original 5000-shuffle tests. Geometry failures necessarily fail that conjunction;",
        "  geometry passes are not thereby validated replay.",
        "- Detection used RUN-QC cells rather than every recorded unit. This changes the event population",
        "  and may reduce detection sensitivity; no recall or replay-prevalence estimate follows.",
        "- Models learn emissions/dynamics from other calibration events, including evaluation-cell activity",
        "  in those OTHER events. Five chronological folds and a one-second guard exclude all test events.",
        "- No claim of uniform physical speed, causal memory function, or a new generative mechanism follows.",
        "",
        "## Verification",
        "",
        f"- Producer commit: {m['code_commit']}.",
        f"- All 33 sessions complete; {sum(x['events'] for x in m['completed'])} candidates; {sum(x['rows'] for x in m['completed'])} event/split/level/model/horizon rows.",
        f"- Independently reconstructed {sum(x['counts_checked'] for x in audit['sessions']):,} count entries,",
        f"  {sum(x['geometry_paths_checked'] for x in audit['sessions']):,} geometric paths and all detections.",
        f"- Reconstructed {sum(x['scores_reconstructed'] for x in audit['sessions']):,} neural-model predictive scores",
        "  from the first chronological event of every fold, both inference levels, all splits and eligible lags.",
        f"  Maximum absolute discrepancy: {max(x['maximum_score_error'] for x in audit['sessions']):.3g} nats.",
        "- All primary animal estimates and intervals were independently recomputed. Spatial-diffusion",
        "  sensitivity scores and model fitting were not independently rerun by that verifier.",
        "- The first verifier stopped on a NumPy Boolean-subtraction bug; the corrected v2 audit passed.",
        "  Scoring outputs were never changed. No raw recordings or large count/fit arrays enter this report.",
        "",
        "![Animal-level predictive advantage and thinning sensitivity](rejected_forecast_validation.png)",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"run_manifest": manifest_path, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(non_rescoring=True, output_sha256={p.name: file_sha256(p) for p in out.iterdir()})
    (out / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


def figure(out, primary, animals, thinning, acceptance):
    colors = {"pfeiffer_foster": "#00786c", "tanni2022": "#b24a62"}
    names = {"pfeiffer_foster": "Pfeiffer-Foster", "tanni2022": "Tanni"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), layout="constrained")
    for x, dataset in enumerate(colors):
        color = colors[dataset]
        a = animals[animals.dataset.eq(dataset) & animals.contrast.eq(PRIMARY)].sort_values("animal")
        axes[0].scatter(x + np.linspace(-0.10, 0.10, len(a)), a.delta_per_spike, c=color, s=32, zorder=3)
        s = primary[primary.dataset.eq(dataset) & primary.contrast.eq(PRIMARY)].iloc[0]
        axes[0].errorbar(x + 0.23, s["mean"], yerr=[[s["mean"] - s.ci_low], [s.ci_high - s["mean"]]], fmt="D", color="black", capsize=4)
        t = thinning[thinning.dataset.eq(dataset)].set_index("level")
        axes[1].plot([0, 1], [t.loc["full", "mean"], t.loc["half", "mean"]], "-o", color=color, label=names[dataset])
        for i, level in enumerate(["full", "half"]):
            z = t.loc[level]
            axes[1].errorbar(i, z["mean"], yerr=[[z["mean"] - z.ci_low], [z.ci_high - z["mean"]]], color=color, capsize=4)
        b = acceptance[acceptance.dataset.eq(dataset)]
        for row in b.itertuples(index=False):
            axes[2].plot([0, 1], [row.full_acceptance * 100, row.half_acceptance * 100], "-o", color=color, alpha=0.55, lw=1)
    axes[0].set(title="Rejected-event future prediction", xticks=[0, 1], xticklabels=list(names.values()), ylabel="Advantage over matched null (nats/spike)")
    axes[1].set(title="Events lost under cell thinning", xticks=[0, 1], xticklabels=["Full inference", "Half inference"], ylabel="Advantage over matched null (nats/spike)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].set(title="Geometric acceptance", xticks=[0, 1], xticklabels=["Full inference", "Half inference"], ylabel="Candidates passing geometry (%)")
    for ax in axes[:2]:
        ax.axhline(0, color="#777777", linestyle="--", lw=0.8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Independent detection and held-out future prediction", fontsize=14)
    fig.savefig(out / "rejected_forecast_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    run(a.run_dir, a.audit, a.output_dir)
