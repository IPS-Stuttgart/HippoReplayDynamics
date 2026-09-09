#!/usr/bin/env python3
"""Non-rescoring, split-internal continuity/prediction stratification."""

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
sys.path.insert(0, str(ROOT / "src"))
from _provenance import build_script_provenance, file_sha256
from audit_training_continuity_prediction import checked

from hipporeplayimm.training_continuity import (
    IDENTITY,
    PRIMARY_CONTRASTS,
    PRIMARY_GROUPS,
    SETTINGS,
    grouped_event_values,
)

EXPECTED_ANIMALS = {"pfeiffer_foster": 4, "tanni2022": 5}
AGGREGATION = SETTINGS + ["group", "contrast"]


def hierarchical_interval(frame, draws=5000, seed=20260910):
    columns = ["delta", "delta_per_heldout_spike"]
    arrays = [[g[columns].to_numpy() for _, g in rat.groupby("session")] for _, rat in frame.groupby("animal")]
    if not arrays or draws < 1:
        raise ValueError("nonempty hierarchical sample required")
    rng, samples = np.random.default_rng(seed), []
    for _ in range(draws):
        rats = []
        for i in rng.integers(len(arrays), size=len(arrays)):
            sessions = arrays[i]
            means = []
            for j in rng.integers(len(sessions), size=len(sessions)):
                values = sessions[j]
                sampled = values[rng.integers(len(values), size=len(values))]
                n = np.isfinite(sampled).sum(axis=0)
                means.append(np.divide(np.nansum(sampled, axis=0), n, out=np.full(2, np.nan), where=n > 0))
            means = np.asarray(means)
            n = np.isfinite(means).sum(axis=0)
            rats.append(np.divide(np.nansum(means, axis=0), n, out=np.full(2, np.nan), where=n > 0))
        rats = np.asarray(rats)
        n = np.isfinite(rats).sum(axis=0)
        samples.append(np.divide(np.nansum(rats, axis=0), n, out=np.full(2, np.nan), where=n > 0))
    samples = np.asarray(samples)
    ci = np.full((2, 2), np.nan)
    for column in range(2):
        finite = samples[np.isfinite(samples[:, column]), column]
        if len(finite):
            ci[:, column] = np.quantile(finite, [0.025, 0.975])
    return ci


def aggregate_points(events):
    sessions = events.groupby(IDENTITY + AGGREGATION, as_index=False).agg(
        delta=("delta", "mean"),
        delta_per_heldout_spike=("delta_per_heldout_spike", "mean"),
        events=("event_id", "size"),
        mean_qualifying_splits=("qualifying_splits", "mean"),
    )
    animals = sessions.groupby(["dataset", "animal", *AGGREGATION], as_index=False).agg(
        delta=("delta", "mean"),
        delta_per_heldout_spike=("delta_per_heldout_spike", "mean"),
        events=("events", "sum"),
        sessions=("session", "size"),
        mean_qualifying_splits=("mean_qualifying_splits", "mean"),
    )
    summary = animals.groupby(["dataset", *AGGREGATION], as_index=False).agg(
        mean=("delta", "mean"),
        mean_per_heldout_spike=("delta_per_heldout_spike", "mean"),
        animals=("animal", "size"),
        positive_animals=("delta", lambda x: int((x > 0).sum())),
        events=("events", "sum"),
        sessions=("sessions", "sum"),
        mean_qualifying_splits=("mean_qualifying_splits", "mean"),
    )
    return sessions, animals, summary


def decision_table(summary):
    rows = []
    for dataset, required in EXPECTED_ANIMALS.items():
        for group in PRIMARY_GROUPS:
            local = summary[summary.dataset.eq(dataset) & summary.group.eq(group) & summary.contrast.isin(PRIMARY_CONTRASTS)]
            complete = set(local.contrast) == set(PRIMARY_CONTRASTS) and len(local) == len(PRIMARY_CONTRASTS)
            available = complete and bool(local.animals.eq(required).all()) and bool(local.events.gt(0).all())
            positive = available and bool(local.ci_low.gt(0).all()) and bool(local.positive_animals.eq(required).all())
            rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "all_primary_contrasts_present": complete,
                    "all_source_animals_represented": available,
                    "joint_predictive_support": positive,
                    "events": int(local.events.max()) if len(local) else 0,
                    "verdict": "bounded_independent_predictive_support" if positive else "predictive_support_not_established" if available else "group_unavailable_or_incomplete",
                    "biological_replay_truth_established": False,
                    "parent_replication_gate_changed": False,
                }
            )
    return pd.DataFrame(rows)


def figure(summary, output):
    names = ["IMM - independent", "IMM - stationary", "IMM - composition", "Original - shuffled", "Order x map"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), squeeze=False)
    for row, group in enumerate(PRIMARY_GROUPS):
        for col, (contrast, name) in enumerate(zip(PRIMARY_CONTRASTS, names, strict=True)):
            ax = axes[row, col]
            for i, (dataset, color, label) in enumerate([("pfeiffer_foster", "#126d82", "PF"), ("tanni2022", "#a95529", "Tanni")]):
                local = summary[summary.dataset.eq(dataset) & summary.group.eq(group) & summary.contrast.eq(contrast)]
                if len(local) == 1:
                    r = local.iloc[0]
                    ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=2)
                    ax.scatter([r["mean"]], [i], c=color, s=30, zorder=3)
            ax.axvline(0, c="0.5", linestyle=":", linewidth=1)
            ax.set_yticks([0, 1], ["PF", "Tanni"])
            ax.set_ylim(-0.5, 1.5)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predictive nats", fontsize=9)
            ax.tick_params(labelsize=9)
            if col == 0:
                ax.set_ylabel("Rejected, sufficient frames" if row == 0 else "Lost after training-cell thinning", fontsize=10)
    fig.suptitle("Training-only geometry versus separate held-out neurons\nEvent medians; equal-animal means and conditional hierarchical 95% intervals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(args):
    root, out = args.run_dir.resolve(), args.output_dir.resolve()
    manifest_path = root / "training_continuity_manifest.json"
    m, a = json.loads(manifest_path.read_text()), json.loads(args.audit.read_text())
    if (
        m.get("status") != "complete"
        or a.get("status") != "pass"
        or a.get("input_file_sha256", {}).get("run_manifest") != file_sha256(manifest_path)
        or sum(r["events"] for r in a.get("records", [])) != 9225
    ):
        raise ValueError("complete independently audited classification required")
    for name, digest in m["output_sha256"].items():
        checked(root / name, digest)
    out.mkdir(parents=True, exist_ok=False)
    label_parts, prediction_parts = [], []
    for item in m["completed"]:
        label_parts.append(pd.read_csv(root / f"{item['tag']}_labels.csv.gz"))
        prediction_parts.append(pd.read_csv(root / f"{item['tag']}_predictions.csv.gz"))
    labels, predictions = pd.concat(label_parts, ignore_index=True), pd.concat(prediction_parts, ignore_index=True)
    event_primary, sessions_all, animals_all, summary_all = [], [], [], []
    for settings, local in labels.groupby(SETTINGS, sort=True):
        events = grouped_event_values(local, predictions)
        sessions, animals, summary = aggregate_points(events)
        sessions_all.append(sessions)
        animals_all.append(animals)
        summary_all.append(summary)
        if settings == ("parent", "edge_only", 10):
            event_primary.append(events)
    event_primary = pd.concat(event_primary, ignore_index=True)
    sessions, animals, summary = pd.concat(sessions_all), pd.concat(animals_all), pd.concat(summary_all)
    primary = event_primary[event_primary.group.isin(PRIMARY_GROUPS) & event_primary.contrast.isin(PRIMARY_CONTRASTS)]
    ci_rows = []
    for key, g in primary.groupby(["dataset", "group", "contrast"]):
        interval = hierarchical_interval(g)
        ci_rows.append(
            dict(zip(["dataset", "group", "contrast"], key, strict=True))
            | {"ci_low": interval[0, 0], "ci_high": interval[1, 0], "per_spike_ci_low": interval[0, 1], "per_spike_ci_high": interval[1, 1]}
        )
        print("interval", *key, flush=True)
    intervals = pd.DataFrame(ci_rows, columns=["dataset", "group", "contrast", "ci_low", "ci_high", "per_spike_ci_low", "per_spike_ci_high"])
    primary_summary = summary[
        summary.support.eq("parent")
        & summary.bin_filter.eq("edge_only")
        & summary.min_frames.eq(10)
        & summary.group.isin(PRIMARY_GROUPS)
        & summary.contrast.isin(PRIMARY_CONTRASTS)
    ].merge(intervals, on=["dataset", "group", "contrast"], validate="one_to_one")
    decisions = decision_table(primary_summary)
    split0 = grouped_event_values(
        labels[labels.split.eq(0) & labels.support.eq("parent") & labels.bin_filter.eq("edge_only") & labels.min_frames.eq(10)], predictions[predictions.split.eq(0)]
    )
    _, _, split0_summary = aggregate_points(split0)
    count_events = labels.groupby(IDENTITY + ["event_id", *SETTINGS], as_index=False).agg(
        full_training_pass_fraction=("full_training_pass", "mean"), nested_half_pass_fraction=("nested_half_pass", "mean")
    )
    count_sessions = count_events.groupby(IDENTITY + SETTINGS, as_index=False)[["full_training_pass_fraction", "nested_half_pass_fraction"]].mean()
    counts = count_sessions.groupby(["dataset", "animal", *SETTINGS], as_index=False)[["full_training_pass_fraction", "nested_half_pass_fraction"]].mean()
    outputs = {
        "training_continuity_primary_event_contrasts.csv.gz": event_primary,
        "training_continuity_by_session.csv": sessions,
        "training_continuity_by_animal.csv": animals,
        "training_continuity_sensitivity_summary.csv": summary,
        "training_continuity_primary_summary.csv": primary_summary,
        "training_continuity_split0_summary.csv": split0_summary,
        "training_continuity_acceptance_by_animal.csv": counts,
        "training_continuity_decisions.csv": decisions,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    figure(primary_summary, out / "training_continuity_prediction.png")
    text = [
        "# Training-Only Continuity And Independent Prediction",
        "",
        "Non-rescoring retrospective validation. Classification uses only training neurons; predictions concern separate held-out neurons. Candidate detection was previously performed using all cells.",
        "",
        "## Primary Results",
        "",
        "| Dataset | Group | Contrast | Equal-animal mean | 95% interval | Positive animals | Events |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for r in primary_summary.itertuples(index=False):
        text.append(f"| {r.dataset} | {r.group} | {r.contrast} | {r.mean:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.positive_animals}/{r.animals} | {r.events} |")
    text += ["", "## Decisions", ""]
    text += [f"- {r.dataset}, {r.group}: `{r.verdict}`." for r in decisions.itertuples(index=False)]
    text += [
        "",
        "## Interpretation Boundaries",
        "",
        "- A geometric failure cannot pass the full transferred continuity-and-shuffle screen. A geometric pass here is not shuffle-significant replay: the two map-shuffle tests were not repeated.",
        "- The primary rejected group has sufficient training-cell-supported frames to attempt the geometric test. Short/unsupported groups and every source candidate remain visible in the sensitivity tables.",
        "- Lost-with-thinning events are tested using predictions inferred from the larger training population, not the smaller one. The result would show independent support for a rejected event, not recovery from the reduced population.",
        "- Events receive one median per group across their qualifying splits. An event may enter multiple groups in different splits; these are not independent group samples. Group differences were not tested.",
        "- Primary CIs condition on the fixed maps, candidate ascertainment, neuron partitions and order permutations. Four PF and five Tanni animals remain the biological sample size.",
        "- Other grid masks, frame/support rules, diffusion contrasts and split 0 are descriptive sensitivities, not ways to replace a failed primary result.",
        "- The parent prediction grid contains some edge-bin centres outside arena bounds. Arena clipping changes only the classifier in the sensitivity. Neither support convention is asserted to be biological truth.",
        "- These endpoints do not prove latent replay truth, accurate physical kinematics, uniform speed or a new mechanism. Whole-cohort replication failures remain unchanged.",
        "",
        "![Independent prediction](training_continuity_prediction.png)",
    ]
    (out / "training_continuity_prediction_report.md").write_text("\n".join(text) + "\n")
    meta = build_script_provenance(
        input_paths={"run_manifest": manifest_path, "classification_audit": args.audit, "reporter": Path(__file__), "library": ROOT / "src/hipporeplayimm/training_continuity.py"},
        cwd=ROOT,
    )
    meta.update(
        status="complete",
        biological_replay_truth_established=False,
        n_events=9225,
        independent_classification_audit_passed=True,
        report_aggregation_audit_required=True,
        output_sha256={p.name: file_sha256(p) for p in out.iterdir()},
    )
    (out / "training_continuity_report_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
