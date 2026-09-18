#!/usr/bin/env python3
"""Non-rescoring report of observation coverage and retained forecast information."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hipporeplayimm.coverage_dose_forecast import retention_gate
from scripts._provenance import build_script_provenance, file_sha256

DATASETS = {"pfeiffer_foster": ("Pfeiffer-Foster", 4), "tanni2022": ("Tanni", 5)}
PRIMARY = "dynamic_minus_matched_own"


def decisions(summary):
    records = []
    for dataset, (_, n) in DATASETS.items():
        change = summary[summary.dataset.eq(dataset) & summary.arm.eq("fixed_codebook") & summary.fraction.eq(0.5) & summary.group.eq("all") & summary.metric.eq("pass_change")]
        classification = bool(len(change) == 1 and change.animals.eq(n).all() and change.ci_high.lt(0).all())
        for arm in ["fixed_codebook", "restricted_calibration"]:
            for group in ["lost", "lost_supported"]:
                supported = retention_gate(summary, dataset, arm, group, n)
                records.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "group": group,
                        "classification_declines_at_half": classification,
                        "retained_prediction_at_half_supported": supported,
                        "joint_statement_supported": classification and supported,
                        "scope": "group_level_geometric_screen_not_ground_truth_replay",
                    }
                )
    return pd.DataFrame(records)


def figures(output, summary, animals):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), layout="constrained")
    colors = {"fixed_codebook": "#087d71", "restricted_calibration": "#b04b65"}
    for row, (dataset, (name, _)) in enumerate(DATASETS.items()):
        ax = axes[row, 0]
        a = animals[animals.dataset.eq(dataset) & animals.arm.eq("fixed_codebook") & animals.group.eq("all")]
        for _, g in a.groupby("animal"):
            g = g.sort_values("fraction")
            ax.plot(g.fraction * 100, g.acceptance * 100, "o-", lw=1, alpha=0.5, color="#59666b")
        s = summary[summary.dataset.eq(dataset) & summary.arm.eq("fixed_codebook") & summary.group.eq("all") & summary.metric.eq("acceptance")].sort_values("fraction")
        ax.plot(s.fraction * 100, s["mean"] * 100, "D-", color="black", label="Equal-animal mean")
        ax.set(title=name + ": geometric acceptance", ylabel="Detected candidates passing (%)", xlabel="Inference neurons retained (%)")
        ax.legend(frameon=False, fontsize=8)
        for column, metric, title in [(1, PRIMARY, "Prediction after geometric failure"), (2, "paired_change_" + PRIMARY, "Paired change from full inference")]:
            ax = axes[row, column]
            for arm, color in colors.items():
                g = summary[summary.dataset.eq(dataset) & summary.arm.eq(arm) & summary.group.eq("lost_supported") & summary.metric.eq(metric)].sort_values("fraction")
                if g.empty:
                    continue
                label = "Frozen model" if arm == "fixed_codebook" else "Restricted-calibration refit"
                ax.plot(g.fraction * 100, g["mean"], "o-", color=color, label=label)
                ax.fill_between(g.fraction * 100, g.ci_low, g.ci_high, alpha=0.15, color=color)
            ax.axhline(0, linestyle="--", color="#777777", linewidth=0.8)
            ax.set(title=title, xlabel="Inference neurons retained (%)", ylabel="Matched-null advantage (nats/spike)" if column == 1 else "Reduced minus full (nats/spike)")
            ax.legend(frameon=False, fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([25, 50, 75, 100])
            ax.invert_xaxis()
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Fewer observed neurons: classification and independent future prediction", fontsize=15)
    fig.savefig(output / "coverage_dose_forecast.png", dpi=180)
    plt.close(fig)


def run(root, audit, output):
    mp = root / "coverage_dose_manifest.json"
    m, v = json.loads(mp.read_text()), json.loads(audit.read_text())
    if m["status"] != "complete" or v["status"] != "pass" or len(m["completed"]) != 33:
        raise ValueError("complete independently audited cohort required")
    if file_sha256(mp) != v["input_file_sha256"]["run_manifest"]:
        raise ValueError("audit references another run")
    for name, sha in m["output_sha256"].items():
        if file_sha256(root / name) != sha:
            raise ValueError("changed output: " + name)
    summary = pd.read_csv(root / "coverage_summary.csv")
    animals = pd.read_csv(root / "coverage_animals.csv")
    events = pd.read_csv(root / "coverage_events.csv.gz")
    status = pd.read_csv(root / "session_status.csv")
    decision = decisions(summary)
    half = summary[summary.fraction.eq(0.5) & summary.group.isin(["lost", "lost_supported"]) & summary.metric.isin([PRIMARY, "paired_change_" + PRIMARY])]
    counts = events.groupby(["dataset", "arm", "fraction", "group"], as_index=False).agg(
        events=("event_id", "size"), informative_events=(PRIMARY, "count"), animals=("animal", "nunique"), sessions=("session", "nunique")
    )
    reasons = pd.concat([pd.read_csv(root / x["tag"] / "failure_reasons.csv") for x in m["completed"]], ignore_index=True)
    technical = [
        ("complete_33_sessions", len(status) == 33 and status.status.eq("complete").all()),
        ("all_2475_restricted_refits_complete", status.refits.eq(75).all()),
        ("all_prior_full_half_scores_reproduced", status.maximum_legacy_error.lt(1e-8).all()),
        ("independent_audit_pass", v["status"] == "pass"),
        ("all_subsets_and_aggregates_verified", v["all_subsets_checked"] and v["all_aggregates_reconstructed"]),
    ]
    technical.append(("overall_technical", all(value for _, value in technical)))
    output.mkdir(parents=True, exist_ok=False)
    tables = {
        "primary_half_summary": half,
        "decision_summary": decision,
        "denominators": counts,
        "failure_reasons_by_session": reasons,
        "technical_gates": pd.DataFrame(technical, columns=["gate", "passed"]),
        "dose_summary": summary,
        "by_animal": animals,
        "session_status": status,
        "by_session": pd.read_csv(root / "coverage_sessions.csv"),
        "leave_one_animal_out": pd.read_csv(root / "coverage_leave_one_animal_out.csv"),
    }
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    figures(output, summary, animals)
    lines = [
        "# Coverage-dependent classification and retained predictive information",
        "",
        "## Frozen design",
        "",
        "The same independently detected candidates, held-out neurons and target windows are used at",
        "100%, 75%, 50% and 25% of the full INFERENCE population. These are not fractions of all recorded cells.",
        "Five inference/evaluation partitions, ten nested cell orderings; full inference is scored once.",
        "Future cell identities are predicted 40 ms ahead across a 20-ms gap, conditional on target spike count.",
        "No target spikes enter test-event detection, classification or inference.",
        "",
        "The fixed-codebook arm isolates test-event information loss. Restricted calibration additionally",
        "refits models without omitted cells in calibration, for the prespecified first ordering.",
        "Both models use evaluation-cell activity only in OTHER calibration events to learn emissions.",
        "",
        "## Prespecified 50% result",
        "",
        "| Dataset | Arm | Group | Advantage (nats/spike) | Descriptive animal-bootstrap interval | Positive animals |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for x in half[half.metric.eq(PRIMARY)].itertuples(index=False):
        lines.append(f"| {DATASETS[x.dataset][0]} | {x.arm} | {x.group} | {x.mean:+.5f} | [{x.ci_low:+.5f}, {x.ci_high:+.5f}] | {x.positive_animals}/{x.animals} |")
    lines += [
        "",
        "Lost means full geometry passes but reduced geometry fails. Lost-supported additionally requires",
        "at least ten supported reduced-population frames, excluding mere loss of enough decodable duration.",
        "Groups are split/repeat-specific; event counts are unions, not individually significant events.",
        "",
        "## Classification dose curve",
        "",
        "| Dataset | 100% | 75% | 50% | 25% |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, (name, _) in DATASETS.items():
        a = summary[summary.dataset.eq(dataset) & summary.arm.eq("fixed_codebook") & summary.group.eq("all") & summary.metric.eq("acceptance")].set_index("fraction")
        lines.append("| " + name + " | " + " | ".join(f"{a.loc[f, 'mean'] * 100:.3f}%" for f in [1, 0.75, 0.5, 0.25]) + " |")
    lines += [
        "",
        "Equal-event session means precede equal-session animal means. Classification fractions average",
        "technical repeat/partition probabilities, not majority labels. Prediction uses repeat medians then",
        "partition medians within events. Paired reduced-minus-full score changes are calculated BEFORE",
        "aggregation on identical evaluation targets. Detectable information is NOT unchanged information.",
        "",
        "## Interpretation",
        "",
    ]
    for dataset, (name, _) in DATASETS.items():
        d = decision[decision.dataset.eq(dataset)]
        supported = bool(d.joint_statement_supported.all())
        lines.append(
            f"- {name}: {'supports the joint statement in both arms and both lost groups' if supported else 'does not meet the full joint-support rule; retain mixed/nonpositive results'}."
        )
    lines += [
        "",
        "The screen is the transferred PF-style geometric stage, not both original 5000-shuffle tests.",
        "Lower acceptance is not a false-negative rate without independent replay ground truth. Positive",
        "future prediction establishes statistical population organization, not continuous spatial replay or",
        "a memory function. This is retrospective validation on already explored recordings, not independent",
        "biological confirmation. Four PF/five Tanni animals limit population inference; the intervals are",
        "conditional/descriptive. More neuron subsets do not create more independent animals. No arbitrary",
        "threshold turns an acceptance decrease into a universal 'strong effect'; report its measured size.",
        "",
        "## Verification",
        "",
        f"Producer commit: {m['code_commit']}. All {len(status)} sessions completed; {int(status.refits.sum())} restricted fits.",
        f"Independent audit reconstructed {sum(x['scores_reconstructed'] for x in v['sessions']):,} scalar predictive scores",
        f"and {sum(x['geometries_reconstructed'] for x in v['sessions']):,} sampled geometric paths/labels, all subset memberships",
        "and ALL event/animal/bootstrap aggregates. Model fitting itself was not independently repeated.",
        f"Maximum sampled score discrepancy: {max(x['maximum_score_error'] for x in v['sessions']):.3g} nats.",
        "",
        "![Coverage curves and paired predictive validation](coverage_dose_forecast.png)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    prov = build_script_provenance(input_paths={"run_manifest": mp, "audit": audit, "reporter": Path(__file__)}, cwd=ROOT)
    prov.update(non_rescoring=True, output_sha256={p.name: file_sha256(p) for p in output.iterdir()})
    (output / "report_manifest.json").write_text(json.dumps(prov, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.run_dir, a.audit, a.output_dir)
