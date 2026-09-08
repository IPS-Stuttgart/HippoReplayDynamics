#!/usr/bin/env python3
"""Non-rescoring report of the frozen burst-time recruitment experiment."""

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

DATASETS = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
COLORS = {"pfeiffer_foster": "#2672a4", "tanni2022": "#b94565"}
LABELS = {
    "phase_minus_global": "Burst time minus global composition",
    "phase_minus_phase_averaged": "Burst time minus its own time average",
    "phase_order_advantage": "Burst time: original minus shuffled",
    "spatial_imm_minus_phase": "Spatial IMM minus burst time",
    "learned_hmm_minus_phase": "Learned HMM minus burst time",
    "spatial_iid_minus_phase": "Spatial IID minus burst time",
    "learned_iid_minus_phase": "Learned IID minus burst time",
    "learned_order_advantage": "Learned HMM: original minus shuffled",
}
MAIN = tuple(LABELS)[:6]


def validate_summary(summary):
    keys = ["dataset", "n_knots", "contrast", "metric"]
    if summary.empty or summary.duplicated(keys).any():
        raise ValueError("nonempty unique summary required")
    expected = {(d, n, c, m) for d in DATASETS for c in LABELS for n in ((5,) if c == "phase_order_advantage" else (3, 5, 10)) for m in ("delta", "delta_per_spike")}
    if set(summary[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("complete frozen contrast and sensitivity grid required")
    values = ["mean", "ci_low", "ci_high", "positive_animals", "animals"]
    if not np.isfinite(summary[values]).all().all():
        raise ValueError("finite summary required")
    if not summary.animals.eq(summary.dataset.map({"pfeiffer_foster": 4, "tanni2022": 5})).all():
        raise ValueError("all nine animals required")
    if (summary.ci_low > summary.ci_high).any() or (summary.positive_animals < 0).any() or (summary.positive_animals > summary.animals).any():
        raise ValueError("invalid interval or animal support")


def classifications(summary):
    validate_summary(summary)
    main = summary[summary.n_knots.eq(5) & summary.metric.eq("delta_per_spike")]
    rows = []
    for dataset, group in main.groupby("dataset"):
        indexed = group.set_index("contrast")
        positive = {c: bool(indexed.loc[c, "ci_low"] > 0 and indexed.loc[c, "positive_animals"] == indexed.loc[c, "animals"]) for c in LABELS}
        recruitment = all(positive[c] for c in ("phase_minus_global", "phase_minus_phase_averaged", "phase_order_advantage"))
        status = (
            "exploratory_recruitment_lead"
            if recruitment
            else "order_sensitive_without_complete_predictive_support"
            if positive["phase_order_advantage"]
            else "recruitment_not_established"
        )
        rows.append(
            {
                "dataset": dataset,
                "status": status,
                "recruitment_lead": recruitment,
                "spatial_imm_beats_this_comparator": positive["spatial_imm_minus_phase"],
                "unique_spatial_mechanism_established": False,
                "independent_confirmation": False,
            }
        )
    return pd.DataFrame(rows)


def load_verified(source, audit_path):
    mp = source / "burst_phase_manifest.json"
    m = json.loads(mp.read_text())
    audit = json.loads(audit_path.read_text())
    if m.get("status") != "complete" or not m.get("gates", {}).get("all_events_present"):
        raise ValueError("technically complete source required")
    if audit.get("status") != "pass" or audit.get("input_file_sha256", {}).get("source_manifest") != file_sha256(mp):
        raise ValueError("matching passing independent audit required")
    if not audit.get("all_summaries_reconstructed"):
        raise ValueError("independent summary reconstruction required")
    for name, digest in m["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed source output: " + name)
    summary = pd.read_csv(source / "burst_phase_summary.csv.gz")
    validate_summary(summary)
    status = classifications(summary)
    if bool(status.recruitment_lead.all()) != m["gates"]["recruitment_lead"]:
        raise ValueError("source recruitment gate disagrees")
    return m, audit, summary, status


def contrast_figure(summary, animals, out, metric):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    unit = "nats / held-out spike" if metric == "delta_per_spike" else "nats / event"
    for ax, contrast in zip(axes.flat, MAIN, strict=True):
        ax.axhline(0, color="0.45", lw=1)
        for x, dataset in enumerate(DATASETS):
            a = animals[animals.n_knots.eq(5) & animals.dataset.eq(dataset) & animals.contrast.eq(contrast)].sort_values("animal")
            ax.scatter(x + np.linspace(-0.15, 0.05, len(a)), a[metric], color=COLORS[dataset], s=32, alpha=0.85, zorder=4)
            s = summary[summary.n_knots.eq(5) & summary.dataset.eq(dataset) & summary.contrast.eq(contrast) & summary.metric.eq(metric)].iloc[0]
            ax.plot([x + 0.18, x + 0.18], [s.ci_low, s.ci_high], color="black", lw=1.8)
            ax.plot(x + 0.18, s["mean"], "s", color="black", ms=5, zorder=5)
            ax.text(x, 0.98, f"{int(s.positive_animals)}/{int(s.animals)} positive", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.set_title(LABELS[contrast], fontsize=11, pad=18)
        ax.set_xticks([0, 1], DATASETS.values())
        ax.set_xlim(-0.4, 1.45)
        ax.margins(y=0.22)
        ax.set_ylabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does a repeatable burst-time pattern explain prediction?", fontsize=15)
    fig.text(
        0.5,
        0.035,
        "Five fixed knots. Colored dots: animals. Black square/line: equal-animal mean and 95% animal-bootstrap interval.\nMedian neural split within event, then equal events, sessions, and animals. Not a causal decomposition.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.89, bottom=0.14, hspace=0.55, wspace=0.34)
    fig.savefig(out / f"burst_phase_contrasts_{metric}.png", dpi=180)
    plt.close(fig)


def sensitivity_figure(summary, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, contrast in zip(axes, ("phase_minus_global", "phase_minus_phase_averaged", "spatial_imm_minus_phase"), strict=True):
        ax.axhline(0, color="0.5", lw=1)
        for i, dataset in enumerate(DATASETS):
            s = summary[summary.dataset.eq(dataset) & summary.contrast.eq(contrast) & summary.metric.eq("delta_per_spike")].sort_values("n_knots")
            x = np.arange(3) + (i - 0.5) * 0.08
            ax.plot(x, s["mean"], "o-", color=COLORS[dataset], label=DATASETS[dataset])
            ax.vlines(x, s.ci_low, s.ci_high, colors=COLORS[dataset], lw=1.5)
        ax.set_xticks(range(3), ["3", "5 (primary)", "10"])
        ax.set_xlabel("Frozen number of time-profile knots")
        ax.set_ylabel("nats / held-out spike")
        ax.set_title(LABELS[contrast], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Prespecified sensitivity: no setting selected after scoring", fontsize=14)
    fig.text(0.5, 0.035, "Lines connect equal-animal means; vertical bars are 95% animal-bootstrap intervals. These are not independent replications.", ha="center", fontsize=10)
    fig.subplots_adjust(top=0.83, bottom=0.2, wspace=0.4)
    fig.savefig(out / "burst_phase_knot_sensitivity.png", dpi=180)
    plt.close(fig)


def report(source, audit_path, out):
    m, audit, summary, status = load_verified(source, audit_path)
    animals = pd.read_csv(source / "burst_phase_animals.csv.gz")
    events = pd.read_csv(source / "burst_phase_events.csv.gz")
    sessions = pd.read_csv(source / "burst_phase_sessions.csv.gz")
    support = events[events.n_knots.eq(5) & events.contrast.eq("phase_minus_global")]
    coverage = support.groupby("dataset").agg(events=("event_id", "size"), zero_heldout_events=("valid_neural_splits", lambda x: int(x.eq(0).sum()))).reset_index()
    coverage["sessions"] = coverage.dataset.map(sessions.groupby("dataset").session.nunique())
    coverage["animals"] = coverage.dataset.map(animals.groupby("dataset").animal.nunique())
    if coverage.set_index("dataset").events.to_dict() != {"pfeiffer_foster": 4001, "tanni2022": 5224}:
        raise ValueError("frozen event denominator changed")
    out.mkdir(parents=True, exist_ok=False)
    primary = summary[summary.n_knots.eq(5) & summary.metric.eq("delta_per_spike")].copy()
    for name, frame in {"primary": primary, "all_sensitivities": summary, "animals": animals, "coverage": coverage, "decisions": status}.items():
        frame.to_csv(out / f"burst_phase_report_{name}.csv", index=False)
    contrast_figure(summary, animals, out, "delta_per_spike")
    contrast_figure(summary, animals, out, "delta")
    sensitivity_figure(summary, out)
    lines = [
        "# Burst-Time Recruitment: Independent Predictive Test",
        "",
        "Non-rescoring report. Exploratory reused data, not independent confirmation.",
        "",
        f"Frozen scoring commit: `{m['code_commit']}`. {m['events']:,} events, {m['original_rows']:,} original split/knot rows, 33 sessions, nine animals. Five chronological folds; five fixed neural partitions; 20 primary whole-bin shuffles.",
        "",
        "| Dataset | Classification |",
        "|---|---|",
    ]
    lines += [f"| {DATASETS[r.dataset]} | {r.status} |" for r in status.itertuples()]
    lines += [
        "",
        "## Primary Five-Knot Results",
        "",
        "Paired score differences per held-out spike; event median across neural splits, then equal-session and equal-animal aggregation. Intervals enumerate animal-bootstrap resamples (four PF, five Tanni), conditional on these fitted models and reused data. Not corrected for the wider hypothesis search.",
        "",
        "| Contrast | Dataset | Mean | 95% animal-bootstrap interval | Positive animals |",
        "|---|---|---:|---:|---:|",
    ]
    for contrast, label in LABELS.items():
        for r in primary[primary.contrast.eq(contrast)].itertuples():
            lines.append(f"| {label} | {DATASETS[r.dataset]} | {r.mean:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {r.positive_animals}/{r.animals} |")
    lines += [
        "",
        "## What This Tests",
        "",
        "The burst-time predictor learns which cells tend to fire early or late from other calibration bursts. At test time it uses only the bin's relative position within the frozen event window. This is elapsed burst time, not theta/ripple oscillation phase. No target spikes, positions, decoded path or spatial rate map fit the predictor.",
        "",
        "Its own time-averaged comparator uses the same learned profiles, averaged across the target's known bin times before renormalizing to held-out cells. This controls for a different overall cell composition or shrinkage. The global comparator instead pools calibration spike counts without time dependence.",
        "",
        "Original-minus-shuffled asks whether alignment matters; burst-time-minus-time-average asks whether this fitted temporal pattern actually improves prediction. A positive shuffle difference cannot replace a successful predictive comparison. A failed burst-time model does not prove biological absence of stereotyped recruitment.",
        "",
        "Spatial IMM, spatial IID and learned HMM also use target training-cell observations to predict held-out neurons; the burst-time predictor does not. Their advantage rejects this particular time-only explanation but cannot, by itself, establish uniquely spatial information or unique IMM dynamics. The coordinate-free HMM can encode spatial information implicitly. These non-nested contrasts are not a fraction of replay explained.",
        "",
        "Scores are normalized count-conditional multinomial predictions, summed over time-bin marginals. They test held-out cell identity composition, not total firing rate or full joint sequence probability. Event boundaries were detected with all cells; inference is conditional on the frozen all-cell candidate definition and duration, not an end-to-end held-out detector.",
        "",
        f"Events with no held-out counts in any split: {int(coverage.zero_heldout_events.sum())}. Raw scores retain zero; normalized event ratios are missing, never assigned zero. Final partial count bins retain their physical timestamps.",
        "",
        "## Verification And Limits",
        "",
        f"Independent audit: {audit['independent_predictions']:,} original/first-shuffle predictions, maximum difference {audit['max_error']:.3g} nats. It reconstructs all calibration profiles, original scores, first permutations, all shuffle coverage/means, reused comparator values, folds, support counts, event medians, session/animal summaries and bootstrap intervals. All hashed inputs and outputs are checked. The other 19 shuffled predictions are coverage/mean/hash checked, not independently recalculated. Native files and source HMM/spatial fitting were not repeated.",
        "",
        "Three and ten knots and raw nats/event remain sensitivities. No animal, event, knot count, normalization or threshold was selected after observing this result. A passing exploratory contrast does not establish a high-importance discovery, causality or a replay classification.",
        "",
        "Close precedents include [stereotyped brain-wide cascades](https://pmc.ncbi.nlm.nih.gov/articles/PMC10983782/) and [independent replay-detector evaluation](https://elifesciences.org/articles/85635). Generic sequence structure and the need for better null controls are not new claims.",
        "",
    ]
    (out / "burst_phase_report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(
        input_paths={"source_manifest": source / "burst_phase_manifest.json", "independent_audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT
    )
    provenance.update(status="complete", rescoring=False, output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "burst_phase_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(status.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    report(a.source_dir, a.audit, a.output_dir)


if __name__ == "__main__":
    main()
