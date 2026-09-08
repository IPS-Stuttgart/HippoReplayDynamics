#!/usr/bin/env python3
"""Non-rescoring report of the audited cross-context diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY = ["mix_iid_minus_current_iid", "mix_iid_minus_mix_global", "mix_iid_minus_event_global"]
LABELS = ["Other arenas allowed\nvs current arena only", "Spatial maps\nvs context rates", "Spatial maps\nvs other-event rates"]
COLORS = ["#21818a", "#c04a45", "#718e36", "#9b65a8", "#ad832e"]


def report(root, audit_dir, out):
    root, audit_dir, out = map(Path, (root, audit_dir, out))
    manifest_file = root / "tanni_multicontext_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    audit = json.loads((audit_dir / "tanni_multicontext_audit.json").read_text())
    checksum = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    if manifest["status"] != "complete" or audit["status"] != "passed" or audit["source_manifest_sha256"] != checksum:
        raise ValueError("matching completed scoring and passed audit required")
    for filename, expected in manifest["output_sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed evidence output: {filename}")
    out.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(root / "tanni_multicontext_summary.csv").set_index("metric")
    animals = pd.read_csv(root / "tanni_multicontext_animals.csv")
    animals = animals[animals.phase.eq("MUA")].sort_values("animal")
    run = pd.read_csv(root / "tanni_multicontext_RUN_validation.csv").sort_values("animal")
    scores = pd.read_csv(root / "tanni_multicontext_split_contrasts.csv")
    decisions = manifest["decisions"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].bar(run.animal, run.iid_context_correct, color=COLORS, width=0.65)
    axes[0].axhline(0.25, color="0.45", linestyle=":", label="Four-context chance")
    axes[0].axhline(0.5, color="0.3", linestyle="--", label="Declared RUN threshold")
    axes[0].set(ylim=(0, 1), ylabel="Held-out RUN context accuracy", title="A. Maps identify the arena during running")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].tick_params(axis="x", labelsize=9)
    for i, row in enumerate(animals.itertuples(index=False)):
        y = [getattr(row, metric) for metric in PRIMARY]
        axes[1].scatter(np.arange(3) + (i - 2) * 0.055, y, color=COLORS[i], s=32, label=row.animal, zorder=3)
    m = summary.loc[PRIMARY]
    axes[1].errorbar(np.arange(3), m["mean"], yerr=np.array([m["mean"] - m.ci_low, m.ci_high - m["mean"]]), fmt="ks", ms=5, capsize=5, label="Equal-animal mean / 95% CI", zorder=4)
    axes[1].axhline(0, color="0.5", linewidth=1)
    axes[1].set(xticks=np.arange(3), xticklabels=LABELS, ylabel="Held-out predictive advantage (nats/event)", title="B. MUA prediction: nonspatial baseline still competitive")
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "tanni_multicontext_diagnostic.png", dpi=180)
    plt.close(fig)

    run_scores = scores[scores.phase.eq("RUN")].copy()
    run_scores["predicted_context"] = np.array(list("ABCD"))[run_scores[[f"iid_p_{c}" for c in "ABCD"]].to_numpy().argmax(axis=1)]
    confusion_rows = []
    for context in "ABCD":
        group = run_scores[run_scores.context.eq(context)]
        for predicted in "ABCD":
            binary = group.assign(match=group.predicted_context.eq(predicted))
            fraction = binary.groupby(["animal", "session"]).match.mean().groupby("animal").mean().mean()
            confusion_rows.append({"true_context": context, "predicted_context": predicted, "equal_animal_fraction": fraction})
    pd.DataFrame(confusion_rows).to_csv(out / "tanni_RUN_context_confusion.csv", index=False)
    lines = [
        "# Tanni cross-context held-out prediction",
        "",
        "Non-rescoring report. Exploratory diagnostic, not a remote-replay prevalence estimate or a new biological mechanism.",
        "",
        "## Frozen experiment",
        "",
        "- Five animals, 25 recordings; four familiar arenas, with arena A visited twice.",
        "- First-half RUN maps and training-only unit QC; five common 70/30 neural splits per animal.",
        f"- {manifest['RUN_windows']:,} second-half RUN windows and {manifest['events']:,} previously frozen high-MUA candidates.",
        "- No IMM, momentum or temporal transition prior. Context weights and within-context position posteriors use training neurons only.",
        "- Held-out cell-identity counts scored using proper normalized multinomial probabilities; held-out spikes cannot update context or location.",
        "- Current-arena comparator includes both A templates. Each physical context has equal prior and summary weight.",
        "",
        "## Results",
        "",
        f"RUN validation passed: **{decisions['RUN_context_validation_passed']}**. All primary MUA contrasts passed: **{decisions['all_candidate_predictive_contrasts_passed']}**.",
        "",
        "| Comparison | Mean nats/event | 95% CI | Positive animals |",
        "|---|---:|---:|---:|",
    ]
    for metric in PRIMARY:
        row = summary.loc[metric]
        lines.append(f"| {metric} | {row['mean']:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {int(row.positive_animals)}/5 |")
    lines += ["", "| Animal | RUN context accuracy | RUN spatial minus current-global prediction |", "|---|---:|---:|"]
    for row in run.itertuples(index=False):
        lines.append(f"| {row.animal} | {row.iid_context_correct:.1%} | {row.current_iid_minus_current_global:+.3f} |")
    lines += [
        "",
        "Scores are paired within splits, then median across splits per event; average events within recording, A visits within physical context, contexts within animal, then animals equally. The 95% intervals resample animals and events inside the fixed recording/context structure. Five animals, not thousands of events, bound biological generality. Per-spike outputs are sensitivity, not a replacement for the raw primary outcome.",
        "",
        "## Interpretation",
        "",
    ]
    if not decisions["all_candidate_predictive_contrasts_passed"]:
        lines += [
            "Allowing other familiar arenas does not satisfy the declared all-candidate predictive rule. A modest improvement over the current map is insufficient if nonspatial event activity remains competitive. Do not label high remote posterior mass as verified remote replay or select a favorable animal/event subset to rescue the result.",
            "",
            "This diagnostic does not establish that remote reactivation is absent. RUN arena identification is useful but does not validate every MUA context label. Context-specific firing, state-dependent encoding changes, or an inadequate spatial observation model remain alternatives. No time-order test was performed.",
        ]
    else:
        lines += [
            "The predeclared diagnostic rule passes, which warrants independent temporal and context-null validation. It does not establish remote replay, its prevalence, or a novel mechanism."
        ]
    lines += [
        "",
        "## Audit and provenance",
        "",
        f"Scoring commit: `{manifest['code_commit']}`; clean working tree at launch: `{not manifest['git_dirty']}`.",
        f"Manifest SHA256: `{checksum}`.",
        f"Independent audit recounted {audit['raw_cached_timestamp_windows_recounted']:,} windows / {audit['raw_cached_timestamp_bins_recounted']:,} bins from cached native timestamps and checked {audit['independent_prediction_rows']:,} predictive rows with independently written equations (maximum difference {audit['maximum_prediction_error']:.3g}).",
        "All event/split contrasts, group means, cell partitions, map-bank construction, calibration exclusion and source/output hashes are checked. Original NWB reopening and map fitting reuse the parent audit. Independent animal-only bootstrap is a separate sensitivity; it is not exact reconstruction of the hierarchical interval.",
        "",
        "## Decision",
        "",
        "Do not promote this experiment to a biological confirmation, broaden to a favorable subset, or treat the high-importance-paper goal as achieved. Retain it as a completed diagnostic with its failed comparisons visible.",
    ]
    (out / "tanni_multicontext_report.md").write_text("\n".join(lines) + "\n")
    (out / "report_manifest.json").write_text(
        json.dumps({"source_manifest_sha256": checksum, "rescored": False, "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report(args.input_dir, args.audit_dir, args.output_dir)
