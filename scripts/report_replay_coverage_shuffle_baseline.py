#!/usr/bin/env python3
"""Non-rescoring report of the transferred PF-style shuffle benchmark."""

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


def order_contrast(animals, seed=20260906, bootstraps=5000):
    """Describe paired order excess, not a corrected biological replay rate."""
    keys = ["dataset", "detector", "animal"]
    columns = keys + ["accepted_fraction_full", "accepted_fraction_half"]
    primary = animals[animals.bin_filter.eq("edge_only") & animals.min_frames.eq(10) & animals.alpha.eq(.02)]
    original = primary[primary.observation.eq("original_order")][columns]
    randomized = primary[primary.observation.eq("order_randomized")][columns]
    paired = original.merge(randomized, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_original", "_randomized"))
    if not paired._merge.eq("both").all():
        raise ValueError("order contrast requires both observations for each animal/cohort")
    for population in ["full", "half"]:
        paired[f"order_excess_{population}"] = paired[f"accepted_fraction_{population}_original"] - paired[f"accepted_fraction_{population}_randomized"]
    paired["half_minus_full_order_excess"] = paired.order_excess_half - paired.order_excess_full
    rng = np.random.default_rng(seed)
    rows = []
    for identity, group in paired.groupby(["dataset", "detector"], sort=True):
        for metric in ["order_excess_full", "order_excess_half", "half_minus_full_order_excess"]:
            values = group[metric].dropna().to_numpy()
            ci = np.quantile(rng.choice(values, (bootstraps, len(values)), replace=True).mean(axis=1), [.025, .975]) if len(values) else [np.nan, np.nan]
            rows.append({"dataset": identity[0], "detector": identity[1], "metric": metric,
                "animals": len(values), "equal_animal_mean_pp": 100 * values.mean() if len(values) else np.nan,
                "ci_low_pp": 100 * ci[0], "ci_high_pp": 100 * ci[1],
                "animals_positive": int((values > 0).sum()), "animals_negative": int((values < 0).sum())})
    return pd.DataFrame(rows)


def report(input_dir, output_dir):
    manifest = input_dir / "coverage_shuffle_baseline_manifest.json"
    audit_path = input_dir / "coverage_shuffle_baseline_reconstruction_audit.json"
    meta, audit = json.loads(manifest.read_text()), json.loads(audit_path.read_text())
    if meta["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["scoring_manifest"] != file_sha256(manifest):
        raise ValueError("successful linked scoring and reconstruction required")
    paths = {k: input_dir / f"coverage_shuffle_baseline_{k}.csv" for k in ["summary", "session_summary", "animal_summary", "gate_summary"]}
    for path in paths.values():
        if file_sha256(path) != meta["output_sha256"][path.name]:
            raise ValueError("scoring output hash changed")
    summary, sessions, animals = [pd.read_csv(paths[k]) for k in ["summary", "session_summary", "animal_summary"]]
    primary = summary[summary.bin_filter.eq("edge_only") & summary.min_frames.eq(10) & summary.alpha.eq(.02)]
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for keys, group in primary.groupby(["dataset", "detector", "observation"], sort=True):
        identity = dict(zip(["dataset", "detector", "observation"], keys, strict=True))
        g = group.set_index("metric")
        available = sessions[sessions.dataset.eq(keys[0]) & sessions.detector.eq(keys[1]) & sessions.observation.eq(keys[2])
            & sessions.bin_filter.eq("edge_only") & sessions.min_frames.eq(10) & sessions.alpha.eq(.02)]
        rows.append({**identity, "eligible_events": int(available.eligible_events.sum()), "sessions_with_events": int(available.eligible_events.gt(0).sum()),
            "animals_with_events": int(g.loc["accepted_fraction_full", "animals_measurable"]),
            "geometric_full_percent": 100 * g.loc["geometric_fraction_full", "equal_animal_mean"],
            "geometric_half_percent": 100 * g.loc["geometric_fraction_half", "equal_animal_mean"],
            "significant_full_percent": 100 * g.loc["accepted_fraction_full", "equal_animal_mean"],
            "significant_half_percent": 100 * g.loc["accepted_fraction_half", "equal_animal_mean"],
            "half_minus_full_pp": 100 * g.loc["accepted_fraction_delta", "equal_animal_mean"],
            "delta_ci_low_pp": 100 * g.loc["accepted_fraction_delta", "ci_low"],
            "delta_ci_high_pp": 100 * g.loc["accepted_fraction_delta", "ci_high"],
            "animals_negative": int(g.loc["accepted_fraction_delta", "animals_negative"])})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "coverage_shuffle_baseline_primary_table.csv", index=False)
    contrasts = order_contrast(animals, meta["seed"] + 1)
    contrasts.to_csv(output_dir / "coverage_shuffle_baseline_order_contrast.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="col", layout="constrained")
    names = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
    for col, dataset in enumerate(["pfeiffer_foster", "tanni2022"]):
        for r, observation in enumerate(["original_order", "order_randomized"]):
            ax = axes[r, col]
            data = primary[primary.dataset.eq(dataset) & primary.observation.eq(observation)]
            labels = []
            for d, detector in enumerate(sorted(data.detector.unique())):
                g = data[data.detector.eq(detector)].set_index("metric")
                for s, suffix in enumerate(["full", "half"]):
                    x = 3 * d + s
                    geo, sig = g.loc[f"geometric_fraction_{suffix}"], g.loc[f"accepted_fraction_{suffix}"]
                    color = "#20756e" if suffix == "full" else "#b54563"
                    ax.bar(x, 100 * geo.equal_animal_mean, width=.74, facecolor="none", edgecolor=color, linestyle="--", linewidth=1.5)
                    ax.bar(x, 100 * sig.equal_animal_mean, width=.50, color=color, alpha=.7)
                    ax.errorbar(x, 100 * sig.equal_animal_mean,
                        yerr=np.array([[100 * (sig.equal_animal_mean - sig.ci_low)], [100 * (sig.ci_high - sig.equal_animal_mean)]]), fmt="none", ecolor="black", capsize=3)
                    dots = animals[animals.dataset.eq(dataset) & animals.detector.eq(detector) & animals.observation.eq(observation)
                        & animals.bin_filter.eq("edge_only") & animals.min_frames.eq(10) & animals.alpha.eq(.02)][f"accepted_fraction_{suffix}"].dropna()
                    ax.scatter(x + np.linspace(-.18, .18, len(dots)), dots * 100, color="black", s=13, zorder=4)
                    label = "MUA" if detector == "source_high_mua" else ("Native ripple" if dataset == "pfeiffer_foster" else "LFP ripple")
                    labels.append((x, f"{label}\n{suffix.capitalize()} cells"))
            ax.set_xticks([x for x, _ in labels], [label for _, label in labels], fontsize=9)
            ax.set_title(f"{names[dataset]}: {'original' if r == 0 else 'order-randomized'} observations")
            ax.set_ylabel("Accepted candidates (%)")
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"PF-style continuity plus two map-shuffle tests (K={meta['shuffles_per_family']:,} each)\nOutline: geometric only; solid: both p < 0.02; dots: animals; bars: animal-bootstrap 95% CI", fontsize=12)
    fig.savefig(output_dir / "coverage_shuffle_baseline_primary.png", dpi=180)
    plt.close(fig)
    lines = ["# Transferred PF-Style Shuffle Baseline", "", f"Scope: **{meta['benchmark_scope']}**; {meta['sessions']} sessions; {meta['eligible_events']:,} eligible core windows; K={meta['shuffles_per_family']:,} per family.", "",
        "Independent flat-prior Poisson MAP, common 8 cm maps, 20 ms / 5 ms stride, edge trimming only, earliest longest <20 cm jump sequence, >=10 frames and >=40 cm displacement.",
        "This is not an exact reproduction of the authors' 2 cm encoding or their candidate cohorts. Circular field translations are an explicitly declared implementation choice.", "",
        "|Dataset|Detector|Observation|N|Geometric full/half %|Significant full/half %|Half-minus-full pp [95% CI]|Negative animals|",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        lines.append(f"|{row.dataset}|{row.detector}|{row.observation}|{row.eligible_events}|{row.geometric_full_percent:.2f}/{row.geometric_half_percent:.2f}|{row.significant_full_percent:.2f}/{row.significant_half_percent:.2f}|{row.half_minus_full_pp:.2f} [{row.delta_ci_low_pp:.2f}, {row.delta_ci_high_pp:.2f}]|{row.animals_negative}/{row.animals_with_events}|")
    lines += ["", "Rates average recording subsets within sessions, sessions within animals, then animals equally. Candidate counts are not independent animal replication. MUA/ripple windows may overlap; do not sum their denominators as distinct biological events.", "",
        "Order-randomized acceptance is a control readout, not an empirical false-positive rate for real replay. It preserves fine-bin population snapshots, not every local overlapping-window spike total. No speed-uniformity claim is tested.", "",
        "## Descriptive Paired Order Contrast", "",
        "Added after inspection of the frozen benchmark; no scoring or selection changes. Original-minus-randomized acceptance and its half-minus-full change are computed within animals. These are not bias-corrected replay prevalence or a new significance gate. There is only one observation-order surrogate per event, so the intervals do not integrate over repeated order surrogates.", "",
        "|Dataset|Detector|Metric|Mean pp [animal-bootstrap 95% CI]|Animals|",
        "|---|---|---|---:|---:|"]
    for row in contrasts.itertuples(index=False):
        lines.append(f"|{row.dataset}|{row.detector}|{row.metric}|{row.equal_animal_mean_pp:.2f} [{row.ci_low_pp:.2f}, {row.ci_high_pp:.2f}]|{row.animals}|")
    lines += ["",
        "The support-filter, 11-frame, and alpha 0.01/0.05 sensitivities remain in the scoring tables; no setting was chosen to optimize these outcomes.", "",
        f"Scoring commit: `{meta['code_commit']}`. Scoring manifest SHA256: `{file_sha256(manifest)}`."]
    (output_dir / "coverage_shuffle_baseline_report.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={**paths, "manifest": manifest, "audit": audit_path, "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir()})
    (output_dir / "coverage_shuffle_baseline_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    report(opts.input_dir, opts.output_dir)
