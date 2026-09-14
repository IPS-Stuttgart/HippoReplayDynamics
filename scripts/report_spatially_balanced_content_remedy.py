#!/usr/bin/env python3
"""Non-rescoring report of external sampling-remedy validation and information limits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def report(root):
    manifest = json.loads((root/"manifest.json").read_text())
    audit = json.loads((root/"independent_audit.json").read_text())
    if not audit["passed"] or audit["producer_manifest_sha256"] != digest(root/"manifest.json"):
        raise ValueError("missing or stale reconstruction audit")
    for name in ("summary", "by_animal", "by_session", "gates"):
        if audit.get("verified_table_sha256", {}).get(name) != digest(root/f"{name}.csv"):
            raise ValueError(f"table not covered by current audit: {name}")
    frames = []
    for row in manifest["results"]:
        if row["status"] != "complete":
            continue
        path = Path(row["artifact_dir"])/"event_readouts.csv.gz"
        if digest(path) != row["readouts_sha256"]:
            raise ValueError("event readouts changed")
        frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    real = data.loc[data.source.eq("real")].copy()
    real["both_silent_fraction"] = (real.a_spikes+real.b_spikes).eq(0).astype(float)
    real["either_silent_fraction"] = (real.a_spikes.eq(0) | real.b_spikes.eq(0)).astype(float)
    real["both_halves_two_cells_three_spikes_fraction"] = (
        real.a_spikes.ge(3) & real.b_spikes.ge(3) & real.a_active.ge(2) & real.b_active.ge(2)).astype(float)
    real["total_spikes"] = real.a_spikes+real.b_spikes
    real["total_active_units"] = real.a_active+real.b_active
    columns = ["both_silent_fraction", "either_silent_fraction", "both_halves_two_cells_three_spikes_fraction",
               "total_spikes", "total_active_units", "pair_entropy"]
    groups = ["dataset", "split", "condition"]
    session = real.groupby(groups+["animal", "session"])[columns].mean()
    animal = session.groupby(groups+["animal"])[columns].mean()
    information = animal.groupby(groups)[columns].mean().reset_index()
    information.to_csv(root/"endpoint_information_summary.csv", index=False)
    animal.reset_index().to_csv(root/"endpoint_information_by_animal.csv", index=False)
    summary = pd.read_csv(root/"summary.csv")
    paired = []
    for (dataset, source, split), group in summary.groupby(["dataset", "source", "split"]):
        g = group.set_index("condition")
        item = dict(dataset=dataset, source=source, split=split)
        for metric in ("regional_tv", "endpoint_separation_cm", "pair_entropy", "pair_truth_error_cm"):
            item[f"random_{metric}"] = g.loc["random", metric]
            item[f"balanced_{metric}"] = g.loc["balanced", metric]
            item[f"balanced_minus_random_{metric}"] = g.loc["balanced", metric]-g.loc["random", metric]
            if metric in ("regional_tv", "endpoint_separation_cm"):
                item[f"relative_reduction_{metric}"] = 1-g.loc["balanced", metric]/g.loc["random", metric]
        paired.append(item)
    comparison = pd.DataFrame(paired)
    comparison.to_csv(root/"paired_sensitivity_summary.csv", index=False)
    gates = pd.read_csv(root/"gates.csv")
    if gates.passed.dtype != bool:
        raise ValueError("invalid gate booleans")
    passed = gates.loc[gates.gate.eq("statistical_validation"), "passed"].item()
    by_animal = pd.read_csv(root/"by_animal.csv", dtype={"animal": str})
    primary = by_animal.loc[by_animal.dataset.eq("blackstad_moser") & by_animal.source.eq("real") & by_animal.split.eq(0)]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = ("#425b76", "#19795e")
    for ax, metric, label in zip(axes[0], ("regional_tv", "endpoint_separation_cm"),
                                 ("Regional posterior disagreement (TV)", "Posterior-mean separation (cm)"), strict=True):
        values = primary.pivot(index="animal", columns="condition", values=metric)
        for i, (name, row) in enumerate(values.iterrows()):
            shift = (i-2)*.025
            ax.plot(np.array([0, 1])+shift, [row.random, row.balanced], color="#b8bdc1", linewidth=1)
            ax.scatter(np.array([0, 1])+shift, [row.random, row.balanced], c=colors, s=32, zorder=3)
            ax.annotate(name, (1+shift, row.balanced), xytext=(7, (i%2)*5), textcoords="offset points", fontsize=8)
        ax.set(xticks=[0, 1], xticklabels=["Random halves", "RUN-balanced halves"], ylabel=label, xlim=(-.3, 1.6))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("A. Spatial content" if metric == "regional_tv" else "B. Decoded position", loc="left", fontsize=11)
    ax = axes[1, 0]
    sources = ["run_test", "sim_matched", "sim_drift"]
    external = comparison.loc[comparison.dataset.eq("blackstad_moser") & comparison.split.eq(0)].set_index("source")
    changes = external.loc[sources, "balanced_minus_random_pair_truth_error_cm"]
    ax.bar(np.arange(3), changes, color=[colors[1] if x <= 0 else "#b6494d" for x in changes])
    ax.axhline(0, color="#222222", linewidth=.8)
    ax.set(xticks=np.arange(3), xticklabels=["Held-out RUN", "Matched-map\nsimulation", "Map-drift\nsimulation"],
           ylabel="Change in true-position error (cm)")
    ax.set_title("C. Accuracy check: lower is better", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1, 1]
    sensitivity = comparison.loc[comparison.dataset.eq("blackstad_moser") & comparison.source.eq("real")].sort_values("split")
    x = np.arange(len(sensitivity))
    ax.bar(x-.18, sensitivity.relative_reduction_regional_tv*100, width=.36, color=colors[0], label="Regional TV")
    ax.bar(x+.18, sensitivity.relative_reduction_endpoint_separation_cm*100, width=.36, color=colors[1], label="Mean separation")
    ax.axhline(10, color="#b6494d", linestyle="--", linewidth=1, label="Frozen 10% threshold")
    labels = ["Primary split" if k == 0 else f"Sensitivity {k}" for k in sensitivity.split]
    reductions = sensitivity[["relative_reduction_regional_tv", "relative_reduction_endpoint_separation_cm"]].to_numpy()*100
    ax.set(xticks=x, xticklabels=labels, ylabel="Disagreement reduction (%)",
           ylim=(min(0, float(np.nanmin(reductions))-1), max(12, float(np.nanmax(reductions))+2)))
    ax.set_title("D. Split sensitivity", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("RUN-balanced sampling: external validation " + ("passed" if passed else "not passed"), fontsize=13)
    figure.savefig(root/"external_sampling_validation.png", dpi=180)
    figure.savefig(root/"external_sampling_validation.pdf")
    plt.close(figure)
    primary_information = information.loc[information.dataset.eq("blackstad_moser") & information.split.eq(0) & information.condition.eq("random")].iloc[0]
    status = "PASS" if passed else "FAIL"
    completed = [r for r in manifest["results"] if r["status"] == "complete"]
    development = [r for r in completed if r["dataset"] == "pfeiffer_foster"]
    validation = [r for r in completed if r["dataset"] == "blackstad_moser"]
    lines = ["# Spatially balanced sampling: audited outcome", "", f"Complete frozen validation: **{status}**.",
        f"Technical audit: passed; {audit['reconstructed_readouts']:,} reconstructed readouts across {audit['sessions']} recordings.",
        f"Development: {len(development)} PF recordings. External validation: {len({r['animal'] for r in validation})} Blackstad--Moser animals, {sum(r['candidates'] for r in validation)} candidates retained.",
        "", "## Primary external results", "", "| Measure | Random halves | RUN-balanced halves |", "|---|---:|---:|"]
    for metric, label in (("regional_tv", "Regional posterior TV"), ("endpoint_separation_cm", "Mean endpoint separation, cm"), ("pair_entropy", "Normalized posterior entropy")):
        row = external.loc["real"]
        lines.append(f"| {label} | {row[f'random_{metric}']:.4f} | {row[f'balanced_{metric}']:.4f} |")
    lines += ["", "### Frozen gates", ""]
    lines += [f"- {r.gate}: {'pass' if r.passed else 'FAIL'}; {r.value}" for r in gates.itertuples(index=False)]
    lines += ["", "No threshold was changed after observing results.",
        "A failed frozen gate blocks a complete validated-remedy claim even when some measures improve.",
        "", "## Information available at the endpoint", "",
        f"- Mean eligible spikes across both halves: {primary_information.total_spikes:.3f}.",
        f"- Both halves silent: {100*primary_information.both_silent_fraction:.1f}%.",
        f"- At least one half silent: {100*primary_information.either_silent_fraction:.1f}%.",
        f"- Each half has >=2 active cells and >=3 spikes: {100*primary_information.both_halves_two_cells_three_spikes_fraction:.1f}%.",
        "These are descriptive endpoint diagnostics, not new exclusion criteria. All events remain in the primary analysis.",
        "Good event-level MUA support does not guarantee useful information in the final 20 ms after unit QC and population splitting.",
        "", "## Claim boundaries", "",
        "Equal session weights within animal, then equal animals. Seeds/draws are not additional animals.",
        "Real candidate endpoints have no known represented position; known-truth validation uses RUN and conditional simulations.",
        "Simulations condition on actual whole-universe spike totals; they are not full biological replay generators.",
        "Native spike sorting and biological replay truth were not independently verified; encoding caches are shared by producer and auditor.",
        "The previous PF-to-Tanni selection diagnostic failed, and is not superseded or rescued by this result.",
        "Do not claim repaired destination inference, multiplexed replay or biological recording bias from improved agreement alone."]
    (root/"audited_outcome.md").write_text("\n".join(lines)+"\n")
    provenance = dict(reporter_sha256=digest(__file__), producer_manifest_sha256=digest(root/"manifest.json"),
                      independent_audit_sha256=digest(root/"independent_audit.json"),
                      verified_inputs=audit["verified_table_sha256"], rescored=False)
    (root/"report_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return information, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    report(parser.parse_args().result_dir)
