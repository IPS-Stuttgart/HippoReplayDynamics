#!/usr/bin/env python3
"""Non-rescoring report of retrospective excluded-animal speed calibration."""

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


def report(input_dir, output_dir):
    paths = {"manifest": input_dir / "speed_population_transfer_manifest.json",
        "audit": input_dir / "speed_population_transfer_reconstruction_audit.json",
        "summary": input_dir / "speed_population_transfer_summary.csv",
        "paired": input_dir / "speed_population_transfer_paired_comparison.csv"}
    meta, audit = [json.loads(paths[k].read_text()) for k in ["manifest", "audit"]]
    if meta["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["manifest"] != file_sha256(paths["manifest"]):
        raise ValueError("complete transfer and linked successful audit required")
    for name in ["summary", "paired"]:
        if file_sha256(paths[name]) != meta["output_sha256"][paths[name].name]:
            raise ValueError("audited summary hash changed")
    summary = pd.read_csv(paths["summary"])
    selected = summary[summary.estimator.eq("posterior_mean") & summary.bin_filter.eq("at_least_2cells_3spikes")
        & summary.method.eq("inverse_conformal") & summary.stratum.eq("uniform")]
    rows = []
    keys = ["dataset", "selection", "generator", "observation", "calibration_scope"]
    for identity, part in selected.groupby(keys, sort=True):
        values = part.set_index("metric")
        row = dict(zip(keys, identity, strict=True))
        for metric in ["coverage", "finite_fraction", "finite_coverage", "median_finite_width", "equivalence_fraction"]:
            item = values.loc[metric]
            for field in ["equal_animal_mean", "ci_low", "ci_high", "animals_measurable"]:
                row[f"{metric}_{field}"] = item[field]
        row["animals_total"] = int(values.animals_total.iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(output_dir / "speed_population_transfer_primary_table.csv", index=False)
    datasets = sorted(selected.dataset.unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 8), squeeze=False, sharey=True, layout="constrained")
    conditions = [("A", "poisson"), ("A", "shared_gain"), ("B", "poisson"), ("B", "shared_gain")]
    for col, dataset in enumerate(datasets):
        for r, selection in enumerate(["all", "selected"]):
            ax = axes[r, col]
            for scope, color, label in [("within_session", "#287c80", "Within-session"), ("leave_one_animal_out", "#b44b64", "Excluded-animal")]:
                data = table[table.dataset.eq(dataset) & table.selection.eq(selection) & table.calibration_scope.eq(scope)].set_index(["generator", "observation"])
                data = data.reindex(pd.MultiIndex.from_tuples(conditions))
                coverage = data.coverage_equal_animal_mean.to_numpy() * 100
                ax.errorbar(np.arange(4), coverage, yerr=np.array([coverage - 100 * data.coverage_ci_low, 100 * data.coverage_ci_high - coverage]),
                    marker="o", color=color, capsize=3, label=f"{label}: coverage")
                ax.plot(np.arange(4), data.finite_fraction_equal_animal_mean * 100, marker="s", markerfacecolor="white",
                    linestyle="--", color=color, label=f"{label}: finite output")
            ax.axhline(95, color="#777777", linewidth=.7, linestyle=":")
            ax.set_xticks(range(4), ["A / Poisson", "A / gain", "B / Poisson", "B / gain"], fontsize=9)
            ax.set_ylim(-3, 103)
            ax.set_ylabel("Test panels (%)")
            label = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}.get(dataset, dataset)
            ax.set_title(f"{label}: {'all-data diagnostic' if selection == 'all' else 'selected-event primary'}")
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Animal-excluded pooled-conformal transfer\nUnbounded intervals count as covered, but are abstentions; error bars: descriptive animal bootstrap", fontsize=12)
    fig.savefig(output_dir / "speed_population_transfer_coverage.png", dpi=180)
    plt.close(fig)
    lines = ["# Retrospective Held-Out-Population Speed Calibration", "",
        f"{meta['scope']}: {meta['sessions']} sessions, {meta['animals']} animals, {meta['fits']} excluded-animal/readout fits and {meta['decisions']:,} predictive interval rows. Source panels are reused; no replay events were rescored.", "",
        "The excluded animal supplies its RUN encoding maps, but NONE of its panels enter the transfer inverse regression or residual calibration. The within-session reference is less demanding, using local fit/calibration panels. Both use the same test panels.", "",
        "Pooled conformal intervals have no asserted finite-sample new-animal guarantee here: panels are clustered and populations can shift. These are empirical transfer results. Leave-one-out fits overlap; the four/five-animal bootstrap is descriptive and conditional, not full refitting uncertainty.", "",
        "## Primary and Diagnostic Readouts", "",
        "Posterior mean, >=2 cells/3 spikes; uniform-g test panels; inverse conformal. Primary is the selected continuous run; all-data is diagnostic. A/Poisson matches the generator family, B/gain uses a disjoint RUN map plus declared shared gain. All four conditions and every estimator/stratum/method remain in the source summaries.", "",
        "|Dataset|Readout|Condition|Calibration|Coverage %|Finite %|Finite-only coverage %|Mean session median finite width (g)|Animals with finite coverage|",
        "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        if (row.generator, row.observation) not in [("A", "poisson"), ("B", "shared_gain")]:
            continue
        lines.append(f"|{row.dataset}|{row.selection}|{row.generator}/{row.observation}|{row.calibration_scope}|{100 * row.coverage_equal_animal_mean:.2f}|{100 * row.finite_fraction_equal_animal_mean:.2f}|{100 * row.finite_coverage_equal_animal_mean:.2f}|{row.median_finite_width_equal_animal_mean:.3f}|{int(row.finite_coverage_animals_measurable)}/{row.animals_total}|")
    lines += ["", "NaN finite-only coverage/width means no measurable finite intervals, not zero error or perfect calibration. High all-panel coverage with zero finite output is abstention, not a useful solution.", "",
        "Gradient truth is v=1000*(1+g*q) cm/s over normalized horizontal coordinate q, NOT physical wall distance. The strict +/-0.25 equivalence band is a predeclared benchmark tolerance, not a biologically established uniformity criterion. No biological speed inference is authorized by this report.", "",
        "The paired comparison file gives transfer-minus-local differences on matched session test cohorts; finite-only rates and widths can condition on different surviving panel subsets. Do not call those matched-panel width effects.", "",
        f"Scoring commit: `{meta['code_commit']}`. Manifest SHA256: `{file_sha256(paths['manifest'])}`."]
    (output_dir / "speed_population_transfer_report.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={**paths, "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir()})
    (output_dir / "speed_population_transfer_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    report(opts.input_dir, opts.output_dir)
