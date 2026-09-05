#!/usr/bin/env python3
"""Non-rescoring, equal-animal readout of frozen candidate-budget sensitivity."""

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
import numpy as np
import pandas as pd
from hipporeplayimm.replay_speed_population_transfer import GROUP, IDENTITY, METRICS, READOUT

from scripts._provenance import build_script_provenance, file_sha256
from scripts.report_replay_speed_identifiability import monte_carlo_intervals

BUDGET = "candidate_budget"
PANEL_GROUP = [BUDGET] + READOUT + ["phase", "generator", "observation", "stratum"]
AVAILABILITY = ["statistic_available_fraction", "too_few_events_fraction", "too_little_spatial_variance_fraction",
    "mean_contributing_events", "mean_continuous_events", "mean_statistic", "mean_spikes", "mean_supported_window_fraction"]
DIRECTIONAL = ["correct_sign_nonzero_fraction", "wrong_sign_nonzero_fraction"]


def directional_endpoints(decisions):
    data = decisions.copy()
    finite = np.isfinite(data.lower) & np.isfinite(data.upper)
    data["correct_sign_nonzero_fraction"] = (finite & (((data.gradient > 0) & (data.lower > 0)) | ((data.gradient < 0) & (data.upper < 0)))).astype(float)
    data["wrong_sign_nonzero_fraction"] = (finite & (((data.gradient < 0) & (data.lower > 0)) | ((data.gradient > 0) & (data.upper < 0)))).astype(float)
    return data.groupby(IDENTITY + [BUDGET] + GROUP, sort=True, dropna=False)[DIRECTIONAL].mean().reset_index()


def panel_endpoints(panels):
    data = panels.copy()
    data["statistic_available_fraction"] = np.isfinite(data.statistic).astype(float)
    data["too_few_events_fraction"] = data.statistic_status.eq("fewer_than_five_events").astype(float)
    data["too_little_spatial_variance_fraction"] = data.statistic_status.eq("insufficient_spatial_variance").astype(float)
    if not np.allclose(data[["statistic_available_fraction", "too_few_events_fraction", "too_little_spatial_variance_fraction"]].sum(axis=1), 1):
        raise ValueError("unclassified statistic availability")
    for metric, source in [("mean_contributing_events", "contributing_events"), ("mean_continuous_events", "continuous_events"),
        ("mean_statistic", "statistic"), ("mean_spikes", "spikes")]:
        data[metric] = data[source]
    data["mean_supported_window_fraction"] = data.valid_windows / data.total_windows
    keys = IDENTITY + PANEL_GROUP
    result = data.groupby(keys, sort=True, dropna=False)[AVAILABILITY].mean()
    return result.join(data.groupby(keys, sort=True, dropna=False).size().rename("panels")).reset_index()


def aggregate_events(table, group, metrics, seed=20260918, bootstraps=5000):
    if table.empty or table.duplicated(IDENTITY + group).any():
        raise ValueError("unique nonempty session endpoints required")
    animal = table.groupby(["dataset", "animal"] + group, sort=True, dropna=False)[metrics].mean().reset_index()
    total = table.groupby(["dataset", "animal"] + group, sort=True, dropna=False).size().rename("sessions").reset_index()
    animal = animal.merge(total, on=["dataset", "animal"] + group, validate="one_to_one")
    rng, rows = np.random.default_rng(seed), []
    for identity, part in animal.groupby(["dataset"] + group, sort=True, dropna=False):
        base = dict(zip(["dataset"] + group, identity, strict=True))
        for metric in metrics:
            values = part[metric].dropna().to_numpy(float)
            ci = np.quantile(rng.choice(values, (bootstraps, len(values)), replace=True).mean(axis=1), [.025, .975]) if len(values) else [np.nan, np.nan]
            rows.append({**base, "metric": metric, "equal_animal_mean": values.mean() if len(values) else np.nan,
                "ci_low": ci[0], "ci_high": ci[1], "animals_available": len(values), "animals_total": len(part),
                "animals_positive": int((values > 0).sum()), "animals_negative": int((values < 0).sum())})
    return animal, pd.DataFrame(rows)


def paired_budgets(table, group, metrics):
    if table.empty or table.duplicated(IDENTITY + [BUDGET] + group).any():
        raise ValueError("unique budget/session endpoints required")
    budgets = sorted(table[BUDGET].unique())
    if len(budgets) < 2:
        raise ValueError("at least two budgets required")
    base = table[table[BUDGET].eq(budgets[0])]
    frames = []
    for budget in budgets[1:]:
        keys = IDENTITY + group
        selected = table[table[BUDGET].eq(budget)]
        joined = selected.merge(base, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_larger", "_reference"))
        if not joined._merge.eq("both").all() or not joined.panels_larger.eq(joined.panels_reference).all():
            raise ValueError("budget comparison requires identical session/draw denominators")
        frame = joined[keys].copy()
        frame["budget_comparison"] = f"{budget}-{budgets[0]}"
        for metric in metrics:
            frame[metric] = joined[f"{metric}_larger"] - joined[f"{metric}_reference"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def primary(data, selection="selected"):
    return data[data.estimator.eq("posterior_mean") & data.bin_filter.eq("at_least_2cells_3spikes") & data.selection.eq(selection)]


def figures(availability, intervals, output):
    datasets = sorted(availability.dataset.unique())
    titles = {"pfeiffer_foster": "Pfeiffer/Foster", "tanni2022": "Tanni"}
    conditions = [("A", "poisson", "-", "A / Poisson"), ("B", "shared_gain", "--", "B / gain")]
    budgets = sorted(availability[BUDGET].unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 7), squeeze=False, sharey=True, layout="constrained")
    for col, dataset in enumerate(datasets):
        for row, selection in enumerate(["all", "selected"]):
            ax = axes[row, col]
            data = primary(availability[availability.dataset.eq(dataset) & availability.phase.eq("test") & availability.stratum.eq("uniform")], selection)
            for index, animal in enumerate(sorted(data.animal.unique())):
                color = plt.get_cmap("tab10")(index)
                for generator, observation, style, _ in conditions:
                    part = data[data.animal.eq(animal) & data.generator.eq(generator) & data.observation.eq(observation)].set_index(BUDGET).reindex(budgets)
                    ax.plot(budgets, 100 * part.statistic_available_fraction, style, marker="o", color=color,
                        label=animal if generator == "A" else None)
            ax.set(title=f"{titles.get(dataset, dataset)}: {selection}", ylabel="Finite speed statistics (%)", xlabel="Candidate events per panel",
                xticks=budgets, ylim=(-3, 103))
            ax.legend(fontsize=8, ncol=3)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does a larger candidate panel restore measurable speed variation?\nOne line per animal; solid: A/Poisson, dashed: disjoint B map + shared gain", fontsize=12)
    fig.savefig(output / "speed_panel_size_availability.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(3, len(datasets), figsize=(6 * len(datasets), 10), squeeze=False, layout="constrained")
    for col, dataset in enumerate(datasets):
        data = primary(intervals[intervals.dataset.eq(dataset) & intervals.stratum.eq("uniform") & intervals.method.eq("inverse_conformal")])
        for row, metric in enumerate(["finite_fraction", "coverage", "median_finite_width"]):
            ax = axes[row, col]
            for scope, color, label in [("within_session", "#287c80", "Local"), ("leave_one_animal_out", "#b44b64", "Excluded animal")]:
                for generator, observation, style, condition in conditions:
                    part = data[data.calibration_scope.eq(scope) & data.generator.eq(generator) & data.observation.eq(observation) & data.metric.eq(metric)].set_index(BUDGET).reindex(budgets)
                    scale = 1 if metric == "median_finite_width" else 100
                    ax.plot(budgets, scale * part.equal_animal_mean, style, marker="o", color=color, label=f"{label}: {condition}")
                    ax.fill_between(budgets, scale * part.ci_low, scale * part.ci_high, color=color, alpha=.08)
            labels = {"finite_fraction": "Finite intervals (%)", "coverage": "All-panel coverage (%)", "median_finite_width": "Finite interval width in g"}
            ax.set(xticks=budgets, xlabel="Candidate events per panel", ylabel=labels[metric])
            if row < 2:
                ax.set_ylim(-3, 103)
            else:
                ax.set_ylim(bottom=0)
            if row == 1:
                ax.axhline(95, color=".5", linestyle=":", linewidth=.8)
            if row == 0:
                ax.set_title(titles.get(dataset, dataset))
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Selected-event primary readout: inverse conformal\nUnbounded intervals cover the truth but abstain; widths condition on finite output\nShading: descriptive animal bootstrap; absent width means no finite output", fontsize=12)
    fig.savefig(output / "speed_panel_size_calibration.png", dpi=170)
    plt.close(fig)


def report(root, out, seed=20260918, bootstraps=5000):
    paths = {"manifest": root / "speed_panel_size_manifest.json", "audit": root / "speed_panel_size_reconstruction_audit.json",
        "sessions": root / "speed_panel_size_session_summary.csv", "batches": root / "speed_panel_size_batches.csv"}
    meta, audit = [json.loads(paths[k].read_text()) for k in ["manifest", "audit"]]
    if meta["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["manifest"] != file_sha256(paths["manifest"]):
        raise ValueError("complete scoring with linked passing audit required")
    def read(path):
        if file_sha256(path) != meta["output_sha256"][path.name]:
            raise ValueError("audited report input changed")
        paths[path.name] = path
        return pd.read_csv(path, float_precision="round_trip")
    sessions, batches = read(paths["sessions"]), read(paths["batches"])
    direction = []
    columns = IDENTITY + [BUDGET] + GROUP + ["gradient", "lower", "upper"]
    decision_paths = [root / f"local_decisions__{row.label}.csv.gz" for row in batches.itertuples()]
    decision_paths += [root / f"transfer_decisions__{budget}.csv.gz" for budget in sorted(sessions[BUDGET].unique())]
    for path in decision_paths:
        if file_sha256(path) != meta["output_sha256"][path.name]:
            raise ValueError("audited decision input changed")
        paths[path.name] = path
        direction.append(directional_endpoints(pd.read_csv(path, usecols=columns, float_precision="round_trip")))
    sessions = sessions.merge(pd.concat(direction, ignore_index=True), on=IDENTITY + [BUDGET] + GROUP,
        how="outer", validate="one_to_one", indicator=True)
    if not sessions._merge.eq("both").all():
        raise ValueError("directional summary membership differs")
    sessions = sessions.drop(columns="_merge")
    interval_metrics = METRICS + DIRECTIONAL
    panels = pd.concat([read(root / f"panels__{row.label}.csv.gz") for row in batches.itertuples()], ignore_index=True)
    availability = panel_endpoints(panels)
    animal_avail, summary_avail = aggregate_events(availability, PANEL_GROUP, AVAILABILITY, seed, bootstraps)
    animal_intervals, summary_intervals = aggregate_events(sessions, [BUDGET] + GROUP, interval_metrics, seed, bootstraps)
    paired = paired_budgets(sessions, GROUP, interval_metrics)
    paired_animal, paired_summary = aggregate_events(paired, ["budget_comparison"] + GROUP, interval_metrics, seed, bootstraps)
    paired_avail = paired_budgets(availability, [k for k in PANEL_GROUP if k != BUDGET], AVAILABILITY)
    _, paired_avail_summary = aggregate_events(paired_avail, ["budget_comparison"] + [k for k in PANEL_GROUP if k != BUDGET], AVAILABILITY, seed, bootstraps)
    monte_carlo = []
    for (budget, scope), part in sessions.groupby([BUDGET, "calibration_scope"], sort=True):
        frame = monte_carlo_intervals(part)
        frame[BUDGET], frame["calibration_scope"] = budget, scope
        monte_carlo.append(frame)
        for metric in DIRECTIONAL:
            changed = part.copy()
            changed["nonzero_fraction"] = changed[metric]
            frame = monte_carlo_intervals(changed)
            frame = frame[frame.metric.eq("nonzero_fraction")].copy()
            frame["metric"], frame[BUDGET], frame["calibration_scope"] = metric, budget, scope
            monte_carlo.append(frame)
    out.mkdir(parents=True, exist_ok=False)
    tables = {"availability_by_session": availability, "availability_by_animal": animal_avail, "availability_summary": summary_avail,
        "intervals_by_session": sessions,
        "intervals_by_animal": animal_intervals, "interval_summary": summary_intervals, "paired_budget_by_animal": paired_animal,
        "paired_budget_summary": paired_summary, "paired_availability_summary": paired_avail_summary,
        "session_monte_carlo": pd.concat(monte_carlo, ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(out / f"speed_panel_size_{name}.csv", index=False)
    figures(animal_avail, summary_intervals, out)
    lines = ["# Candidate-Panel Size and Speed Recoverability", "",
        f"Non-rescoring report of {meta['scope']}: {meta['sessions']} sessions, {meta['animals']} animals. All thresholds and calibration methods were frozen before this experiment.", "",
        "Budgets are strict prefixes of identical generated observations. Cycling a source duration profile generates a fresh path and spike draw, not a duplicate replay. There are no additional biological animals or ground-truth replay observations.", "",
        "Primary: posterior mean, >=2 cells/3 spikes, selected continuous core. All-data/MAP/unfiltered readouts remain in tables. A/Poisson is matching calibration; B/gain combines disjoint RUN-map and shared-gain stress.", "",
        "## Primary Test-Statistic Availability", "",
        "|Dataset|Candidates|Condition|Finite statistic %|Contributing events (mean)|",
        "|---|---:|---|---:|---:|"]
    p = primary(summary_avail[summary_avail.phase.eq("test") & summary_avail.stratum.eq("uniform")])
    for (dataset, budget, generator, observation), part in p.groupby(["dataset", BUDGET, "generator", "observation"]):
        if (generator, observation) not in [("A", "poisson"), ("B", "shared_gain")]:
            continue
        values = part.set_index("metric").equal_animal_mean
        lines.append(f"|{dataset}|{budget}|{generator}/{observation}|{100 * values.statistic_available_fraction:.2f}|{values.mean_contributing_events:.2f}|")
    lines += ["", "## Primary Inverse-Conformal Outcomes", "",
        "Uniform-g test panels; session means within animals, then animals equally. Width is the mean of session medians among finite intervals; its conditioning set can differ between budgets.", "",
        "|Dataset|Candidates|Condition|Calibration|Finite %|Coverage %|Finite-only coverage %|Width g|",
        "|---|---:|---|---|---:|---:|---:|---:|"]
    p = primary(summary_intervals[summary_intervals.stratum.eq("uniform") & summary_intervals.method.eq("inverse_conformal")])
    for (dataset, budget, generator, observation, scope), part in p.groupby(["dataset", BUDGET, "generator", "observation", "calibration_scope"]):
        if (generator, observation) not in [("A", "poisson"), ("B", "shared_gain")]:
            continue
        v = part.set_index("metric").equal_animal_mean
        lines.append(f"|{dataset}|{budget}|{generator}/{observation}|{scope}|{100 * v.finite_fraction:.2f}|{100 * v.coverage:.2f}|{100 * v.finite_coverage:.2f}|{v.median_finite_width:.3f}|")
    lines += ["", "## Known Zero/Nonzero Gradient Checks", "",
        "Primary excluded-animal inverse-conformal output. g=0 uses the strict +/-0.25 equivalence rate. g=+/-0.50 uses the CORRECT-SIGN nonzero-interval rate (not proof of correctly estimated magnitude). Wrong-sign confidence and all conditions/strata and false-equivalence rates remain in the CSV tables.", "",
        "|Dataset|Candidates|Condition|Truth g|Decision|Rate %|Animals available|",
        "|---|---:|---|---|---|---:|---:|"]
    p = primary(summary_intervals[summary_intervals.method.eq("inverse_conformal") & summary_intervals.calibration_scope.eq("leave_one_animal_out")])
    for row in p.itertuples(index=False):
        metric = "equivalence_fraction" if row.stratum == "fixed_+0.00" else "correct_sign_nonzero_fraction"
        if row.stratum not in ["fixed_+0.00", "fixed_-0.50", "fixed_+0.50"] or row.metric != metric or (row.generator, row.observation) not in [("A", "poisson"), ("B", "shared_gain")]:
            continue
        lines.append(f"|{row.dataset}|{row.candidate_budget}|{row.generator}/{row.observation}|{row.stratum}|{metric}|{100 * row.equal_animal_mean:.2f}|{row.animals_available}/{row.animals_total}|")
    lines += ["", "## Interpretation Boundaries", "",
        "Finite statistics do not establish informative inference. Unbounded intervals count as covered but are abstentions. Interpret availability, coverage, finite-only coverage, width, false equivalence and known-gradient recovery together. A zero observed false-equivalence rate is not proof of zero risk: the session Monte Carlo table retains Wilson intervals and denominators.", "",
        "The source durations and maps are fixed surrogates. q is normalized horizontal position, NOT physical wall distance. A result for these gradients does not establish real replay uniformity. The +/-0.25 band is a predeclared benchmark tolerance, with +/-0.10 and +/-0.50 sensitivities retained.", "",
        "No fit/calibration panel from an excluded animal enters its pooled transfer fit. Overlapping training folds and four/five animals limit population inference; bootstrap CIs are descriptive and conditional, not refitting uncertainty or a new-animal conformal guarantee. More candidate draws are not more biological replication.", "",
        "The paired-budget tables compare identical session/draw cohorts. Finite-only widths/coverage may condition on different surviving panels; they are not matched-finite-panel effects. Do not select a favorable budget after inspecting these results.", "",
        f"Scoring commit: `{meta['code_commit']}`. Manifest SHA256: `{file_sha256(paths['manifest'])}`."]
    (out / "speed_panel_size_report.md").write_text("\n".join(lines) + "\n")
    paths.update({"reporter": Path(__file__), "monte_carlo_helper": ROOT / "scripts/report_replay_speed_identifiability.py"})
    provenance = build_script_provenance(input_paths=paths, cwd=ROOT)
    provenance.update(seed=seed, bootstraps=bootstraps, output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "speed_panel_size_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--bootstraps", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error("positive bootstrap count required")
    report(args.input_dir.resolve(), args.output_dir.resolve(), args.seed, args.bootstraps)
