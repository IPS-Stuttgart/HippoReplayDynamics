#!/usr/bin/env python3
"""Non-rescoring paired population summaries for the geometry factorial."""

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

from hipporeplayimm.replay_coverage_geometry import FACTORS
from scripts._provenance import build_script_provenance, file_sha256

GROUP = [f for f in FACTORS if f != "population_seed"]
MEASURES = ["acceptance_fraction", "eligible_recovery_fraction", "spikes", "valid_bins", "all_steps", "selected_steps",
            "large_jump_fraction", "median_position_error_cm", "median_posterior_rms_cm",
            "all_median_speed_cm_s", "selected_median_speed_cm_s"]


def interval(values, rng, draws):
    data = np.asarray(values, float)
    data = data[np.isfinite(data)]
    ci = [np.nan, np.nan]
    if len(data) > 1:
        ci = np.quantile(rng.choice(data, (draws, len(data))).mean(axis=1), [.025, .975])
    return {"populations_available": len(data), "mean": float(data.mean()) if len(data) else np.nan,
            "ci95_low": ci[0], "ci95_high": ci[1]}


def aggregate_endpoints(frame, rng, draws):
    rows = []
    for key, group in frame.groupby(GROUP + ["truth_kind", "gradient"], sort=True):
        if group.population_seed.duplicated().any():
            raise ValueError("duplicate population endpoint")
        identity = dict(zip(GROUP + ["truth_kind", "gradient"], key, strict=True))
        for metric in MEASURES:
            rows.append({**identity, "metric": metric, "populations_expected": len(group),
                         **interval(group[metric], rng, draws)})
    return pd.DataFrame(rows)


def gradient_responses(frame):
    index = FACTORS + ["selection", "coordinate", "readout"]
    local = frame[frame.gradient.isin([-.5, .5])]
    pivot = local.pivot(index=index, columns="gradient", values="normalized_slope").reindex(columns=[-.5, .5])
    result = pivot.index.to_frame(index=False)
    result["gradient_response"] = (pivot[.5] - pivot[-.5]).to_numpy()
    result["both_gradients_available"] = pivot.notna().all(axis=1).to_numpy()
    return result


def aggregate_responses(frame, rng, draws):
    rows = []
    for key, group in frame.groupby(GROUP + ["selection", "coordinate", "readout"], sort=True):
        rows.append({**dict(zip(GROUP + ["selection", "coordinate", "readout"], key, strict=True)),
                     "populations_expected": len(group), **interval(group.gradient_response, rng, draws)})
    return pd.DataFrame(rows)


def rule_alias(frame):
    """5 ms rules coincide; reuse once per labeled comparison, not extra samples."""
    alias = frame[frame.stride_ms.eq(5)].assign(continuity_rule="time_scaled_4000cm_s_45ms")
    return pd.concat([frame, alias], ignore_index=True)


def paired_contrasts(endpoints, responses, rng, draws):
    base = endpoints[endpoints.truth_kind.eq("continuous") & endpoints.gradient.eq(0)]
    blocks = [base.assign(metric=metric, value=base[metric]) for metric in
              ["eligible_recovery_fraction", "all_median_speed_cm_s", "median_position_error_cm"]]
    reduced = responses[responses.selection.eq("all") & responses.coordinate.eq("true_coordinate") & responses.readout.eq("decoded")]
    blocks.append(reduced.assign(metric="gradient_response", value=reduced.gradient_response))
    values = rule_alias(pd.concat([b[FACTORS + ["metric", "value"]] for b in blocks], ignore_index=True))
    references = {"n_cells": 128, "area_m2": 3.7, "aspect": 1., "sigma_cm": 30.,
                  "grid_cm": 8, "window_ms": 20, "stride_ms": 5, "support_domain": "common_core"}
    out = []
    for factor, reference in references.items():
        index = [col for col in FACTORS if col != factor] + ["metric"]
        ref = values[values[factor].eq(reference)][index + ["value"]].rename(columns={"value": "reference_value"})
        for level, changed in values[~values[factor].eq(reference)].groupby(factor, sort=True):
            paired = changed.merge(ref, on=index, how="left", validate="one_to_one", indicator=True)
            paired["difference"] = paired.value - paired.reference_value
            grouping = [col for col in index if col != "population_seed"]
            for key, group in paired.groupby(grouping, sort=True):
                out.append({**dict(zip(grouping, key, strict=True)), "varied_factor": factor,
                            "reference_level": reference, "changed_level": level,
                            "populations_requested": len(group), "matched_populations": int(group._merge.eq("both").sum()),
                            **interval(group.difference, rng, draws)})
    return pd.DataFrame(out)


def figures(endpoints, responses, out):
    base = endpoints[(endpoints.grid_cm.eq(8)) & endpoints.window_ms.eq(20) & endpoints.stride_ms.eq(5)
                     & endpoints.aspect.eq(1.4) & endpoints.estimator.eq("map") & endpoints.bin_filter.eq("unfiltered")
                     & endpoints.truth_kind.eq("continuous") & endpoints.gradient.eq(0)
                     & endpoints.metric.eq("eligible_recovery_fraction")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, observation in enumerate(["native_poisson", "common_count_schedule"]):
        for col, support in enumerate(["common_core", "full_arena"]):
            ax = axes[row, col]
            local = base[base.observation.eq(observation) & base.support_domain.eq(support)]
            for (area, cells), group in local.groupby(["area_m2", "n_cells"]):
                group = group.sort_values("sigma_cm")
                ax.plot(group.sigma_cm, 100 * group["mean"], marker="o", label=f"{area:g} m2; {cells} cells")
                ax.fill_between(group.sigma_cm, 100 * group.ci95_low, 100 * group.ci95_high, alpha=.12)
            ax.set(xlabel="Gaussian field sigma (cm)", ylabel="Eligible continuous paths recovered (%)",
                   title=f"{observation.replace('_', ' ')}\n{support.replace('_', ' ')} support", ylim=(0, 100), xticks=[20, 30, 40])
            ax.legend(fontsize=8)
    fig.suptitle("Known 1000 cm/s paths; MAP; 8 cm / 20 ms / 5 ms; aspect 1.4\nPopulation-bootstrap intervals, not biological confidence intervals")
    fig.savefig(out / "geometry_field_factorial.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, area in enumerate([3.7, 8.75]):
        common = {"area_m2": area, "n_cells": 128, "sigma_cm": 30., "aspect": 1.4, "support_domain": "full_arena",
                  "observation": "native_poisson", "stride_ms": 5, "bin_filter": "unfiltered"}
        top, bottom = endpoints, responses
        for name, value in common.items():
            top, bottom = top[top[name].eq(value)], bottom[bottom[name].eq(value)]
        top = top[top.estimator.eq("map") & top.truth_kind.eq("continuous") & top.gradient.eq(0) & top.metric.eq("eligible_recovery_fraction")]
        bottom = bottom[bottom.estimator.eq("posterior_mean") & bottom.selection.eq("all") & bottom.coordinate.eq("true_coordinate") & bottom.readout.eq("decoded")]
        for ax, table, scale in [(axes[0, col], top, 100), (axes[1, col], bottom, 1)]:
            for grid, group in table.groupby("grid_cm"):
                group = group.sort_values("window_ms")
                ax.plot(group.window_ms, scale * group["mean"], marker="o", label=f"{grid:g} cm grid")
                ax.fill_between(group.window_ms, scale * group.ci95_low, scale * group.ci95_high, alpha=.12)
            ax.set(xlabel="Window duration (ms)", xticks=[10, 20, 40], title=f"{area:g} m2")
            ax.legend(fontsize=8)
        axes[0, col].set(ylabel="MAP continuous-path recovery (%)", ylim=(0, 100))
        axes[1, col].axhline(0, color="black", lw=.7)
        axes[1, col].axhline(1, color="gray", ls="--", label="injected response")
        axes[1, col].set(ylabel="Posterior-mean gradient response")
    fig.suptitle("Same paths and counts across decoder resolutions\n128 cells; sigma30 cm; native Poisson; full support; no bin filter")
    fig.savefig(out / "geometry_resolution_factorial.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.input_dir / f"geometry_{name}.csv" for name in ["population_summary", "population_gradients"]}
    rng = np.random.default_rng(args.seed)
    summary = pd.read_csv(paths["population_summary"])
    gradient = gradient_responses(pd.read_csv(paths["population_gradients"]))
    endpoints = aggregate_endpoints(summary, rng, args.bootstraps)
    responses = aggregate_responses(gradient, rng, args.bootstraps)
    contrasts = paired_contrasts(summary, gradient, rng, args.bootstraps)
    for name, table in [("endpoint_summary", endpoints), ("gradient_response_by_population", gradient),
                        ("gradient_response_summary", responses), ("paired_factor_contrasts", contrasts)]:
        table.to_csv(args.output_dir / f"geometry_{name}.csv", index=False)
    figures(endpoints, responses, args.output_dir)
    (args.output_dir / "geometry_report.md").write_text(
        "# Controlled Geometry and Resolution Benchmark\n\n"
        "Non-rescoring synthetic study. Confidence intervals resample synthetic population seeds, not animals.\n\n"
        "Positive-path recovery is conditional on setting-specific true geometric eligibility. Null acceptance is not biological false-positive rate. "
        "Gradient summaries require both signs available within the same population; missing values are not zero. "
        "Changed-minus-reference contrasts pair population seeds and every other setting. Missing matched settings remain explicitly unavailable.\n\n"
        "The common-count family assigns new labels using known position and must not be described as a count-only intervention on recordings. "
        "Common-core decoder support is an optimistic restriction. Grids/time windows use the same fine-bin counts. "
        "Time-scaled continuity comparisons reuse the identical 5 ms reference without increasing sample size.\n\n"
        "No biological uniformity, cross-dataset replay-prevalence, or neural-mechanism claim is authorized by these results.\n")
    provenance = build_script_provenance(input_paths={**paths, "reporter": __file__, "scoring_manifest": args.input_dir / "geometry_manifest.json"})
    (args.output_dir / "geometry_report_manifest.json").write_text(json.dumps({**provenance,
        "output_sha256": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file() and p.name != "geometry_report_manifest.json"}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
