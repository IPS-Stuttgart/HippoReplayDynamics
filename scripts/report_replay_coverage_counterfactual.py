#!/usr/bin/env python3
"""Equal-animal counterfactual summaries after reconstruction audit; no rescore."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.simulate_replay_coverage_counterfactual import analysis_sets

CONDITION = ["dataset", "rate_scale", "truth_kind", "gradient", "regime", "cell_fraction", "likelihood", "estimator", "bin_filter"]
METRICS = ["geometric_pass_fraction", "continuous_eligible_recovery_fraction", "null_geometric_acceptance_fraction",
           "median_speed_cm_s", "median_selected_speed_cm_s", "median_position_error_cm", "mean_hpd95_coverage"]


def endpoint_summary(table, primary_scale, dose_replicates, seed, bootstraps):
    if table.empty or table.duplicated(CONDITION + ["animal", "session", "replicate"]).any():
        raise ValueError("empty or duplicated session endpoints")
    table = analysis_sets(table, primary_scale, dose_replicates)
    group = ["analysis_set"] + CONDITION
    sessions = table.groupby(group + ["animal", "session"], as_index=False)[METRICS].mean()
    animals = sessions.groupby(group + ["animal"], as_index=False)[METRICS].mean()
    rows, rng = [], np.random.default_rng(seed)
    for key, local in animals.groupby(group, sort=True):
        for metric in METRICS:
            finite = local[metric].dropna().to_numpy()
            low, high = np.nan, np.nan
            if len(finite) > 1:
                low, high = np.quantile(rng.choice(finite, (bootstraps, len(finite))).mean(axis=1), [.025, .975])
            rows.append({**dict(zip(group, key, strict=True)), "metric": metric, "animals": len(local),
                         "finite_animals": len(finite), "estimate": float(finite.mean()) if len(finite) else np.nan,
                         "ci95_low": low, "ci95_high": high})
    return animals, pd.DataFrame(rows)


def gradient_availability(table, primary_scale, dose_replicates):
    group = ["analysis_set", "dataset", "rate_scale", "regime", "cell_fraction", "likelihood", "estimator", "bin_filter",
             "selection", "coordinate", "readout"]
    table = analysis_sets(table, primary_scale, dose_replicates)
    key = group + ["animal", "session", "replicate"]
    if table.empty or table.duplicated(key + ["gradient"]).any():
        raise ValueError("empty or duplicated gradient endpoints")
    pivot = table[table.gradient.ne(0)].pivot(index=key, columns="gradient", values="normalized_slope")
    if -.5 not in pivot or .5 not in pivot:
        raise ValueError("both opposite injected gradients required")
    available = np.isfinite(pivot[-.5]) & np.isfinite(pivot[.5])
    pairs = available.rename("both_gradients_available").reset_index()
    rows = []
    for values, local in pairs.groupby(group, sort=True):
        retained = local[local.both_gradients_available]
        rows.append({**dict(zip(group, values, strict=True)), "session_replicate_pairs": len(local),
                     "available_session_replicate_pairs": len(retained), "available_fraction": len(retained) / len(local),
                     "sessions": local[["animal", "session"]].drop_duplicates().shape[0],
                     "sessions_with_any_available_replicate": retained[["animal", "session"]].drop_duplicates().shape[0],
                     "animals": local.animal.nunique(), "animals_with_any_available_replicate": retained.animal.nunique()})
    return pairs, pd.DataFrame(rows)


def require_completed_audit(out, manifest):
    path = out / "coverage_counterfactual_reconstruction_audit.json"
    audit = json.loads(path.read_text())
    if audit["input_file_sha256"]["manifest"] != file_sha256(out / "coverage_counterfactual_manifest.json"):
        raise ValueError("audit refers to a different manifest")
    for flag in ["all_output_hashes_verified", "all_cache_hashes_verified", "all_dependency_and_snapshot_hashes_verified"]:
        if audit.get(flag) is not True:
            raise ValueError(f"audit incomplete: {flag}")
    if audit["truth_trials"] != manifest["truth_trials_including_shuffle"] or audit["metric_rows_support_recounted"] != manifest["metric_rows"] or audit["metric_rows_support_recounted"] <= 0:
        raise ValueError("audit does not cover the complete run")
    return path


def spike_budget_summary(trials, observations, primary_scale, dose_replicates):
    keys = ["dataset", "animal", "session", "source_event_index", "replicate", "truth_kind", "gradient"]
    specs = trials[trials.truth_kind.eq("continuous") & trials.gradient.eq(0)]
    observed = observations[observations.truth_kind.eq("continuous") & observations.gradient.eq(0)
                            & observations.regime.eq("native") & observations.cell_fraction.eq(1.)]
    if specs.duplicated(keys).any() or observed.duplicated(keys + ["rate_scale"]).any():
        raise ValueError("duplicate spike-budget keys")
    merged = observed.merge(specs[keys + ["source_spikes", "source_duration_s", "simulated_duration_s"]],
                            on=keys, how="left", validate="many_to_one", indicator=True)
    if merged.empty or not merged._merge.eq("both").all():
        raise ValueError("missing source spike budget")
    merged["simulated_population_rate_hz"] = merged.spikes / merged.simulated_duration_s
    merged["source_population_rate_hz"] = merged.source_spikes / merged.source_duration_s
    merged["simulated_to_source_rate_ratio"] = merged.simulated_population_rate_hz / merged.source_population_rate_hz.replace(0, np.nan)
    merged = analysis_sets(merged, primary_scale, dose_replicates)
    group = ["analysis_set", "dataset", "rate_scale"]
    values = ["simulated_population_rate_hz", "source_population_rate_hz", "simulated_to_source_rate_ratio"]
    by_rep = merged.groupby(group + ["animal", "session", "replicate"], as_index=False)[values].median()
    sessions = by_rep.groupby(group + ["animal", "session"], as_index=False)[values].mean()
    animals = sessions.groupby(group + ["animal"], as_index=False)[values].mean()
    result = animals.groupby(group, as_index=False).agg(**{key: (key, "mean") for key in values}, animals=("animal", "nunique"))
    return result


def markdown_table(table, columns):
    def cell(value):
        if pd.isna(value):
            return "unavailable"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in table[columns].itertuples(index=False, name=None)])


def run(args):
    out = args.result_dir.resolve()
    manifest_path = out / "coverage_counterfactual_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit_path = require_completed_audit(out, manifest)
    inputs = {"manifest": manifest_path, "audit": audit_path, "reporter": Path(__file__)}
    tables = {}
    for name in ["session_summary", "session_gradients", "decomposition_summary", "gradient_response_summary", "gate_summary", "trials", "observations"]:
        path = out / f"coverage_counterfactual_{name}.csv"
        if file_sha256(path) != manifest["output_sha256"][path.name]:
            raise ValueError(f"input table changed: {name}")
        inputs[name] = path
        tables[name] = pd.read_csv(path)
    gates = tables["gate_summary"]
    if gates.empty or not gates.passed.eq(True).all() or not gates.gate.eq("overall_technical").any():
        raise ValueError("technical gates must pass before interpretation")
    params = manifest["parameters"]
    primary, dose = params["primary_rate_scale"], params["dose_replicates"]
    animals, endpoints = endpoint_summary(tables["session_summary"], primary, dose, args.seed, args.bootstraps)
    pairs, availability = gradient_availability(tables["session_gradients"], primary, dose)
    budgets = spike_budget_summary(tables["trials"], tables["observations"], primary, dose)
    outputs = []
    for name, frame in [("animal_endpoints", animals), ("endpoint_summary", endpoints),
                        ("gradient_pair_availability", pairs), ("gradient_availability_summary", availability),
                        ("spike_budget_summary", budgets)]:
        path = out / f"coverage_counterfactual_{name}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
    part = tables["decomposition_summary"]
    part = part[part.analysis_set.eq("primary_replication") & part.cell_fraction.eq(.5) & part.estimator.eq("map") & part.bin_filter.eq("unfiltered") & part.contrast.isin(["native_minus_full", "pooled_minus_full", "native_minus_pooled"])]
    part = part.assign(effect_pp=100 * part.effect, low_pp=100 * part.ci95_low, high_pp=100 * part.ci95_high)
    response = tables["gradient_response_summary"]
    response = response[response.analysis_set.eq("paired_dose") & response.regime.eq("native") & response.cell_fraction.eq(1.) & response.likelihood.eq("poisson") & response.estimator.eq("posterior_mean") & response.bin_filter.eq("unfiltered") & response.selection.eq("all") & response.coordinate.eq("true_coordinate") & response.readout.isin(["decoded", "true_chord"])]
    nulls = endpoints[endpoints.analysis_set.eq("primary_replication") & endpoints.regime.eq("native") & endpoints.cell_fraction.eq(1.) & endpoints.likelihood.eq("poisson") & endpoints.estimator.eq("map") & endpoints.bin_filter.eq("unfiltered") & endpoints.metric.eq("null_geometric_acceptance_fraction") & endpoints.truth_kind.ne("continuous")]
    selected = availability[availability.analysis_set.eq("primary_replication") & availability.regime.eq("native") & availability.cell_fraction.eq(1.) & availability.likelihood.eq("poisson") & availability.estimator.eq("posterior_mean") & availability.bin_filter.eq("unfiltered") & availability.selection.eq("selected") & availability.coordinate.eq("true_coordinate") & availability.readout.eq("decoded")]
    text = f"""# Paired Counterfactual Recovery Report

Non-rescoring report. All technical and reconstruction checks passed.
Scoring commit: `{manifest['code_commit']}`. {manifest['sessions']} sessions,
{manifest['source_profiles']} fixed source duration profiles,
{manifest['truth_trials_including_shuffle']} truth-trial records (including shuffled derivatives),
{manifest['metric_rows']} repeated metric rows. Not independent biological replicates.
{'SMOKE ONLY: capped sources or dirty scoring tree; not production evidence.' if params['max_profiles'] is not None or manifest['git_dirty'] else 'Clean production invocation.'}

## Primary Count-Preserving Partition

Constant true speed 1,000 cm/s; truth-eligible events; half population; MAP,
no per-bin support filter; rate multiplier {primary:g}; all {params['replicates']}
synthetic randomizations. Events averaged within session/randomization,
randomizations within session, sessions within animal, then animals equally.
Intervals bootstrap animals conditional on these fixed maps and simulations.

{markdown_table(part, ['dataset', 'likelihood', 'contrast', 'animals', 'effect_pp', 'low_pp', 'high_pp'])}

`pooled_minus_full` removes individual identities but keeps every spike.
`native_minus_pooled` removes the pooled channel, losing both its counts AND
aggregate spatial information. It is not a pure count-only effect. The two
paired components sum to `native_minus_full`. The separate oracle-restoration
tables can add information through true-position-dependent relabeling and must
not be substituted for this primary data-only coarsening.

## Paired Dose and Gradient Recovery

Replicate zero only at all doses. Opposite horizontal speed fields span
500-1,500 cm/s. Injected arclength-slope difference = 1.0. Truth chord response
accounts for 20 ms window averaging and reflection. Independent Poisson decoder,
posterior mean, full population, before continuity selection, KNOWN coordinate.
This is an optimistic simulation diagnostic, not a wall-distance result or an
available real-data covariate. High multipliers test an information limit, not
physiological replay rates.

{markdown_table(response, ['dataset', 'rate_scale', 'readout', 'finite_animals', 'response', 'ci95_low', 'ci95_high'])}

## Null Acceptance and Selected-Sample Availability

Geometric null acceptance is not empirical replay FPR or shuffle significance.
Whole-bin shuffles preserve the population snapshots and within-bin movement;
overlapping 20 ms windows partly average neighboring shuffled 5 ms bins.

{markdown_table(nulls, ['dataset', 'truth_kind', 'estimate', 'ci95_low', 'ci95_high'])}

Selected-core gradient contrast requires BOTH opposite-gradient estimates.
Keep unavailable session/randomization combinations in the denominator.

{markdown_table(selected, ['dataset', 'session_replicate_pairs', 'available_session_replicate_pairs', 'sessions', 'sessions_with_any_available_replicate', 'animals', 'animals_with_any_available_replicate'])}

## Simulated Versus Source Spike Budgets

Full population, constant-speed synthetic paths. Population rates use each
window's actual duration, including the source's possible partial tail. Source
counts use the same RUN-QC cell set. Values average within-replicate session
medians, randomizations, sessions within animal, and animals equally. Ratios
are paired source-profile rate ratios, not ratios of these aggregate means.
Source events are high-MUA candidates; synthetic paths were not selected by
MUA strength. This is a representativeness check, not an inferred replay rate
multiplier or a reason to choose a new dose after seeing recovery outcomes.

{markdown_table(budgets, ['analysis_set', 'dataset', 'rate_scale', 'simulated_population_rate_hz', 'source_population_rate_hz', 'simulated_to_source_rate_ratio'])}

## Claim Boundary

These are new synthetic randomizations on existing empirical maps, not new
biological animals, held-out tuning maps, or an independently validated bias
correction. This measures controlled information dependence and gradient
recoverability; it cannot establish biological speed uniformity or fully
explain the PF/Tanni difference. Finite estimates and positive dose responses
do not alone validate equivalence testing. Map mismatch, broader factor
isolation, event-definition sensitivity, and held-out calibration remain open.
"""
    report = out / "coverage_counterfactual_report.md"
    report.write_text(text)
    outputs.append(report)
    provenance = {**build_script_provenance(input_paths=inputs, cwd=ROOT),
                  "scope": "non-rescoring equal-animal summaries with unavailable estimates retained",
                  "output_sha256": {p.name: file_sha256(p) for p in outputs}}
    (out / "coverage_counterfactual_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstraps < 100:
        parser.error("at least 100 bootstrap draws required")
    run(args)


if __name__ == "__main__":
    main()
