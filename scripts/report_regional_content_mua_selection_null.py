"""Non-rescoring report of the fixed-tuning MUA-selection counterfactual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

KEY = ["animal", "session", "replicate", "peak_gain", "cohort", "readout", "calibration"]
CONDITION = ["peak_gain", "cohort", "readout", "calibration"]
LABELS = {"fixed": "Fixed windows", "selected": "MUA endpoints", "independent_same_window": "Independent, same windows"}
COLORS = {"fixed": "#007f86", "selected": "#b72b50", "independent_same_window": "#616161"}


def technical_gates(manifest, sessions, detection, fits, audited):
    expected = len(sessions)*manifest["parameters"]["replicates"]*3
    expected_fits = int((4+8*(detection.selected_endpoints > 0)).sum()) if len(detection) else 0
    primary = detection.loc[detection.peak_gain.eq(6)]
    checks = {
        "input_run_complete": manifest["status"] == "complete",
        "independent_audit_pass": audited.get("status") == "pass",
        "all_sessions_complete": len(sessions) > 0 and sessions.status.eq("complete").all(),
        "all_simulations_present": expected > 0 and len(detection) == expected and not detection.duplicated(["session", "replicate", "peak_gain"]).any(),
        "nonempty_primary_selected": len(primary) > 0 and len(primary) == len(sessions)*manifest["parameters"]["replicates"] and primary.selected_endpoints.gt(0).all(),
        "calibration_rows_complete": expected_fits > 0 and len(fits) == expected_fits and not fits.duplicated(KEY).any(),
        "finite_calibration_fits": len(fits) > 0 and fits.status.eq("fit").all() and np.isfinite(fits[["prevalence", "fit_tv", "auc"]]).all().all(),
    }
    checks["overall"] = all(checks.values())
    return pd.DataFrame([{"gate": k, "status": "pass" if v else "fail"} for k, v in checks.items()])


def tables(fits, bounds):
    zero = bounds.loc[bounds.slack.eq(0), KEY+["lower", "upper", "status", "contains_truth"]].rename(columns={"status": "bound_status"})
    joined = fits.merge(zero, on=KEY, validate="one_to_one")
    if len(joined) != len(fits):
        raise ValueError("missing zero-slack bounds")
    joined["abs_prevalence_bias"] = joined.prevalence_bias.abs()
    joined["incompatible"] = joined.bound_status.eq("incompatible")
    joined["feasible_wrong_truth"] = joined.bound_status.eq("feasible") & ~joined.contains_truth
    joined["bound_width"] = (joined.upper-joined.lower).where(joined.bound_status.eq("feasible"))
    metrics = ["fit_tv", "prevalence", "true_prevalence", "prevalence_bias", "abs_prevalence_bias", "auc",
               "mean_spikes", "silent_fraction", "incompatible", "contains_truth", "feasible_wrong_truth", "bound_width"]
    by_session = joined.groupby(["animal", "session"]+CONDITION, as_index=False)[metrics].mean()
    by_animal = by_session.groupby(["animal"]+CONDITION, as_index=False)[metrics].mean()
    summary = by_animal.groupby(CONDITION, as_index=False)[metrics].mean()
    counts = joined.groupby(CONDITION, as_index=False).agg(simulation_runs=("session", "size"),
        incompatible_runs=("incompatible", "sum"), truth_contained_runs=("contains_truth", "sum"),
        feasible_wrong_truth_runs=("feasible_wrong_truth", "sum"))
    summary = summary.merge(counts, on=CONDITION, validate="one_to_one")
    keys = [x for x in KEY if x != "cohort"]
    selected = joined.loc[joined.cohort.eq("selected")]
    independent = joined.loc[joined.cohort.eq("independent_same_window")]
    paired = selected.merge(independent, on=keys, suffixes=("_selected", "_independent"), validate="one_to_one")
    np.testing.assert_allclose(paired.true_prevalence_selected, paired.true_prevalence_independent)
    for metric in ("fit_tv", "abs_prevalence_bias", "mean_spikes", "auc"):
        paired[metric+"_selection_excess"] = paired[metric+"_selected"]-paired[metric+"_independent"]
    return joined, by_session, by_animal, summary, paired


def figures(by_session, summary, joined, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for cohort, label in LABELS.items():
        part = summary.loc[summary.cohort.eq(cohort) & summary.calibration.eq("generator_unselected") & summary.readout.eq("frozen_poisson")].sort_values("peak_gain")
        axes[0, 0].plot(part.peak_gain, part.fit_tv, "o-", label=label, color=COLORS[cohort])
        axes[0, 1].plot(part.peak_gain, part.abs_prevalence_bias*100, "o-", color=COLORS[cohort])
    axes[0, 0].set(title="Calibration mismatch despite fixed tuning", xlabel="Peak shared activity gain", ylabel="Mixture residual (total variation)", xticks=[1, 3, 6])
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Mismatch is not the same as prevalence error", xlabel="Peak shared activity gain", ylabel="Mean absolute prevalence error (percentage points)", xticks=[1, 3, 6])
    for i, readout in enumerate(("frozen_poisson", "oracle_gain")):
        part = by_session.loc[by_session.peak_gain.eq(6) & by_session.calibration.eq("generator_unselected") & by_session.readout.eq(readout)]
        pivot = part.pivot(index="session", columns="cohort", values="fit_tv")
        for session, row in pivot.iterrows():
            axes[1, 0].plot([i-.14, i+.14], [row.independent_same_window, row.selected], "o-", alpha=.5, color="#007f86" if i else "#b72b50", markersize=4)
    axes[1, 0].set(title="Same times and states: selection increases mismatch", xticks=[-.14, .14, .86, 1.14], xticklabels=["Independent\nfrozen", "Selected\nfrozen", "Independent\noracle", "Selected\noracle"], ylabel="Total variation; each line is one recording")
    for cohort in LABELS:
        part = joined.loc[joined.peak_gain.eq(6) & joined.calibration.eq("generator_unselected") & joined.readout.eq("frozen_poisson") & joined.cohort.eq(cohort)]
        axes[1, 1].scatter(part.true_prevalence, part.prevalence, color=COLORS[cohort], alpha=.6, s=17)
    axes[1, 1].plot([0, .6], [0, .6], "--", color="black", linewidth=1)
    axes[1, 1].set(title="Known truth versus inferred prevalence (gain 6)", xlabel="True Home fraction", ylabel="Estimated Home fraction", xlim=(0, .6), ylim=(0, .6))
    fig.suptitle("MUA selection null: PF maps unchanged in every condition", fontsize=14)
    fig.savefig(output/"regional_content_mua_selection_null.png", dpi=160)
    plt.close(fig)


def markdown_table(summary, calibration):
    part = summary.loc[summary.calibration.eq(calibration) & summary.readout.eq("frozen_poisson")]
    text = ["| Gain | Windows | Mismatch TV | AUC | Absolute prevalence error | Incompatible runs |",
            "|---|---|---:|---:|---:|---:|"]
    for r in part.sort_values(["peak_gain", "cohort"]).itertuples():
        text.append(f"| {r.peak_gain} | {LABELS[r.cohort]} | {r.fit_tv:.3f} | {r.auc:.3f} | {100*r.abs_prevalence_bias:.2f} pp | {r.incompatible_runs}/{r.simulation_runs} |")
    return "\n".join(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.input_dir
    manifest = json.loads((root/"manifest.json").read_text())
    audited = json.loads(args.audit.read_text())
    inputs = {n: root/(n+".csv") for n in ("sessions", "detection", "calibration_fits", "compatibility", "call_distributions")}
    inputs.update(manifest=root/"manifest.json", audit=args.audit, reporter=Path(__file__))
    data = {n: pd.read_csv(path) for n, path in inputs.items() if path.suffix == ".csv"}
    gates = technical_gates(manifest, data["sessions"], data["detection"], data["calibration_fits"], audited)
    joined, by_session, by_animal, summary, paired = tables(data["calibration_fits"], data["compatibility"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {"by_session": by_session, "by_animal": by_animal, "condition_summary": summary,
                   "paired_selection_effects": paired, "gate_summary": gates, "simulation_results": joined}
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir/("regional_content_mua_selection_null_"+name+".csv"), index=False)
    figures(by_session, summary, joined, args.output_dir)
    primary = by_session.loc[by_session.peak_gain.eq(6) & by_session.calibration.eq("generator_unselected") & by_session.readout.eq("frozen_poisson")]
    comparison = primary.pivot(index="session", columns="cohort", values="fit_tv")
    n_increase = int((comparison.selected > comparison.independent_same_window).sum())
    complete = gates.loc[gates.gate.eq("overall"), "status"].iloc[0]
    report = f"""# Fixed-tuning MUA-selection null

Technical verification: **{complete}**. Eight-recording design; actual completed
recordings: {len(data['sessions'])}; animals: {data['sessions'].animal.nunique()}.
{len(data['detection'])} simulation runs, with four replicas per recording in the
full run. Replicas are Monte Carlo repeats, not independent animals.

## Question and design

Can activity gain and selection on spikes break calibration even when every
cell's spatial tuning is unchanged? This is a known-truth counterfactual, not
a rescore or ground-truth labeling of real replay.

The original PF detector was loaded unchanged: all recorded sorted cells,
1-ms bins, 10-ms Gaussian SD, mean+3SD peak threshold, mean-crossing boundaries,
50-2000 ms, at least 10% of cells active. Physical speed is zero in simulation.
Decoding uses the original RUN-qualified subset and fixed first-half RUN maps.
Represented locations are constant within two-second epochs; Home prevalence
is generated at 0.30. We use the actual finite-cohort prevalence for evaluation.
This simple regional-content null does not simulate moving replay trajectories.

The fixed windows are chosen before observing spikes. Selected windows are the
last complete 20 ms of detected events. Independent controls use new Poisson
counts at those identical times, locations, and gains, without redetection.
Thus selected versus independent isolates conditioning on the measured spikes.
Fixed versus selected additionally includes location and timing selection.

## Model-consistent calibration

Independent, unselected gain-one calibration has known labels and the same
class-conditional represented-position distribution as the generator. This
removes real-RUN model mismatch from the primary counterexample.
Values below average replicas within recording, recordings within animal,
then animals equally. Counts of incompatible runs are descriptive Monte Carlo
counts, not biological sample sizes or certified coverage probabilities.

{markdown_table(summary, 'generator_unselected')}

At gain 6, selection increases mismatch relative to independent counts at the
same windows in **{n_increase}/{len(comparison)} recordings** (replica averages).
See the paired table for the size, direction, and prevalence-error consequences.

## Transfer from observed RUN calibration

This secondary comparison uses the expanded known-position RUN calibration
from the preceding experiment. Its mismatch includes ordinary simulator-versus-
real-RUN differences, so it cannot isolate selection as cleanly as the primary.

{markdown_table(summary, 'observed_RUN')}

## Interpretation

An incompatible mixture means the fixed calibration distributions cannot
reproduce the target ternary readout distribution within the declared bootstrap
ranges at zero transfer slack. It does **not** itself mean that tuning changed,
that spatial decoding failed, or that estimated Home prevalence is far from truth.
Inspect mismatch, AUC, prevalence error, and truth inclusion separately.

The oracle-gain sensitivity changes the likelihood's exposure using the true
integrated gain. It does not condition on detection, and it does not change the
gain-one calibration readout distribution. Better likelihood specification alone
therefore need not restore sensitivity/false-alarm transfer. Full oracle results
are retained in the CSVs and figure; no thresholds were changed after outcomes.

This control can establish that fixed tuning plus selection/gain is sufficient
to produce a calibration failure. It cannot establish that these mechanisms
fully explain real replay, validate an unsupervised latent-class estimator, or
recover true real-replay Home content. It suggests testing selection- and
activity-matched calibration before attributing failure to changed representations.

## Verification and provenance

Independent audit: {audited['simulation_runs']} simulated trains,
{audited['generated_spikes']:,} spikes, {audited['selected_windows']:,} retained
endpoints; original detector rerun, timestamps and counts independently checked,
likelihoods recalculated, unchanged maps and input/output hashes verified.
Source commit: `{manifest['code_commit']}`; dirty: `{manifest['git_dirty']}`.
New code is uncommitted; exact source SHA256 hashes in the measurement manifest
are required to identify it, rather than the commit alone.

![Selection null](regional_content_mua_selection_null.png)
"""
    (args.output_dir/"regional_content_mua_selection_null_report.md").write_text(report)
    provenance = build_script_provenance(input_paths=inputs, cwd=Path(__file__).resolve().parents[1])
    provenance["outputs"] = {path.name: file_sha256(path) for path in args.output_dir.iterdir() if path.is_file()}
    (args.output_dir/"report_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
