"""Non-rescoring validation, power and real-data status for regional calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


from hipporeplayimm.selection_matched_regional import threshold
from scripts._provenance import build_script_provenance, file_sha256


def summarize_validation(fits):
    selected = fits.loc[fits.phase.eq("validation") & ~fits.requested_prevalence.eq(.38)].copy()
    selected["inside_budget"] = selected.absolute_error.le(.05)
    keys = ["animal", "session", "population", "family", "side", "generator", "requested_prevalence"]
    grouped = selected.groupby(keys, as_index=False).agg(
        replicas=("replica", "size"), valid_fits=("status", lambda x: x.eq("fit").sum()),
        mean_absolute_error=("absolute_error", "mean"), median_absolute_error=("absolute_error", "median"),
        p90_absolute_error=("absolute_error", lambda x: x.quantile(.90)),
        signed_bias=("signed_error", "mean"), budget_coverage=("inside_budget", "mean"),
        flag_rate=("flagged", "mean"), mean_events=("events", "mean"))
    grouped["budget_pass"] = grouped.budget_coverage.ge(.90) & grouped.mean_absolute_error.le(.05) & grouped.valid_fits.eq(grouped.replicas)
    grouped["false_flag_tolerance_pass"] = grouped.flag_rate.le(.10)
    return grouped


def paired_checks(fits):
    keys = ["animal", "session", "family", "phase", "requested_prevalence", "generator", "perturbation", "replica"]
    high = fits.loc[fits.side.eq("high")]
    low = fits.loc[fits.side.eq("low")]
    pair = high.merge(low, on=keys, suffixes=("_high", "_low"), validate="one_to_one")
    pair["estimated_difference"] = pair.prevalence_high-pair.prevalence_low
    pair["same_true_content"] = np.isclose(pair.true_prevalence_high, pair.true_prevalence_low)
    if len(pair) and not pair.same_true_content.all():
        raise ValueError("same-panel populations do not share retained truth")
    cutoffs = []
    for (session, family), group in pair.loc[pair.phase.eq("null")].groupby(["session", "family"]):
        limit = max(threshold(g.estimated_difference.abs()) for _, g in group.groupby("requested_prevalence"))
        cutoffs.append({"session": session, "family": family, "difference_cutoff": limit})
    if not cutoffs:
        return pair, pd.DataFrame(), pd.DataFrame()
    pair = pair.merge(pd.DataFrame(cutoffs), on=["session", "family"], validate="many_to_one")
    pair["within_shared_budget"] = pair.estimated_difference.abs().le(.05)
    pair["difference_flag"] = pair.estimated_difference.abs() > pair.difference_cutoff
    validation = pair.loc[pair.phase.eq("validation") & pair.generator.eq("mix") & ~pair.requested_prevalence.eq(.38)]
    agreement = validation.groupby(["animal", "session", "family"], as_index=False).agg(
        replicas=("replica", "size"), agreement_fraction=("within_shared_budget", "mean"),
        false_difference_flag_rate=("difference_flag", "mean"), difference_cutoff=("difference_cutoff", "first"))
    agreement["agreement_pass"] = agreement.agreement_fraction.ge(.90)
    power_rows = []
    for (animal, session, family), group in pair.groupby(["animal", "session", "family"]):
        g = group.loc[group.phase.eq("validation") & group.generator.eq("mix")]
        a = g.loc[g.requested_prevalence.eq(.38)].set_index("replica")
        b = g.loc[g.requested_prevalence.eq(.30)].set_index("replica")
        if not len(a) or set(a.index) != set(b.index):
            power_rows.append({"animal": animal, "session": session, "family": family, "status": "missing_panels", "power_pass": False})
            continue
        contrast = a.prevalence_high-b.prevalence_low
        truth = a.true_prevalence_high-b.true_prevalence_low
        limit = float(g.difference_cutoff.iloc[0])
        fraction = float((contrast > limit).mean())
        power_rows.append({"animal": animal, "session": session, "family": family, "status": "available",
            "independent_panel_replicas": len(contrast), "intended_difference": .08, "mean_true_retained_difference": truth.mean(),
            "min_true_retained_difference": truth.min(), "max_true_retained_difference": truth.max(),
            "mean_estimated_difference": contrast.mean(), "positive_difference_detection_fraction": fraction,
            "difference_cutoff": limit, "power_pass": fraction >= .80,
            "interpretation": "separate_cohorts_not_different_truth_for_shared_cells_in_one_event"})
    return pair, agreement, pd.DataFrame(power_rows)


def perturbations(fits):
    p = fits.loc[fits.phase.eq("perturbation")].copy()
    p["outside_budget"] = p.absolute_error.gt(.05)
    p["flagged_bad"] = p.flagged & p.outside_budget
    result = p.groupby(["animal", "session", "population", "perturbation"], as_index=False).agg(
        replicas=("replica", "size"), flag_rate=("flagged", "mean"), budget_failure_rate=("outside_budget", "mean"),
        mean_absolute_error=("absolute_error", "mean"), large_error_count=("outside_budget", "sum"),
        flagged_large_error_count=("flagged_bad", "sum"))
    result["sensitivity_to_large_errors"] = result.flagged_large_error_count/result.large_error_count.replace(0, np.nan)
    return result


def real_status(real):
    r = real.copy()
    r["matched_null_flag"] = r.fit_tv > r.tv_cutoff
    r["calibrated_claim_available"] = r.calibrated_estimate_available & ~r.matched_null_flag
    r["status_for_claims"] = np.where(~r.calibrated_estimate_available, "synthetic_validation_failed",
        np.where(r.matched_null_flag, "real_distribution_flagged", "conditional_calibration_available"))
    r["enrichment_claim"] = "not_established"
    residual_names = ["residual_against", "residual_neutral", "residual_for"]
    if all(name in r for name in residual_names):
        r["largest_excess_readout"] = r[residual_names].idxmax(axis=1).str.replace("residual_", "", regex=False)
    # A bounded point error is not a confidence interval or an enrichment test.
    return r


def plot_validation(validation, real, output):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    full = validation.loc[validation.population.eq("full")]
    kinds = ["stationary", "moving", "late_jump", "mix"]
    for i, kind in enumerate(kinds):
        g = full.loc[full.generator.eq(kind)].groupby("session").mean(numeric_only=True)
        offsets = np.linspace(-.16, .16, len(g))
        ax[0].scatter(i+offsets, 100*g.mean_absolute_error, s=30, label=kind)
        ax[1].scatter(i+offsets, g.budget_coverage, s=30)
    ax[0].axhline(5, color="black", linestyle="--", linewidth=1)
    ax[1].axhline(.90, color="black", linestyle="--", linewidth=1)
    ax[0].set(ylabel="Mean absolute prevalence error (pp)", title="Full-population validation", xticks=range(4), xticklabels=kinds)
    ax[1].set(ylabel="Fraction with error <=5 percentage points", title="Proposed budget: >=90%", xticks=range(4), xticklabels=kinds, ylim=(-.02, 1.02))
    q = real.loc[real.population.eq("full")].sort_values("session")
    y = np.arange(len(q))
    ax[2].barh(y, q.fit_tv, color=np.where(q.matched_null_flag, "#b72b50", "#007f86"), alpha=.8, label="Real TV")
    ax[2].scatter(q.tv_cutoff, y, marker="|", s=120, color="black", label="Null cutoff")
    ax[2].set(yticks=y, yticklabels=q.session, xlabel="Distribution mismatch (TV)", title="Fit diagnostic, not content truth")
    ax[2].invert_yaxis()
    ax[2].legend(fontsize=8, loc="lower right")
    ax[0].set_ylim(bottom=0)
    fig.savefig(output/"selection_matched_regional_calibration.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {n:args.input_dir/(n+".csv") for n in ("sessions", "validation_and_null", "real_descriptive", "detector_equivalence")}
    paths.update(manifest=args.input_dir/"manifest.json", audit=args.audit, reporter=Path(__file__))
    data = {n:read_csv(path) for n,path in paths.items() if path.suffix == ".csv"}
    manifest = json.loads(paths["manifest"].read_text())
    audit = json.loads(args.audit.read_text())
    fits = data["validation_and_null"]
    validation = summarize_validation(fits)
    paired, agreement, power = paired_checks(fits)
    sensitivity = perturbations(fits)
    real = real_status(data["real_descriptive"])
    rows = {"validation_by_population": validation, "paired_validation": paired, "paired_agreement": agreement,
                "eight_point_power": power, "perturbation_sensitivity": sensitivity, "real_status": real}
    checks = {"measurement_complete": manifest["status"] == "complete",
        "independent_audit_pass": audit["status"] == "pass", "all_sessions_present": len(data["sessions"]) == 8,
        "original_detector_equivalence": len(data["detector_equivalence"]) == 24 and data["detector_equivalence"].status.eq("pass").all(),
        "all_population_budgets_pass": len(validation)>0 and validation.budget_pass.all(),
        "same_content_population_agreement": len(agreement)>0 and agreement.agreement_pass.all(),
        "eight_point_difference_power": len(power)>0 and power.power_pass.all(),
        "mixed_null_false_flag_validation": len(real)>0 and real.false_flag_validation_pass.all()}
    checks["validated_calibration_ready"] = all(checks.values())
    gates = pd.DataFrame([{"gate": k, "status": "pass" if v else "fail"} for k,v in checks.items()])
    rows["gate_summary"] = gates
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in rows.items():
        frame.to_csv(args.output_dir/(name+".csv"), index=False)
    plot_validation(validation, real, args.output_dir)
    animal_validation = validation.groupby(["animal", "population", "generator"], as_index=False).mean(numeric_only=True)
    animal_validation.to_csv(args.output_dir/"validation_by_animal.csv", index=False)
    summary = animal_validation.groupby(["population", "generator"], as_index=False).agg(
        mean_absolute_error=("mean_absolute_error", "mean"), budget_coverage=("budget_coverage", "mean"), flag_rate=("flag_rate", "mean"))
    summary.to_csv(args.output_dir/"descriptive_validation_summary.csv", index=False)
    lines = ["# Selection-matched regional calibration", "", "## Decision", "",
        f"Validated calibration ready: **{str(checks['validated_calibration_ready']).lower()}**.",
        f"Populations meeting synthetic budget and mixed false-flag checks: {int(real.calibrated_estimate_available.sum())}/{len(real)}.",
        f"Paired groups meeting agreement: {int(agreement.agreement_pass.sum())}/{len(agreement)}; eight-point power: {int(power.power_pass.sum())}/{len(power)}.",
        "No hc-11 transfer was run. Failed populations cannot support calibrated enrichment or reconciliation claims.",
        "", "## Full-population validation", "", "| Generator | Mean absolute error | Fraction within 5 pp | Flag rate |",
        "|---|---:|---:|---:|"]
    for row in summary.loc[summary.population.eq("full")].itertuples():
        lines.append(f"| {row.generator} | {100*row.mean_absolute_error:.2f} pp | {row.budget_coverage:.1%} | {row.flag_rate:.1%} |")
    lines += ["", "Rows average session-by-prevalence summaries; they are descriptive, not independent-animal confidence intervals.",
        "The gates apply separately to every session/population/generator/prevalence, not just these averages.",
        "The mix row is matched-mixture validation. Pure stationary, moving and late-jump rows stress transfer from the same frozen three-generator calibration; they are not generator-specific calibrations.",
        "", "## What is matched", "",
        "Each original candidate's exact whole-population spike timestamps and the surrounding count trace are preserved. Cell identities are simulated from unchanged first-half RUN maps. The native detector and terminal complete-window rule are preserved; active-cell gates are recomputed and independently audited.",
        "The simulation is conditional on the frozen 200 candidate templates per session; it does not claim to reproduce the unconditional candidate discovery process or entirely new activity profiles.",
        "Calibration pools stationary, moving and late-jump generators equally before selection. The primary readout remains the frozen ternary regional Bayes-factor call. Null and validation replicas use disjoint random seeds; real-data mismatch did not set thresholds.",
        "", "## Perturbations and power", "",
        "The sensitivity table distinguishes flags, actual prevalence errors, and whether flags catch large errors. A common gain is exactly invisible under fixed-total allocation; its zero incremental sensitivity is expected and verified, not evidence for robustness to arbitrary gain changes.",
        "The eight-point power control compares separate .38/.30 panels. It is not evidence for two distinct simultaneous truths in overlapping cell populations. Actual retained truth differences are reported after active-cell selection.",
        "", "## Real-data diagnostics", "", "| Session (full population) | Real TV | Null cutoff | Flag | Calibration validated |", "|---|---:|---:|---|---|"]
    for row in real.loc[real.population.eq("full")].itertuples():
        lines.append(f"| {row.session} | {row.fit_tv:.3f} | {row.tv_cutoff:.3f} | {row.matched_null_flag} | {row.calibrated_estimate_available} |")
    lines += ["", "These are distribution diagnostics. A real flag does not identify a mechanism or prove tuning drift. A non-flag does not validate content if synthetic estimation or diagnostic sensitivity failed.",
        "The real table retains nuisance mixture estimates explicitly as descriptive and leaves calibrated estimates unavailable after failed validation. No confidence interval or enrichment test is manufactured from the point-error budget. The area fraction is only a uniform endpoint-location reference, not a selected-event biological null.",
        "Original matched groups and whole-tetrode groups are retained with shared-cell counts; they are not independent detectors. Their older full-RUN eligibility restriction remains a limitation.",
        "", "## Verification", "", f"Audit: {audit['status']}; source commit `{manifest['code_commit']}`, dirty `{manifest['git_dirty']}`. Exact source/input hashes and simulation seeds identify the run; code is uncommitted.",
        "", "![Validation and flags](selection_matched_regional_calibration.png)"]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n")
    provenance = build_script_provenance(input_paths=paths, cwd=Path(__file__).resolve().parents[1])
    provenance["outputs"] = {p.name:file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()}
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(gates.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
