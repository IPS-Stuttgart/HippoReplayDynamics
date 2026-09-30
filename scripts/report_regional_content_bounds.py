#!/usr/bin/env python3
"""Non-rescoring report of regional identification, calibration and failure limits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256


def aggregate(bounds, scores, conditional):
    keys = ["source", "calibration", "transfer_slack"]
    work = bounds.copy()
    work["feasible"] = work.status.eq("feasible").astype(float)
    work["covered"] = pd.to_numeric(work.covers_truth, errors="coerce")
    session = work.groupby(["animal", "session"]+keys)[["width", "useful", "feasible", "covered"]].mean().reset_index()
    animal = session.groupby(["animal"]+keys)[["width", "useful", "feasible", "covered"]].mean().reset_index()
    summary = animal.groupby(keys)[["width", "useful", "feasible", "covered"]].mean().reset_index()
    metric = [k for k in scores if k.endswith(("prevalence", "brier", "log_loss")) or k in ("naive_mean_mass", "converged", "multistart_range")]
    score_session = scores.groupby(["animal", "session", "source"])[metric].mean().reset_index()
    score_animal = score_session.groupby(["animal", "source"])[metric].mean().reset_index()
    score_summary = score_animal.groupby("source")[metric].mean().reset_index()
    c = conditional.copy()
    c["weighted_width"] = c.width*c.events
    c["supported_home"] = (c.lower > .5)*c.events
    c["supported_elsewhere"] = (c.upper < .5)*c.events
    ck = ["animal", "session", "calibration", "transfer_slack"]
    c = c.groupby(ck)[["events", "weighted_width", "supported_home", "supported_elsewhere"]].sum().reset_index()
    for key in ("weighted_width", "supported_home", "supported_elsewhere"):
        c[key] /= c.events
    ca = c.groupby(["animal", "calibration", "transfer_slack"])[["weighted_width", "supported_home", "supported_elsewhere"]].mean().reset_index()
    cs = ca.groupby(["calibration", "transfer_slack"])[["weighted_width", "supported_home", "supported_elsewhere"]].mean().reset_index()
    return summary, animal, score_summary, score_animal, cs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "pass" or audit["input_file_sha256"]["manifest"] != file_sha256(args.input_dir/"manifest.json"):
        raise ValueError("matching passing independent audit required")
    for path, sha in audit["result_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("audited table changed: "+path)
    names = ("bounds", "scores", "conditional_bounds", "sessions", "matched_home_bayes_factor_summary")
    inputs = {name: args.input_dir/f"{name}.csv" for name in names}
    frames = {name: pd.read_csv(path) for name, path in inputs.items()}
    b, scores, cond, sessions, stage = (frames[name] for name in names)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary, animal, score_summary, score_animal, event_summary = aggregate(b, scores, cond)
    for name, frame in (("bounds_summary", summary), ("bounds_by_animal", animal),
                        ("score_summary", score_summary), ("scores_by_animal", score_animal),
                        ("conditional_bounds_summary", event_summary)):
        frame.to_csv(args.output_dir/f"{name}.csv", index=False)
    real = b.loc[b.source.eq("real")]
    real.to_csv(args.output_dir/"real_session_compatibility_bounds.csv", index=False)
    stage = stage.loc[stage.encoding.eq("full_run") & stage.cohort.eq("all_fixed_candidates")]
    columns = ["mean_home_mass", "positive_fraction", "negative_fraction", "near_prior_fraction", "silent_fraction"]
    stage_animal = stage.groupby(["animal", "population"])[columns].mean().reset_index()
    stage_summary = stage_animal.groupby("population")[columns].mean()
    stage_summary.to_csv(args.output_dir/"matched_home_equal_animal_summary.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), layout="constrained")
    ax = axes[0, 0]
    colors = {0.: "#197f79", .1: "#a95b25"}
    for slack in (0., .1):
        d = real.loc[real.calibration.eq("synthetic_native") & real.transfer_slack.eq(slack)].sort_values("session")
        y = np.arange(len(d)) + (-.10 if slack == 0 else .10)
        ax.hlines(y, d.lower, d.upper, color=colors[slack], linewidth=3, label=f"Synthetic calibration, slack {slack:.2f}")
        ax.scatter(d.upper, y, color=colors[slack], s=18)
    ax.set(yticks=np.arange(len(d)), yticklabels=d.session, xlim=(-.02, 1.02),
           xlabel="Compatible Home prevalence", title="A. Real fixed endpoints: assumptions matter")
    ax.set_ylim(-.7, 9.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(.02, .98, "RUN-derived bounds: [0, 1] in all sessions", transform=ax.transAxes, va="top", fontsize=9)
    ax = axes[0, 1]
    short = {"native": "Native", "eightfold": "8x spikes", "map_drift_eightfold": "8x + map drift", "shared_assembly_eightfold": "8x + shared assembly"}
    for source, label in short.items():
        d = summary.loc[summary.source.eq(source) & summary.transfer_slack.le(.20)]
        ax.plot(d.transfer_slack, d.width, marker="o", label=label)
    ax.axhline(.20, ls=":", color="black", label="Frozen utility width")
    ax.set(xlabel="Allowed calibration transfer error", ylabel="Mean prevalence interval width",
           ylim=(0, 1.04), title="B. Known-truth bounds remain broad")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    d = score_summary.set_index("source").loc[list(short)]
    x = np.arange(len(d))
    ax.bar(x-.19, d.latent_prevalence, .38, color="#71658a", label="Blind latent-class estimate")
    ax.bar(x+.19, d.naive_mean_mass, .38, color="#819e82", label="Mean decoded Home mass")
    ax.axhline(.30, color="black", ls="--", label="True simulated prevalence")
    ax.set(xticks=x, xticklabels=list(short.values()), ylabel="Home prevalence / posterior mass",
           title="C. Shared activity misleads agreement-based inference", ylim=(0, .75))
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    bottom = np.zeros(2)
    for column, color, label in (("negative_fraction", "#bc6558", "Evidence against Home"),
                                  ("near_prior_fraction", "#c9ced0", "Within BF 1/3 to 3"),
                                  ("positive_fraction", "#438aab", "Evidence for Home")):
        value = stage_summary.loc[["high", "low"], column].to_numpy()
        ax.bar([0, 1], value, bottom=bottom, color=color, label=label)
        bottom += value
    ax.set(xticks=[0, 1], xticklabels=["Home-rich", "Home-poor"], ylabel="Fraction of fixed endpoint readouts",
           title="D. A near-prior mean is not event-level uncertainty", ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Regional content: identification depends on calibration transfer", fontsize=15)
    fig.savefig(args.output_dir/"regional_content_bounds.png", dpi=170)
    fig.savefig(args.output_dir/"regional_content_bounds.pdf")
    plt.close(fig)
    real_zero = real.loc[real.calibration.eq("synthetic_native") & real.transfer_slack.eq(0)]
    syn = summary.loc[summary.source.eq("native") & summary.transfer_slack.eq(0)].iloc[0]
    wrong = score_summary.loc[score_summary.source.eq("shared_assembly_eightfold")].iloc[0]
    source_manifest = json.loads((args.input_dir/"manifest.json").read_text())
    n_targets = len(sessions)*4*source_manifest["parameters"]["replicates"]*source_manifest["parameters"]["target_events"]
    lines = ["# Regional content bounds: completed feasibility test", "",
        "**Decision: no validated useful real-content remedy at the frozen operating point.**",
        "The implementation is technically verified. Narrow-looking latent-class probabilities are not calibrated spatial truth.", "",
        (f"Eight PF sessions/four rats; {sessions.real_events.sum()} unchanged last-complete 20-ms MUA endpoints, "
        f"{sessions.run_events.sum()} known-position RUN controls; {n_targets:,} fresh synthetic target windows."),
        "Four disjoint RUN-coverage-ordered views. Different populations, support and denominators from the separate matched-Home audit.", "",
        "## What the bounds show", "",
        "- Without a transfer/calibration assumption, all prevalence and event bounds are [0,1].",
        ("- RUN-calibrated real and held-out RUN bounds are [0,1] in every session. Only "
        f"{sessions.calibration_run_home.min()}-{sessions.calibration_run_home.max()} Home examples occur in each 100-window calibration half."),
        f"- Exact synthetic-calibration transfer gives real prevalence bounds [0, upper], with upper {real_zero.upper.min():.3f}-{real_zero.upper.max():.3f}.",
        "- No real interval meets the prospectively frozen width <=.20 utility criterion. This is a diagnostic convention, not a biological threshold.",
        f"- Native known-truth simulation bounds have mean width {syn.width:.3f}; covering truth with broad intervals does not demonstrate informative recovery.",
        "- Allowing .10 absolute drift in each calibration probability makes all real event-level conditional bounds [0,1].",
        "- Bonferroni over 81 response patterns is conservative. This result does not prove that all alternative confidence constructions or larger calibration sets fail.", "",
        "## The latent-class comparison", "",
        "Conditional-independent categorical EM sometimes improves Brier scores in matching simulations, but agreement does not establish spatial meaning.",
        (f"Under the shared-assembly stress, true prevalence=.300; mean fitted prevalence={wrong.latent_prevalence:.3f}. "
        f"Its Brier score={wrong.latent_brier:.3f}, versus {wrong.oracle_prevalence_brier:.3f} for the known-prevalence constant baseline."),
        "The stress has one true spatial state and a different shared activation state. It does not create genuine multiplexed replay.",
        f"Numerically converged fits: {int(scores.loc[scores.source.ne('real'), 'converged'].sum())}/{int(scores.source.ne('real').sum())}. This is not proof of a global optimum or model validity.", "",
        "## Correction to the proposal's Home interpretation", "",
        (f"The old matched-Home mass averages reproduce {100*stage_summary.loc['high','mean_home_mass']:.2f}% vs "
        f"{100*stage_summary.loc['low','mean_home_mass']:.2f}% (four sessions, three rats; equal-rat weighting)."),
        (f"Home-poor readouts have {100*stage_summary.loc['low','negative_fraction']:.1f}% BF<1/3, "
        f"{100*stage_summary.loc['low','near_prior_fraction']:.1f}% BF between1/3 and3, and "
        f"{100*stage_summary.loc['low','positive_fraction']:.1f}% BF>3."),
        "Thus the low mean is not simply universal non-detection. These are decoder evidence categories, not verified true/false content calls.", "",
        "## Assumptions and boundaries", "",
        "Calibration/error bounds require transfer of full ternary emission distributions. View-error dependence is allowed; shared misspecification is not thereby corrected.",
        "The .05 error budget is per simulated analysis, under independent windows. Real MUA and RUN data are temporally dependent: real outputs are compatibility intervals, not certified biological confidence intervals.",
        "Synthetic truth positions are uniform within Home/complement; spike totals are sampled from real endpoints and allocated conditionally. Shared assembly and quarter-map drift are explicit stresses, not a complete biological generator.",
        "Calibration is reused across twelve target panels per source/condition. Coverage frequencies are conditional stress diagnostics, not independent calibration experiments.",
        "RUN-Q4 calibration is not count-matched to replay; transporting it is explicitly an unverified assumption. Bounds concern mean position within the fixed RUN window, not a relocated endpoint.",
        "Region annotations are inferred. No event was moved, reclassified as replay, or selected by a favorable bound. No hc11 confirmation or speed-uniformity claim follows.", "",
        "## Verification", "",
        "Passing independent reconstruction: all recorded/simulated counts, all population readouts and calibration intervals, all prevalence bounds and observed conditional-pattern bounds. Raw spike sorting and biological replay truth remain unaudited.",
        "See reconstruction.json and manifest.json for exact source/code hashes. Complete source snapshots remain on gpuserver6000.", "",
        "## What remains worth testing", "",
        "A future test would need substantially better independent regional calibration, validated transfer, or an alternative informative readout. Increasing certainty by assuming independent errors is not itself a remedy.",
        "This experiment answers the proposed frozen feasibility question negatively; it does not prove regional content can never be bounded.", ""]
    (args.output_dir/"regional_content_bounds_report.md").write_text("\n".join(lines))
    inputs.update(audit=args.audit, manifest=args.input_dir/"manifest.json", script=Path(__file__))
    provenance = build_script_provenance(input_paths=inputs)
    provenance["outputs"] = {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()}
    (args.output_dir/"report_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()



