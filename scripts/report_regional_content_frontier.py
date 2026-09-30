#!/usr/bin/env python3
"""Non-rescoring regional calibration and discrimination report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256


def summarize_animals(frame, keys, metrics):
    animal = frame.groupby([*keys, "animal"], dropna=False)[metrics].mean().reset_index()
    pooled = animal.groupby(keys, dropna=False)[metrics].mean().reset_index()
    return animal, pooled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.input_dir/"manifest.json").read_text())
    audit = json.loads(args.audit.read_text())
    if source["status"] != "complete" or audit["status"] != "pass":
        raise ValueError("complete audited input required")
    for name, digest in audit["result_sha256"].items():
        if file_sha256(args.input_dir/name) != digest:
            raise ValueError("audited table changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    names = ("sessions", "run_calibration", "prevalence_fits", "prevalence_compatibility", "discrimination_frontier", "populations")
    frames = {name: pd.read_csv(args.input_dir/(name+".csv")) for name in names}
    sessions, run, fits, bounds, frontier = [frames[x] for x in names[:5]]
    for name in ("sessions", "run_calibration", "prevalence_fits", "prevalence_compatibility", "populations"):
        frames[name].to_csv(args.output_dir/(name+".csv"), index=False)
    keys = ["population", "duration_ms", "multiplier", "generator", "readout"]
    animal, mean = summarize_animals(frontier, keys, ["auc", "balanced_error", "balanced_brier", "balanced_log_loss", "mean_spikes", "cells", "home_peak_cells"])
    animal.to_csv(args.output_dir/"frontier_by_animal.csv", index=False)
    mean.to_csv(args.output_dir/"frontier_summary.csv", index=False)
    full = frontier.query('population == "full" and duration_ms == 20 and multiplier == 1 and readout == "matched_likelihood"')
    validation = run.query('method == "native_run" and subset == "validation"')
    calibration = run.query('method == "native_run" and subset == "calibration"')
    real = bounds.query('target == "real" and calibration == "native_run" and slack == 0')
    heldout = bounds.query('target == "heldout_run" and calibration == "native_run" and slack == 0')
    covered = (heldout.lower <= heldout.empirical_truth_prevalence) & (heldout.upper >= heldout.empirical_truth_prevalence)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    x = np.arange(len(calibration))
    ax.bar(x, calibration.home_windows, color="#287c8e")
    ax.set_xticks(x, calibration.session, rotation=45, ha="right")
    ax.set(title="A. Expanded RUN calibration", ylabel="Home windows (20 ms)")
    ax.text(.02, .96, "Alternate whole running bouts; first-half maps frozen", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_ylim(0, calibration.home_windows.max()*1.2)
    ax = axes[0, 1]
    labels = {"poisson": "Exact Poisson / matched likelihood", "conditional_multinomial": "Fixed observed totals / matched likelihood"}
    for gen, color in (("poisson", "#a14770"), ("conditional_multinomial", "#287c8e")):
        d = mean.query('population == "full" and duration_ms == 20 and readout == "matched_likelihood" and generator == @gen')
        ax.plot(d.multiplier, d.auc, "o-", color=color, label=labels[gen])
    ax.axhline(.5, color=".5", linestyle=":")
    ax.set(title="B. Synthetic full-population discrimination", xlabel="Spike-total / gain multiplier", ylabel="AUC", ylim=(.48, 1.01), xticks=[1, 2, 4, 8])
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 0]
    compare = validation[["session", "auc"]].merge(full.query('generator == "poisson"')[["session", "auc"]], on="session", suffixes=("_run", "_synthetic"))
    x = np.arange(len(compare))
    ax.plot(x, compare.auc_synthetic, "o-", label="Synthetic, native mean count", color="#a14770")
    ax.plot(x, compare.auc_run, "s-", label="Actual held-out RUN", color="#287c8e")
    ax.set_xticks(x, compare.session, rotation=45, ha="right")
    ax.set(title="C. Simulation is not observed RUN or replay", ylabel="AUC", ylim=(.48, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1, 1]
    gap = fits.query('calibration == "native_run"').pivot(index="session", columns="target", values="fit_tv")
    x = np.arange(len(gap))
    ax.bar(x-.18, gap.heldout_run, width=.36, color="#287c8e", label="Held-out RUN")
    ax.bar(x+.18, gap.real, width=.36, color="#a14770", label="Real candidate endpoints")
    ax.set_xticks(x, gap.index, rotation=45, ha="right")
    ax.set(title="D. Calibration-mixture mismatch", ylabel="Category-distribution total variation")
    ax.legend(fontsize=8)
    fig.suptitle("PF regional content: information versus calibration transfer", fontsize=15)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir/("regional_content_frontier."+ext), dpi=170)
    plt.close(fig)

    rows = []
    for g in ("poisson", "conditional_multinomial"):
        d = full[full.generator.eq(g)]
        rows.append(f"| {g} | {d.auc.mean():.3f} | {d.auc.min():.3f}-{d.auc.max():.3f} | {d.balanced_error.mean():.3f} |")
    long = mean.query('population == "full" and multiplier == 1 and readout == "matched_likelihood" and generator == "conditional_multinomial"')
    text = f"""# Regional content: expanded calibration and discrimination

## Result

The earlier broad bounds did not establish that these populations lack regional
information. Full-population simulated Home discrimination is strong under the
specified models, and actual held-out RUN is informative but worse. Increasing
calibration data and removing the 81-pattern agreement model does NOT establish
calibrated replay prevalence: direct RUN-to-candidate calibration is incompatible
at zero transfer slack in {int(real.status.eq('incompatible').sum())}/{len(real)} sessions.
This is a transfer/selection diagnostic, not proof of replay-specific tuning change,
multiplexing, uniform replay speed, or a universal impossibility result.

## Data and frozen estimands

- {len(sessions)} PF sessions, {sessions.animal.nunique()} animals, {int(sessions.real_events.sum())} unchanged candidate endpoints.
- {int(sessions.run_windows.sum()):,} eligible second-half RUN windows, {int(sessions.home_windows.sum()):,} inside Home.
- Calibration halves have {calibration.home_windows.min()}-{calibration.home_windows.max()} Home windows in {calibration.home_bouts.min()}-{calibration.home_bouts.max()} Home-containing bouts per session, rather than 2-11 windows.
- All maps and unit QC use first-half RUN only; alternate running bouts calibrate/evaluate.
- {int(frontier.generated_windows.sum()/2):,} simulated windows (each decoded two ways). Synthetic repeats are not animals.
- No spatial replay truth was supplied. Home remains an inferred 20-cm disc; only RUN/simulation has known position.

## Discrimination, not guaranteed endpoint recovery

| Full population, 20 ms, native spike setting | Mean AUC | Session range | Balanced decision error |
|---|---:|---:|---:|
{chr(10).join(rows)}

Mean observed held-out RUN AUC is {validation.auc.mean():.3f}, range
{validation.auc.min():.3f}-{validation.auc.max():.3f}. The likelihood-ratio statistic
is informative, but individual windows can still be wrong or uncertain. AUC
measures ranking, not calibrated prevalence, and not the chance each event is right.
In the exact Poisson simulation, count is random; a known gain aligns its mean
to native endpoint activity. In the conditional simulation, sampled totals are
preserved and the likelihood removes the position-dependent total rate. The frozen
gain-one Poisson decoder is a separate, sometimes mismatched, comparator.

Coverage-rich/poor halves and Home-peak-cell removals are in frontier_summary.csv.
Original targeted populations are included only after intersection with current
training-only QC; they can lose cells and are NOT the original unchanged matched
populations. Removed Home peaks are not proof of removing all information about
Home: broad fields, negative evidence and other cells can still discriminate.

## What happened to the calibration bounds?

Native-RUN-calibrated zero-slack intervals cover actual validation RUN prevalence
in {int(covered.sum())}/{len(covered)} sessions. This is a descriptive check, not
estimated frequentist coverage from eight independent experiments.
On real endpoints, {int(real.status.eq('incompatible').sum())}/{len(real)} sets are
incompatible: no prevalence mixture of the calibration Home/non-Home category
distributions fits within the block-bootstrap compatibility ranges.
Fitted mixture prevalences are retained for debugging, NOT reported as corrected
biological content. Incompatible bounds remain missing, never converted to zero.

The remaining assumptions include within-region position distribution, activity,
candidate-selection conditioning, RUN-to-replay tuning stability, temporal block
choices and the silence abstention convention. Changing these can change the
calibration. More RUN samples do not verify transfer; nor do they prove the
transfer failure is specifically a Home-cell participation mechanism.

## Thinning limitation

Mean full-population endpoint count across sessions: {sessions.mean_native_endpoint_spikes.mean():.2f};
mean RUN count: {sessions.mean_run_spikes.mean():.2f}; after thinning: {sessions.mean_thinned_spikes.mean():.2f}.
Desired endpoint totals exceed available RUN spikes in
{100*sessions.thinning_unattainable_fraction.min():.1f}-{100*sessions.thinning_unattainable_fraction.max():.1f}% of windows.
Binomial thinning cannot create spikes, so this is NOT successfully count-matched
RUN. Exposure-adjusted thinning is provided separately, not sold as a remedy.
Even its retention probability depends on observed count, so it is diagnostic.

## Longer windows are a different question

Full-population conditional-likelihood AUCs for the constant-region terminal-segment
simulation: {', '.join(f'{int(d.duration_ms)} ms: {d.auc:.3f}' for d in long.itertuples())}.
These use eligible observed terminal-count pools, with short-event omissions
recorded. The generator holds the region fixed throughout. They do not validate
the original endpoint, late jumps, or moving replay content.

## Interpretation and next step

The information-limited and transfer-limited questions are distinct. These results
favor testing activity/selection-conditioned calibration before fitting the
proposed cell-participation correction. Any such correction needs event-held-out
predictive tests plus known-truth perturbation controls. A scalar global gain
cannot absorb every shared assembly, and a better predictive likelihood alone
does not certify spatial truth. The proposed distance-to-Home discriminator also
requires assumptions about content prevalence and position-dependent false alarms;
it is not independent ground truth. Neither follow-up was run here.

The ternary mixture/bootstrap analysis uses full-population readouts and avoids
conditional independence between quarter populations. Its intervals are model-based
compatibility ranges, NOT certified 95% biological CIs. No claim of publishability
or proof that no alternative decoder can work follows from this experiment.

## Verification

Independent reconstruction audit: {audit['status']}. Source counts, RUN position,
fixed bout partitions, simulated RNG draws, likelihood ratios, AUC/error/losses,
and pooled table hashes checked. The audit does not certify biological truth or
bootstrap coverage. Protocol, input/code hashes, detailed session tables and
unmodified candidate identities are retained. This report does not rescore events.
"""
    (args.output_dir/"regional_content_frontier_report.md").write_text(text)
    manifest = build_script_provenance(input_paths={**{name: args.input_dir/(name+".csv") for name in names},
        "audit": args.audit, "source_manifest": args.input_dir/"manifest.json", "reporter": Path(__file__)}, cwd=ROOT)
    manifest.update(outputs={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    (args.output_dir/"report_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
