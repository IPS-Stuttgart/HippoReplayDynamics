"""Non-rescoring synthesis of frozen diagnostic and post-hoc zero-mass control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.diagnose_regional_readout_endpoint import read_csv


def verified_manifest(path):
    m = json.loads((path/"manifest.json").read_text())
    if m["status"] != "complete":
        raise ValueError("incomplete source")
    for name, expected in m["outputs"].items():
        if file_sha256(path/name) != expected:
            raise ValueError(f"changed output {path/name}")
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-dir", type=Path, required=True)
    p.add_argument("--zero-mass-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    primary = verified_manifest(args.primary_dir)
    zero = verified_manifest(args.zero_mass_dir)
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "pass" or not zero["posthoc_control"]:
        raise ValueError("completed audited primary and labeled posthoc control required")
    assert audit["provenance"]["input_file_sha256"]["manifest"] == file_sha256(args.primary_dir/"manifest.json")
    desc = pd.concat([read_csv(d/"descriptive_summary.csv") for d in (args.primary_dir, args.zero_mass_dir)], ignore_index=True)
    summary = pd.concat([read_csv(d/"validation_summary.csv") for d in (args.primary_dir, args.zero_mass_dir)], ignore_index=True)
    animals = pd.concat([read_csv(d/"by_animal.csv") for d in (args.primary_dir, args.zero_mass_dir)], ignore_index=True)
    frame = pd.concat([read_csv(d/"validation_panels.csv") for d in (args.primary_dir, args.zero_mass_dir)], ignore_index=True)
    keys = ["session", "population", "window_ms", "generator", "readout", "calibration_mode", "job"]
    assert not frame.duplicated(keys).any()
    expected_zero = len(primary["sessions"])*(320*2+160)
    assert len(frame) == audit["rows"]+expected_zero
    np.testing.assert_allclose(frame.absolute_error, np.abs(frame.estimate-frame.truth), atol=1e-12)
    assert np.isfinite(frame.estimate).all()
    desc["posthoc_control"] = desc.readout.eq("continuous_zero_mass")
    desc.to_csv(args.output_dir/"combined_descriptive_summary.csv", index=False)
    animals.to_csv(args.output_dir/"combined_by_animal.csv", index=False)
    summary.to_csv(args.output_dir/"combined_validation_summary.csv", index=False)
    group_keys = ["window_ms", "readout", "calibration_mode", "generator", "family"]
    groups = summary.groupby(["session", "population"]+group_keys).budget_pass.all().reset_index()
    gates = groups.groupby(group_keys).agg(populations=("budget_pass", "size"), populations_with_all_prevalences_passing=("budget_pass", "sum")).reset_index()
    gates.to_csv(args.output_dir/"budget_gate_summary.csv", index=False)
    # Paired event-template panels, not independent biological samples.
    full = frame.loc[frame.family.eq("full")]
    key = ["session", "animal", "generator", "requested_prevalence", "replica", "job"]
    a = full.loc[full.window_ms.eq(20) & full.readout.eq("continuous_neutral") & full.calibration_mode.eq("pooled")]
    b = full.loc[full.window_ms.eq(20) & full.readout.eq("continuous_neutral") & full.calibration_mode.eq("oracle_generator")]
    paired = a.merge(b, on=key, suffixes=("_pooled", "_oracle"), validate="one_to_one")
    paired["oracle_error_reduction_pp"] = 100*(paired.absolute_error_pooled-paired.absolute_error_oracle)
    paired[key+["oracle_error_reduction_pp"]].to_csv(args.output_dir/"generator_oracle_contrasts.csv", index=False)
    t20 = full.loc[full.window_ms.eq(20) & full.readout.eq("continuous_zero_mass") & full.calibration_mode.eq("oracle_generator")]
    t5 = full.loc[full.window_ms.eq(5) & full.readout.eq("continuous_zero_mass")]
    temporal = t20.merge(t5, on=key, suffixes=("_20ms", "_5ms"), validate="one_to_one")
    temporal["short_window_error_increase_pp"] = 100*(temporal.absolute_error_5ms-temporal.absolute_error_20ms)
    temporal[key+["short_window_error_increase_pp"]].to_csv(args.output_dir/"temporal_contrasts.csv", index=False)

    def row(window, readout, mode="oracle_generator", generator="mix"):
        return desc.loc[desc.window_ms.eq(window) & desc.readout.eq(readout) & desc.calibration_mode.eq(mode) & desc.generator.eq(generator)].iloc[0]
    choices = [("Three-category, pooled", row(20, "ternary", "pooled")),
               ("Continuous, pooled", row(20, "continuous_neutral", "pooled")),
               ("Three-category, generator known", row(20, "ternary")),
               ("Continuous, generator known", row(20, "continuous_neutral"))]
    lines = ["# What limits regional endpoint calibration?", "", "## Answer", "",
        ("In these simulations, unknown within-window dynamics matter much more than ternary thresholding. "
        "The final 5 ms do not provide a simple fix: they contain far fewer spikes. A continuous-density treatment of mostly silent windows also introduced an avoidable estimation error, separated below."), "",
        ("Eight PF sessions, four rats, 22 population definitions; 1,600 saved simulation panels reconstructed exactly. "
        "No real replay rescoring, no calibrated real-content claims and no hc-11 transfer."), "",
        "## Frozen diagnostic: compression versus generator uncertainty", "",
        "Full-population, 20-ms mixed-generator validation. Equal-rat descriptive means.", "",
        "| Calibration | Mean absolute prevalence error | Estimates within 5 pp |", "|---|---:|---:|"]
    for name, r in choices:
        lines.append(f"| {name} | {r.mean_absolute_error*100:.2f} pp | {r.within_5pp_fraction*100:.1f}% |")
    lines += ["", "The continuous-native sensitivity (including silent Poisson evidence) and the later zero-mass control produce essentially the same 20-ms conclusion; all variants are retained in the CSVs.", "",
        "Knowing the generator is an ORACLE control, not a usable real-data calibration. Even the oracle mixed result does not achieve the required 90% within-budget rate overall; each session/population/prevalence must also pass separately.", "",
        "## Why trajectory type changes the meaning of the readout", "",
        (f"At 20 ms, raw regional BF AUC is {row(20, 'continuous_neutral', generator='stationary').raw_bf_auc:.3f} for stationary events and "
        f"{row(20, 'continuous_neutral', generator='late_jump').raw_bf_auc:.3f} for late jumps. "
        "This is an inversion, not simply noisy decoding. The late-jump generator spends the preceding part of the window in the opposite region. Its origin is chosen by the generator, so an oracle can infer the endpoint by interpreting that earlier information in reverse."), "",
        "Consequently, strong generator-calibrated discrimination is not evidence that the last 5 ms alone encode the endpoint well. The result is conditional on this synthetic late-jump rule, not proof that real events follow it.", "",
        "## Short-window diagnostic and post-hoc density check", "",
        "The final 5 ms are spike-free in about 61% of accepted windows, versus about 2.5% for 20 ms. Mean QC-population counts fall from about 4.6 to 0.51. Zero-spike windows were retained, not excluded.", "",
        "| Known late-jump generator | 20-ms error | 5-ms error |", "|---|---:|---:|"]
    for name, method in [("Three-category", "ternary"), ("Continuous KDE (frozen primary)", "continuous_neutral"), ("Continuous + explicit zero mass (post-hoc)", "continuous_zero_mass")]:
        lines.append(f"| {name} | {100*row(20, method, generator='late_jump').mean_absolute_error:.2f} pp | {100*row(5, method, generator='late_jump').mean_absolute_error:.2f} pp |")
    lines += ["", "The frozen continuous KDE smooths a large discrete mass at zero. The separately declared post-hoc control models that mass explicitly; its improvement shows the ~21-pp error must not be attributed entirely to biological information loss. After this correction, 5-ms errors remain about 7 pp and fewer than half of estimates are within 5 pp. Stationary controls show the same short-window deterioration.", "",
        "## What this licenses", "",
        "- Three-category compression is not the main limitation of the tested 20-ms pooled calibration.",
        "- A regional statement about an instantaneous endpoint depends on assumptions about preceding within-window content.",
        "- Shorter windows trade temporal specificity for spike support; they did not restore the five-point accuracy target here.",
        "- This does not establish universal unidentifiability or invalidate window-averaged content. It does not validate a new real-data estimator.", "",
        "## Verification and provenance", "",
        (f"The primary independent audit checked {audit['rows']:,} estimates, all 320,000 saved short-window likelihoods/calls, input/output hashes, exact spike totals and unchanged acceptance, plus 128 refits using a second optimizer. "
        "The zero-mass extension has separate unit tests, source hashes and retained outputs; its 6,400 estimates are labeled post-hoc. Simulation repeats are Monte Carlo replicates, not independent animals. No animal-level confidence intervals or external-validation claims are inferred from pooled repeats."), "",
        "![Readout and endpoint diagnostic](regional_readout_endpoint_synthesis.png)"]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    names = ["Ternary\npooled", "Continuous\npooled", "Ternary\ngenerator known", "Continuous\ngenerator known"]
    axes[0, 0].bar(np.arange(4), [100*r.mean_absolute_error for _, r in choices], color=["#777777", "#007f83", "#777777", "#007f83"])
    axes[0, 0].axhline(5, ls=":", color="black")
    axes[0, 0].set(xticks=np.arange(4), xticklabels=names, ylabel="Mean prevalence error (pp)", title="20 ms: generator knowledge matters more")
    kinds = ["stationary", "moving", "late_jump"]
    axes[0, 1].bar(np.arange(3), [row(20, "continuous_neutral", generator=g).raw_bf_auc for g in kinds], color=["#ae4256", "#dcac38", "#007f83"])
    axes[0, 1].axhline(.5, ls=":", color="black")
    axes[0, 1].set(xticks=np.arange(3), xticklabels=["Stationary", "Moving", "Late jump"], ylabel="Raw endpoint readout AUC", ylim=(0, 1), title="The late-jump readout is inverted")
    methods = [("ternary", "Ternary", "#777777"), ("continuous_neutral", "Continuous KDE", "#ae4256"), ("continuous_zero_mass", "Explicit zero mass (post-hoc)", "#007f83")]
    for j, (method, label, color) in enumerate(methods):
        axes[1, 0].plot([5, 20], [100*row(w, method, generator="late_jump").mean_absolute_error for w in [5, 20]], "o-", label=label, color=color)
    axes[1, 0].axhline(5, ls=":", color="black")
    axes[1, 0].set(xticks=[5, 20], xlabel="Readout window (ms)", ylabel="Mean prevalence error (pp)", title="Late jumps: silence needs explicit treatment")
    axes[1, 0].legend(fontsize=9)
    for kind, color in [("stationary", "#ae4256"), ("late_jump", "#007f83")]:
        axes[1, 1].plot([5, 20], [100*row(w, "continuous_neutral", generator=kind).silent_fraction for w in [5, 20]], "o-", label=kind.replace("_", " "), color=color)
    axes[1, 1].set(xticks=[5, 20], xlabel="Readout window (ms)", ylabel="Spike-free windows (%)", title="Short windows lose spike support")
    axes[1, 1].legend()
    fig.savefig(args.output_dir/"regional_readout_endpoint_synthesis.png", dpi=180)
    plt.close(fig)
    inputs = {"primary_manifest": args.primary_dir/"manifest.json", "zero_mass_manifest": args.zero_mass_dir/"manifest.json", "audit": args.audit, "reporter": Path(__file__)}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="complete", non_rescoring=True, outputs={f.name:file_sha256(f) for f in args.output_dir.iterdir() if f.is_file()})
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print((args.output_dir/"report.md").read_text())


if __name__ == "__main__":
    main()
