"""Non-rescoring report for frozen terminal-segment mixture calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.build_regional_blind_bank import read_csv, write_json

KEYS = ["population", "family", "readout", "representation", "delta_ms"]


def summarize_envelopes(frame):
    required = [*KEYS, "session", "animal", "prevalence", "replica", "worst_absolute_error", "within_5pp", "identified"]
    if frame.empty or not set(required).issubset(frame):
        raise ValueError("nonempty complete envelope table required")
    if not np.isfinite(frame.worst_absolute_error).all():
        raise ValueError("missing envelope errors must not be omitted")
    if frame.duplicated([*KEYS, "session", "prevalence", "replica"]).any():
        raise ValueError("duplicate envelope")
    rows = []
    for key, group in frame.groupby(KEYS, observed=True):
        cells = group.groupby(["session", "prevalence"]).worst_absolute_error.mean()
        unidentified = int((~group.identified).sum())
        budget = bool(cells.max() <= .05 and group.within_5pp.mean() >= .90 and unidentified == 0)
        rows.append(dict(zip(KEYS, key, strict=True)) | {
            "session_prevalence_replicas": len(group), "sessions": group.session.nunique(), "animals": group.animal.nunique(),
            "mean_worst_mixture_error": float(group.worst_absolute_error.mean()),
            "worst_session_prevalence_mae": float(cells.max()),
            "maximum_single_panel_error": float(group.worst_absolute_error.max()),
            "within_5pp_fraction": float(group.within_5pp.mean()), "unidentified": unidentified,
            "development_budget_pass": budget})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.output_dir)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "complete_pending_report_audit":
        raise ValueError("incomplete experiment")
    for name, expected in manifest["outputs"].items():
        if file_sha256(root/name) != expected:
            raise ValueError("experiment output changed")
    checked = json.loads((Path(args.audit)/"manifest.json").read_text())
    if checked["status"] != "technical_pass" or checked["input_file_sha256"]["experiment"] != file_sha256(root/"manifest.json"):
        raise ValueError("independent audit does not validate this experiment")
    out.mkdir(parents=True, exist_ok=False)
    envelope, pure = read_csv(root/"mixture_envelopes.csv"), read_csv(root/"pure_validation.csv")
    summary = summarize_envelopes(envelope)
    summary.to_csv(out/"terminal_length_worst_mixture_summary.csv", index=False)
    cell_keys = [*KEYS, "animal", "session", "prevalence"]
    cells = envelope.groupby(cell_keys, observed=True).agg(
        replicas=("replica", "size"), mean_error=("worst_absolute_error", "mean"),
        replica_sd=("worst_absolute_error", "std"), min_error=("worst_absolute_error", "min"),
        max_error=("worst_absolute_error", "max"), within_5pp_fraction=("within_5pp", "mean"),
        all_identified=("identified", "all")).reset_index()
    cells.to_csv(out/"session_prevalence_uncertainty.csv", index=False)
    by_animal = cells.groupby([*KEYS, "animal"], observed=True).agg(
        mean_error=("mean_error", "mean"), within_5pp_fraction=("within_5pp_fraction", "mean"),
        worst_cell_mae=("mean_error", "max")).reset_index()
    by_animal.to_csv(out/"by_animal.csv", index=False)
    bootstrap = []
    rng = np.random.default_rng(2026091617)
    for key, group in by_animal.groupby(KEYS, observed=True):
        values = group[["mean_error", "within_5pp_fraction"]].to_numpy()
        draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
        q = np.quantile(draws, [.025, .975], axis=0)
        bootstrap.append(dict(zip(KEYS, key, strict=True)) | {
            "animals": len(values), "mean_error_lower": q[0, 0], "mean_error_upper": q[1, 0],
            "within_fraction_lower": q[0, 1], "within_fraction_upper": q[1, 1],
            "interpretation": "descriptive_small_cluster_bootstrap_not_coverage_certification"})
    pd.DataFrame(bootstrap).to_csv(out/"descriptive_animal_bootstrap.csv", index=False)
    pure_summary = pure.groupby([*KEYS, "calibration_mode", "generator", "scale"], observed=True).agg(
        mean_absolute_error=("absolute_error", "mean"), maximum_error=("absolute_error", "max"),
        within_5pp_fraction=("within_5pp", "mean"), estimates=("estimate", "size"),
        identified=("estimate", "count")).reset_index()
    pure_summary.to_csv(out/"pure_dynamics_summary.csv", index=False)
    full = summary.loc[summary.population.eq("full")]
    primary = full.loc[full.readout.eq("independent_bin_any_home") & full.representation.eq("continuous_zero_mass")].sort_values("delta_ms")
    passing = primary.loc[primary.development_budget_pass, "delta_ms"]
    selected = int(passing.min()) if len(passing) else None
    status = "candidate_requires_independent_confirmation" if selected is not None else "no_tested_length_meets_development_budget"
    write_json(out/"decision.json", {"status": status, "development_delta_star_ms": selected,
        "primary": "independent_bin_any_home_continuous_zero_mass", "error_budget_pp": 5,
        "required_within_budget_fraction": .9, "real_data_calibration_authorized": False,
        "null_panels_used": False, "limitation": "conditional finite-bank envelope; four replicas per configuration"})
    gates = [
        {"gate": "completed_tasks", "status": "pass" if manifest["tasks"] == 32 else "fail", "detail": str(manifest["tasks"])},
        {"gate": "nonempty_primary_lengths", "status": "pass" if set(primary.delta_ms) == {20, 40, 60, 100} else "fail", "detail": "four frozen durations"},
        {"gate": "primary_identified", "status": "pass" if primary.unidentified.sum() == 0 else "fail", "detail": str(int(primary.unidentified.sum()))},
        {"gate": "primary_development_budget", "status": "pass" if selected is not None else "fail", "detail": status},
        {"gate": "independent_empirical_audit", "status": "pass", "detail": "32 task audits, see separate artifact"},
        {"gate": "real_content_calibration", "status": "not_tested", "detail": "no real-data inference or hc-11 transfer"},
    ]
    pd.DataFrame(gates).to_csv(out/"gate_summary.csv", index=False)
    audit_path = Path(manifest["parameters"]["bank_audit"])
    reference = read_csv(audit_path/"unconditioned_geometry_reference_summary.csv")
    reference.to_csv(out/"geometry_reference_by_session.csv", index=False)
    reference.groupby(["generator", "scale", "delta_ms"], observed=True).natural_positive_fraction.agg(
        ["mean", "min", "max"]).reset_index().to_csv(out/"geometry_reference_summary.csv", index=False)
    for name in ("occupancy_and_crossing_readouts.csv", "spike_support.csv"):
        (out/name).write_bytes((root/name).read_bytes())
    figure(full, pure_summary, out)
    lines = ["# Terminal-segment calibration under unknown dynamics", "",
             "Known-truth synthetic development experiment; no real replay content estimates.", "",
             f"Decision: **{status}**. Development Delta*: **{selected if selected is not None else 'none'}**.", "",
             "Primary readout: independent flat-prior Poisson decodes in non-overlapping 20-ms bins,",
             "combined into an any-Home scalar and calibrated against >=20-ms geometric Home dwell.",
             "No HMM or inferred dynamics type. Silent bins are neutral. The independent-bin union",
             "is a calibrated surrogate, not an exact posterior for continuous cumulative dwell.", "",
             "| Delta | Mean worst-mixture error | Worst session/prevalence MAE | Within 5 pp |",
             "| --- | ---: | ---: | ---: |"]
    for r in primary.itertuples():
        lines.append(f"| {r.delta_ms} ms | {100*r.mean_worst_mixture_error:.2f} pp | {100*r.worst_session_prevalence_mae:.2f} pp | {100*r.within_5pp_fraction:.1f}% |")
    lines += ["", "## What the envelope means", "",
              "Seven pure panels share identical true prevalence and event count. Under one fixed pooled",
              "calibration, every common-weight empirical mixture MLE lies between their pure MLEs.",
              "Thus the worst absolute error over this entire empirical mixture simplex is obtained",
              "at a pure stratum. Numerical grid checks test this statement on the saved likelihoods.",
              "This is NOT a real-prevalence confidence bound, a new independent N-event simulation,",
              "or a guarantee over unseen occupancy/spike profiles and class-dependent mixtures.", "",
              "## Interpretation limits", "",
              "1139 common count-trace templates from eight sessions/four rats; 461 shorter original events excluded.",
              "The target changes with segment length. Geometry-reference prevalence is reported separately",
              "from the iid decoder prior, and longer targets must not be described as endpoint recovery.",
              "Four calibration and four validation replicas per prevalence are development only.",
              "Replica dispersion and four-rat bootstrap ranges are descriptive, not certified 90% coverage.",
              "Oracle generator calibration is a diagnostic using simulated truth, unavailable for real events.",
              "A failed gate rules out these frozen readouts under this bank; it does not prove that all",
              "terminal estimators are unidentifiable or that an IMM would necessarily fix the problem.",
              "The preserved bank caveats include decoder-derived anchors, legacy overlapping populations,",
              "fixed spike totals and arena-limited realization of some 2x jump requests.", "",
              "No new real events, replay model evidence, null panels, threshold tuning or hc-11 analysis were used.", ""]
    (out/"report.md").write_text("\n".join(lines))
    provenance = build_script_provenance(input_paths={"experiment": root/"manifest.json", "reporter": __file__, "audit": Path(args.audit)/"manifest.json"})
    provenance.update(status="complete", decision=status, outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out/"manifest.json", provenance)
    print(primary.to_string(index=False), flush=True)


def figure(full, pure, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for (readout, rep), group in full.groupby(["readout", "representation"]):
        s = group.sort_values("delta_ms")
        label = ("independent bins" if readout == "independent_bin_any_home" else "pooled spikes") + (", continuous" if rep == "continuous_zero_mass" else ", ternary")
        axes[0, 0].plot(s.delta_ms, 100*s.worst_session_prevalence_mae, marker="o", label=label)
        axes[0, 1].plot(s.delta_ms, 100*s.within_5pp_fraction, marker="o", label=label)
    axes[0, 0].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Worst session/prevalence mean error", ylabel="Percentage points", xlabel="Segment length (ms)")
    axes[0, 1].set(title="Worst-mixture estimates within 5 pp", ylabel="Percent", xlabel="Segment length (ms)", ylim=(0, 102))
    axes[0, 0].legend(fontsize=8)
    primary = pure.loc[pure.population.eq("full") & pure.readout.eq("independent_bin_any_home") & pure.representation.eq("continuous_zero_mass")]
    heat = primary.loc[primary.calibration_mode.eq("pooled")].pivot(index=["generator", "scale"], columns="delta_ms", values="mean_absolute_error")
    im = axes[1, 0].imshow(100*heat.to_numpy(), aspect="auto", cmap="magma")
    axes[1, 0].set(xticks=range(4), xticklabels=heat.columns, yticks=range(len(heat)),
                   yticklabels=[f"{k} x{s:g}" for k, s in heat.index], title="Primary: pure-stratum mean error", xlabel="Segment length (ms)")
    fig.colorbar(im, ax=axes[1, 0], label="Percentage points")
    for mode, group in primary.groupby("calibration_mode"):
        s = group.groupby("delta_ms").mean_absolute_error.mean()
        axes[1, 1].plot(s.index, 100*s, marker="o", label=mode)
    axes[1, 1].set(title="Oracle diagnostic, not deployable", xlabel="Segment length (ms)", ylabel="Mean absolute error (pp)")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Terminal-content robustness: synthetic PF bank\n8 sessions, 4 rats; fixed pooled calibration, no HMM")
    fig.savefig(out/"terminal_mixture_robustness.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
