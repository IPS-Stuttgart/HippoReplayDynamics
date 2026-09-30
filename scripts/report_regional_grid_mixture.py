"""Non-rescoring summary of the independently audited grid-mixture pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np

from scripts._provenance import build_script_provenance, file_sha256
from scripts.build_regional_blind_bank import read_csv, write_json


def summarize_frames(estimates, pairs):
    if len(estimates) != 4224 or len(pairs) != 1344:
        raise ValueError("complete audited pilot required")
    full = (
        estimates[estimates.population.eq("full")]
        .groupby(["observation", "fit_kind"])
        .agg(
            fits=("estimate", "size"),
            mean_error=("absolute_error", "mean"),
            maximum_error=("absolute_error", "max"),
            first_mean_error=("first_absolute_error", "mean"),
            converged=("converged", "sum"),
        )
        .reset_index()
    )
    gap = (
        pairs.groupby(["family", "observation", "fit_kind"])
        .agg(
            comparisons=("fitted_gap", "size"),
            first_mean_gap=("absolute_first_gap", "mean"),
            fitted_mean_gap=("absolute_fitted_gap", "mean"),
            maximum_gap=("absolute_fitted_gap", "max"),
        )
        .reset_index()
    )
    animal = (
        pairs.groupby(["family", "observation", "fit_kind", "animal", "session"])[["absolute_first_gap", "absolute_fitted_gap"]]
        .mean()
        .groupby(["family", "observation", "fit_kind", "animal"])
        .mean()
        .reset_index()
    )
    return full, gap, animal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.output_dir)
    audit = json.loads((Path(args.audit) / "manifest.json").read_text())
    if audit["status"] != "technical_pass" or audit["input_file_sha256"]["experiment"] != file_sha256(root / "manifest.json"):
        raise ValueError("audit does not validate this run")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, h in manifest["outputs"].items():
        if file_sha256(root / name) != h:
            raise ValueError("experiment output changed")
    out.mkdir(parents=True, exist_ok=False)
    d, p = read_csv(root / "estimates.csv"), read_csv(root / "paired_population_gaps.csv")
    full, gap, animal = summarize_frames(d, p)
    full.to_csv(out / "full_population_accuracy.csv", index=False)
    gap.to_csv(out / "matched_population_gaps.csv", index=False)
    animal.to_csv(out / "matched_gaps_by_animal.csv", index=False)
    animal.groupby(["family", "observation", "fit_kind"])[["absolute_first_gap", "absolute_fitted_gap"]].mean().reset_index().to_csv(
        out / "equal_animal_gap_summary.csv", index=False
    )
    d.groupby(["observation", "fit_kind", "population", "generator", "scale"], dropna=False).agg(
        mean_error=("absolute_error", "mean"), maximum_error=("absolute_error", "max"), first_mean_error=("first_absolute_error", "mean")
    ).reset_index().to_csv(out / "by_dynamics.csv", index=False)
    primary = d[d.observation.eq("conditional_multinomial") & d.fit_kind.eq("selected_penalty")]
    primary.nlargest(20, "absolute_error").to_csv(out / "largest_primary_errors.csv", index=False)
    penalty = json.loads((root / "frozen_penalty.json").read_text())
    decision = json.loads((root / "decision.json").read_text())
    write_json(out / "decision.json", decision)
    lines = [
        "# Grid-level content inference: synthetic pilot",
        "",
        f"Decision: **{decision['status']}**. Shared CV penalty: **{penalty['penalty']}**.",
        "",
        "Eight sessions/four rats, 1,139 fixed event templates, 22 frozen populations.",
        "40-ms terminal occupancy; two independent 20-ms windows. One held-out simulation replica,",
        "four binary-label quotas, seven pure dynamics strata and their equal empirical mixture.",
        "",
        "## Full-population occupancy error",
        "",
        "Mean absolute errors below are percentage points. No regional calibration was used.",
        "",
        "| Likelihood | First EM | Unpenalized NPMLE | CV-regularized | Worst regularized |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for observation in full.observation.unique():
        f = full[full.observation.eq(observation)].set_index("fit_kind")
        u, s = f.loc["unpenalized"], f.loc["selected_penalty"]
        lines.append(f"| {observation} | {100 * s.first_mean_error:.2f} | {100 * u.mean_error:.2f} | {100 * s.mean_error:.2f} | {100 * s.maximum_error:.2f} |")
    lines += [
        "",
        "## Same-event high/low discrepancy",
        "",
        "Mean absolute gaps are equal-comparison/equal-session within each family, not an animal-level CI.",
        "Separate per-animal and equally weighted animal summaries accompany this table.",
        "",
        "| Family / likelihood | First EM gap | Unpenalized gap | Regularized gap | Worst regularized gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for (family, observation), f in gap.groupby(["family", "observation"]):
        f = f.set_index("fit_kind")
        u, s = f.loc["unpenalized"], f.loc["selected_penalty"]
        lines.append(f"| {family} / {observation} | {100 * s.first_mean_gap:.2f} | {100 * u.fitted_mean_gap:.2f} | {100 * s.fitted_mean_gap:.2f} | {100 * s.maximum_gap:.2f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "First EM equals the mean flat-prior posterior, as proposed. Further grid inference improves",
        "average full-population truth error, but does not yield coverage-invariant fixed points in this pilot.",
        "Conditional-multinomial targeted high/low gaps widen. CV-selected spatial regularization improves",
        "held-out predictive likelihood yet worsens full-population regional accuracy relative to the",
        "unpenalized fit. Predictive tuning is not a guarantee for a particular regional functional.",
        "",
        "The fixed-total censored diagnostic retains excluded spikes as one category using their summed",
        "training rates. It improves matched-population agreement compared with the conditional model,",
        "but uses extra whole-recording information and still misses the worst-case screen.",
        "Even the frozen full decoder excludes cells. Subset counts vary with location; conditioning them",
        "away is not generally equivalent to fitting a common location prior under this fixed-total bank.",
        "",
        "The bank's active-cell support selection was nearly inactive in this subset (see exact rejection",
        "counts), but remains a formal model-consistency caveat. Moving AND jumping can change position",
        "within bins. Each legacy population's native grid was reconstructed by coordinate matching;",
        "the aborted first preflight is not an experiment result.",
        "",
        "All 4,224 validation fits and 264 CV fits passed numerical checks and an independent audit.",
        "Convergence is not regional identification. One replica and eight specified mixtures cannot",
        "certify bias, every dynamics mixture or recording-design impossibility. Lambda=0.1 is the",
        "strongest tested choice, not a proven global optimum over penalties.",
        "",
        "No real-data reconciliation, biological content claim, real event bootstrap, nonspatial",
        "perturbation panel or own-population detector experiment was performed after the first two",
        "primary screens failed. The outcome rules out this proposed remedy at the tested settings,",
        "not every grid-level estimator or all approaches to regional-content bounds.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    figure(d, p, read_csv(root / "penalty_scores.csv"), out)
    m = build_script_provenance(input_paths={"experiment": root / "manifest.json", "audit": Path(args.audit) / "manifest.json", "reporter": __file__})
    m.update(status="complete", outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out / "manifest.json", m)


def figure(d, p, cv, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    cv = cv.sort_values("penalty")
    axes[0, 0].plot(cv.penalty, cv.score - cv.loc[cv.penalty.eq(0), "score"].iloc[0], marker="o", color="#257a72")
    axes[0, 0].set_xscale("symlog", linthresh=0.001)
    axes[0, 0].set(title="Event-fold penalty selection", xlabel="Shared spatial penalty", ylabel="Held-out gain over no penalty (nats/bin)")
    full = d[d.population.eq("full") & d.observation.eq("conditional_multinomial")]
    u = full[full.fit_kind.eq("unpenalized")]
    s = full[full.fit_kind.eq("selected_penalty")]
    arrays = [100 * u.first_absolute_error.to_numpy(), 100 * u.absolute_error.to_numpy(), 100 * s.absolute_error.to_numpy()]
    axes[0, 1].boxplot(arrays, tick_labels=["First EM", "Unpenalized", "CV regularized"], showfliers=True)
    axes[0, 1].scatter([1, 2, 3], [np.mean(x) for x in arrays], color="#b24a38", label="Mean", zorder=4)
    axes[0, 1].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Full population: conditional multinomial", ylabel="Absolute occupancy error (pp)")
    axes[0, 1].legend(fontsize=8)
    colors = {"conditional_multinomial": "#327c91", "poisson": "#b05c3c", "fixed_total_censored": "#6b7b36"}
    for obs, color in colors.items():
        q = p[p.family.eq("targeted") & p.observation.eq(obs) & p.fit_kind.eq("selected_penalty")]
        axes[1, 0].scatter(100 * q.absolute_first_gap, 100 * q.absolute_fitted_gap, s=12, alpha=0.55, label=obs, color=color)
    axes[1, 0].plot([0, 40], [0, 40], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Targeted high/low populations", xlabel="First-EM absolute gap (pp)", ylabel="Regularized absolute gap (pp)")
    axes[1, 0].legend(fontsize=7)
    q = d[d.observation.eq("conditional_multinomial") & d.fit_kind.eq("selected_penalty")]
    for name, color in [("targeted_high", "#327c91"), ("targeted_low", "#b05c3c")]:
        a = q[q.population.eq(name)]
        axes[1, 1].scatter(100 * a.true_occupancy, 100 * a.estimate, s=12, alpha=0.6, label=name, color=color)
    axes[1, 1].plot([0, 60], [0, 60], color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Grid inference does not remove coverage dependence", xlabel="True terminal Home occupancy (%)", ylabel="Regularized estimate (%)")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Grid-level marginal inference: capped synthetic PF test\nNo real Home-content correction")
    fig.savefig(out / "grid_mixture_pilot.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
