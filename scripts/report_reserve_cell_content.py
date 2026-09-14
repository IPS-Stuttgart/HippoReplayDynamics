#!/usr/bin/env python3
"""Non-rescoring report of audited reserve-cell acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256

ORDER = ("baseline", "random_mean", "targeted")
LABELS = ("Original populations", "Random added cells", "RUN-selected added cells")
COLORS = ("#737373", "#b46925", "#16786d")


def markdown(table):
    def cell(value):
        if pd.isna(value):
            return "unavailable"
        if isinstance(value, (float, np.floating)):
            return f"{value:.5f}"
        return str(value).replace("|", "/").replace("\n", " ")

    rows = [list(table.columns)] + [[cell(v) for v in row] for row in table.itertuples(index=False, name=None)]
    rows.insert(1, ["---"] * len(table.columns))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def error_decomposition(events):
    known = events[events.true_home.isin((0, 1))].copy()
    required = ["high_error", "low_error", "separation"]
    if known.empty or not np.isfinite(known[required]).all().all():
        raise ValueError("complete known-position errors required")
    if (
        (known[required] < 0).any().any()
        or (known.separation > known.high_error + known.low_error + 1e-8).any()
        or (known.separation + 1e-8 < (known.high_error - known.low_error).abs()).any()
    ):
        raise ValueError("inconsistent geometric errors")
    known["sum_position_mse_cm2"] = known.high_error**2 + known.low_error**2
    known["pair_separation_mse_cm2"] = known.separation**2
    known["twice_error_dot_cm2"] = known.sum_position_mse_cm2 - known.pair_separation_mse_cm2
    metrics = ["sum_position_mse_cm2", "pair_separation_mse_cm2", "twice_error_dot_cm2"]
    keys = ["session", "source", "encoding", "method"]
    class_means = known.groupby(keys + ["true_home"], as_index=False)[metrics].mean()
    if not class_means.groupby(keys).size().eq(2).all():
        raise ValueError("both truth classes required")
    session = class_means.groupby(keys, as_index=False)[metrics].mean()
    session["animal"] = session.session.str.split("/").str[0]
    animal = session.groupby(["animal", "source", "encoding", "method"], as_index=False)[metrics].mean()
    summary = animal.groupby(["source", "encoding", "method"], as_index=False)[metrics].mean()
    np.testing.assert_allclose(summary.sum_position_mse_cm2 - summary.twice_error_dot_cm2, summary.pair_separation_mse_cm2, atol=1e-8)
    return animal, summary


def report(result_dir, audit_path, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("report requires a passing audit bound to this manifest")
    for name, sha in manifest["output_sha256"].items():
        if file_sha256(result_dir / name) != sha:
            raise ValueError(f"altered result: {name}")
    gates = pd.read_csv(result_dir / "gates.csv")
    if dict(zip(gates.gate, gates.passed, strict=True)) != audit["gates"]:
        raise ValueError("audit and scientific gate decisions disagree")
    summary = pd.read_csv(result_dir / "summary.csv")
    animal = pd.read_csv(result_dir / "animal_summary.csv")
    session = pd.read_csv(result_dir / "session_summary.csv")
    draws = pd.read_csv(result_dir / "session_draws.csv")
    frozen = json.loads((result_dir / "pre_scoring.json").read_text())
    acquisitions = []
    for name, assignment in frozen["assignments"].items():
        first = [x for x in assignment["trace"] if x["side"] == "high"][0]
        last_high = [x for x in assignment["trace"] if x["side"] == "high"][-1]
        first_low = [x for x in assignment["trace"] if x["side"] == "low"][0]
        last_low = [x for x in assignment["trace"] if x["side"] == "low"][-1]
        acquisitions.append(
            dict(
                session=name,
                reserve_cells=len(assignment["reserve"]),
                added_per_side=assignment["quota"],
                calibration_parents=len(assignment["calibration_rows"]),
                high_q3_brier_before=first["before"],
                high_q3_brier_after=last_high["after"],
                low_q3_brier_before=first_low["before"],
                low_q3_brier_after=last_low["after"],
            )
        )
    acquisitions = pd.DataFrame(acquisitions)
    real = summary[summary.source.isin(("all_fixed_candidates", "full_accepted_segment"))].copy()
    real["home_gap_pp"] = 100 * real.home_gap
    real = real[["source", "encoding", "method", "home_gap_pp", "separation", "regional_tv", "high_entropy", "low_entropy"]]
    truth = summary[~summary.source.isin(real.source)].copy()
    truth["mean_population_balanced_error_cm"] = (truth.balanced_high_error + truth.balanced_low_error) / 2
    truth["mean_population_balanced_brier"] = (truth.balanced_high_brier + truth.balanced_low_brier) / 2
    truth = truth[["source", "method", "mean_population_balanced_error_cm", "mean_population_balanced_brier"]]
    primary_events = pd.concat([pd.read_csv(result_dir / name, dtype={"event_id": str}) for name in manifest["output_sha256"] if name.endswith("_events.csv.gz")], ignore_index=True)
    alignment_animal, alignment_summary = error_decomposition(primary_events)
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = dict(
        real_summary=real,
        truth_summary=truth,
        animal_summary=animal,
        session_summary=session,
        random_draw_summary=draws[draws.method.str.startswith("random_")],
        acquisition_summary=acquisitions,
        gates=gates,
        error_decomposition_by_animal=alignment_animal,
        error_decomposition_summary=alignment_summary,
    )
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    primary = summary[(summary.source == "all_fixed_candidates") & (summary.encoding == "early_run")].set_index("method")
    gaps = [
        ("Candidates\nearly map", "all_fixed_candidates", "early_run"),
        ("Candidates\nfull map", "all_fixed_candidates", "full_run"),
        ("Accepted\nearly map", "full_accepted_segment", "early_run"),
        ("Accepted\nfull map", "full_accepted_segment", "full_run"),
    ]
    x = np.arange(len(gaps))
    for j, (method, label, color) in enumerate(zip(ORDER, LABELS, COLORS, strict=True)):
        y = [real.loc[(real.source == src) & (real.encoding == enc) & (real.method == method), "home_gap_pp"].item() for _, src, enc in gaps]
        axs[0, 0].bar(x + (j - 1) * 0.24, y, width=0.24, color=color, label=label)
        delta = [100 * (primary.loc[method, k] / primary.loc["baseline", k] - 1) for k in ("separation", "regional_tv")]
        axs[0, 1].bar(np.arange(2) + (j - 1) * 0.24, delta, width=0.24, color=color)
    axs[0, 0].set(xticks=x, xticklabels=[g[0] for g in gaps], ylabel="Home-content gap (percentage points)", title="Original regional-content target")
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].axhline(-10, linestyle="--", color="black", linewidth=1)
    axs[0, 1].axhline(0, color="black", linewidth=0.7)
    axs[0, 1].set(
        xticks=np.arange(2), xticklabels=["Position separation", "Regional TV"], ylabel="Change from original (%)", title="Primary candidate disagreement; lower is better"
    )
    source_names = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
    short_names = ("RUN-Q4", "Poisson", "Gain x4", "Fixed total", "Map drift", "Assembly")
    for ax, metric, title, unit in (
        (axs[1, 0], "mean_population_balanced_error_cm", "Known-position localization", "Mean error (cm)"),
        (axs[1, 1], "mean_population_balanced_brier", "Known-region probability accuracy", "Brier loss"),
    ):
        for method, label, color in zip(ORDER, LABELS, COLORS, strict=True):
            y = [truth.loc[(truth.source == src) & (truth.method == method), metric].item() for src in source_names]
            ax.plot(np.arange(6), y, "o-", color=color, label=label, markersize=4)
        ax.set(xticks=np.arange(6), xticklabels=short_names, ylabel=unit, title=title)
        ax.tick_params(axis="x", labelrotation=20)
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Disjoint reserve-cell acquisition: fixed windows, unchanged shared-cell overlap", fontsize=13)
    fig.savefig(output_dir / "reserve_cell_content.png", dpi=170)
    plt.close(fig)
    passed = bool(audit["gates"]["development_numerical_screen"])
    failed = gates[~gates.passed]
    status = "Development screen passes; independent confirmation still required" if passed else "Development screen fails; do not promote as a validated remedy"
    original = draws[draws.method.eq("baseline") & draws.encoding.eq("early_run")]
    n_candidates = int(original.loc[original.source.eq("all_fixed_candidates"), "events"].sum())
    n_accepted = int(original.loc[original.source.eq("full_accepted_segment"), "events"].sum())
    text = f"""# Reserve-cell content diagnostic

**{status}.** Technical reconstruction passes; external validation is absent.

This is a non-rescoring report of the frozen test. The original {draws.session.nunique()} matched PF
pairs ({draws.animal.nunique()} rats), {n_candidates:,} candidate endpoints and {n_accepted:,} accepted-segment endpoints
remain. New cells were taken only from the reserve present in neither original
population; different cells were added to each side. Shared-cell overlap did not
increase. Flat-prior 20-ms Poisson decoding, event times and maps were unchanged.

RUN-only greedy selection minimized class-balanced regional Brier loss on the
first 20-ms bin of each Q3 parent block. Assignments were frozen before target or
Q4 outcomes were read. The random comparator averages 20 equal-budget allocations,
not their posteriors; these draws are not independent animals. Both populations
are independently decoded, not forced to agree. This requires additional recorded
cells and cannot manufacture missing neurons in an existing recording.

## Acquisition and calibration

{markdown(acquisitions)}

Q3 improvement is a training result, not held-out validation. Baseline population
eligibility and matching were already established in the original study, so the
development cohort is conditional on that selection, not pristine held-out data.

## Original content and disagreement

{markdown(real)}

Home gap is the absolute difference of session-mean posterior mass within 20 cm
of inferred Home, then equal sessions within rat and equal rats. These are NOT
fractions of true Home replay. Accepted-segment results are sensitivity analyses
and cannot rescue failure on the original candidate set.

## Known-truth safeguards

{markdown(truth)}

Truth tables equally weight Home and non-Home within session. This overview
averages population sides; animal_summary.csv retains every side and rat used
by the stricter no-harm gates. The synthetic conditions share fixed whole-universe
spike draws across assignments. They are model checks, not true biological replay.
A failed point-estimate no-harm gate is not proof of statistically certain harm.

## Exploratory error-alignment decomposition

This post-outcome diagnostic uses the already audited known-position errors;
it does not rescore spikes, change the allocation or replace the frozen gates.
For errors eA = decodedA - truth and eB = decodedB - truth, exactly:

    ||decodedA - decodedB||^2 = ||eA||^2 + ||eB||^2 - 2 eA dot eB

{markdown(alignment_summary[alignment_summary.source.eq("run_q4")])}

Both truth classes, sessions and rats receive the same balanced weights as above.
The dot-product term is error alignment INCLUDING systematic bias, not a
mean-centered covariance or proof of shared neuronal noise. Localization error
and between-decoder separation can change in opposite directions when error
alignment changes. This identity has known truth here; its decomposition is not
observable as true error on real replay and does not solve content instability.

## Failed gates

{markdown(failed)}

The full gates.csv includes all passes and failures. Neither technical audit
success nor gains on training data establish independent-data utility. Thresholds
and assignments were not adjusted to the real outcomes. No biological claim.

## Provenance

- Source result directory: {result_dir}
- Manifest SHA256: {file_sha256(manifest_path)}
- Verified event/method rows: {audit["reconstructed_event_method_rows"]:,}
- Verified population posteriors: {audit["reconstructed_population_posteriors"]:,}
- These totals include reused simulations, map sensitivities and random draws,
  NOT that many independent replay events.
- Source commit: {manifest["code_commit"]}
"""
    (output_dir / "report.md").write_text(text)
    report_manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file_sha256": {str(manifest_path): file_sha256(manifest_path), str(audit_path): file_sha256(audit_path)},
        "output_sha256": {p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
        "non_rescoring": True,
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2) + "\n")
    return tables


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report(args.result_dir, args.audit, args.output_dir)
