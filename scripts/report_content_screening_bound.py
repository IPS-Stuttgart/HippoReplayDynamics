#!/usr/bin/env python3
"""Report certified oracle bounds without selecting or decoding new events."""

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

REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
LABELS = dict(zip(REAL + TRUTH, ("Candidates", "Accepted segments", "Native Q4", "Poisson", "Gain 4", "Conditional", "Map drift", "Shared assembly"), strict=True))


def markdown(frame):
    def cell(x):
        if pd.isna(x):
            return "unavailable"
        if isinstance(x, (float, np.floating)):
            return f"{x:.4f}"
        return str(x).replace("|", "/")

    rows = [list(frame.columns), ["---"] * len(frame.columns)]
    rows.extend([[cell(x) for x in row] for row in frame.itertuples(index=False, name=None)])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def tables(frame):
    keys = ["session", "source", "encoding", "coverage", "oracle", "guard"]
    if frame.empty or frame.duplicated(keys).any() or not np.isfinite(frame.max_progress).all():
        raise ValueError("nonempty unique finite bounds required")
    if not frame.oracle.isin(("free_event", "count_pattern")).all() or not frame.guard.isin(("agreement_only", "truth_guarded")).all():
        raise ValueError("unknown bound class")
    if not frame.source.isin(REAL + TRUTH).all():
        raise ValueError("unknown source")
    summary = frame.groupby(["source", "encoding", "coverage", "oracle", "guard"], as_index=False).agg(
        sessions=("session", "size"),
        attainable_sessions=("targets_attainable", "sum"),
        min_progress=("max_progress", "min"),
        median_progress=("max_progress", "median"),
        max_progress=("max_progress", "max"),
    )
    index = ["session", "source", "encoding", "coverage"]
    wide = frame.pivot(index=index, columns=["oracle", "guard"], values="max_progress")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for guard in ("agreement_only", "truth_guarded"):
        free, tied = wide[f"free_event_{guard}"], wide[f"count_pattern_{guard}"]
        if ((free - tied).dropna() < -1e-6).any():
            raise ValueError("free oracle cannot be worse than the nested tied oracle")
        wide[f"indistinguishability_penalty_{guard}"] = free - tied
    for oracle in ("free_event", "count_pattern"):
        penalty = wide[f"{oracle}_agreement_only"] - wide[f"{oracle}_truth_guarded"]
        if (penalty.dropna() < -1e-6).any():
            raise ValueError("truth guards cannot improve an optimal bound")
        wide[f"truth_guard_penalty_{oracle}"] = penalty
    metadata = frame[frame.oracle.eq("count_pattern") & frame.guard.eq("agreement_only")][index + ["observations", "unique_patterns"]]
    wide = wide.merge(metadata, on=index, validate="one_to_one")
    wide["unique_pattern_fraction"] = wide.unique_patterns / wide.observations
    return summary, wide


def plot_primary(frame, path):
    primary = frame[frame.coverage.eq(0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 11), layout="constrained", gridspec_kw={"width_ratios": [1, 1.4]})
    blocks = [
        (primary[primary.source.isin(REAL)], [("free_event", "agreement_only"), ("count_pattern", "agreement_only")], "Real endpoints: no known-truth guard"),
        (
            primary[primary.source.isin(TRUTH)],
            [("free_event", "agreement_only"), ("count_pattern", "agreement_only"), ("free_event", "truth_guarded"), ("count_pattern", "truth_guarded")],
            "Known-position controls",
        ),
    ]
    for ax, (block, columns, title) in zip(axes, blocks, strict=True):
        values = block.pivot(index=["source", "encoding", "session"], columns=["oracle", "guard"], values="max_progress")
        order = sorted(values.index, key=lambda k: ((REAL + TRUTH).index(k[0]), k[1], k[2]))
        values = values.loc[order, columns]
        matrix = values.to_numpy()
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=5, cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(values)), [f"{LABELS[s]} / {e.removesuffix('_run')} / {session}" for s, e, session in values.index], fontsize=8)
        ax.set_xticks(
            range(len(columns)),
            [f"{'Free' if o == 'free_event' else 'Same-count'}\n{'accuracy guarded' if g == 'truth_guarded' else 'agreement only'}" for o, g in columns],
            fontsize=9,
        )
        ax.set_title(title, fontsize=11)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value < 2.4 else "black")
    fig.colorbar(im, ax=axes, label="Optimal progress t (1 meets numerical targets)", shrink=0.5)
    fig.suptitle("Fixed-window screening bounds, retain 50%\nOutcome-informed oracles, not validated selectors", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def report(root, audit_path, output):
    manifest_path = root / "manifest.json"
    manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("report requires independently certified bounds")
    for name, value in manifest["output_sha256"].items():
        if file_sha256(root / name) != value:
            raise ValueError(f"altered bound artifact: {name}")
    frame = pd.read_csv(root / "bounds.csv")
    if audit["cases"] != len(frame):
        raise ValueError("audit case count mismatch")
    summary, comparison = tables(frame)
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "attainability_summary.csv", index=False)
    comparison.to_csv(output / "oracle_comparison.csv", index=False)
    primary = frame[frame.coverage.eq(0.5)]
    primary.to_csv(output / "primary_cases.csv", index=False)
    plot_primary(frame, output / "screening_bound.png")
    short = summary[summary.coverage.eq(0.5)].drop(columns="coverage")
    failed = primary[primary.oracle.eq("count_pattern") & primary.guard.eq("truth_guarded") & ~primary.targets_attainable]
    cases = ["session", "source", "max_progress", "observations", "unique_patterns"]
    note = "\n".join(
        [
            "# Fixed-window screening attainability",
            "",
            "Status: numerical bound independently certified; NO validated remedy or external confirmation.",
            "",
            "This asks whether outcome-informed selection could satisfy the frozen requirements, not whether an observable rule can generalize. Original cells, likelihoods and event endpoints remain unchanged. No events were rescored.",
            "",
            "At progress t=1, Home-gap reduction is at least 20%, position-separation and regional-TV reductions are at least 10%, and neither population's entropy increases. Truth-guarded cases additionally forbid an increase in either population's class-balanced localization error or Home Brier loss. Zero baseline disagreement cannot count as success.",
            "",
            "## Primary 50% retention bounds",
            "",
            markdown(short),
            "",
            "Free-event selection may distinguish identical spike vectors using their true outcomes. Same-count selection must give identical joint original-population count vectors identical probabilities. Both are outcome-informed oracles fit separately per source/session/map; unique patterns may be memorized. Neither is a deployable predictor.",
            "",
            "## Same-count, truth-guarded cases below target",
            "",
            markdown(failed[cases]) if len(failed) else "None at 50% retention. This means the constraints are compatible, not that a transferable selector has been found.",
            "",
            "## Limits",
            "",
            "All weights are fractional. Retention is exactly 50% per real session and per true Home/non-Home class for controls; 25% and 75% are frozen sensitivities in the CSVs. These per-session/equal-class constraints are stricter than earlier pooled empirical screens. Failure is not a proof against all possible selectors. Real replay truth remains unknown. Control-bank truth is native RUN or simulated, not a true replay destination label.",
            "",
            "A feasible bound is a reason to develop and independently test an observable selection rule. An infeasible bound identifies a limitation only under the stated assumptions. No threshold is changed and no external run is authorized by this report.",
            "",
            f"Certified cases: {len(frame)}. Source commit: {manifest.get('code_commit', 'unavailable')}.",
            "",
        ]
    )
    (output / "report.md").write_text(note)
    inputs = {
        str(manifest_path): file_sha256(manifest_path),
        str(audit_path): file_sha256(audit_path),
        str(root / "bounds.csv"): file_sha256(root / "bounds.csv"),
        str(Path(__file__)): file_sha256(__file__),
    }
    provenance = {
        **build_script_provenance(),
        "input_file_sha256": inputs,
        "non_rescoring": True,
        "oracle_not_deployable": True,
        "validated_remedy": False,
        "external_validation": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return summary, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.result_dir, args.audit, args.output_dir)
