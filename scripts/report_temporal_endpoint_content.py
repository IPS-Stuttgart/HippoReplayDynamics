#!/usr/bin/env python3
"""Non-rescoring report for fixed-endpoint temporal-context falsification."""

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
from scripts.audit_edge_support_content import SOURCES
from scripts.measure_temporal_endpoint_content import METHODS, PRIMARY

CORE = ("separation_cm", "regional_tv", "a_entropy", "b_entropy")
TRUTH = ("a_error", "b_error", "a_brier", "b_brier", "a_nll", "b_nll")
GROUP = ("dataset", "animal", "session", "split", "source", "method")


def summaries(frame, methods=METHODS):
    if frame.empty or frame.duplicated([*GROUP, "event_index"]).any():
        raise ValueError("empty or duplicate observations")
    if frame.dataset.nunique() != 1 or frame.animal.nunique() != 4 or frame.session.nunique() != 8:
        raise ValueError("full eight-session four-animal cohort required")
    rows = []
    for keys, sub in frame.groupby(["dataset", "animal", "session", "split"]):
        if set(sub.source) != set(SOURCES):
            raise ValueError("missing source")
        for source, local in sub.groupby("source"):
            if set(local.method) != set(methods):
                raise ValueError("missing method")
            reference = local.loc[local.method.eq("independent")].set_index("event_index").sort_index()
            for method, view in local.groupby("method"):
                view = view.set_index("event_index").sort_index()
                if len(view) != len(reference) or not view.index.equals(reference.index):
                    raise ValueError("method changed endpoint cohort")
                for clock in ("original_start_s", "original_end_s", "context_ms"):
                    np.testing.assert_allclose(view[clock], reference[clock], atol=1e-10, rtol=0)
                fields = CORE if source == "real" else (*CORE, *TRUTH)
                if not np.isfinite(view[list(fields)]).all().all():
                    raise ValueError("missing finite required measurements")
                if source == "real" and not view[list(TRUTH)].isna().all().all():
                    raise ValueError("real replay truth must be unknown")
                base = dict(zip(GROUP, (*keys, source, method), strict=True))
                for metric in fields:
                    rows.append(dict(base, metric=metric, value=view[metric].mean(), n_events=len(view)))
                if source != "real":
                    for side in ("a", "b"):
                        rows.append(dict(base, metric=side + "_error_p90", value=view[side + "_error"].quantile(0.9), n_events=len(view)))
    sessions = pd.DataFrame(rows)
    animals = sessions.groupby(["dataset", "animal", "split", "source", "method", "metric"], as_index=False).value.mean()
    output = []
    for keys, local in animals.groupby(["dataset", "split", "source", "metric"]):
        pivot = local.pivot(index="animal", columns="method", values="value")
        if len(pivot) != 4 or pivot.isna().any().any():
            raise ValueError("missing animal/method outcome")
        baseline = pivot["independent"]
        for method in methods:
            reduction = baseline - pivot[method]
            rng = np.random.default_rng(20260914)
            interval = np.quantile(rng.choice(reduction.to_numpy(), (5000, 4), replace=True).mean(axis=1), [0.025, 0.975])
            output.append(
                dict(zip(["dataset", "split", "source", "metric"], keys, strict=True))
                | dict(
                    method=method,
                    baseline=baseline.mean(),
                    value=pivot[method].mean(),
                    reduction=reduction.mean(),
                    relative_reduction=reduction.mean() / baseline.mean() if baseline.mean() > 0 else np.nan,
                    reduction_ci_low=interval[0],
                    reduction_ci_high=interval[1],
                    animals=4,
                    animals_improved=int((reduction > 0).sum()),
                )
            )
    return sessions, animals, pd.DataFrame(output)


def gates(summary, frame, audited, primary=PRIMARY):
    if set(frame.split) != {0, 1, 2}:
        raise ValueError("all frozen splits required")
    table = summary.loc[summary.split.eq(0) & summary.method.eq(primary)].set_index(["source", "metric"])
    control = summary.loc[summary.split.eq(0) & summary.method.eq("entropy_matched")].set_index(["source", "metric"])
    rows = [dict(gate="independently_verified", passed=bool(audited))]
    for metric in ("separation_cm", "regional_tv"):
        value = table.loc[("real", metric)]
        rows.append(
            dict(
                gate="real_reduction_" + metric, passed=bool(value.animals == 4 and value.relative_reduction >= 0.1 and value.animals_improved >= 3 and value.reduction_ci_low > 0)
            )
        )
        other = control.loc[("real", metric), "value"]
        rows.append(dict(gate="beats_entropy_control_" + metric, passed=bool(other > 0 and (other - value.value) / other >= 0.05)))
    for side in ("a", "b"):
        rows.append(dict(gate="no_increased_" + side + "_entropy", passed=bool(table.loc[("real", side + "_entropy"), "reduction"] >= -1e-8)))
        for source in set(SOURCES) - {"real"}:
            for metric in (side + "_error", side + "_error_p90", side + "_brier"):
                rows.append(dict(gate=source + "_no_worse_" + metric, passed=bool(table.loc[(source, metric), "reduction"] >= -1e-8)))
    primary_rows = frame.loc[frame.split.eq(0) & frame.method.eq(primary)]
    matches = primary_rows[["a_entropy_control_available", "b_entropy_control_available"]]
    complete = not matches.empty and matches.notna().all().all() and matches.eq(True).all().all()
    rows.append(dict(gate="all_primary_entropy_controls_available", passed=bool(complete)))
    rows.append(dict(gate="advance_external_validation", passed=all(row["passed"] for row in rows)))
    return pd.DataFrame(rows)


def plot_summary(animals, summary, output, primary_method=PRIMARY, primary_label="Up to 200 ms, diffusion + reset", cohort_label="PF development"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"independent": "#555555", primary_method: "#197c87"}
    labels = {"independent": "Independent final 20 ms", primary_method: primary_label}
    primary = animals.loc[animals.split.eq(0)]
    for axis, metric, title, ylabel in (
        (axes[0, 0], "separation_cm", "A. Real population separation", "A/B endpoint distance (cm)"),
        (axes[0, 1], "regional_tv", "B. Real regional disagreement", "Regional total variation"),
        (axes[1, 1], "a_entropy", "D. Real posterior concentration", "Mean A/B normalized entropy"),
    ):
        for method in ("independent", primary_method):
            local = primary.loc[primary.source.eq("real") & primary.method.eq(method)]
            if metric == "a_entropy":
                values = local.loc[local.metric.isin(["a_entropy", "b_entropy"])].groupby("animal").value.mean()
            else:
                values = local.loc[local.metric.eq(metric)].set_index("animal").value.sort_index()
            axis.plot(values.index, values, "o-", color=colors[method], label=labels[method])
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
    known = ["run_q4", "sim_stationary", "sim_moving", "sim_moving_gain", "sim_late_jump"]
    axis = axes[1, 0]
    for method in ("independent", primary_method):
        local = summary.loc[summary.split.eq(0) & summary.method.eq(method) & summary.metric.isin(["a_error", "b_error"])]
        values = local.groupby("source").value.mean().reindex(known)
        axis.plot(np.arange(5), values, "o-", color=colors[method], label=labels[method])
    axis.set(title="C. Known-position falsification", ylabel="Mean physical error, A/B (cm)")
    axis.set_xticks(np.arange(5), ["RUN", "Static", "Moving", "Moving\n+ gains", "Late jump"])
    axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Fixed endpoint: agreement gains must survive known-truth controls\n{cohort_label}; split 0; equal animal weights", fontsize=13)
    fig.savefig(output / "temporal_endpoint_falsification.png", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.audit_dir / "independent_audit.json").read_text())
    producer = json.loads((args.measurement_dir / "manifest.json").read_text())
    if audit["status"] != "passed" or producer["status"] != "complete":
        raise ValueError("complete independently audited measurement required")
    if audit["input_file_sha256"]["producer"] != file_sha256(args.measurement_dir / "manifest.json"):
        raise ValueError("audit from another or changed measurement")
    inputs = dict(
        audit=args.audit_dir / "independent_audit.json",
        source=args.measurement_dir / "manifest.json",
        script=Path(__file__),
        protocol=ROOT / "docs/temporal_endpoint_content_protocol.md",
    )
    frames = []
    for row in producer["results"]:
        path = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
        record = [r for r in audit["results"] if r["session"] == row["session"] and r["animal"] == row["animal"]]
        if len(record) != 1 or record[0]["readout_sha256"] != file_sha256(path):
            raise ValueError("missing or mismatched audited readout")
        inputs[row["session"]] = path
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    frame = pd.concat(frames, ignore_index=True)
    sessions, animals, summary = summaries(frame)
    checks = gates(summary, frame, audited=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    sessions.to_csv(args.output_dir / "by_session.csv", index=False)
    animals.to_csv(args.output_dir / "by_animal.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    checks.to_csv(args.output_dir / "gate_summary.csv", index=False)
    plot_summary(animals, summary, args.output_dir)
    primary = summary.loc[summary.split.eq(0) & summary.method.eq(PRIMARY)]
    selected = frame.loc[frame.split.eq(0) & frame.method.eq(PRIMARY)]
    denominator = selected.groupby(["source", "animal", "session"], as_index=False).agg(events=("event_index", "size"), mean_context_ms=("context_ms", "mean"))
    denominator.to_csv(args.output_dir / "denominators.csv", index=False)
    lines = [
        "# Fixed-endpoint temporal-context result",
        "",
        "No event resampling or endpoint shifts. Primary: diffusion+25% reset; sigma20cm/20ms; context<=200ms.",
        "Independent dense filtering and native context recounting passed.",
        "",
        "## Primary split: equal-animal means",
        "",
        "| Source | Metric | Independent | Context | Reduction |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        if row.metric in ("separation_cm", "regional_tv", "a_error", "b_error", "a_entropy", "b_entropy"):
            lines.append(f"| {row.source} | {row.metric} | {row.baseline:.5f} | {row.value:.5f} | {row.reduction:+.5f} |")
    failed = checks.loc[~checks.passed, "gate"].tolist()
    lines += [
        "",
        "## Frozen decision",
        "",
        "Advancement: " + ("PASS" if not failed else "FAIL"),
        "",
        *["- " + x for x in failed],
        "",
        "Agreement is not known truth. A gain caused by smoothing away a genuine late jump is not a fixed-endpoint remedy. Secondary controls/splits cannot replace the frozen primary.",
        "",
        "PF is development; hc11 has not been run by this report. The original targeted Home contrast has not been remedied. No independent-data validation or biological claim follows from a development-only result.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if any(file_sha256(v) != manifest["input_file_sha256"][k] for k, v in inputs.items()):
        raise ValueError("report inputs changed")
    manifest.update(
        status="complete",
        non_rescoring=True,
        primary_advanced=not failed,
        scientific_goal_achieved=False,
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
