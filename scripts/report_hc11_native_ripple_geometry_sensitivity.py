#!/usr/bin/env python3
"""Compare existing hc-11 native-ripple pilot runs without rescoring events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_OUTPUT = "hc11_native_ripple_sensitivity_summary.csv"
GEOMETRY_OUTPUT = "hc11_native_ripple_sensitivity_by_geometry.csv"
SESSION_OUTPUT = "hc11_native_ripple_sensitivity_by_session.csv"
STRICT_OUTPUT = "hc11_native_ripple_strict_clean_imm_events.csv"
OVERLAP_OUTPUT = "hc11_native_ripple_selection_overlap.csv"
GATE_OUTPUT = "hc11_native_ripple_sensitivity_gate_summary.csv"
MANIFEST_OUTPUT = "hc11_native_ripple_sensitivity_manifest.json"
REPORT_OUTPUT = "hc11_native_ripple_sensitivity_report.md"


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have LABEL=OUTPUT_DIR form")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must have LABEL=OUTPUT_DIR form")
    return label, Path(path)


def read_run(label: str, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(directory / "hc11_native_ripple_model_claim_decisions.csv")
    selection = pd.read_csv(directory / "hc11_native_ripple_event_selection.csv")
    gates = pd.read_csv(directory / "hc11_native_ripple_gate_summary.csv")
    decisions = decisions.copy()
    decisions["run"] = label
    selection = selection.copy()
    selection["run"] = label
    for prefix in ("delta_trajectory_minus_stationary", "delta_imm_minus_fragmented"):
        per_bin = f"{prefix}_per_time_bin"
        per_spike = f"{prefix}_per_spike"
        if per_bin not in decisions:
            decisions[per_bin] = decisions[prefix] / decisions["n_time_bins"].clip(lower=1)
        if per_spike not in decisions:
            decisions[per_spike] = decisions[prefix] / decisions["n_spikes"].clip(lower=1)
    decisions["strict_clean_imm"] = (
        decisions["trajectory_confident_claim"].astype(bool)
        & decisions["imm_confident_over_fragmented"].astype(bool)
        & decisions["best_model"].eq("first_order_imm")
    )
    return decisions, selection, gates


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        best = group["best_model"].value_counts()
        strict = group[group["strict_clean_imm"]]
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "events": int(len(group)),
                "trajectory_confident_count": int(group["trajectory_confident_claim"].sum()),
                "trajectory_confident_fraction": float(group["trajectory_confident_claim"].mean()),
                "stationary_confident_count": int(group["stationary_confident_claim"].sum()),
                "median_trajectory_minus_stationary": float(group["delta_trajectory_minus_stationary"].median()),
                "median_trajectory_minus_stationary_per_time_bin": float(group["delta_trajectory_minus_stationary_per_time_bin"].median()),
                "median_trajectory_minus_stationary_per_spike": float(group["delta_trajectory_minus_stationary_per_spike"].median()),
                "imm_raw_win_count": int((group["delta_imm_minus_fragmented"] > 0.0).sum()),
                "imm_confident_over_fragmented_count": int(group["imm_confident_over_fragmented"].sum()),
                "fragmented_confident_over_imm_count": int(group["fragmented_confident_over_imm"].sum()),
                "median_imm_minus_fragmented": float(group["delta_imm_minus_fragmented"].median()),
                "median_imm_minus_fragmented_per_time_bin": float(group["delta_imm_minus_fragmented_per_time_bin"].median()),
                "median_imm_minus_fragmented_per_spike": float(group["delta_imm_minus_fragmented_per_spike"].median()),
                "stationary_best_count": int(best.get("stationary", 0)),
                "diffusion_best_count": int(best.get("diffusion", 0)),
                "fragmented_best_count": int(best.get("fragmented", 0)),
                "first_order_imm_best_count": int(best.get("first_order_imm", 0)),
                "strict_clean_imm_count": int(len(strict)),
                "strict_clean_imm_animals": int(strict["animal"].nunique()),
                "strict_clean_imm_sessions": int(strict["session"].nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def selection_overlap(selection: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(selection["run"].unique())
    event_sets = {
        label: set(zip(group["session"].astype(str), group["event_id"].astype(int), strict=True))
        for label, group in selection.groupby("run")
    }
    rows = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            intersection = event_sets[left] & event_sets[right]
            union = event_sets[left] | event_sets[right]
            rows.append(
                {
                    "left_run": left,
                    "right_run": right,
                    "overlap_events": len(intersection),
                    "left_events": len(event_sets[left]),
                    "right_events": len(event_sets[right]),
                    "jaccard": len(intersection) / len(union) if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_gates(gates_by_run: dict[str, pd.DataFrame], summary: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    all_technical = all(
        bool(frame.loc[frame["gate"] == "overall_technical", "passed"].astype(bool).iloc[0])
        for frame in gates_by_run.values()
    )
    both_geometries = all(set(group["geometry"]) == {"linear", "circular"} for _, group in strict.groupby("run")) if not strict.empty else False
    checks = [
        ("all_input_runs_technical_pass", all_technical, f"runs={len(gates_by_run)}"),
        ("all_runs_have_events", bool(not summary.empty and (summary["events"] > 0).all()), f"rows={len(summary)}"),
        ("both_geometries_evaluated", both_geometries, "linear and circular represented in each decision table"),
        (
            "external_clean_imm_replication_supported",
            False,
            "descriptive stopgate: strict clean-IMM events are sparse/localized; Gate 2/3/4 not launched",
        ),
    ]
    return pd.DataFrame([{"gate": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks])


def build_report(summary: pd.DataFrame, by_geometry: pd.DataFrame, gates: pd.DataFrame) -> str:
    strongest = summary.sort_values("trajectory_confident_count", ascending=False).iloc[0]
    strict_best = summary.sort_values("strict_clean_imm_count", ascending=False).iloc[0]
    lines = [
        "# hc-11 native-ripple geometry and selection sensitivity",
        "",
        "This is a non-rescoring comparison of frozen native-ripple pilot outputs.",
        "",
        "## Main readout",
        "",
        f"The strongest trajectory subset is `{strongest['run']}`: {int(strongest['trajectory_confident_count'])}/{int(strongest['events'])} events are trajectory-confident, with median trajectory-minus-stationary {strongest['median_trajectory_minus_stationary']:+.3f}.",
        f"The largest strict clean-IMM intersection is still only {int(strict_best['strict_clean_imm_count'])}/{int(strict_best['events'])} in `{strict_best['run']}`, spanning {int(strict_best['strict_clean_imm_sessions'])} sessions and {int(strict_best['strict_clean_imm_animals'])} animals.",
        "",
        "Linear and circular summaries are similar enough that maze topology is not a persuasive explanation for the weak external IMM result. Session/animal heterogeneity and spike/decoder support are larger effects.",
        "",
        "## Overall tiers",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Geometry strata",
        "",
        "```text",
        by_geometry.to_string(index=False),
        "```",
        "",
        "## Decision",
        "",
        "hc-11 supports a replay-rich native-ripple subset, especially under pre-evidence spike-support selection, but it does not currently replicate Pfeiffer/Foster's clean-IMM result. Do not run the full Gate 2/3/4 ladder on the localized strict subset as if it were a dataset-wide confirmation.",
        "The appropriate role is external trajectory-subset/specificity evidence, with the clean-IMM claim remaining Pfeiffer/Foster-specific until another dataset passes the strict intersection.",
        "",
        "## Gates",
        "",
        "```text",
        gates.to_string(index=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def run(runs: list[tuple[str, Path]], output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    gates_by_run: dict[str, pd.DataFrame] = {}
    for label, directory in runs:
        decisions, selection, gates = read_run(label, directory)
        decision_frames.append(decisions)
        selection_frames.append(selection)
        gates_by_run[label] = gates
    decisions = pd.concat(decision_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    summary = summarize(decisions, ["run"])
    by_geometry = summarize(decisions, ["run", "geometry"])
    by_session = summarize(decisions, ["run", "animal", "session", "geometry"])
    strict = decisions[decisions["strict_clean_imm"]].copy()
    overlap = selection_overlap(selection)
    gates = build_gates(gates_by_run, summary, decisions)
    outputs = {
        SUMMARY_OUTPUT: summary,
        GEOMETRY_OUTPUT: by_geometry,
        SESSION_OUTPUT: by_session,
        STRICT_OUTPUT: strict,
        OVERLAP_OUTPUT: overlap,
        GATE_OUTPUT: gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "non_rescoring_hc11_native_ripple_sensitivity",
        "input_runs": {label: str(path) for label, path in runs},
        "claim_boundary": "descriptive external trajectory subset; not clean-IMM replication",
    }
    (output_dir / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / REPORT_OUTPUT).write_text(build_report(summary, by_geometry, gates), encoding="utf-8")
    return {"summary": summary, "by_geometry": by_geometry, "by_session": by_session, "strict": strict, "overlap": overlap, "gates": gates}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="LABEL=OUTPUT_DIR; repeat for each frozen run")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args.run, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
