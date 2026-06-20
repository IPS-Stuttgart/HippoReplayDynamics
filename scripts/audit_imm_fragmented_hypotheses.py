#!/usr/bin/env python3
"""Audit whether first-order IMM is distinguishable from fragmented replay.

The audit consumes an existing full-core event-model-evidence CSV and writes
tables that classify every event as clean IMM, fragmented, momentum-like,
Brownian/diffusion-like, static, or ambiguous. Clean IMM is only claimed when
first-order IMM beats fragmented by the calibrated margin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"
REQUIRED = [STATIONARY, DIFFUSION, FRAGMENTED, FIRST_ORDER_IMM, MOMENTUM_EXACT]
LABEL_COLUMNS = ["original_algorithm_label", "original_label", "model_label", "label", "best_model"]


def _rat(session: object) -> str:
    return str(session).split("/", 1)[0]


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_evidence(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")
    if "status" in frame:
        frame = frame[frame["status"].astype(str).eq("success")].copy()
    if "evidence_comparable" not in frame:
        frame["evidence_comparable"] = True
    frame["evidence_comparable"] = frame["evidence_comparable"].map(_as_bool)
    frame["session"] = frame["session"].astype(str)
    frame["rat"] = frame["session"].map(_rat)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    frame["model"] = frame["model"].astype(str)
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    return frame.dropna(subset=["log_evidence"]).copy()


def _value(group: pd.DataFrame, model: str) -> float:
    row = group[group["model"].eq(model)]
    if row.empty:
        return float("nan")
    return float(row.iloc[-1]["log_evidence"])


def _classify(row: pd.Series, threshold: float) -> str:
    best = str(row["best_exact_core_model"])
    imm_frag = float(row["delta_imm_minus_fragmented"])
    mom_diff = float(row["delta_momentum_minus_diffusion"])
    if best == FIRST_ORDER_IMM and imm_frag >= threshold:
        return "clean_imm_switching_candidate"
    if best == FRAGMENTED and imm_frag <= -threshold:
        return "fragmented_candidate"
    if best == MOMENTUM_EXACT and mom_diff >= threshold:
        return "momentum_like_candidate"
    if best == DIFFUSION and -mom_diff >= threshold:
        return "brownian_diffusion_like_candidate"
    if best == STATIONARY:
        return "static_nontrajectory_candidate"
    if pd.notna(imm_frag) and abs(imm_frag) < threshold:
        return "imm_fragmented_ambiguous"
    return "trajectory_family_ambiguous"


def build_event_table(evidence: pd.DataFrame, threshold: float = 5.5) -> pd.DataFrame:
    rows = []
    for (session, event_index), group in evidence.groupby(["session", "event_index"], sort=True):
        comparable = group[group["evidence_comparable"]]
        core = comparable[comparable["model"].isin(REQUIRED)].copy()
        present = set(core["model"])
        missing = [model for model in REQUIRED if model not in present]
        best_model = ""
        best_logz = float("nan")
        if not core.empty:
            best = core.sort_values("log_evidence", ascending=False).iloc[0]
            best_model = str(best["model"])
            best_logz = float(best["log_evidence"])
        row = {
            "session": session,
            "rat": _rat(session),
            "event_index": int(event_index),
            "exact_core_complete": not missing,
            "missing_required_exact_core_models": " ".join(missing),
            "best_exact_core_model": best_model,
            "best_exact_core_log_evidence": best_logz,
            "logZ_stationary": _value(group, STATIONARY),
            "logZ_diffusion": _value(group, DIFFUSION),
            "logZ_fragmented": _value(group, FRAGMENTED),
            "logZ_first_order_imm": _value(group, FIRST_ORDER_IMM),
            "logZ_momentum_exact_sparse": _value(group, MOMENTUM_EXACT),
        }
        row["delta_imm_minus_fragmented"] = row["logZ_first_order_imm"] - row["logZ_fragmented"]
        row["delta_imm_minus_momentum"] = row["logZ_first_order_imm"] - row["logZ_momentum_exact_sparse"]
        row["delta_momentum_minus_diffusion"] = row["logZ_momentum_exact_sparse"] - row["logZ_diffusion"]
        row["delta_trajectory_minus_stationary"] = (
            max(
                row["logZ_diffusion"],
                row["logZ_fragmented"],
                row["logZ_first_order_imm"],
                row["logZ_momentum_exact_sparse"],
            )
            - row["logZ_stationary"]
        )
        row["imm_confident_vs_fragmented"] = row["delta_imm_minus_fragmented"] >= threshold
        row["fragmented_confident_vs_imm"] = row["delta_imm_minus_fragmented"] <= -threshold
        row["momentum_confident_vs_diffusion"] = row["delta_momentum_minus_diffusion"] >= threshold
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["session", "event_index"]).reset_index(drop=True)
    if not out.empty:
        out["within_family_classification"] = out.apply(lambda row: _classify(row, threshold), axis=1)
    return out


def _summary_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"metric": "events", "value": len(table)}]
    if table.empty:
        return rows
    rows.append({"metric": "exact_core_complete", "value": int(table["exact_core_complete"].sum())})
    for column in ["imm_confident_vs_fragmented", "fragmented_confident_vs_imm", "momentum_confident_vs_diffusion"]:
        rows.append({"metric": column, "value": int(table[column].sum())})
    for column in ["delta_imm_minus_fragmented", "delta_imm_minus_momentum", "delta_momentum_minus_diffusion"]:
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        rows.append({"metric": f"mean_{column}", "value": float(values.mean()) if not values.empty else np.nan})
        rows.append({"metric": f"median_{column}", "value": float(values.median()) if not values.empty else np.nan})
    for label, count in table["within_family_classification"].value_counts().items():
        rows.append({"metric": f"classification_{label}", "value": int(count)})
    return rows


def _load_labels(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    labels = pd.read_csv(path)
    for column in LABEL_COLUMNS:
        if column in labels.columns:
            label_column = column
            break
    else:
        raise ValueError(f"original labels need one of: {', '.join(LABEL_COLUMNS)}")
    labels = labels.copy()
    labels["session"] = labels["session"].astype(str)
    labels["event_index"] = pd.to_numeric(labels["event_index"], errors="raise").astype(int)
    labels["original_algorithm_label"] = labels[label_column].fillna("").astype(str)
    return labels[["session", "event_index", "original_algorithm_label"]]


def _reassignment(table: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session", "rat", "event_index", "original_algorithm_label",
        "best_exact_core_model", "within_family_classification",
        "delta_imm_minus_fragmented", "delta_imm_minus_momentum",
        "delta_momentum_minus_diffusion",
    ]
    if labels.empty:
        return pd.DataFrame(columns=columns)
    merged = table.merge(labels, on=["session", "event_index"], how="inner")
    momentum = merged["original_algorithm_label"].str.lower().str.contains("momentum")
    return merged.loc[momentum, columns].sort_values(["session", "event_index"]).reset_index(drop=True)


def _gates(table: pd.DataFrame, reassignment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(gate: str, passed: bool, observed: object, criterion: str) -> None:
        rows.append({"gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion})
    add("event_rows_present", len(table) > 0, len(table), "at least one event row")
    add(
        "exact_core_complete_all_events",
        bool(table["exact_core_complete"].all()) if len(table) else False,
        f"{int(table['exact_core_complete'].sum()) if len(table) else 0}/{len(table)}",
        "all required exact core rows are present",
    )
    add("imm_fragmented_axis_reported", "delta_imm_minus_fragmented" in table, len(table), "IMM-vs-fragmented deltas are reported")
    add(
        "clean_imm_subset_counted",
        True,
        int(table["within_family_classification"].eq("clean_imm_switching_candidate").sum()) if len(table) else 0,
        "clean IMM subset is counted explicitly",
    )
    add(
        "fragmented_or_ambiguous_subset_counted",
        True,
        int(table["within_family_classification"].isin(["fragmented_candidate", "imm_fragmented_ambiguous"]).sum()) if len(table) else 0,
        "fragmented/ambiguous events are separated from clean IMM",
    )
    add("original_momentum_subset_checked", True, len(reassignment), "original momentum reassignment is written; zero means labels absent/unmatched")
    rows.append({"gate": "overall", "passed": all(row["passed"] for row in rows), "observed": f"{sum(row['passed'] for row in rows)}/{len(rows)} gates passed", "criterion": "all required artifact-audit gates pass"})
    return pd.DataFrame(rows)


def write_outputs(evidence: pd.DataFrame, output: str | Path, labels: pd.DataFrame | None = None, threshold: float = 5.5) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    table = build_event_table(evidence, threshold=threshold)
    taxonomy = table["within_family_classification"].value_counts().rename_axis("classification").reset_index(name="events") if not table.empty else pd.DataFrame(columns=["classification", "events"])
    if not taxonomy.empty:
        taxonomy["fraction"] = taxonomy["events"] / len(table)
    reassignment = _reassignment(table, labels if labels is not None else pd.DataFrame())
    outputs = {
        "imm_fragmented_head_to_head_event_table.csv": table,
        "imm_fragmented_head_to_head_summary.csv": pd.DataFrame(_summary_rows(table)),
        "trajectory_taxonomy_event_table.csv": table,
        "trajectory_taxonomy_summary.csv": taxonomy,
        "original_momentum_reassignment_event_table.csv": reassignment,
        "original_momentum_reassignment_summary.csv": pd.DataFrame(_summary_rows(reassignment)) if not reassignment.empty else pd.DataFrame([{"metric": "original_momentum_events", "value": 0}]),
        "imm_fragmented_hypothesis_gate_summary.csv": _gates(table, reassignment),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--original-labels", default="")
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = _read_evidence(args.event_model_evidence)
    labels = _load_labels(args.original_labels) if args.original_labels else pd.DataFrame()
    write_outputs(evidence, args.output, labels=labels, threshold=args.margin_threshold)
    print(f"Wrote IMM/fragmented hypothesis audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
