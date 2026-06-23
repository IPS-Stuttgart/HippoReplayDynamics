#!/usr/bin/env python3
"""Audit model-evidence tables for exact/truncated support mixing.

The model-evidence pipeline can contain both exact full-grid evidences and
candidate-pruned truncated lower bounds.  This helper writes small CSV/text
artifacts that make that distinction visible after a sharded aggregate run.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    _coerce_bool_series,
    ensure_evidence_support_columns,
)
from hipporeplayimm.evidence_status_coercion import _status_success_mask

_EVENT_COLUMNS = ("session", "event_index")
_AUDIT_FILENAMES = {
    "support_summary": "evidence_support_summary.csv",
    "event_support": "event_evidence_support_audit.csv",
    "pairwise_support": "pairwise_evidence_support_audit.csv",
    "warnings": "evidence_support_warnings.txt",
}


def _successful_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Return successful rows with evidence-support metadata attached."""

    rows = ensure_evidence_support_columns(scores)
    if rows.empty:
        return rows
    rows = rows[_status_success_mask(rows)].copy()
    rows["evidence_comparable"] = _coerce_bool_series(rows["evidence_comparable"])
    return rows


def _event_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    return int(rows.loc[:, _EVENT_COLUMNS].drop_duplicates().shape[0])


def _joined(values: Iterable[object]) -> str:
    unique = sorted({str(value) for value in values if pd.notna(value) and str(value)})
    return ",".join(unique)


def evidence_support_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize rows, events, and diagnostics by model/support class."""

    rows = _successful_rows(scores)
    columns = [
        "model",
        "model_family",
        "evidence_support",
        "evidence_comparable",
        "sessions",
        "events",
        "rows",
        "mean_log_evidence",
        "median_log_evidence",
        "mean_runtime_s",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    if "model_family" not in rows:
        rows["model_family"] = "unknown"
    if "runtime_s" not in rows:
        rows["runtime_s"] = pd.NA

    records: list[dict[str, object]] = []
    group_cols = ["model", "model_family", "evidence_support", "evidence_comparable"]
    for key, group in rows.groupby(group_cols, dropna=False, sort=True):
        model, family, support, comparable = key
        records.append(
            {
                "model": model,
                "model_family": family,
                "evidence_support": support,
                "evidence_comparable": bool(comparable),
                "sessions": int(group["session"].nunique()),
                "events": _event_count(group),
                "rows": int(len(group)),
                "mean_log_evidence": float(group["log_evidence"].mean()),
                "median_log_evidence": float(group["log_evidence"].median()),
                "mean_runtime_s": (
                    float(group["runtime_s"].mean())
                    if group["runtime_s"].notna().any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def event_support_audit(scores: pd.DataFrame) -> pd.DataFrame:
    """Return one row per event describing which support classes are present."""

    rows = _successful_rows(scores)
    columns = [
        "session",
        "event_index",
        "models",
        "exact_models",
        "truncated_models",
        "other_support_models",
        "exact_rows",
        "truncated_rows",
        "other_support_rows",
        "comparable_rows",
        "has_mixed_exact_truncated",
        "has_uncomparable_rows",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    for (session, event_index), group in rows.groupby(list(_EVENT_COLUMNS), sort=True):
        exact = group[group["evidence_support"].eq(EXACT_EVIDENCE_SUPPORT)]
        truncated = group[group["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
        other = group[
            ~group["evidence_support"].isin(
                [EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT]
            )
        ]
        comparable = group[_coerce_bool_series(group["evidence_comparable"])]
        records.append(
            {
                "session": session,
                "event_index": int(event_index),
                "models": _joined(group["model"]),
                "exact_models": _joined(exact["model"]),
                "truncated_models": _joined(truncated["model"]),
                "other_support_models": _joined(other["model"]),
                "exact_rows": int(len(exact)),
                "truncated_rows": int(len(truncated)),
                "other_support_rows": int(len(other)),
                "comparable_rows": int(len(comparable)),
                "has_mixed_exact_truncated": bool(len(exact) and len(truncated)),
                "has_uncomparable_rows": bool(
                    (~_coerce_bool_series(group["evidence_comparable"])).any()
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def pairwise_support_audit(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize support classes for every model pair observed on an event."""

    rows = _successful_rows(scores)
    columns = [
        "model_a",
        "model_b",
        "support_a",
        "support_b",
        "model_a_comparable",
        "model_b_comparable",
        "events",
        "sessions",
        "comparison_support",
        "mixes_exact_and_truncated",
        "both_exact_comparable",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    event_pair_rows: list[dict[str, object]] = []
    for (session, event_index), group in rows.groupby(list(_EVENT_COLUMNS), sort=True):
        per_model = group.sort_values("model").drop_duplicates("model", keep="first")
        model_rows = {str(row.model): row for row in per_model.itertuples(index=False)}
        for model_a, model_b in combinations(sorted(model_rows), 2):
            row_a = model_rows[model_a]
            row_b = model_rows[model_b]
            support_a = str(row_a.evidence_support)
            support_b = str(row_b.evidence_support)
            comparable_a = bool(row_a.evidence_comparable)
            comparable_b = bool(row_b.evidence_comparable)
            event_pair_rows.append(
                {
                    "session": session,
                    "event_index": int(event_index),
                    "model_a": model_a,
                    "model_b": model_b,
                    "support_a": support_a,
                    "support_b": support_b,
                    "model_a_comparable": comparable_a,
                    "model_b_comparable": comparable_b,
                }
            )

    if not event_pair_rows:
        return pd.DataFrame(columns=columns)

    pair_rows = pd.DataFrame.from_records(event_pair_rows)
    records: list[dict[str, object]] = []
    group_cols = [
        "model_a",
        "model_b",
        "support_a",
        "support_b",
        "model_a_comparable",
        "model_b_comparable",
    ]
    for key, group in pair_rows.groupby(group_cols, dropna=False, sort=True):
        model_a, model_b, support_a, support_b, comparable_a, comparable_b = key
        mixes = {
            str(support_a),
            str(support_b),
        } == {EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT}
        both_exact = (
            str(support_a) == EXACT_EVIDENCE_SUPPORT
            and str(support_b) == EXACT_EVIDENCE_SUPPORT
            and bool(comparable_a)
            and bool(comparable_b)
        )
        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "support_a": support_a,
                "support_b": support_b,
                "model_a_comparable": bool(comparable_a),
                "model_b_comparable": bool(comparable_b),
                "events": _event_count(group),
                "sessions": int(group["session"].nunique()),
                "comparison_support": f"{support_a}_vs_{support_b}",
                "mixes_exact_and_truncated": bool(mixes),
                "both_exact_comparable": bool(both_exact),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def evidence_support_warnings(
    event_audit: pd.DataFrame,
    pairwise_audit: pd.DataFrame,
    *,
    max_pairs: int = 50,
) -> str:
    """Build a reviewer-facing text warning for mixed-support comparisons."""

    lines: list[str] = []
    mixed_events = (
        int(event_audit["has_mixed_exact_truncated"].sum())
        if not event_audit.empty
        else 0
    )
    total_events = int(len(event_audit))
    mixed_pairs = (
        pairwise_audit[pairwise_audit["mixes_exact_and_truncated"]]
        if not pairwise_audit.empty
        else pd.DataFrame()
    )

    if mixed_events == 0 and mixed_pairs.empty:
        return "No mixed exact/truncated evidence-support comparisons were detected.\n"

    lines.append(
        "Mixed evidence-support audit: exact full-grid evidences and candidate-pruned "
        "truncated lower bounds appear in the same event/model comparison set."
    )
    lines.append(f"Events with both exact and truncated rows: {mixed_events} / {total_events}")
    if not mixed_pairs.empty:
        lines.append("Model-pair support combinations requiring cautious interpretation:")
        display = mixed_pairs.sort_values(
            ["events", "model_a", "model_b"],
            ascending=[False, True, True],
        )
        for _, row in display.head(max_pairs).iterrows():
            lines.append(
                f"- {row['model_a']} ({row['support_a']}) vs {row['model_b']} "
                f"({row['support_b']}): {int(row['events'])} events"
            )
        if len(display) > max_pairs:
            lines.append(
                f"- ... {len(display) - max_pairs} additional "
                "mixed-support model pairs omitted"
            )
    lines.append(
        "Recommended reporting: label candidate-supported momentum/IMM comparisons as "
        "truncated lower-bound or approximation-based comparisons unless a separate "
        "candidate-support convergence check establishes stability."
    )
    return "\n".join(lines) + "\n"


def write_evidence_support_audit(
    scores: pd.DataFrame,
    outdir: Path | str,
) -> dict[str, pd.DataFrame | str]:
    """Write support-audit artifacts next to the usual model-evidence outputs."""

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support = evidence_support_summary(scores)
    events = event_support_audit(scores)
    pairwise = pairwise_support_audit(scores)
    warnings = evidence_support_warnings(events, pairwise)

    support.to_csv(output_dir / _AUDIT_FILENAMES["support_summary"], index=False)
    events.to_csv(output_dir / _AUDIT_FILENAMES["event_support"], index=False)
    pairwise.to_csv(output_dir / _AUDIT_FILENAMES["pairwise_support"], index=False)
    (output_dir / _AUDIT_FILENAMES["warnings"]).write_text(warnings, encoding="utf-8")

    return {
        "support_summary": support,
        "event_support": events,
        "pairwise_support": pairwise,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit model-evidence CSVs for exact/truncated support mixing."
    )
    parser.add_argument("scores_csv", help="Path to event_model_evidence.csv")
    parser.add_argument(
        "--output",
        default="results/model-evidence",
        help="Directory for audit CSV/text outputs",
    )
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    outputs = write_evidence_support_audit(scores, Path(args.output))
    print(outputs["warnings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
