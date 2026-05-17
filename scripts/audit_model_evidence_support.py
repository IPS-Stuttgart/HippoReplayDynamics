#!/usr/bin/env python3
"""Audit exact versus candidate-supported model-evidence summaries.

This utility is intentionally independent of the all-session aggregator.  It can
be run on existing ``event_model_evidence.csv`` or
``all_sessions_event_model_evidence.csv`` artifacts to generate paper-side
checks that do not mix exact full-grid evidences with candidate-supported
truncated lower bounds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXACT_EVIDENCE_SUPPORT = "exact_full_grid"
TRUNCATED_EVIDENCE_SUPPORT = "truncated_full_grid"
_SUPPORT_DIAGNOSTICS = (
    "diagnostic_candidate_evidence_support",
    "diagnostic_state_space_momentum_evidence_support",
    "diagnostic_state_space_imm_evidence_support",
)
_STATE_SPACE_PREFIXES = (
    "clusterless-state-space-",
    "sorted-spike-state-space-",
    "state-space-",
)


def infer_evidence_support(row: pd.Series) -> str:
    """Infer evidence support from diagnostic columns used by the benchmark."""

    if str(row.get("status", "success")) != "success":
        return "not_scored"
    for column in _SUPPORT_DIAGNOSTICS:
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value)
        if text in {EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT}:
            return text
    return EXACT_EVIDENCE_SUPPORT


def ensure_evidence_support_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``evidence_support`` and ``evidence_comparable`` if absent."""

    out = df.copy()
    if out.empty:
        return out
    inferred = out.apply(infer_evidence_support, axis=1)
    if "evidence_support" in out:
        existing = out["evidence_support"].astype(object)
        missing = existing.isna() | existing.astype(str).str.len().eq(0)
        out["evidence_support"] = existing.where(~missing, inferred)
    else:
        out["evidence_support"] = inferred
    status_ok = out["status"].eq("success") if "status" in out else pd.Series(True, index=out.index)
    out["evidence_comparable"] = status_ok & out["evidence_support"].eq(EXACT_EVIDENCE_SUPPORT)
    return out


def evidence_support_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact and truncated evidence rows by model."""

    df = ensure_evidence_support_columns(df)
    ok = df[df.get("status", "success") == "success"].copy()
    columns = [
        "model",
        "model_family",
        "evidence_support",
        "evidence_comparable",
        "sessions",
        "events",
        "wins",
        "truncated_lower_bound_wins",
        "mean_log_evidence",
        "mean_relative_log_evidence",
        "mean_truncated_relative_log_evidence",
    ]
    if ok.empty:
        return pd.DataFrame(columns=columns)
    if "model_family" not in ok:
        ok["model_family"] = "unknown"
    for name, default in (
        ("is_best_model", False),
        ("is_best_truncated_lower_bound", False),
        ("relative_log_evidence", np.nan),
        ("truncated_relative_log_evidence", np.nan),
    ):
        if name not in ok:
            ok[name] = default
    summary = ok.groupby(["model", "model_family", "evidence_support", "evidence_comparable"], as_index=False).agg(
        sessions=("session", "nunique"),
        events=("event_index", "count"),
        wins=("is_best_model", "sum"),
        truncated_lower_bound_wins=("is_best_truncated_lower_bound", "sum"),
        mean_log_evidence=("log_evidence", "mean"),
        mean_relative_log_evidence=("relative_log_evidence", "mean"),
        mean_truncated_relative_log_evidence=("truncated_relative_log_evidence", "mean"),
    )
    return summary[columns].sort_values(
        ["evidence_comparable", "truncated_lower_bound_wins", "wins", "mean_log_evidence"],
        ascending=[False, False, False, False],
    )


def _model_prefix(model: str | None) -> str | None:
    if not model:
        return None
    for prefix in _STATE_SPACE_PREFIXES:
        if str(model).startswith(prefix):
            return prefix
    return None


def resolve_model(models: list[str], role: str, *, reference_model: str | None = None) -> str | None:
    """Resolve ``momentum`` to a matching concrete model name, if present."""

    available = list(dict.fromkeys(str(model) for model in models if str(model)))
    available_set = set(available)
    ref_prefix = _model_prefix(reference_model)
    candidates: list[str] = []
    if ref_prefix is not None:
        candidates.append(f"{ref_prefix}{role}")
    candidates.append(role)
    candidates.extend(f"{prefix}{role}" for prefix in _STATE_SPACE_PREFIXES)
    for candidate in candidates:
        if candidate in available_set:
            return candidate
    suffix = f"-{role}"
    matches = [model for model in available if model.endswith(suffix)]
    return sorted(matches)[-1] if matches else None


def comparison_support(left_support: object, right_support: object, left_comparable: object, right_comparable: object) -> str:
    """Label whether a paired delta compares exact evidence or lower bounds."""

    left_exact = bool(left_comparable) and str(left_support) == EXACT_EVIDENCE_SUPPORT
    right_exact = bool(right_comparable) and str(right_support) == EXACT_EVIDENCE_SUPPORT
    left_truncated = str(left_support) == TRUNCATED_EVIDENCE_SUPPORT
    right_truncated = str(right_support) == TRUNCATED_EVIDENCE_SUPPORT
    if left_exact and right_exact:
        return EXACT_EVIDENCE_SUPPORT
    if left_truncated and right_truncated:
        return TRUNCATED_EVIDENCE_SUPPORT
    if left_truncated and right_exact:
        return "truncated_lower_bound_vs_exact"
    if left_exact and right_truncated:
        return "exact_vs_truncated_lower_bound"
    return f"{left_support}_vs_{right_support}"


def _pair_delta_rows(log_pivot: pd.DataFrame, support_pivot: pd.DataFrame, comparable_pivot: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    if left not in log_pivot or right not in log_pivot:
        return pd.DataFrame()
    rows = pd.DataFrame(
        {
            "delta": log_pivot[left] - log_pivot[right],
            "left_support": support_pivot[left],
            "right_support": support_pivot[right],
            "left_comparable": comparable_pivot[left].fillna(False).astype(bool),
            "right_comparable": comparable_pivot[right].fillna(False).astype(bool),
        }
    ).dropna(subset=["delta"])
    if rows.empty:
        return rows
    rows = rows.reset_index()
    rows["comparison_support"] = [
        comparison_support(ls, rs, lc, rc)
        for ls, rs, lc, rc in zip(rows["left_support"], rows["right_support"], rows["left_comparable"], rows["right_comparable"], strict=True)
    ]
    return rows


def _summarize_delta_rows(rows: pd.DataFrame, comparison: str, left: str, right: str) -> list[dict[str, object]]:
    if rows.empty:
        return []
    out: list[dict[str, object]] = []
    for (session, support), group in rows.groupby(["session", "comparison_support"], sort=True):
        deltas = group["delta"].astype(float)
        events = int(deltas.shape[0])
        positives = int((deltas > 0.0).sum())
        out.append(
            {
                "session": session,
                "comparison": comparison,
                "comparison_support": support,
                "left_model": left,
                "right_model": right,
                "events": events,
                "positive_events": positives,
                "positive_fraction": positives / events if events else float("nan"),
                "mean_delta": float(deltas.mean()),
                "median_delta": float(deltas.median()),
                "min_delta": float(deltas.min()),
                "max_delta": float(deltas.max()),
            }
        )
    return out


def paired_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create support-labelled session-level paired-delta summaries."""

    df = ensure_evidence_support_columns(df)
    ok = df[df.get("status", "success") == "success"].copy()
    columns = [
        "session",
        "comparison",
        "comparison_support",
        "left_model",
        "right_model",
        "events",
        "positive_events",
        "positive_fraction",
        "mean_delta",
        "median_delta",
        "min_delta",
        "max_delta",
    ]
    if ok.empty:
        return pd.DataFrame(columns=columns)
    models = sorted(ok["model"].dropna().astype(str).unique())
    log_pivot = ok.pivot_table(index=["session", "event_index"], columns="model", values="log_evidence", aggfunc="first")
    support_pivot = ok.pivot_table(index=["session", "event_index"], columns="model", values="evidence_support", aggfunc="first")
    comparable_pivot = ok.pivot_table(index=["session", "event_index"], columns="model", values="evidence_comparable", aggfunc="first")

    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_pair(left: str | None, right: str | None, label: str | None = None) -> None:
        if left is None or right is None or left == right:
            return
        comparison = label or f"{left}_minus_{right}"
        key = (comparison, left, right)
        if key in seen:
            return
        seen.add(key)
        pair_rows = _pair_delta_rows(log_pivot, support_pivot, comparable_pivot, left, right)
        out.extend(_summarize_delta_rows(pair_rows, comparison, left, right))

    momentum = resolve_model(models, "momentum")
    diffusion = resolve_model(models, "diffusion", reference_model=momentum)
    add_pair(momentum, diffusion)

    imm = resolve_model(models, "imm")
    if imm is not None:
        for role in ("momentum", "diffusion", "stationary", "random"):
            add_pair(imm, resolve_model(models, role, reference_model=imm))
    if not out:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out)[columns].sort_values(["comparison", "comparison_support", "session"])


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((phat * (1.0 - phat) + z**2 / (4 * total)) / total) / denom
    return (center - spread, center + spread)


def binomial_two_sided_p_value(successes: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    smaller_tail = min(successes, total - successes)
    tail = 0.0
    for k in range(smaller_tail + 1):
        log_prob = math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1) + total * math.log(0.5)
        tail += math.exp(log_prob)
    return min(1.0, 2.0 * tail)


def pooled_paired_delta_summary(delta_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison",
        "comparison_support",
        "left_model",
        "right_model",
        "sessions",
        "events",
        "positive_events",
        "positive_fraction",
        "wilson_95_low",
        "wilson_95_high",
        "mean_delta",
        "median_of_session_medians",
        "two_sided_sign_test_p",
    ]
    if delta_table.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in delta_table.groupby(["comparison", "comparison_support", "left_model", "right_model"], sort=True):
        comparison, support, left, right = keys
        events = int(group["events"].sum())
        positives = int(group["positive_events"].sum())
        low, high = wilson_interval(positives, events)
        weights = group["events"].astype(float)
        rows.append(
            {
                "comparison": comparison,
                "comparison_support": support,
                "left_model": left,
                "right_model": right,
                "sessions": int(group["session"].nunique()),
                "events": events,
                "positive_events": positives,
                "positive_fraction": positives / events if events else float("nan"),
                "wilson_95_low": low,
                "wilson_95_high": high,
                "mean_delta": float(np.average(group["mean_delta"], weights=weights)) if events else float("nan"),
                "median_of_session_medians": float(group["median_delta"].median()),
                "two_sided_sign_test_p": binomial_two_sided_p_value(positives, events),
            }
        )
    return pd.DataFrame(rows)[columns]


def write_audit(input_csv: Path, output_dir: Path) -> None:
    scores = pd.read_csv(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    support = evidence_support_summary(scores)
    session_delta = paired_delta_summary(scores)
    pooled_delta = pooled_paired_delta_summary(session_delta)
    support.to_csv(output_dir / "evidence_support_summary.csv", index=False)
    session_delta.to_csv(output_dir / "session_paired_delta_summary.csv", index=False)
    pooled_delta.to_csv(output_dir / "pooled_paired_delta_summary.csv", index=False)
    print("Evidence-support summary:")
    print(support.to_string(index=False))
    if not pooled_delta.empty:
        print("\nPooled paired deltas:")
        print(pooled_delta.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exact/truncated support in model-evidence artifacts.")
    parser.add_argument("input_csv", type=Path, help="event_model_evidence.csv or all_sessions_event_model_evidence.csv")
    parser.add_argument("--output", type=Path, default=Path("results/model-evidence-support-audit"))
    args = parser.parse_args()
    write_audit(args.input_csv, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
