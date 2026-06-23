#!/usr/bin/env python3
"""Compare replay model-evidence artifacts with support-aware reporting.

This utility is meant for paper-side rerun triage.  It accepts either a single
CSV file or an artifact directory containing one of:

* event_model_evidence.csv
* all_sessions_event_model_evidence.csv

It canonicalizes model names across KD-aligned, sorted-spike state-space, and
clusterless state-space runs; optionally filters to exact-comparable evidence;
and writes compact tables that answer whether a rerun changed the model story.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    _coerce_bool_series,
    ensure_evidence_support_columns,
)

_SCORE_FILENAMES = ("event_model_evidence.csv", "all_sessions_event_model_evidence.csv")
_CANONICAL_MODELS = {
    "random",
    "stationary",
    "stationary-gaussian",
    "diffusion",
    "momentum",
    "imm",
    "fragmented",
}
_MISSING_STATUS_VALUES = {"", "nan", "none", "null", "<na>"}


def canonical_model_name(model: object) -> str:
    """Map implementation-specific model names onto dynamics labels."""

    name = str(model).strip().lower()
    for prefix in ("sorted-spike-state-space-", "clusterless-state-space-", "state-space-"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    if name == "jump":
        return "fragmented"
    if name == "momentum-exact-sparse":
        return "momentum"
    return name if name in _CANONICAL_MODELS else name


def compare_artifacts(
    left: str | Path,
    right: str | Path,
    *,
    left_label: str = "left",
    right_label: str = "right",
    output: str | Path = "results/model-evidence-artifact-comparison",
    exact_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compare two model-evidence artifacts and write CSV summaries."""

    left_scores = load_scores(left, left_label, exact_only=exact_only)
    right_scores = load_scores(right, right_label, exact_only=exact_only)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_best = event_best_models(left_scores, left_label)
    right_best = event_best_models(right_scores, right_label)
    best_comparison = left_best.merge(right_best, on=["session", "event_index"], how="inner")
    if not best_comparison.empty:
        best_comparison["canonical_best_agree"] = (
            best_comparison[f"{left_label}_canonical_best_model"]
            == best_comparison[f"{right_label}_canonical_best_model"]
        )
    else:
        best_comparison["canonical_best_agree"] = pd.Series(dtype=bool)
    best_comparison.to_csv(out_dir / "event_best_model_comparison.csv", index=False)

    crosstab = best_model_crosstab(best_comparison, left_label, right_label)
    crosstab.to_csv(out_dir / "canonical_best_model_crosstab.csv", index=False)

    support_counts = pd.concat(
        [
            evidence_support_counts(left_scores, left_label),
            evidence_support_counts(right_scores, right_label),
        ],
        ignore_index=True,
    )
    support_counts.to_csv(out_dir / "evidence_support_counts.csv", index=False)

    relative = shared_relative_evidence(left_scores, right_scores, left_label, right_label)
    relative.to_csv(out_dir / "shared_relative_evidence.csv", index=False)

    relative_summary = shared_relative_evidence_summary(relative, left_label, right_label)
    relative_summary.to_csv(out_dir / "shared_relative_evidence_summary.csv", index=False)

    session_summary = session_story_shift_summary(best_comparison, relative, left_label, right_label)
    session_summary.to_csv(out_dir / "session_story_shift_summary.csv", index=False)

    summary = run_comparison_summary(
        left_scores,
        right_scores,
        best_comparison,
        relative,
        left_label,
        right_label,
        exact_only=exact_only,
    )
    summary.to_csv(out_dir / "model_evidence_artifact_comparison_summary.csv", index=False)

    return {
        "summary": summary,
        "support_counts": support_counts,
        "best_comparison": best_comparison,
        "crosstab": crosstab,
        "relative": relative,
        "relative_summary": relative_summary,
        "session_summary": session_summary,
    }


def find_score_file(root: str | Path) -> Path:
    """Return a model-evidence CSV from a file path or artifact directory."""

    path = Path(root)
    if path.is_file():
        return path
    for name in _SCORE_FILENAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    searched = ", ".join(_SCORE_FILENAMES)
    raise FileNotFoundError(f"No score CSV found under {path}; searched: {searched}")


def load_scores(root: str | Path, run_label: str, *, exact_only: bool = False) -> pd.DataFrame:
    """Load and normalize one model-evidence artifact."""

    source = find_score_file(root)
    frame = pd.read_csv(source)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if "status" in frame:
        frame = frame[_status_success_mask(frame)].copy()
    frame = ensure_evidence_support_columns(frame)
    frame["evidence_comparable"] = _coerce_bool_series(frame["evidence_comparable"])
    if exact_only:
        frame = frame[frame["evidence_comparable"]].copy()
    frame["source_score_file"] = str(source)
    frame["run_label"] = run_label
    frame["canonical_model"] = frame["model"].map(canonical_model_name)
    frame = add_relative_log_evidence(frame, force=exact_only)
    return frame


def _status_success_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    return frame["status"].map(_status_is_success).astype(bool)


def _status_is_success(value: object) -> bool:
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text == "success" or text in _MISSING_STATUS_VALUES


def add_relative_log_evidence(frame: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Add within-event relative log evidence if not already available."""

    out = frame.copy()
    if "relative_log_evidence" in out and not force:
        return out
    if out.empty:
        out["relative_log_evidence"] = pd.Series(dtype=float)
        return out
    out["relative_log_evidence"] = (
        out["log_evidence"]
        - out.groupby(["session", "event_index"])["log_evidence"].transform("max")
    )
    return out


def event_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame[["session", "event_index"]].drop_duplicates().shape[0])


def source_file(frame: pd.DataFrame) -> str:
    if frame.empty or "source_score_file" not in frame:
        return ""
    return str(frame["source_score_file"].iloc[0])


def event_best_models(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    columns = [
        "session",
        "event_index",
        f"{label}_best_model",
        f"{label}_canonical_best_model",
        f"{label}_best_log_evidence",
        f"{label}_best_relative_log_evidence",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    best = frame.sort_values(
        ["session", "event_index", "log_evidence"],
        ascending=[True, True, False],
    ).drop_duplicates(["session", "event_index"], keep="first")
    return best[
        ["session", "event_index", "model", "canonical_model", "log_evidence", "relative_log_evidence"]
    ].rename(
        columns={
            "model": f"{label}_best_model",
            "canonical_model": f"{label}_canonical_best_model",
            "log_evidence": f"{label}_best_log_evidence",
            "relative_log_evidence": f"{label}_best_relative_log_evidence",
        }
    )


def best_model_crosstab(best_comparison: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    left_col = f"{left_label}_canonical_best_model"
    right_col = f"{right_label}_canonical_best_model"
    if best_comparison.empty:
        return pd.DataFrame(columns=[left_col])
    return pd.crosstab(best_comparison[left_col], best_comparison[right_col]).reset_index()


def evidence_support_counts(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    columns = ["run_label", "evidence_support", "evidence_comparable", "rows", "session_events", "models"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (support, comparable), group in frame.groupby(["evidence_support", "evidence_comparable"], sort=True):
        rows.append(
            {
                "run_label": label,
                "evidence_support": support,
                "evidence_comparable": bool(comparable),
                "rows": int(len(group)),
                "session_events": int(group[["session", "event_index"]].drop_duplicates().shape[0]),
                "models": int(group["model"].nunique()),
            }
        )
    return pd.DataFrame(rows)[columns]


def canonical_relative_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["session", "event_index", "canonical_model", "model", "relative_log_evidence"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    best_by_canonical = frame.sort_values(
        ["session", "event_index", "canonical_model", "log_evidence"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["session", "event_index", "canonical_model"], keep="first")
    return best_by_canonical[columns]


def shared_relative_evidence(left: pd.DataFrame, right: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    key = ["session", "event_index", "canonical_model"]
    delta_col = f"{right_label}_minus_{left_label}_relative_log_evidence"
    columns = [
        *key,
        f"{left_label}_model",
        f"{left_label}_relative_log_evidence",
        f"{right_label}_model",
        f"{right_label}_relative_log_evidence",
        delta_col,
    ]
    if left.empty or right.empty:
        return pd.DataFrame(columns=columns)

    left_rel = canonical_relative_table(left).rename(
        columns={
            "model": f"{left_label}_model",
            "relative_log_evidence": f"{left_label}_relative_log_evidence",
        }
    )
    right_rel = canonical_relative_table(right).rename(
        columns={
            "model": f"{right_label}_model",
            "relative_log_evidence": f"{right_label}_relative_log_evidence",
        }
    )
    joined = left_rel.merge(right_rel, on=key, how="inner")
    joined[delta_col] = (
        joined[f"{right_label}_relative_log_evidence"]
        - joined[f"{left_label}_relative_log_evidence"]
    )
    return joined[columns]


def shared_relative_evidence_summary(relative: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    if relative.empty:
        return pd.DataFrame()
    delta_col = f"{right_label}_minus_{left_label}_relative_log_evidence"
    return (
        relative.groupby("canonical_model", as_index=False)
        .agg(
            matched_events=("event_index", "count"),
            left_mean_relative_log_evidence=(f"{left_label}_relative_log_evidence", "mean"),
            right_mean_relative_log_evidence=(f"{right_label}_relative_log_evidence", "mean"),
            mean_right_minus_left_relative_log_evidence=(delta_col, "mean"),
            median_right_minus_left_relative_log_evidence=(delta_col, "median"),
            positive_right_minus_left_events=(delta_col, lambda values: int((values > 0.0).sum())),
        )
        .assign(
            positive_right_minus_left_fraction=lambda df: df["positive_right_minus_left_events"]
            / df["matched_events"].clip(lower=1)
        )
        .sort_values("matched_events", ascending=False)
    )


def session_story_shift_summary(
    best_comparison: pd.DataFrame,
    relative: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    columns = [
        "session",
        "matched_events",
        "canonical_best_agreement_fraction",
        f"{left_label}_momentum_wins",
        f"{right_label}_momentum_wins",
        "momentum_win_delta",
        "mean_momentum_relative_evidence_delta",
    ]
    if best_comparison.empty:
        return pd.DataFrame(columns=columns)

    sessions = (
        best_comparison.groupby("session", as_index=False)
        .agg(
            matched_events=("event_index", "count"),
            canonical_best_agreement_fraction=("canonical_best_agree", "mean"),
        )
        .set_index("session")
    )
    for label in (left_label, right_label):
        col = f"{label}_canonical_best_model"
        wins = (
            best_comparison.loc[best_comparison[col].eq("momentum")]
            .groupby("session")
            .size()
            .rename(f"{label}_momentum_wins")
        )
        sessions = sessions.join(wins, how="left")
        sessions[f"{label}_momentum_wins"] = sessions[f"{label}_momentum_wins"].fillna(0).astype(int)

    sessions["momentum_win_delta"] = sessions[f"{right_label}_momentum_wins"] - sessions[f"{left_label}_momentum_wins"]
    delta_col = f"{right_label}_minus_{left_label}_relative_log_evidence"
    if delta_col in relative:
        momentum_delta = (
            relative.loc[relative["canonical_model"].eq("momentum")]
            .groupby("session")[delta_col]
            .mean()
            .rename("mean_momentum_relative_evidence_delta")
        )
        sessions = sessions.join(momentum_delta, how="left")
    else:
        sessions["mean_momentum_relative_evidence_delta"] = np.nan

    return sessions.reset_index()[columns]


def run_comparison_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    best_comparison: pd.DataFrame,
    relative: pd.DataFrame,
    left_label: str,
    right_label: str,
    *,
    exact_only: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "left_label": left_label,
                "right_label": right_label,
                "left_score_file": source_file(left),
                "right_score_file": source_file(right),
                "exact_only": bool(exact_only),
                "left_events": event_count(left),
                "right_events": event_count(right),
                "matched_events": int(len(best_comparison)),
                "canonical_best_agreements": int(best_comparison["canonical_best_agree"].sum())
                if "canonical_best_agree" in best_comparison
                else 0,
                "canonical_best_agreement_fraction": float(best_comparison["canonical_best_agree"].mean())
                if len(best_comparison)
                else float("nan"),
                "shared_relative_evidence_rows": int(len(relative)),
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two model-evidence artifacts.")
    parser.add_argument("--left", required=True, help="Left artifact directory or score CSV")
    parser.add_argument("--right", required=True, help="Right artifact directory or score CSV")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", default="results/model-evidence-artifact-comparison")
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Compare only rows marked exact_full_grid/evidence_comparable.",
    )
    args = parser.parse_args()

    tables = compare_artifacts(
        args.left,
        args.right,
        left_label=args.left_label,
        right_label=args.right_label,
        output=args.output,
        exact_only=args.exact_only,
    )
    print(tables["summary"].to_string(index=False))
    print("\nEvidence-support counts:")
    print(tables["support_counts"].to_string(index=False))
    print("\nShared relative-evidence summary:")
    print(tables["relative_summary"].to_string(index=False))
    print("\nSession story-shift summary:")
    print(tables["session_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
