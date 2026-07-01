#!/usr/bin/env python3
"""Compare two replay model-evidence artifact directories.

Absolute log evidence can differ across preprocessing pipelines. This comparison
therefore emphasizes event-level best-model agreement and within-run relative
evidence after mapping model names onto canonical dynamics labels.
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

_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_EVENT_KEY_COLUMNS = (
    "session",
    "simulation_random_seed",
    "random_seed",
    "benchmark_random_seed",
    "null_random_seed",
    "benchmark_event_subset_seed",
    "simulation_event_index",
    "event_index",
    "event_id",
    "window_index",
    "event_window_variant",
    "window_role",
    "benchmark_cell_split_index",
    "cell_split_index",
    "benchmark_cell_split_seed",
    "cell_split_seed",
    "cell_split_count",
    "cell_split_shard_index",
    "cell_split_shard_count",
    "split_shard_index",
    "split_shard_count",
)


def canonical_model_name(model: str) -> str:
    name = str(model).strip().lower()
    if name.startswith("sorted-spike-state-space-"):
        name = name.removeprefix("sorted-spike-state-space-")
    elif name.startswith("clusterless-state-space-"):
        name = name.removeprefix("clusterless-state-space-")
    elif name.startswith("state-space-"):
        name = name.removeprefix("state-space-")
    if name == "jump":
        return "fragmented"
    if name in {
        "momentum-exact-sparse",
        "displacement-momentum",
        "velocity-momentum",
    }:
        return "momentum"
    if name in {
        "first-order-imm",
        "trajectory-imm-exact-sparse",
        "displacement-imm",
    }:
        return "imm"
    if name in {"random", "stationary", "stationary-gaussian", "diffusion", "momentum", "imm", "fragmented"}:
        return name
    return name


def compare_runs(
    left_dir: str | Path,
    right_dir: str | Path,
    *,
    left_label: str,
    right_label: str,
    output: str | Path,
    exact_only: bool = False,
) -> dict[str, pd.DataFrame]:
    left = _load_event_scores(left_dir, left_label, exact_only=exact_only)
    right = _load_event_scores(right_dir, right_label, exact_only=exact_only)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_key = _comparison_event_key_columns(left, right)
    left_best = _event_best(left, left_label)
    right_best = _event_best(right, right_label)
    left_best, right_best = _align_event_key_columns(left_best, right_best, event_key)
    event_comparison = left_best.merge(right_best, on=event_key, how="inner")
    event_comparison["canonical_best_agree"] = (
        event_comparison[f"{left_label}_canonical_best_model"] == event_comparison[f"{right_label}_canonical_best_model"]
    )
    event_comparison.to_csv(out_dir / "event_best_model_comparison.csv", index=False)

    counts = pd.concat(
        [
            _best_counts(event_comparison, left_label),
            _best_counts(event_comparison, right_label),
        ],
        ignore_index=True,
    )
    counts.to_csv(out_dir / "best_model_counts_comparison.csv", index=False)

    support_counts = pd.concat(
        [
            _evidence_support_counts(left, left_label),
            _evidence_support_counts(right, right_label),
        ],
        ignore_index=True,
    )
    support_counts.to_csv(out_dir / "evidence_support_counts.csv", index=False)

    cross_tab = pd.crosstab(
        event_comparison[f"{left_label}_canonical_best_model"],
        event_comparison[f"{right_label}_canonical_best_model"],
    ).reset_index()
    cross_tab.to_csv(out_dir / "best_model_canonical_crosstab.csv", index=False)

    relative = _relative_evidence_comparison(left, right, left_label, right_label)
    relative.to_csv(out_dir / "shared_model_relative_evidence_comparison.csv", index=False)
    relative_summary = _relative_evidence_summary(relative, left_label, right_label)
    relative_summary.to_csv(out_dir / "shared_model_relative_evidence_summary.csv", index=False)
    session_comparison = _session_model_evidence_comparison(event_comparison, relative, left_label, right_label)
    session_comparison.to_csv(out_dir / "session_model_evidence_comparison.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "left_label": left_label,
                "right_label": right_label,
                "left_events": _event_count(left),
                "right_events": _event_count(right),
                "matched_events": int(len(event_comparison)),
                "canonical_best_agreements": int(event_comparison["canonical_best_agree"].sum()),
                "canonical_best_agreement_fraction": float(event_comparison["canonical_best_agree"].mean())
                if len(event_comparison)
                else float("nan"),
                "shared_relative_evidence_rows": int(len(relative)),
                "exact_only": bool(exact_only),
            }
        ]
    )
    summary.to_csv(out_dir / "model_evidence_run_comparison_summary.csv", index=False)
    return {
        "event_comparison": event_comparison,
        "counts": counts,
        "support_counts": support_counts,
        "cross_tab": cross_tab,
        "relative": relative,
        "relative_summary": relative_summary,
        "session_comparison": session_comparison,
        "summary": summary,
    }


def _load_event_scores(root: str | Path, run_label: str, *, exact_only: bool = False) -> pd.DataFrame:
    path = Path(root) / "event_model_evidence.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = _successful_score_rows(frame)
    frame = _finite_log_evidence_rows(frame)
    frame = ensure_evidence_support_columns(frame)
    if "evidence_comparable" in frame:
        frame["evidence_comparable"] = _coerce_bool_series(frame["evidence_comparable"])
    else:
        frame["evidence_comparable"] = pd.Series(True, index=frame.index, dtype=bool)
    if exact_only:
        frame = frame[frame["evidence_comparable"]].copy()
    frame["run_label"] = run_label
    frame["canonical_model"] = frame["model"].map(canonical_model_name)
    frame = _add_relative_log_evidence(frame)
    return frame


def _successful_score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return frame.copy()
    normalized = frame["status"].astype("string").str.strip().str.lower()
    missing = normalized.isna() | normalized.fillna("").isin(_MISSING_STATUS_VALUES)
    success = normalized.fillna("").eq("success")
    out = frame[missing | success].copy()
    out["status"] = "success"
    return out


def _finite_log_evidence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    if out.empty:
        return out
    finite = np.isfinite(out["log_evidence"].to_numpy(dtype=float))
    return out.loc[finite].copy()


def _evidence_support_counts(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    columns = [
        "run_label",
        "evidence_support",
        "evidence_comparable",
        "rows",
        "session_events",
        "models",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (support, comparable), group in frame.groupby(
        ["evidence_support", "evidence_comparable"], dropna=False, sort=True
    ):
        rows.append(
            {
                "run_label": label,
                "evidence_support": support,
                "evidence_comparable": bool(comparable),
                "rows": int(len(group)),
                "session_events": _event_count(group),
                "models": int(group["model"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _add_relative_log_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame = frame.copy()
        frame["relative_log_evidence"] = pd.Series(index=frame.index, dtype=float)
        return frame
    group_columns = _event_group_columns(frame)
    groups = []
    if group_columns:
        iterator = frame.groupby(group_columns, sort=False, dropna=False)
    else:
        iterator = [((), frame)]
    for _, group in iterator:
        group = group.copy()
        group["relative_log_evidence"] = group["log_evidence"] - group["log_evidence"].max()
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def _event_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    group_columns = _event_group_columns(frame)
    if not group_columns:
        return int(len(frame))
    return int(frame[group_columns].drop_duplicates().shape[0])


def _event_best(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    group_columns = _event_group_columns(frame)
    sort_columns = [*group_columns, "log_evidence"]
    ascending = [True] * len(group_columns) + [False]
    best = frame.sort_values(sort_columns, ascending=ascending)
    if group_columns:
        best = best.drop_duplicates(group_columns, keep="first")
    else:
        best = best.head(1)
    return best[
        [
            *group_columns,
            "model",
            "canonical_model",
            "log_evidence",
            "relative_log_evidence",
        ]
    ].rename(
        columns={
            "model": f"{label}_best_model",
            "canonical_model": f"{label}_canonical_best_model",
            "log_evidence": f"{label}_best_log_evidence",
            "relative_log_evidence": f"{label}_best_relative_log_evidence",
        }
    )


def _best_counts(event_comparison: pd.DataFrame, label: str) -> pd.DataFrame:
    column = f"{label}_canonical_best_model"
    counts = event_comparison[column].value_counts().rename_axis("canonical_model").reset_index(name="events")
    counts["run_label"] = label
    counts["event_fraction"] = counts["events"] / max(1, len(event_comparison))
    return counts[["run_label", "canonical_model", "events", "event_fraction"]]


def _relative_evidence_comparison(left: pd.DataFrame, right: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    event_key = _comparison_event_key_columns(left, right)
    key = [*event_key, "canonical_model"]
    left_relative = _canonical_relative_table(left).rename(
        columns={
            "model": f"{left_label}_model",
            "relative_log_evidence": f"{left_label}_relative_log_evidence",
        }
    )
    right_relative = _canonical_relative_table(right).rename(
        columns={
            "model": f"{right_label}_model",
            "relative_log_evidence": f"{right_label}_relative_log_evidence",
        }
    )
    left_relative, right_relative = _align_event_key_columns(left_relative, right_relative, event_key)
    joined = left_relative.merge(right_relative, on=key, how="inner")
    joined[f"{right_label}_minus_{left_label}_relative_log_evidence"] = (
        joined[f"{right_label}_relative_log_evidence"] - joined[f"{left_label}_relative_log_evidence"]
    )
    return joined


def _canonical_relative_table(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = _event_group_columns(frame)
    sort_columns = [*group_columns, "canonical_model", "log_evidence"]
    ascending = [True] * (len(group_columns) + 1) + [False]
    deduplicate_columns = [*group_columns, "canonical_model"]
    best_by_canonical = frame.sort_values(
        sort_columns,
        ascending=ascending,
    ).drop_duplicates(deduplicate_columns, keep="first")
    return best_by_canonical[
        [
            *group_columns,
            "canonical_model",
            "model",
            "relative_log_evidence",
        ]
    ]


def _relative_evidence_summary(relative: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    if relative.empty:
        return pd.DataFrame()
    delta_col = f"{right_label}_minus_{left_label}_relative_log_evidence"
    return (
        relative.groupby("canonical_model", as_index=False)
        .agg(
            matched_events=("canonical_model", "count"),
            left_mean_relative_log_evidence=(f"{left_label}_relative_log_evidence", "mean"),
            right_mean_relative_log_evidence=(f"{right_label}_relative_log_evidence", "mean"),
            mean_right_minus_left_relative_log_evidence=(delta_col, "mean"),
            median_right_minus_left_relative_log_evidence=(delta_col, "median"),
        )
        .sort_values("matched_events", ascending=False)
    )


def _session_model_evidence_comparison(
    event_comparison: pd.DataFrame,
    relative: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    if "session" in event_comparison.columns:
        sessions = pd.DataFrame({"session": sorted(event_comparison["session"].dropna().unique())}).set_index("session")
    else:
        sessions = pd.DataFrame(index=pd.Index([], name="session"))
    for label in [left_label, right_label]:
        model_col = f"{label}_canonical_best_model"
        for model in ["diffusion", "momentum"]:
            counts = (
                event_comparison.loc[event_comparison[model_col] == model]
                .groupby("session")
                .size()
                .rename(f"{label}_{model}_wins")
            )
            sessions = sessions.join(counts, how="left")

    for label in [left_label, right_label]:
        for model in ["diffusion", "momentum"]:
            col = f"{label}_{model}_wins"
            sessions[col] = sessions[col].fillna(0).astype(int)

    sessions["momentum_win_delta"] = sessions[f"{right_label}_momentum_wins"] - sessions[f"{left_label}_momentum_wins"]
    sessions["canonical_best_agreement_fraction"] = event_comparison.groupby("session")["canonical_best_agree"].mean()

    delta_col = f"{right_label}_minus_{left_label}_relative_log_evidence"
    if delta_col in relative.columns:
        momentum_delta = (
            relative.loc[relative["canonical_model"] == "momentum"]
            .groupby("session")[delta_col]
            .mean()
            .rename("mean_momentum_relative_evidence_delta")
        )
        sessions = sessions.join(momentum_delta, how="left")
    else:
        sessions["mean_momentum_relative_evidence_delta"] = float("nan")

    columns = [
        "session",
        f"{left_label}_diffusion_wins",
        f"{right_label}_diffusion_wins",
        f"{left_label}_momentum_wins",
        f"{right_label}_momentum_wins",
        "momentum_win_delta",
        "canonical_best_agreement_fraction",
        "mean_momentum_relative_evidence_delta",
    ]
    return sessions.reset_index()[columns]


def _event_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _EVENT_KEY_COLUMNS if column in frame.columns]


def _comparison_event_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Return event-key columns that can be compared on both score tables."""

    return [column for column in _EVENT_KEY_COLUMNS if column in left.columns and column in right.columns]


def _align_event_key_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    event_key: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.copy()
    right = right.copy()
    for column in event_key:
        if column not in left:
            left[column] = pd.NA
        if column not in right:
            right[column] = pd.NA
        left[column] = left[column].astype(object)
        right[column] = right[column].astype(object)
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two replay model-evidence artifact directories.")
    parser.add_argument("--left", required=True, help="Left artifact directory containing event_model_evidence.csv")
    parser.add_argument("--right", required=True, help="Right artifact directory containing event_model_evidence.csv")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", default="results/model-evidence-comparison")
    parser.add_argument("--exact-only", action="store_true", help="Compare only exact-comparable evidence rows.")
    args = parser.parse_args()

    tables = compare_runs(args.left, args.right, left_label=args.left_label, right_label=args.right_label, output=args.output, exact_only=args.exact_only)
    print(tables["summary"].to_string(index=False))
    print("\nEvidence-support counts:")
    print(tables["support_counts"].to_string(index=False))
    print("\nCanonical best-model counts:")
    print(tables["counts"].to_string(index=False))
    print("\nShared relative-evidence summary:")
    print(tables["relative_summary"].to_string(index=False))
    print("\nSession model-evidence comparison:")
    print(tables["session_comparison"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
