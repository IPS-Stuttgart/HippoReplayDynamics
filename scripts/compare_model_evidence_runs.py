#!/usr/bin/env python3
"""Compare two replay model-evidence artifact directories.

Absolute log evidence can differ across preprocessing pipelines. This comparison
therefore emphasizes event-level best-model agreement and within-run relative
evidence after mapping model names onto canonical dynamics labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def canonical_model_name(model: str) -> str:
    name = str(model).strip().lower()
    if name.startswith("sorted-spike-state-space-"):
        name = name.removeprefix("sorted-spike-state-space-")
    elif name.startswith("state-space-"):
        name = name.removeprefix("state-space-")
    if name == "jump":
        return "fragmented"
    if name in {"random", "stationary", "stationary-gaussian", "diffusion", "momentum", "imm", "fragmented"}:
        return name
    return name


def compare_runs(left_dir: str | Path, right_dir: str | Path, *, left_label: str, right_label: str, output: str | Path) -> dict[str, pd.DataFrame]:
    left = _load_event_scores(left_dir, left_label)
    right = _load_event_scores(right_dir, right_label)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_best = _event_best(left, left_label)
    right_best = _event_best(right, right_label)
    event_comparison = left_best.merge(right_best, on=["session", "event_index"], how="inner")
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

    cross_tab = pd.crosstab(
        event_comparison[f"{left_label}_canonical_best_model"],
        event_comparison[f"{right_label}_canonical_best_model"],
    ).reset_index()
    cross_tab.to_csv(out_dir / "best_model_canonical_crosstab.csv", index=False)

    relative = _relative_evidence_comparison(left, right, left_label, right_label)
    relative.to_csv(out_dir / "shared_model_relative_evidence_comparison.csv", index=False)
    relative_summary = _relative_evidence_summary(relative, left_label, right_label)
    relative_summary.to_csv(out_dir / "shared_model_relative_evidence_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "left_label": left_label,
                "right_label": right_label,
                "left_events": int(left["event_index"].nunique()),
                "right_events": int(right["event_index"].nunique()),
                "matched_events": int(len(event_comparison)),
                "canonical_best_agreements": int(event_comparison["canonical_best_agree"].sum()),
                "canonical_best_agreement_fraction": float(event_comparison["canonical_best_agree"].mean())
                if len(event_comparison)
                else float("nan"),
                "shared_relative_evidence_rows": int(len(relative)),
            }
        ]
    )
    summary.to_csv(out_dir / "model_evidence_run_comparison_summary.csv", index=False)
    return {
        "event_comparison": event_comparison,
        "counts": counts,
        "cross_tab": cross_tab,
        "relative": relative,
        "relative_summary": relative_summary,
        "summary": summary,
    }


def _load_event_scores(root: str | Path, run_label: str) -> pd.DataFrame:
    path = Path(root) / "event_model_evidence.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "status" in frame.columns:
        frame = frame[frame["status"] == "success"].copy()
    frame["run_label"] = run_label
    frame["canonical_model"] = frame["model"].map(canonical_model_name)
    if "relative_log_evidence" not in frame.columns:
        frame = _add_relative_log_evidence(frame)
    return frame


def _add_relative_log_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, group in frame.groupby(["session", "event_index"], sort=False):
        group = group.copy()
        group["relative_log_evidence"] = group["log_evidence"] - group["log_evidence"].max()
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def _event_best(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    best = frame.sort_values(["session", "event_index", "log_evidence"], ascending=[True, True, False])
    best = best.drop_duplicates(["session", "event_index"], keep="first")
    return best[
        [
            "session",
            "event_index",
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
    key = ["session", "event_index", "canonical_model"]
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
    joined = left_relative.merge(right_relative, on=key, how="inner")
    joined[f"{right_label}_minus_{left_label}_relative_log_evidence"] = (
        joined[f"{right_label}_relative_log_evidence"] - joined[f"{left_label}_relative_log_evidence"]
    )
    return joined


def _canonical_relative_table(frame: pd.DataFrame) -> pd.DataFrame:
    best_by_canonical = frame.sort_values(
        ["session", "event_index", "canonical_model", "log_evidence"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["session", "event_index", "canonical_model"], keep="first")
    return best_by_canonical[
        [
            "session",
            "event_index",
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
            matched_events=("event_index", "count"),
            left_mean_relative_log_evidence=(f"{left_label}_relative_log_evidence", "mean"),
            right_mean_relative_log_evidence=(f"{right_label}_relative_log_evidence", "mean"),
            mean_right_minus_left_relative_log_evidence=(delta_col, "mean"),
            median_right_minus_left_relative_log_evidence=(delta_col, "median"),
        )
        .sort_values("matched_events", ascending=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two replay model-evidence artifact directories.")
    parser.add_argument("--left", required=True, help="Left artifact directory containing event_model_evidence.csv")
    parser.add_argument("--right", required=True, help="Right artifact directory containing event_model_evidence.csv")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", default="results/model-evidence-comparison")
    args = parser.parse_args()

    tables = compare_runs(args.left, args.right, left_label=args.left_label, right_label=args.right_label, output=args.output)
    print(tables["summary"].to_string(index=False))
    print("\nCanonical best-model counts:")
    print(tables["counts"].to_string(index=False))
    print("\nShared relative-evidence summary:")
    print(tables["relative_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
