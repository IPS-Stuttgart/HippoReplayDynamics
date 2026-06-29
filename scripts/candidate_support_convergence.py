#!/usr/bin/env python3
"""Compare model-evidence outputs across candidate-support settings.

Candidate-pruned momentum and IMM recursions are useful for full-session replay
benchmarks, but their rankings should be checked for stability as candidate
support changes. This script compares multiple `event_model_evidence.csv`
outputs, typically produced with different `state_space_momentum_candidate_top_k`
values, and writes convergence-audit tables.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns

EVENT_COLUMNS = ("session", "event_index")
OPTIONAL_ALIGNMENT_COLUMNS = (
    "benchmark_random_seed",
    "random_seed",
    "benchmark_cell_split_index",
    "cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
)
CANDIDATE_TOP_K_COLUMNS = (
    "diagnostic_state_space_momentum_candidate_top_k",
    "diagnostic_state_space_imm_candidate_top_k",
    "state_space_momentum_candidate_top_k",
    "state_space_imm_candidate_top_k",
    "diagnostic_candidate_top_k",
    "candidate_top_k",
)
CANDIDATE_PREDICTED_TOP_K_COLUMNS = (
    "diagnostic_state_space_momentum_predicted_candidate_top_k",
    "diagnostic_state_space_imm_predicted_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_imm_predicted_candidate_top_k",
    "diagnostic_predicted_candidate_top_k",
    "predicted_candidate_top_k",
)
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def _score_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    direct = candidate / "event_model_evidence.csv"
    if direct.is_file():
        return direct
    matches = sorted(candidate.rglob("event_model_evidence.csv"))
    if not matches:
        raise FileNotFoundError(f"No event_model_evidence.csv found under {candidate}")
    return matches[0]


def _numeric_unique_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    for column in columns:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        if numeric.size == 0:
            continue
        rounded = np.rint(numeric)
        if not np.all(np.isclose(numeric, rounded, rtol=0.0, atol=0.0)):
            raise ValueError(f"{column} must contain integer-valued candidate counts")
        if np.any(rounded < 0.0):
            raise ValueError(f"{column} must contain nonnegative candidate counts")
        values.extend(int(value) for value in rounded.astype(int))
    return sorted(set(values))


def _candidate_values(frame: pd.DataFrame) -> list[int]:
    return _numeric_unique_values(frame, CANDIDATE_TOP_K_COLUMNS)


def _predicted_candidate_values(frame: pd.DataFrame) -> list[int]:
    return _numeric_unique_values(frame, CANDIDATE_PREDICTED_TOP_K_COLUMNS)


def _format_values(values: list[int]) -> str:
    return "+".join(str(value) for value in values)


def _status_success_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    return frame["status"].map(_status_is_success_or_missing).astype(bool)


def _status_is_success_or_missing(value: object) -> bool:
    if _is_missing_scalar(value):
        return True
    return str(value).strip().lower() == "success"


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True
    return str(value).strip().lower() in _MISSING_STATUS_VALUES


def _column_has_nonmissing_values(frame: pd.DataFrame, column: str) -> bool:
    return bool(frame[column].map(lambda value: not _is_missing_scalar(value)).any())


def _alignment_columns(frames: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...]) -> tuple[str, ...]:
    """Return columns that uniquely identify comparable event rows.

    Candidate-support runs can contain repeated ``session/event_index`` pairs when
    they aggregate multiple random seeds, cell splits, or randomized event subsets.
    Align on those metadata columns whenever all compared frames provide real
    values for them; otherwise fall back to the legacy event key for old tables.
    """

    if isinstance(frames, pd.DataFrame):
        frame_list = (frames,)
    else:
        frame_list = tuple(frames)
    columns = list(EVENT_COLUMNS)
    if not frame_list:
        return tuple(columns)
    for column in OPTIONAL_ALIGNMENT_COLUMNS:
        if not all(column in frame.columns for frame in frame_list):
            continue
        if all(_column_has_nonmissing_values(frame, column) for frame in frame_list):
            columns.append(column)
    return tuple(columns)


def _event_count(frame: pd.DataFrame, alignment_columns: tuple[str, ...] | None = None) -> int:
    if frame.empty:
        return 0
    columns = tuple(alignment_columns or _alignment_columns(frame))
    return int(frame.loc[:, columns].drop_duplicates().shape[0])


def _format_alignment_value(value: object) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _format_event_key(row: pd.Series, alignment_columns: tuple[str, ...]) -> str:
    parts = [f"{row['session']}:{int(row['event_index'])}"]
    for column in alignment_columns:
        if column in EVENT_COLUMNS or column not in row.index:
            continue
        value = row[column]
        if _is_missing_scalar(value):
            continue
        parts.append(f"{column}={_format_alignment_value(value)}")
    return ":".join(parts)


def infer_run_label(frame: pd.DataFrame, fallback: str) -> str:
    parts: list[str] = []
    top_k_values = _candidate_values(frame)
    if top_k_values:
        parts.append(f"top_k={_format_values(top_k_values)}")
    predicted_values = _predicted_candidate_values(frame)
    if predicted_values:
        parts.append(f"pred_k={_format_values(predicted_values)}")
    return ",".join(parts) if parts else fallback


def parse_labels(spec: str | None, n_expected: int) -> list[str] | None:
    if spec is None or not spec.strip():
        return None
    labels = [part.strip() for part in spec.split(",") if part.strip()]
    if len(labels) != n_expected:
        raise ValueError(f"Expected {n_expected} labels, got {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate-support labels must be unique")
    return labels


def load_candidate_support_run(path: str | Path, label: str | None = None) -> pd.DataFrame:
    score_file = _score_file(path)
    frame = ensure_evidence_support_columns(pd.read_csv(score_file))
    frame = frame[_status_success_mask(frame)].copy()
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{score_file} is missing required columns: {missing}")
    frame["run_label"] = label or infer_run_label(frame, score_file.parent.name)
    frame["source_score_file"] = str(score_file)
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    return frame


def _load_runs(paths: list[str | Path], labels: list[str] | None = None) -> list[pd.DataFrame]:
    if len(paths) < 2:
        raise ValueError("At least two candidate-support runs are required")
    labels = labels or [None] * len(paths)
    runs = [load_candidate_support_run(path, label) for path, label in zip(paths, labels, strict=True)]
    run_labels = [str(run["run_label"].iloc[0]) for run in runs]
    if len(set(run_labels)) != len(run_labels):
        raise ValueError(f"Run labels must be unique; observed {run_labels}")
    return runs


def evidence_delta_summary(runs: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "run_a",
        "run_b",
        "model",
        "support_a",
        "support_b",
        "events",
        "sessions",
        "mean_delta_b_minus_a",
        "median_delta_b_minus_a",
        "mean_abs_delta",
        "median_abs_delta",
        "p95_abs_delta",
        "max_abs_delta",
        "fraction_abs_delta_le_0_1",
        "fraction_abs_delta_le_1_0",
    ]
    records: list[dict[str, object]] = []
    for left, right in combinations(runs, 2):
        run_a = str(left["run_label"].iloc[0])
        run_b = str(right["run_label"].iloc[0])
        alignment_columns = _alignment_columns((left, right))
        merged = left.merge(
            right,
            on=[*alignment_columns, "model"],
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            continue
        merged["delta_b_minus_a"] = merged["log_evidence_b"] - merged["log_evidence_a"]
        merged["abs_delta"] = merged["delta_b_minus_a"].abs()
        group_cols = ["model", "evidence_support_a", "evidence_support_b"]
        for (model, support_a, support_b), group in merged.groupby(group_cols, dropna=False, sort=True):
            abs_delta = group["abs_delta"].dropna().to_numpy(float)
            if abs_delta.size == 0:
                continue
            records.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "model": model,
                    "support_a": support_a,
                    "support_b": support_b,
                    "events": _event_count(group, alignment_columns),
                    "sessions": int(group["session"].nunique()),
                    "mean_delta_b_minus_a": float(group["delta_b_minus_a"].mean()),
                    "median_delta_b_minus_a": float(group["delta_b_minus_a"].median()),
                    "mean_abs_delta": float(np.mean(abs_delta)),
                    "median_abs_delta": float(np.median(abs_delta)),
                    "p95_abs_delta": float(np.quantile(abs_delta, 0.95)),
                    "max_abs_delta": float(np.max(abs_delta)),
                    "fraction_abs_delta_le_0_1": float(np.mean(abs_delta <= 0.1)),
                    "fraction_abs_delta_le_1_0": float(np.mean(abs_delta <= 1.0)),
                }
            )
    return pd.DataFrame.from_records(records, columns=columns)


def _best_model_by_event(
    frame: pd.DataFrame,
    allowed_models: set[str],
    alignment_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    columns = tuple(alignment_columns or _alignment_columns(frame))
    subset = frame[frame["model"].astype(str).isin(allowed_models)].dropna(subset=["log_evidence"])
    if subset.empty:
        return pd.DataFrame(columns=[*columns, "best_model"])
    best_indices = subset.groupby(list(columns), sort=False)["log_evidence"].idxmax()
    return subset.loc[best_indices, [*columns, "model"]].rename(
        columns={"model": "best_model"}
    )


def best_model_agreement(runs: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "run_a",
        "run_b",
        "events",
        "sessions",
        "common_models",
        "best_model_agreements",
        "best_model_disagreements",
        "best_model_agreement_fraction",
        "changed_events",
    ]
    records: list[dict[str, object]] = []
    for left, right in combinations(runs, 2):
        run_a = str(left["run_label"].iloc[0])
        run_b = str(right["run_label"].iloc[0])
        common_models = set(left["model"].astype(str)) & set(right["model"].astype(str))
        if not common_models:
            continue
        alignment_columns = _alignment_columns((left, right))
        best_a = _best_model_by_event(left, common_models, alignment_columns)
        best_b = _best_model_by_event(right, common_models, alignment_columns)
        merged = best_a.merge(best_b, on=list(alignment_columns), suffixes=("_a", "_b"))
        if merged.empty:
            continue
        agree = merged["best_model_a"].eq(merged["best_model_b"])
        changed = merged.loc[~agree, [*alignment_columns, "best_model_a", "best_model_b"]]
        changed_events = ";".join(
            f"{_format_event_key(row, alignment_columns)}:{row['best_model_a']}->{row['best_model_b']}"
            for _, row in changed.head(25).iterrows()
        )
        records.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "events": _event_count(merged, alignment_columns),
                "sessions": int(merged["session"].nunique()),
                "common_models": ",".join(sorted(common_models)),
                "best_model_agreements": int(agree.sum()),
                "best_model_disagreements": int((~agree).sum()),
                "best_model_agreement_fraction": float(agree.mean()),
                "changed_events": changed_events,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def convergence_warnings(
    delta_summary: pd.DataFrame,
    agreement: pd.DataFrame,
    *,
    delta_tolerance: float = 1.0,
    agreement_threshold: float = 0.95,
) -> str:
    if delta_summary.empty and agreement.empty:
        return "No aligned candidate-support comparisons were available.\n"

    large_delta = (
        delta_summary[delta_summary["p95_abs_delta"] > float(delta_tolerance)]
        if not delta_summary.empty
        else pd.DataFrame()
    )
    low_agreement = (
        agreement[agreement["best_model_agreement_fraction"] < float(agreement_threshold)]
        if not agreement.empty
        else pd.DataFrame()
    )
    if large_delta.empty and low_agreement.empty:
        return (
            "Candidate-support convergence audit passed the configured heuristics: "
            f"p95 absolute log-evidence deltas <= {delta_tolerance:g} and best-model "
            f"agreement >= {agreement_threshold:g} where aligned comparisons exist.\n"
        )

    lines = ["Candidate-support convergence audit found potential instability."]
    if not large_delta.empty:
        lines.append(f"Model/run comparisons with p95 absolute log-evidence delta > {delta_tolerance:g}:")
        for _, row in large_delta.sort_values("p95_abs_delta", ascending=False).head(50).iterrows():
            lines.append(
                f"- {row['run_a']} vs {row['run_b']} / {row['model']}: "
                f"p95={float(row['p95_abs_delta']):.3g}, max={float(row['max_abs_delta']):.3g}, "
                f"events={int(row['events'])}"
            )
    if not low_agreement.empty:
        lines.append(f"Run pairs with best-model agreement below {agreement_threshold:g}:")
        for _, row in low_agreement.sort_values("best_model_agreement_fraction").head(50).iterrows():
            lines.append(
                f"- {row['run_a']} vs {row['run_b']}: "
                f"agreement={float(row['best_model_agreement_fraction']):.3f}, "
                f"events={int(row['events'])}, disagreements={int(row['best_model_disagreements'])}"
            )
    lines.append(
        "Recommended reporting: avoid treating candidate-pruned rankings as stable until top-k "
        "sensitivity is acceptably small for the relevant model pair."
    )
    return "\n".join(lines) + "\n"


def write_candidate_support_convergence(
    paths: list[str | Path],
    outdir: str | Path,
    *,
    labels: list[str] | None = None,
    delta_tolerance: float = 1.0,
    agreement_threshold: float = 0.95,
) -> dict[str, pd.DataFrame | str]:
    runs = _load_runs(paths, labels)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_summary = evidence_delta_summary(runs)
    agreement = best_model_agreement(runs)
    warnings = convergence_warnings(
        delta_summary,
        agreement,
        delta_tolerance=delta_tolerance,
        agreement_threshold=agreement_threshold,
    )
    delta_summary.to_csv(output_dir / "candidate_support_delta_summary.csv", index=False)
    agreement.to_csv(output_dir / "candidate_support_best_model_agreement.csv", index=False)
    (output_dir / "candidate_support_convergence_warnings.txt").write_text(warnings, encoding="utf-8")
    return {"delta_summary": delta_summary, "best_model_agreement": agreement, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare model-evidence outputs across candidate-support settings.")
    parser.add_argument("runs", nargs="+", help="Run directories or event_model_evidence.csv files to compare")
    parser.add_argument("--labels", default="", help="Optional comma-separated labels, one per run")
    parser.add_argument("--output", default="results/candidate-support-convergence")
    parser.add_argument("--delta-tolerance", type=float, default=1.0)
    parser.add_argument("--agreement-threshold", type=float, default=0.95)
    args = parser.parse_args()

    outputs = write_candidate_support_convergence(
        args.runs,
        args.output,
        labels=parse_labels(args.labels, len(args.runs)),
        delta_tolerance=args.delta_tolerance,
        agreement_threshold=args.agreement_threshold,
    )
    print(outputs["warnings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
