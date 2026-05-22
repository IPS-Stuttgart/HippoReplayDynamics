"""Diagnostics for synthetic replay-dynamics recovery runs.

The strict simulation-recovery table intentionally chooses event winners only
among exact-comparable evidence rows.  That is the right conservative default,
but it can make candidate-pruned momentum/IMM rows look like recovery failures
even when their truncated lower bound already certifies a win over every exact
baseline.  This module turns that distinction into explicit event- and
configuration-level diagnostic tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .simulation_recovery import (
    add_evidence_columns,
    certified_vs_exact_event_recovery,
    certified_vs_exact_recovery_summary,
)


RECOVERY_SCORE_FILENAMES = (
    "simulation_recovery_event_scores.csv",
    "simulation_recovery_sweep_event_scores.csv",
    "event_scores.csv",
    "scores.csv",
)

EXACT_SUPPORT = "exact_full_grid"
TRUNCATED_SUPPORT = "truncated_full_grid"
_CANDIDATE_METRIC_COLUMNS = (
    "candidate_true_bin_coverage",
    "candidate_true_pair_coverage",
    "candidate_true_triplet_coverage",
    "candidate_true_path_fully_supported",
    "candidate_true_path_missing_bins",
)


@dataclass
class RecoveryDiagnosticTables:
    """Output tables for recovery failure-mode diagnosis."""

    event_diagnostics: pd.DataFrame
    summary: pd.DataFrame
    certified_vs_exact_events: pd.DataFrame
    certified_vs_exact_summary: pd.DataFrame
    manifest: dict[str, object]

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_diagnostics.to_csv(out_dir / "simulation_recovery_diagnostic_event_table.csv", index=False)
        self.summary.to_csv(out_dir / "simulation_recovery_diagnostic_summary.csv", index=False)
        self.certified_vs_exact_events.to_csv(out_dir / "simulation_recovery_certified_vs_exact_events.csv", index=False)
        self.certified_vs_exact_summary.to_csv(out_dir / "simulation_recovery_certified_vs_exact_summary.csv", index=False)
        (out_dir / "simulation_recovery_diagnostic_report.md").write_text(
            render_recovery_diagnostic_markdown(self),
            encoding="utf-8",
        )
        (out_dir / "simulation_recovery_diagnostic_manifest.json").write_text(
            json.dumps(_json_ready(self.manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_recovery_score_tables(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load one or more simulation-recovery score CSVs or result directories."""

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = _resolve_recovery_score_path(Path(raw_path))
        frame = pd.read_csv(path)
        frame["source_recovery_score_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one recovery score table is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_recovery_diagnostic_tables(
    event_scores: pd.DataFrame,
    *,
    source_paths: Iterable[str | Path] | None = None,
) -> RecoveryDiagnosticTables:
    """Build strict-vs-certified recovery and candidate-support diagnostics."""

    scores = _ensure_recovery_score_columns(event_scores)
    certified_events = certified_vs_exact_event_recovery(scores)
    certified_summary = certified_vs_exact_recovery_summary(scores)
    event_diagnostics = _event_diagnostics(scores, certified_events)
    summary = _diagnostic_summary(event_diagnostics)
    manifest = {
        "schema_version": 1,
        "source_paths": [] if source_paths is None else [str(path) for path in source_paths],
        "n_input_rows": int(len(event_scores)),
        "n_scored_rows": int(len(scores)),
        "n_diagnostic_events": int(len(event_diagnostics)),
        "strict_recovery_definition": "best exact-comparable evidence row equals expected model",
        "certified_vs_exact_definition": (
            "expected model is comparable and exact-best, or expected truncated lower bound exceeds the best exact-comparable row"
        ),
        "candidate_support_metrics": list(_CANDIDATE_METRIC_COLUMNS),
        "failure_modes": sorted(event_diagnostics["failure_mode"].dropna().astype(str).unique()) if not event_diagnostics.empty else [],
    }
    return RecoveryDiagnosticTables(
        event_diagnostics=event_diagnostics,
        summary=summary,
        certified_vs_exact_events=certified_events,
        certified_vs_exact_summary=certified_summary,
        manifest=manifest,
    )


def render_recovery_diagnostic_markdown(tables: RecoveryDiagnosticTables) -> str:
    """Render a compact Markdown report for recovery diagnostics."""

    if tables.summary.empty:
        return "# Simulation-recovery diagnostics\n\nNo diagnostic events were available.\n"
    overall = tables.summary[tables.summary["true_model"].astype(str).eq("overall")]
    row = overall.iloc[0] if not overall.empty else tables.summary.iloc[-1]
    lines = [
        "# Simulation-recovery diagnostics",
        "",
        f"Diagnostic events: {int(row['events'])}",
        f"Strict recovery accuracy: {float(row['strict_recovery_accuracy']):.3f}",
        f"Certified-vs-exact recovery accuracy: {float(row['certified_vs_exact_recovery_accuracy']):.3f}",
        "",
        "## Interpretation",
        "",
        (
            "Strict recovery uses only exact-comparable evidence rows. Candidate-pruned momentum/IMM rows are therefore not allowed "
            "to win strict recovery unless they were scored on full support. Certified-vs-exact recovery counts a pruned expected model "
            "only when its reported lower bound already exceeds the best exact-comparable row."
        ),
        "",
        "## Failure-mode counts",
        "",
    ]
    failure_columns = [col for col in row.index if str(col).startswith("failure_mode_") and str(col).endswith("_events")]
    if not failure_columns:
        lines.append("No failure modes were recorded.")
    else:
        for column in sorted(failure_columns):
            label = column.removeprefix("failure_mode_").removesuffix("_events").replace("_", " ")
            lines.append(f"- {label}: {int(row[column])}")
    lines.append("")
    return "\n".join(lines)


def _ensure_recovery_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("recovery event score table is empty")
    required = {"session", "event_index", "true_model", "expected_model", "model", "log_evidence"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"recovery score table is missing required columns: {sorted(missing)}")
    out = frame.copy()
    if "status" not in out:
        out["status"] = "success"
    if "evidence_support" not in out or "evidence_comparable" not in out:
        out = add_evidence_columns(out)
    return out


def _event_diagnostics(scores: pd.DataFrame, certified_events: pd.DataFrame) -> pd.DataFrame:
    certified_lookup = {
        _event_key(row["session"], row["event_index"]): row
        for _, row in certified_events.iterrows()
    }
    rows: list[dict[str, object]] = []
    for (session, event_index), group in scores.groupby(["session", "event_index"], sort=False, dropna=False):
        rows.append(_event_diagnostic_row(str(session), event_index, group, certified_lookup.get(_event_key(session, event_index))))
    return pd.DataFrame(rows).sort_values(["session", "event_index"]).reset_index(drop=True)


def _event_diagnostic_row(
    session: str,
    event_index: object,
    group: pd.DataFrame,
    certified_row: pd.Series | None,
) -> dict[str, object]:
    first = group.iloc[0]
    expected_model = str(first.get("expected_model", ""))
    true_model = str(first.get("true_model", ""))
    scored = _successful_finite_scores(group)
    comparable = scored[_comparable_mask(scored)] if not scored.empty else scored
    strict_best = _best_log_evidence_row(comparable)
    expected_rows = scored[scored["model"].astype(str).eq(expected_model)] if not scored.empty else scored
    expected = _best_log_evidence_row(expected_rows)
    strict_best_model = "" if strict_best is None else str(strict_best["model"])
    strict_best_log_evidence = np.nan if strict_best is None else float(strict_best["log_evidence"])
    expected_log_evidence = np.nan if expected is None else float(expected["log_evidence"])
    expected_support = "" if expected is None else _row_str(expected, "evidence_support", "")
    expected_comparable = False if expected is None else _row_bool(expected, "evidence_comparable", expected_support == EXACT_SUPPORT)
    certified_recovered = False if certified_row is None else _row_bool(certified_row, "certified_vs_exact_recovered_expected_model", False)
    certified_reason = "" if certified_row is None else _row_str(certified_row, "certified_vs_exact_reason", "")
    expected_minus_best_comparable = (
        np.nan
        if certified_row is None
        else _row_float(certified_row, "expected_minus_best_comparable_log_evidence", np.nan)
    )
    strict_recovered = bool(strict_best_model and strict_best_model == expected_model)

    row: dict[str, object] = {
        "session": session,
        "event_index": _event_index_value(event_index),
        "true_model": true_model,
        "expected_model": expected_model,
        "successful_scores": int(len(scored)),
        "comparable_scores": int(len(comparable)),
        "strict_best_model": strict_best_model,
        "strict_best_log_evidence": strict_best_log_evidence,
        "strict_recovered_expected_model": strict_recovered,
        "certified_vs_exact_recovered_expected_model": certified_recovered,
        "certified_vs_exact_reason": certified_reason,
        "expected_model_scored": expected is not None,
        "expected_model_log_evidence": expected_log_evidence,
        "expected_model_evidence_support": expected_support,
        "expected_model_evidence_comparable": expected_comparable,
        "best_comparable_model": strict_best_model,
        "best_comparable_log_evidence": strict_best_log_evidence,
        "expected_minus_best_comparable_log_evidence": expected_minus_best_comparable,
    }
    row.update(_candidate_metric_values(expected))
    row["failure_mode"] = _classify_failure_mode(row)
    return row


def _diagnostic_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = [_diagnostic_summary_row(str(true_model), group) for true_model, group in events.groupby("true_model", sort=False)]
    rows.append(_diagnostic_summary_row("overall", events))
    return pd.DataFrame(rows)


def _diagnostic_summary_row(label: str, group: pd.DataFrame) -> dict[str, object]:
    n_events = int(len(group))
    strict = group["strict_recovered_expected_model"].fillna(False).astype(bool)
    certified = group["certified_vs_exact_recovered_expected_model"].fillna(False).astype(bool)
    row: dict[str, object] = {
        "true_model": label,
        "events": n_events,
        "strict_recovered_events": int(strict.sum()),
        "strict_recovery_accuracy": _safe_fraction(int(strict.sum()), n_events),
        "certified_vs_exact_recovered_events": int(certified.sum()),
        "certified_vs_exact_recovery_accuracy": _safe_fraction(int(certified.sum()), n_events),
        "events_with_comparable_scores": int((group["comparable_scores"].fillna(0).astype(int) > 0).sum()),
        "events_with_expected_model_scored": int(group["expected_model_scored"].fillna(False).astype(bool).sum()),
    }
    for column in _CANDIDATE_METRIC_COLUMNS:
        diagnostic_column = f"expected_{column}"
        if diagnostic_column not in group:
            continue
        values = pd.to_numeric(group[diagnostic_column], errors="coerce")
        row[f"mean_{diagnostic_column}"] = float(values.mean()) if values.notna().any() else np.nan
        if diagnostic_column == "expected_candidate_true_path_fully_supported":
            row["expected_true_path_fully_supported_events"] = int(values.fillna(0).astype(bool).sum())
    for mode, count in group["failure_mode"].astype(str).value_counts().items():
        row[f"failure_mode_{_slug(mode)}_events"] = int(count)
    return row


def _classify_failure_mode(row: dict[str, object]) -> str:
    if int(row.get("successful_scores", 0)) <= 0:
        return "no_successful_scores"
    if not bool(row.get("expected_model_scored", False)):
        return "expected_model_not_scored"
    if bool(row.get("strict_recovered_expected_model", False)):
        return "strict_recovered"
    if bool(row.get("certified_vs_exact_recovered_expected_model", False)):
        return "strict_gate_excluded_certified_lower_bound"
    triplet_coverage = _coerce_float(row.get("expected_candidate_true_triplet_coverage"), np.nan)
    full_path_supported = _coerce_float(row.get("expected_candidate_true_path_fully_supported"), np.nan)
    if np.isfinite(full_path_supported) and full_path_supported < 1.0:
        return "candidate_support_misses_true_path"
    if np.isfinite(triplet_coverage) and triplet_coverage < 1.0:
        return "candidate_support_misses_true_path"
    if str(row.get("expected_model_evidence_support", "")) == TRUNCATED_SUPPORT:
        margin = _coerce_float(row.get("expected_minus_best_comparable_log_evidence"), np.nan)
        if np.isfinite(margin) and margin <= 0.0:
            return "lower_bound_not_decisive"
    if int(row.get("comparable_scores", 0)) <= 0:
        return "no_comparable_exact_reference"
    if bool(row.get("expected_model_evidence_comparable", False)):
        return "expected_exact_not_best"
    return "nondecisive_unknown"


def _candidate_metric_values(expected: pd.Series | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for column in _CANDIDATE_METRIC_COLUMNS:
        out[f"expected_{column}"] = np.nan if expected is None else _row_float(expected, column, np.nan)
    return out


def _successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
    status_ok = group["status"].astype(str).eq("success") if "status" in group else pd.Series(True, index=group.index)
    finite = pd.Series(np.isfinite(pd.to_numeric(group["log_evidence"], errors="coerce")), index=group.index)
    return group[status_ok & finite].copy()


def _comparable_mask(frame: pd.DataFrame) -> pd.Series:
    if "evidence_comparable" in frame:
        return frame["evidence_comparable"].fillna(False).astype(bool)
    if "evidence_support" in frame:
        return frame["evidence_support"].fillna("").astype(str).eq(EXACT_SUPPORT)
    return pd.Series(True, index=frame.index)


def _best_log_evidence_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    values = pd.to_numeric(frame["log_evidence"], errors="coerce")
    finite = values.notna() & np.isfinite(values)
    if not finite.any():
        return None
    valid = frame.loc[finite].copy()
    valid_values = values.loc[valid.index].to_numpy(float)
    return valid.iloc[int(np.argmax(valid_values))]


def _row_float(row: pd.Series, column: str, default: float) -> float:
    if column not in row.index or pd.isna(row[column]):
        return default
    return _coerce_float(row[column], default)


def _row_str(row: pd.Series, column: str, default: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return default
    return str(row[column])


def _row_bool(row: pd.Series, column: str, default: bool) -> bool:
    if column not in row.index or pd.isna(row[column]):
        return default
    value = row[column]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _event_key(session: object, event_index: object) -> tuple[str, str]:
    return (str(session), str(_event_index_value(event_index)))


def _event_index_value(value: object) -> object:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer():
        return int(numeric)
    return value


def _resolve_recovery_score_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    for name in RECOVERY_SCORE_FILENAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    csvs = sorted(path.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(
        f"Could not infer recovery score CSV inside {path}; expected one of {RECOVERY_SCORE_FILENAMES}"
    )


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float("nan") if denominator <= 0 else float(numerator) / float(denominator)


def _slug(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower()).strip("_")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, float):
        return None if not math.isfinite(scalar) else float(scalar)
    if isinstance(scalar, (int, bool, str)) or scalar is None:
        return scalar
    return str(scalar)
