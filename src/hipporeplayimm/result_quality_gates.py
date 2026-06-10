"""Result-quality gates and evidence-margin summaries for replay evidence runs.

These helpers are deliberately conservative: exact full-grid model evidences
are summarized separately from candidate-pruned lower bounds, and support/null
diagnostics are exposed as CSV tables rather than silently mixed into headline
model-probability summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evidence_reporting import (
    TRUNCATED_EVIDENCE_SUPPORT,
    _coerce_bool_series,
    ensure_evidence_support_columns,
)

MARGIN_TIE = "tie"
MARGIN_WEAK = "weak"
MARGIN_STRONG = "strong"
MARGIN_DECISIVE = "decisive"
MARGIN_UNKNOWN = "unknown"


def evidence_margin_label(margin: object) -> str:
    """Return a readable evidence-margin category for a log-evidence gap."""

    try:
        value = float(margin)
    except (TypeError, ValueError):
        return MARGIN_UNKNOWN
    if not np.isfinite(value):
        return MARGIN_DECISIVE if value > 0.0 else MARGIN_UNKNOWN
    if value <= 1.0:
        return MARGIN_TIE
    if value <= 3.0:
        return MARGIN_WEAK
    if value <= 10.0:
        return MARGIN_STRONG
    return MARGIN_DECISIVE


def event_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns identifying one model-comparison unit."""

    columns = ["session", "event_index"]
    for optional in ("window_index", "benchmark_cell_split_index"):
        if optional in frame.columns:
            columns.append(optional)
    return columns


def add_evidence_margin_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate event/model rows with exact and lower-bound evidence margins.

    Exact-comparable rows receive ``exact_model_*`` rank/margin columns.
    Candidate-pruned rows receive ``truncated_lower_bound_*`` columns.  The two
    scopes are intentionally separate so a truncated lower bound is never
    normalized or ranked against an exact full-grid evidence by accident.
    """

    if frame.empty:
        return frame.copy()
    if "log_evidence" not in frame.columns:
        return ensure_evidence_support_columns(frame)

    out = ensure_evidence_support_columns(frame)
    for prefix in ("exact_model", "truncated_lower_bound"):
        out[f"{prefix}_best_model"] = ""
        out[f"{prefix}_log_evidence_margin"] = np.nan
        out[f"{prefix}_margin_category"] = ""
        out[f"{prefix}_rank"] = np.nan
        out[f"{prefix}_relative_log_evidence"] = np.nan

    group_columns = event_group_columns(out)
    for _, group in out.groupby(group_columns, sort=False):
        successful = _successful_rows(group)
        exact = successful[_coerce_bool_series(successful["evidence_comparable"])]
        _annotate_margin_scope(out, group.index, exact, prefix="exact_model")
        truncated = successful[successful["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
        _annotate_margin_scope(
            out,
            group.index,
            truncated,
            prefix="truncated_lower_bound",
        )
    return out


def event_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one quality/diagnostic row per event-comparison unit."""

    if frame.empty:
        return pd.DataFrame()
    rows = add_evidence_margin_columns(frame)
    group_columns = event_group_columns(rows)
    records: list[dict[str, object]] = []
    for key, group in rows.groupby(group_columns, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, values, strict=True))
        successful = _successful_rows(group)
        exact = successful[_coerce_bool_series(successful["evidence_comparable"])]
        truncated = successful[successful["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
        record.update(
            {
                "score_rows": int(group.shape[0]),
                "successful_rows": int(successful.shape[0]),
                "exact_comparable_models": int(exact["model"].nunique()) if "model" in exact else int(exact.shape[0]),
                "truncated_lower_bound_models": int(truncated["model"].nunique()) if "model" in truncated else int(truncated.shape[0]),
                "exact_best_model": _first_nonempty(group, "exact_model_best_model"),
                "exact_log_evidence_margin": _first_numeric(group, "exact_model_log_evidence_margin"),
                "exact_margin_category": _first_nonempty(group, "exact_model_margin_category"),
                "truncated_best_model": _first_nonempty(group, "truncated_lower_bound_best_model"),
                "truncated_log_evidence_margin": _first_numeric(group, "truncated_lower_bound_log_evidence_margin"),
                "truncated_margin_category": _first_nonempty(group, "truncated_lower_bound_margin_category"),
                "candidate_support_good_fraction": _candidate_good_fraction(successful),
                "min_candidate_log_mass": _min_numeric(successful, "candidate_min_log_mass"),
            }
        )
        if "spatial_shuffle_null_empirical_p_value" in group:
            record["spatial_shuffle_min_p_value"] = _min_numeric(
                successful,
                "spatial_shuffle_null_empirical_p_value",
            )
            record["spatial_shuffle_min_delta_vs_null_median"] = _min_numeric(
                successful,
                "spatial_shuffle_delta_vs_null_median",
            )
        if "event_reliable" in group:
            reliable = _coerce_bool_series(successful["event_reliable"])
            record["event_reliable_fraction"] = float(reliable.mean()) if reliable.size else np.nan
        records.append(record)
    return pd.DataFrame(records)


def model_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize evidence support and quality diagnostics per model."""

    if frame.empty:
        return pd.DataFrame()
    rows = add_evidence_margin_columns(frame)
    records: list[dict[str, object]] = []
    for model, group in rows.groupby("model", sort=True):
        successful = _successful_rows(group)
        exact = successful[_coerce_bool_series(successful["evidence_comparable"])]
        truncated = successful[successful["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
        record = {
            "model": str(model),
            "rows": int(group.shape[0]),
            "successful_rows": int(successful.shape[0]),
            "exact_comparable_rows": int(exact.shape[0]),
            "truncated_lower_bound_rows": int(truncated.shape[0]),
            "candidate_support_good_fraction": _candidate_good_fraction(successful),
            "min_candidate_log_mass": _min_numeric(successful, "candidate_min_log_mass"),
            "mean_runtime_s": _mean_numeric(successful, "runtime_s"),
            "mean_exact_relative_log_evidence": _mean_numeric(successful, "exact_model_relative_log_evidence"),
            "mean_truncated_relative_log_evidence": _mean_numeric(successful, "truncated_lower_bound_relative_log_evidence"),
        }
        if "spatial_shuffle_null_empirical_p_value" in successful:
            record["spatial_shuffle_min_p_value"] = _min_numeric(
                successful,
                "spatial_shuffle_null_empirical_p_value",
            )
            record["spatial_shuffle_mean_delta_vs_null_median"] = _mean_numeric(
                successful,
                "spatial_shuffle_delta_vs_null_median",
            )
        records.append(record)
    return pd.DataFrame(records)


def quality_gate_summary(
    frame: pd.DataFrame,
    *,
    min_exact_models_per_event: int = 2,
    min_candidate_good_fraction: float = 0.95,
) -> pd.DataFrame:
    """Return compact pass/warn/fail style checks for a score table."""

    if frame.empty:
        return pd.DataFrame(
            [{"gate": "nonempty_scores", "status": "fail", "value": 0, "note": "No score rows."}]
        )
    rows = add_evidence_margin_columns(frame)
    events = event_quality_summary(rows)
    successful = _successful_rows(rows)
    exact = successful[_coerce_bool_series(successful["evidence_comparable"])]
    truncated = successful[successful["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
    failed = rows.shape[0] - successful.shape[0]
    candidate_good_fraction = _candidate_good_fraction(truncated if not truncated.empty else successful)
    if events.empty:
        exact_event_fraction = 0.0
        strong_exact_fraction = 0.0
    else:
        exact_event_fraction = float((events["exact_comparable_models"] >= int(min_exact_models_per_event)).mean())
        strong_exact_fraction = float(
            events["exact_margin_category"].isin({MARGIN_STRONG, MARGIN_DECISIVE}).mean()
        )

    records = [
        {
            "gate": "nonempty_scores",
            "status": "pass" if rows.shape[0] > 0 else "fail",
            "value": int(rows.shape[0]),
            "note": "Total score rows.",
        },
        {
            "gate": "no_failed_rows",
            "status": "pass" if failed == 0 else "fail",
            "value": int(failed),
            "note": "Failed model/event rows should be resolved before final claims.",
        },
        {
            "gate": "exact_comparable_rows",
            "status": "pass" if exact.shape[0] > 0 else "warn",
            "value": int(exact.shape[0]),
            "note": "Rows safe for posterior model probabilities within each event.",
        },
        {
            "gate": "truncated_lower_bound_rows",
            "status": "info" if truncated.shape[0] > 0 else "pass",
            "value": int(truncated.shape[0]),
            "note": "Candidate-pruned lower bounds; do not mix with exact evidences in headline rankings.",
        },
        {
            "gate": "events_with_min_exact_models",
            "status": "pass" if exact_event_fraction >= 1.0 else "warn",
            "value": exact_event_fraction,
            "note": f"Fraction of event groups with at least {min_exact_models_per_event} exact-comparable models.",
        },
        {
            "gate": "candidate_support_good_fraction",
            "status": "pass" if np.isnan(candidate_good_fraction) or candidate_good_fraction >= min_candidate_good_fraction else "warn",
            "value": candidate_good_fraction,
            "note": "Fraction of candidate-pruned rows labelled exact/good by support mass diagnostics.",
        },
        {
            "gate": "strong_exact_margin_fraction",
            "status": "info",
            "value": strong_exact_fraction,
            "note": "Fraction of event groups whose exact evidence margin is strong or decisive.",
        },
    ]
    if "spatial_shuffle_null_empirical_p_value" in rows:
        null_p = pd.to_numeric(successful["spatial_shuffle_null_empirical_p_value"], errors="coerce")
        tested = null_p.dropna()
        records.append(
            {
                "gate": "spatial_shuffle_tested_rows",
                "status": "info" if tested.size else "warn",
                "value": int(tested.size),
                "note": "Rows with spatial-bin permutation null p-values.",
            }
        )
        if tested.size:
            records.append(
                {
                    "gate": "spatial_shuffle_significant_fraction",
                    "status": "info",
                    "value": float((tested <= 0.05).mean()),
                    "note": "Rows beating spatial-bin permutation null at p <= 0.05.",
                }
            )
    return pd.DataFrame.from_records(records)


def write_result_quality_tables(frame: pd.DataFrame, outdir: str | Path) -> None:
    """Write standardized quality-gate CSV tables beside benchmark outputs."""

    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    rows = add_evidence_margin_columns(frame)
    rows.to_csv(path / "event_model_evidence_with_margins.csv", index=False)
    event_quality_summary(rows).to_csv(path / "result_quality_event_summary.csv", index=False)
    model_quality_summary(rows).to_csv(path / "result_quality_model_summary.csv", index=False)
    quality_gate_summary(rows).to_csv(path / "result_quality_gate_summary.csv", index=False)


def _successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame:
        return frame.copy()
    return frame[frame["status"].astype(str).eq("success")].copy()


def _annotate_margin_scope(
    out: pd.DataFrame,
    group_index: pd.Index,
    rows: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    if rows.empty or "log_evidence" not in rows:
        return
    values = pd.to_numeric(rows["log_evidence"], errors="coerce")
    rows = rows.loc[values.notna()].copy()
    values = values.loc[rows.index].to_numpy(dtype=float)
    if rows.empty:
        return
    order = np.argsort(-values, kind="mergesort")
    ordered_index = rows.index.to_numpy()[order]
    ordered_values = values[order]
    best_index = ordered_index[0]
    best_model = str(out.loc[best_index, "model"]) if "model" in out else ""
    margin = float(ordered_values[0] - ordered_values[1]) if ordered_values.size > 1 else float("inf")
    out.loc[group_index, f"{prefix}_best_model"] = best_model
    out.loc[group_index, f"{prefix}_log_evidence_margin"] = margin
    out.loc[group_index, f"{prefix}_margin_category"] = evidence_margin_label(margin)
    out.loc[ordered_index, f"{prefix}_rank"] = np.arange(1, ordered_index.shape[0] + 1, dtype=float)
    out.loc[ordered_index, f"{prefix}_relative_log_evidence"] = ordered_values - ordered_values[0]


def _first_nonempty(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    values = values[values != ""]
    return "" if values.empty else str(values.iloc[0])


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float("nan") if values.empty else float(values.iloc[0])


def _min_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float("nan") if values.empty else float(values.min())


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float("nan") if values.empty else float(values.mean())


def _candidate_good_fraction(frame: pd.DataFrame) -> float:
    if frame.empty or "candidate_support_quality_good" not in frame:
        return float("nan")
    values = frame["candidate_support_quality_good"].dropna()
    return float("nan") if values.empty else float(_coerce_bool_series(values).mean())
