"""Guard simulation-recovery event decisions against stale flags and cross-run mixing."""

from __future__ import annotations

from functools import wraps
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.special import logsumexp


_GROUP_COLUMNS = (
    "session",
    "simulation_random_seed",
    "random_seed",
    "benchmark_random_seed",
    "simulation_event_index",
    "event_index",
    "window_index",
    "benchmark_cell_split_index",
    "event_window_variant",
)
_PATCHED_FLAG = "_simulation_best_row_flag_scope_patch_applied"
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def apply_simulation_best_row_flags_patch() -> None:
    """Install per-event guarded handling of explicit best-model flags.

    Synthetic recovery tables are often concatenated across independent random
    seeds or windowed rescoring passes.  Those outputs can reuse the same
    ``session``/``event_index`` keys, so every event-level decision must include
    available run-level metadata rather than collapsing all rows with the same
    template event index.
    """

    from . import evidence_reporting as reporting
    from . import simulation_recovery as recovery

    if getattr(reporting, _PATCHED_FLAG, False) and getattr(recovery, _PATCHED_FLAG, False):
        return

    original_run_session_simulation_recovery = recovery.run_session_simulation_recovery

    @wraps(reporting.simulation_add_evidence_columns)
    def simulation_add_evidence_columns_with_scoped_events(df: pd.DataFrame) -> pd.DataFrame:
        return _simulation_add_evidence_columns(df, reporting)

    @wraps(reporting.simulation_event_best_rows)
    def simulation_event_best_rows_with_scoped_flags(event_scores: pd.DataFrame) -> pd.DataFrame:
        scored = reporting.ensure_evidence_support_columns(event_scores)
        if scored.empty:
            return _empty_like(scored)
        comparable = reporting._coerce_bool_series(scored["evidence_comparable"])
        status_ok = _status_success_mask(scored)
        ok = scored[status_ok & comparable]
        if ok.empty:
            return _empty_like(ok)
        ok = _finite_log_evidence_rows(ok)
        if ok.empty:
            return _empty_like(ok)
        if "is_best_model" not in ok.columns:
            return _best_by_log_evidence(ok)
        return _best_rows_with_guarded_flags(ok, reporting)

    @wraps(recovery.recovery_summary)
    def recovery_summary_with_scoped_events(event_scores: pd.DataFrame) -> pd.DataFrame:
        return _recovery_summary(event_scores, recovery)

    @wraps(recovery.certified_vs_exact_event_recovery)
    def certified_vs_exact_event_recovery_with_scoped_events(event_scores: pd.DataFrame) -> pd.DataFrame:
        return _certified_vs_exact_event_recovery(event_scores, reporting, recovery)

    @wraps(recovery.certified_vs_exact_recovery_summary)
    def certified_vs_exact_recovery_summary_with_scoped_events(event_scores: pd.DataFrame) -> pd.DataFrame:
        return _certified_vs_exact_recovery_summary(event_scores, recovery)

    @wraps(original_run_session_simulation_recovery)
    def run_session_simulation_recovery_with_seed_scope(dataset_root: object, session_id: str, config: object) -> object:
        result = original_run_session_simulation_recovery(dataset_root, session_id, config)
        if result.event_scores.empty or "simulation_random_seed" in result.event_scores:
            return result

        event_scores = result.event_scores.copy()
        event_scores["simulation_random_seed"] = int(config.random_seed)
        result.event_scores = recovery.add_evidence_columns(event_scores)
        result.confusion_matrix = recovery.confusion_matrix(result.event_scores, config.scoring_models)
        result.summary = recovery.recovery_summary(result.event_scores)
        result.certified_vs_exact_summary = recovery.certified_vs_exact_recovery_summary(result.event_scores)
        return result

    for function in (
        simulation_add_evidence_columns_with_scoped_events,
        simulation_event_best_rows_with_scoped_flags,
        recovery_summary_with_scoped_events,
        certified_vs_exact_event_recovery_with_scoped_events,
        certified_vs_exact_recovery_summary_with_scoped_events,
        run_session_simulation_recovery_with_seed_scope,
    ):
        setattr(function, _PATCHED_FLAG, True)

    reporting.simulation_add_evidence_columns = simulation_add_evidence_columns_with_scoped_events
    reporting.simulation_event_best_rows = simulation_event_best_rows_with_scoped_flags
    reporting._simulation_event_group_columns = _event_group_columns

    recovery.add_evidence_columns = simulation_add_evidence_columns_with_scoped_events
    recovery._event_best_rows = simulation_event_best_rows_with_scoped_flags
    recovery.recovery_summary = recovery_summary_with_scoped_events
    recovery.certified_vs_exact_event_recovery = certified_vs_exact_event_recovery_with_scoped_events
    recovery.certified_vs_exact_recovery_summary = certified_vs_exact_recovery_summary_with_scoped_events
    recovery.run_session_simulation_recovery = run_session_simulation_recovery_with_seed_scope

    setattr(reporting, _PATCHED_FLAG, True)
    setattr(recovery, _PATCHED_FLAG, True)


def _simulation_add_evidence_columns(df: pd.DataFrame, reporting: Any) -> pd.DataFrame:
    if df.empty:
        return df

    df = reporting._coerce_log_evidence_column(reporting.ensure_evidence_support_columns(df))
    groups: list[pd.DataFrame] = []
    for _, group in _iter_event_groups(df):
        group = group.copy()
        status_ok = reporting._status_success_series(group)
        scored = group[status_ok]
        group["relative_log_evidence"] = np.nan
        group["model_probability"] = np.nan
        group["is_best_model"] = False
        group["best_model"] = ""
        group["truncated_relative_log_evidence"] = np.nan
        group["is_best_truncated_lower_bound"] = False
        group["best_truncated_lower_bound_model"] = ""
        group["exact_surrogate_best_model"] = ""
        group["exact_surrogate_recovered_expected_model"] = False
        group["exact_surrogate_log_evidence"] = np.nan
        group["exact_surrogate_minus_best_comparable_log_evidence"] = np.nan

        if not scored.empty:
            finite_log_evidence = pd.Series(
                np.isfinite(scored["log_evidence"].to_numpy(float)),
                index=scored.index,
            )
            nonfinite_index = finite_log_evidence.index[~finite_log_evidence.to_numpy()]
            if len(nonfinite_index):
                group.loc[nonfinite_index, "evidence_comparable"] = False
            scored = scored.loc[finite_log_evidence]

        if scored.empty:
            if "expected_model" in group:
                group["recovered_expected_model"] = False
                group["lower_bound_recovered_expected_model"] = False
            groups.append(group)
            continue

        best = ""
        exact = scored[reporting._coerce_bool_series(scored["evidence_comparable"])]
        if not exact.empty:
            values = exact["log_evidence"].to_numpy(float)
            max_value = float(np.max(values))
            probabilities = np.exp(values - logsumexp(values))
            best_index = exact.index[int(np.argmax(values))]
            best = str(group.loc[best_index, "model"])
            group.loc[exact.index, "relative_log_evidence"] = values - max_value
            group.loc[exact.index, "model_probability"] = probabilities
            group.loc[best_index, "is_best_model"] = True
            group["best_model"] = best
            surrogate_models = reporting._simulation_exact_surrogate_models(group)
            surrogate_rows = exact[exact["model"].astype(str).isin(surrogate_models)]
            if not surrogate_rows.empty:
                surrogate = _best_log_evidence_row(surrogate_rows)
                surrogate_log_evidence = float(surrogate["log_evidence"])
                group["exact_surrogate_best_model"] = str(surrogate["model"])
                group["exact_surrogate_log_evidence"] = surrogate_log_evidence
                group["exact_surrogate_minus_best_comparable_log_evidence"] = surrogate_log_evidence - max_value
                group["exact_surrogate_recovered_expected_model"] = bool(str(surrogate["model"]) == best)

        truncated = scored[scored["evidence_support"].eq(reporting.TRUNCATED_EVIDENCE_SUPPORT)]
        if not truncated.empty:
            lower_bounds = truncated["log_evidence"].to_numpy(float)
            max_lower_bound = float(np.max(lower_bounds))
            best_truncated_index = truncated.index[int(np.argmax(lower_bounds))]
            best_truncated = str(group.loc[best_truncated_index, "model"])
            group.loc[truncated.index, "truncated_relative_log_evidence"] = lower_bounds - max_lower_bound
            group.loc[best_truncated_index, "is_best_truncated_lower_bound"] = True
            group["best_truncated_lower_bound_model"] = best_truncated

        if "expected_model" in group:
            group["recovered_expected_model"] = best in reporting._simulation_acceptable_recovery_models(group)
            group["lower_bound_recovered_expected_model"] = group["best_truncated_lower_bound_model"] == group["expected_model"]
        groups.append(group)

    if not groups:
        return _empty_like(df)
    return _sort_scoped_rows(pd.concat(groups, ignore_index=True, sort=False))


def _recovery_summary(event_scores: pd.DataFrame, recovery: Any) -> pd.DataFrame:
    best = recovery._event_best_rows(event_scores)
    if best.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for true_model, group in best.groupby("true_model", sort=False):
        best_counts = group["best_model"].value_counts()
        expected = recovery.expected_scoring_model(str(true_model))
        recovered = recovery._recovered_expected_series(group, str(true_model))
        surrogate_recovered = recovery._surrogate_recovered_series(group)
        rows.append(
            {
                "true_model": true_model,
                "expected_model": expected,
                "simulated_events": _event_count(group),
                "recovered_events": int(recovered.sum()),
                "recovery_accuracy": float(recovered.mean()),
                "exact_surrogate_recovered_events": int(surrogate_recovered.sum()),
                "exact_surrogate_recovery_accuracy": float(surrogate_recovered.mean()),
                "most_common_best_model": str(best_counts.index[0]),
                "most_common_best_model_events": int(best_counts.iloc[0]),
                "mean_n_time": float(group["n_time"].mean()),
                "mean_n_spikes": float(group["n_spikes"].mean()),
            }
        )
    rows.append(
        {
            "true_model": "overall",
            "expected_model": "",
            "simulated_events": _event_count(best),
            "recovered_events": int(best["recovered_expected_model"].sum()),
            "recovery_accuracy": float(best["recovered_expected_model"].mean()),
            "exact_surrogate_recovered_events": int(recovery._surrogate_recovered_series(best).sum()),
            "exact_surrogate_recovery_accuracy": float(recovery._surrogate_recovered_series(best).mean()),
            "most_common_best_model": str(best["best_model"].value_counts().index[0]),
            "most_common_best_model_events": int(best["best_model"].value_counts().iloc[0]),
            "mean_n_time": float(best["n_time"].mean()),
            "mean_n_spikes": float(best["n_spikes"].mean()),
        }
    )
    return pd.DataFrame(rows)


def _certified_vs_exact_event_recovery(event_scores: pd.DataFrame, reporting: Any, recovery: Any) -> pd.DataFrame:
    if event_scores.empty:
        return pd.DataFrame()

    event_scores = reporting._coerce_log_evidence_column(reporting.ensure_evidence_support_columns(event_scores))
    rows: list[dict[str, object]] = []
    for _, group in _iter_event_groups(event_scores):
        group = group.copy()
        first = group.iloc[0]
        expected_model = str(first.get("expected_model", ""))
        base: dict[str, object] = {
            "session": _event_scalar(group, "session"),
            "event_index": _event_scalar(group, "event_index"),
            "true_model": str(first.get("true_model", "")),
            "expected_model": expected_model,
            "n_time": _event_scalar(group, "n_time"),
            "n_spikes": _event_scalar(group, "n_spikes"),
        }
        for column in _event_group_columns(group):
            if column not in base:
                base[column] = _event_scalar(group, column)

        scored = group[reporting._status_success_series(group)].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_successful_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        finite = np.isfinite(scored["log_evidence"].to_numpy(float))
        scored = scored.loc[finite].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_finite_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        comparable_mask = recovery._comparable_mask(scored)
        comparable_rows = scored.loc[comparable_mask].copy()
        best_comparable_row: pd.Series | None = None
        best_comparable_model = ""
        best_comparable_log_evidence = np.nan
        if not comparable_rows.empty:
            best_comparable_row = _best_log_evidence_row(comparable_rows)
            best_comparable_model = str(best_comparable_row["model"])
            best_comparable_log_evidence = float(best_comparable_row["log_evidence"])

        acceptable_models = recovery._event_acceptable_recovery_models(group)
        acceptable_rows = scored[scored["model"].astype(str).isin(acceptable_models)].copy()
        expected_rows = scored[scored["model"].astype(str) == expected_model]
        if acceptable_rows.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "expected_model_not_scored",
                    "certified_reference_model": "",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": best_comparable_model,
                    "best_comparable_log_evidence": best_comparable_log_evidence,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        if best_comparable_row is not None and best_comparable_model in acceptable_models:
            expected = best_comparable_row
        elif not expected_rows.empty:
            expected = _best_log_evidence_row(expected_rows)
        else:
            expected = _best_log_evidence_row(acceptable_rows)

        certified_reference_model = str(expected["model"])
        expected_log_evidence = float(expected["log_evidence"])
        expected_support = str(expected.get("evidence_support", ""))
        raw_expected_comparable = expected.get("evidence_comparable", recovery._evidence_is_comparable(expected_support))
        expected_comparable = (
            recovery._evidence_is_comparable(expected_support)
            if pd.isna(raw_expected_comparable)
            else bool(reporting._coerce_bool_series(pd.Series([raw_expected_comparable])).iloc[0])
        )
        margin = expected_log_evidence - best_comparable_log_evidence

        if best_comparable_row is not None and best_comparable_model in acceptable_models:
            recovered = True
            reason = "expected_comparable_best" if best_comparable_model == expected_model else "exact_surrogate_comparable_best"
        elif expected_comparable:
            recovered = False
            reason = "expected_comparable_not_best" if certified_reference_model == expected_model else "exact_surrogate_comparable_not_best"
        elif expected_support != "truncated_full_grid":
            recovered = False
            reason = "expected_noncomparable_not_certified"
        elif not np.isfinite(best_comparable_log_evidence):
            recovered = False
            reason = "no_comparable_exact_reference"
        else:
            recovered = bool(margin > 0.0)
            reason = "expected_lower_bound_beats_best_comparable" if recovered else "expected_lower_bound_not_above_best_comparable"

        rows.append(
            {
                **base,
                "certified_vs_exact_recovered_expected_model": recovered,
                "certified_vs_exact_reason": reason,
                "certified_reference_model": certified_reference_model,
                "expected_model_log_evidence": expected_log_evidence,
                "expected_model_evidence_support": expected_support,
                "expected_model_evidence_comparable": expected_comparable,
                "best_comparable_model": best_comparable_model,
                "best_comparable_log_evidence": best_comparable_log_evidence,
                "expected_minus_best_comparable_log_evidence": float(margin),
            }
        )
    return pd.DataFrame(rows)


def _certified_vs_exact_recovery_summary(event_scores: pd.DataFrame, recovery: Any) -> pd.DataFrame:
    events = recovery.certified_vs_exact_event_recovery(event_scores)
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for true_model, group in events.groupby("true_model", sort=False):
        rows.append(_certified_vs_exact_summary_row(str(true_model), group))
    rows.append(_certified_vs_exact_summary_row("overall", events))
    return pd.DataFrame(rows)


def _certified_vs_exact_summary_row(label: str, group: pd.DataFrame) -> dict[str, object]:
    recovered = _coerce_bool_series(group["certified_vs_exact_recovered_expected_model"])
    margins = pd.to_numeric(group["expected_minus_best_comparable_log_evidence"], errors="coerce")
    expected_model = "" if label == "overall" else str(group["expected_model"].iloc[0])
    return {
        "true_model": label,
        "expected_model": expected_model,
        "simulated_events": _event_count(group),
        "certified_vs_exact_recovered_events": int(recovered.sum()),
        "certified_vs_exact_recovery_accuracy": float(recovered.mean()),
        "mean_expected_minus_best_comparable_log_evidence": float(margins.mean()),
        "median_expected_minus_best_comparable_log_evidence": float(margins.median()),
        "events_without_comparable_exact_reference": int((group["certified_vs_exact_reason"] == "no_comparable_exact_reference").sum()),
    }


def _event_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    group_columns = _event_group_columns(frame)
    if not group_columns:
        return int(len(frame))
    return int(frame[group_columns].drop_duplicates().shape[0])


def _iter_event_groups(frame: pd.DataFrame) -> Iterator[tuple[object, pd.DataFrame]]:
    group_columns = _event_group_columns(frame)
    if not group_columns:
        yield (), frame
        return
    yield from frame.groupby(group_columns, sort=False, dropna=False)


def _event_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _GROUP_COLUMNS if column in frame.columns]


def _best_rows_with_guarded_flags(frame: pd.DataFrame, reporting: Any) -> pd.DataFrame:
    group_columns = _event_group_columns(frame)
    if not group_columns:
        flags = reporting._coerce_bool_series(frame["is_best_model"])
        if int(flags.sum()) == 1:
            return frame.loc[flags].reset_index(drop=True)
        return _best_by_log_evidence(frame)

    pieces = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        flags = reporting._coerce_bool_series(group["is_best_model"])
        if int(flags.sum()) == 1:
            pieces.append(group.loc[flags])
        else:
            pieces.append(_best_by_log_evidence(group))
    if not pieces:
        return _empty_like(frame)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _finite_log_evidence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_like(frame)
    working = frame.copy()
    working["log_evidence"] = pd.to_numeric(working["log_evidence"], errors="coerce")
    return working[np.isfinite(working["log_evidence"].to_numpy(dtype=float))].copy()


def _best_by_log_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_like(frame)
    working = _finite_log_evidence_rows(frame)
    if working.empty:
        return _empty_like(working)
    group_columns = _event_group_columns(working)
    sort_columns = [*group_columns, "log_evidence"]
    ascending = [True] * len(group_columns) + [False]
    best = working.sort_values(sort_columns, ascending=ascending)
    if group_columns:
        best = best.drop_duplicates(group_columns, keep="first")
    else:
        best = best.head(1)
    return best.reset_index(drop=True)


def _best_log_evidence_row(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame["log_evidence"], errors="coerce").to_numpy(float)
    return frame.iloc[int(np.nanargmax(values))]


def _sort_scoped_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_like(frame).reset_index(drop=True)
    sort_columns = [column for column in (*_event_group_columns(frame), "model") if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _event_scalar(group: pd.DataFrame, column: str) -> object:
    return group[column].iloc[0] if column in group.columns and not group.empty else np.nan


def _status_success_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["status"].map(_status_is_success).astype(bool)


def _status_is_success(value: object) -> bool:
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text == "success" or text in _MISSING_STATUS_VALUES


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy()


def _coerce_bool_series(values: pd.Series) -> pd.Series:
    def coerce(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            return False
        text = str(value).strip().lower()
        if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"", "0", "0.0", "false", "f", "no", "n", "off", "nan", "none", "null"}:
            return False
        try:
            numeric = float(text)
        except ValueError:
            return False
        return bool(np.isfinite(numeric) and numeric != 0.0)

    return values.map(coerce).astype(bool)


__all__ = ["apply_simulation_best_row_flags_patch"]
