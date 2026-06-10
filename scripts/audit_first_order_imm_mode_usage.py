#!/usr/bin/env python3
"""Audit whether first-order IMM wins are driven by nonstationary modes.

The all-session trajectory-family claim has two different strengths:

* a model-class claim: trajectory-capable exact models beat pure stationary
  alternatives;
* a posterior-content claim: the leading first-order IMM actually allocates
  within-event posterior mass to nonstationary modes.

This script keeps those claims separate. Older artifacts contain terminal IMM
mode probabilities only; newer artifacts should also contain event-mean mode
probabilities emitted by the scorer.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"

REQUIRED_EXACT_CORE_MODELS = (
    STATIONARY,
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
TRAJECTORY_CAPABLE_EXACT_MODELS = (
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)

TERMINAL_MODE_COLUMNS = {
    "stationary": "diagnostic_state_space_mode_stationary_terminal_probability",
    "diffusion": "diagnostic_state_space_mode_diffusion_terminal_probability",
    "fragmented": "diagnostic_state_space_mode_fragmented_terminal_probability",
}
EVENT_MODE_COLUMNS = {
    "stationary": "diagnostic_state_space_mode_stationary_event_probability",
    "diffusion": "diagnostic_state_space_mode_diffusion_event_probability",
    "fragmented": "diagnostic_state_space_mode_fragmented_event_probability",
}

DEFAULT_GROUP_COLUMNS = ("session", "event_index")
OFF_SWR_GROUP_COLUMNS = ("session", "event_index", "null_index")
EVENT_METADATA_COLUMNS = (
    "event_class",
    "selection_rule",
    "window_role",
    "null_index",
    "source_event_group_id",
    "candidate_rank",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "n_spikes",
    "active_cell_count",
    "run_or_immobility_state",
    "animal_speed_mean",
    "distance_to_nearest_swr_s",
)

EVENT_COLUMNS = (
    "event_class",
    "session",
    "rat",
    "event_index",
    "selection_rule",
    "window_role",
    "null_index",
    "source_event_group_id",
    "candidate_rank",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "n_spikes",
    "active_cell_count",
    "run_or_immobility_state",
    "animal_speed_mean",
    "distance_to_nearest_swr_s",
    "exact_core_complete",
    "best_exact_core_model",
    "best_exact_core_margin_to_runner_up",
    "best_trajectory_capable_model",
    "trajectory_capable_minus_stationary",
    "trajectory_capable_confident_vs_stationary",
    "first_order_imm_is_best_exact_core",
    "first_order_imm_confident_exact_core_best",
    "logZ_stationary",
    "logZ_diffusion",
    "logZ_fragmented",
    "logZ_first_order_imm",
    "logZ_momentum_exact_sparse",
    "terminal_mode_diagnostics_present",
    "event_mean_mode_diagnostics_present",
    "stationary_terminal_probability",
    "diffusion_terminal_probability",
    "fragmented_terminal_probability",
    "nonstationary_terminal_probability",
    "terminal_dominant_mode",
    "terminal_nonstationary_majority",
    "stationary_event_probability",
    "diffusion_event_probability",
    "fragmented_event_probability",
    "nonstationary_event_probability",
    "event_dominant_mode",
    "event_nonstationary_majority",
    "posterior_content_status",
)

SUMMARY_COLUMNS = (
    "scope",
    "events",
    "exact_core_complete_events",
    "first_order_imm_best_events",
    "first_order_imm_confident_exact_core_best_events",
    "trajectory_capable_confident_vs_stationary_events",
    "terminal_mode_diagnostic_events",
    "terminal_nonstationary_majority_events",
    "terminal_nonstationary_majority_fraction",
    "median_stationary_terminal_probability",
    "median_nonstationary_terminal_probability",
    "event_mean_mode_diagnostic_events",
    "event_nonstationary_majority_events",
    "event_nonstationary_majority_fraction",
    "median_stationary_event_probability",
    "median_nonstationary_event_probability",
    "posterior_content_status",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "claim_level", "required_for_audit_overall")
CLASS_GATE_COLUMNS = ("event_class", "selection_rule", *GATE_COLUMNS)
COMPARISON_COLUMNS = ("event_class", "selection_rule", *SUMMARY_COLUMNS)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _clean_optional_text(value: object) -> object:
    if pd.isna(value):
        return ""
    return value


def _metadata_value(group: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in group:
        return default
    value = group.iloc[0][column]
    return _clean_optional_text(value)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _read_event_model_evidence(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")
    if "status" in frame:
        frame = frame[frame["status"].astype(str).eq("success")].copy()
    frame["session"] = frame["session"].astype(str)
    frame["rat"] = frame["session"].map(_rat_from_session)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    frame["model"] = frame["model"].astype(str)
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    frame = frame.dropna(subset=["log_evidence"]).copy()
    if "evidence_comparable" not in frame:
        frame["evidence_comparable"] = True
    frame["evidence_comparable"] = frame["evidence_comparable"].map(_as_bool)
    return frame


def _model_value(group: pd.DataFrame, model: str) -> float:
    rows = group[group["model"].eq(model)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[-1]["log_evidence"])


def _model_diag_value(group: pd.DataFrame, model: str, column: str) -> float:
    rows = group[group["model"].eq(model)]
    if rows.empty or column not in rows:
        return float("nan")
    value = pd.to_numeric(rows.iloc[-1][column], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _missing_models(group: pd.DataFrame, models: Iterable[str]) -> tuple[str, ...]:
    present = set(group["model"].astype(str))
    return tuple(model for model in models if model not in present)


def _best_in(group: pd.DataFrame, models: Iterable[str]) -> tuple[str, float]:
    subset = group[group["model"].isin(tuple(models))].copy()
    if subset.empty:
        return "", float("nan")
    row = subset.sort_values("log_evidence", ascending=False).iloc[0]
    return str(row["model"]), float(row["log_evidence"])


def _winner_margin_to_runner_up(group: pd.DataFrame, winner: str, models: Iterable[str]) -> float:
    subset = group[group["model"].isin(tuple(models))].copy()
    subset = subset.sort_values("log_evidence", ascending=False)
    if len(subset) < 2 or not winner:
        return float("nan")
    if str(subset.iloc[0]["model"]) != winner:
        return float("nan")
    return float(subset.iloc[0]["log_evidence"] - subset.iloc[1]["log_evidence"])


def _dominant_mode(mode_values: dict[str, float]) -> str:
    finite = {mode: value for mode, value in mode_values.items() if np.isfinite(value)}
    if not finite:
        return ""
    return max(finite.items(), key=lambda item: item[1])[0]


def _mode_status(row: dict[str, object]) -> str:
    if row["event_mean_mode_diagnostics_present"]:
        if row["event_nonstationary_majority"]:
            return "event_mean_nonstationary_majority"
        return "event_mean_stationary_or_mixed"
    if row["terminal_mode_diagnostics_present"]:
        if row["terminal_nonstationary_majority"]:
            return "terminal_only_nonstationary_majority"
        return "terminal_only_stationary_or_mixed"
    return "missing_mode_diagnostics"


def build_first_order_imm_mode_usage_event_table(
    event_model_evidence: pd.DataFrame,
    *,
    event_class: str = "detected_replay_or_swr",
    selection_rule: str = "",
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return one mode-usage audit row per event."""

    evidence = _read_event_model_evidence_from_frame(event_model_evidence)
    group_columns = tuple(group_columns)
    missing_group_columns = sorted(set(group_columns).difference(evidence.columns))
    if missing_group_columns:
        raise ValueError(f"event evidence is missing group columns: {missing_group_columns}")
    rows: list[dict[str, object]] = []
    for _, group in evidence.groupby(list(group_columns), sort=True, dropna=False):
        session = str(group.iloc[0]["session"])
        event_index = int(group.iloc[0]["event_index"])
        exact_group = group[
            group["model"].isin(REQUIRED_EXACT_CORE_MODELS)
            & group["evidence_comparable"].astype(bool)
        ].copy()
        exact_core_complete = not _missing_models(exact_group, REQUIRED_EXACT_CORE_MODELS)
        best_exact_model, _ = _best_in(exact_group, REQUIRED_EXACT_CORE_MODELS)
        best_exact_margin = _winner_margin_to_runner_up(
            exact_group,
            best_exact_model,
            REQUIRED_EXACT_CORE_MODELS,
        )
        best_trajectory_model, best_trajectory_logz = _best_in(
            exact_group,
            TRAJECTORY_CAPABLE_EXACT_MODELS,
        )
        stationary_logz = _model_value(exact_group, STATIONARY)
        first_order_logz = _model_value(exact_group, FIRST_ORDER_IMM)
        trajectory_margin = (
            float(best_trajectory_logz - stationary_logz)
            if np.isfinite(best_trajectory_logz) and np.isfinite(stationary_logz)
            else float("nan")
        )
        terminal = {
            mode: _model_diag_value(group, FIRST_ORDER_IMM, column)
            for mode, column in TERMINAL_MODE_COLUMNS.items()
        }
        event = {
            mode: _model_diag_value(group, FIRST_ORDER_IMM, column)
            for mode, column in EVENT_MODE_COLUMNS.items()
        }
        terminal_present = all(np.isfinite(value) for value in terminal.values())
        event_present = all(np.isfinite(value) for value in event.values())
        terminal_nonstationary = terminal["diffusion"] + terminal["fragmented"] if terminal_present else float("nan")
        event_nonstationary = event["diffusion"] + event["fragmented"] if event_present else float("nan")
        row = {
            "event_class": str(_metadata_value(group, "event_class", event_class) or event_class),
            "session": session,
            "rat": _rat_from_session(session),
            "event_index": int(event_index),
            "selection_rule": str(
                _metadata_value(group, "selection_rule", selection_rule) or selection_rule
            ),
            "window_role": _metadata_value(group, "window_role"),
            "null_index": _metadata_value(group, "null_index"),
            "source_event_group_id": _metadata_value(group, "source_event_group_id"),
            "candidate_rank": _metadata_value(group, "candidate_rank", float("nan")),
            "window_start_s": _metadata_value(group, "window_start_s", float("nan")),
            "window_end_s": _metadata_value(group, "window_end_s", float("nan")),
            "window_duration_s": _metadata_value(group, "window_duration_s", float("nan")),
            "n_spikes": _metadata_value(group, "n_spikes", float("nan")),
            "active_cell_count": _metadata_value(group, "active_cell_count", float("nan")),
            "run_or_immobility_state": _metadata_value(group, "run_or_immobility_state"),
            "animal_speed_mean": _metadata_value(group, "animal_speed_mean", float("nan")),
            "distance_to_nearest_swr_s": _metadata_value(
                group,
                "distance_to_nearest_swr_s",
                float("nan"),
            ),
            "exact_core_complete": bool(exact_core_complete),
            "best_exact_core_model": best_exact_model,
            "best_exact_core_margin_to_runner_up": best_exact_margin,
            "best_trajectory_capable_model": best_trajectory_model,
            "trajectory_capable_minus_stationary": trajectory_margin,
            "trajectory_capable_confident_vs_stationary": bool(trajectory_margin >= margin_threshold),
            "first_order_imm_is_best_exact_core": best_exact_model == FIRST_ORDER_IMM,
            "first_order_imm_confident_exact_core_best": bool(
                best_exact_model == FIRST_ORDER_IMM
                and np.isfinite(best_exact_margin)
                and best_exact_margin >= margin_threshold
            ),
            "logZ_stationary": stationary_logz,
            "logZ_diffusion": _model_value(exact_group, DIFFUSION),
            "logZ_fragmented": _model_value(exact_group, FRAGMENTED),
            "logZ_first_order_imm": first_order_logz,
            "logZ_momentum_exact_sparse": _model_value(exact_group, MOMENTUM_EXACT),
            "terminal_mode_diagnostics_present": bool(terminal_present),
            "event_mean_mode_diagnostics_present": bool(event_present),
            "stationary_terminal_probability": terminal["stationary"],
            "diffusion_terminal_probability": terminal["diffusion"],
            "fragmented_terminal_probability": terminal["fragmented"],
            "nonstationary_terminal_probability": terminal_nonstationary,
            "terminal_dominant_mode": _dominant_mode(terminal),
            "terminal_nonstationary_majority": bool(
                terminal_present and terminal_nonstationary > 0.5
            ),
            "stationary_event_probability": event["stationary"],
            "diffusion_event_probability": event["diffusion"],
            "fragmented_event_probability": event["fragmented"],
            "nonstationary_event_probability": event_nonstationary,
            "event_dominant_mode": _dominant_mode(event),
            "event_nonstationary_majority": bool(event_present and event_nonstationary > 0.5),
        }
        row["posterior_content_status"] = _mode_status(row)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _read_event_model_evidence_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.copy()
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(tmp.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")
    if "status" in tmp:
        tmp = tmp[tmp["status"].astype(str).eq("success")].copy()
    tmp["session"] = tmp["session"].astype(str)
    tmp["rat"] = tmp["session"].map(_rat_from_session)
    tmp["event_index"] = pd.to_numeric(tmp["event_index"], errors="raise").astype(int)
    tmp["model"] = tmp["model"].astype(str)
    tmp["log_evidence"] = pd.to_numeric(tmp["log_evidence"], errors="coerce")
    tmp = tmp.dropna(subset=["log_evidence"]).copy()
    if "evidence_comparable" not in tmp:
        tmp["evidence_comparable"] = True
    tmp["evidence_comparable"] = tmp["evidence_comparable"].map(_as_bool)
    return tmp


def _summary_row(scope: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "scope": scope,
            "events": 0,
            "posterior_content_status": "empty_scope",
        }
    terminal_present = frame["terminal_mode_diagnostics_present"].astype(bool)
    event_present = frame["event_mean_mode_diagnostics_present"].astype(bool)
    event_status = (
        "event_mean_mode_mass_available"
        if bool(event_present.all())
        else "terminal_only_mode_audit"
        if bool(terminal_present.all())
        else "missing_mode_diagnostics"
    )
    terminal_diag = frame[terminal_present]
    event_diag = frame[event_present]
    return {
        "scope": scope,
        "events": int(len(frame)),
        "exact_core_complete_events": int(frame["exact_core_complete"].astype(bool).sum()),
        "first_order_imm_best_events": int(frame["first_order_imm_is_best_exact_core"].astype(bool).sum()),
        "first_order_imm_confident_exact_core_best_events": int(
            frame["first_order_imm_confident_exact_core_best"].astype(bool).sum()
        ),
        "trajectory_capable_confident_vs_stationary_events": int(
            frame["trajectory_capable_confident_vs_stationary"].astype(bool).sum()
        ),
        "terminal_mode_diagnostic_events": int(terminal_present.sum()),
        "terminal_nonstationary_majority_events": int(
            terminal_diag["terminal_nonstationary_majority"].astype(bool).sum()
        ),
        "terminal_nonstationary_majority_fraction": _safe_fraction(
            int(terminal_diag["terminal_nonstationary_majority"].astype(bool).sum()),
            int(terminal_present.sum()),
        ),
        "median_stationary_terminal_probability": _median(terminal_diag, "stationary_terminal_probability"),
        "median_nonstationary_terminal_probability": _median(terminal_diag, "nonstationary_terminal_probability"),
        "event_mean_mode_diagnostic_events": int(event_present.sum()),
        "event_nonstationary_majority_events": int(
            event_diag["event_nonstationary_majority"].astype(bool).sum()
        ),
        "event_nonstationary_majority_fraction": _safe_fraction(
            int(event_diag["event_nonstationary_majority"].astype(bool).sum()),
            int(event_present.sum()),
        ),
        "median_stationary_event_probability": _median(event_diag, "stationary_event_probability"),
        "median_nonstationary_event_probability": _median(event_diag, "nonstationary_event_probability"),
        "posterior_content_status": event_status,
    }


def build_first_order_imm_mode_usage_summary(event_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-order IMM mode usage for paper-facing scopes."""

    complete = event_table[event_table["exact_core_complete"].astype(bool)].copy()
    first_order_best = complete[complete["first_order_imm_is_best_exact_core"].astype(bool)].copy()
    first_order_confident = complete[
        complete["first_order_imm_confident_exact_core_best"].astype(bool)
    ].copy()
    trajectory_confident = complete[
        complete["trajectory_capable_confident_vs_stationary"].astype(bool)
    ].copy()
    rows = [
        _summary_row("complete_exact_core_events", complete),
        _summary_row("first_order_imm_exact_core_best_events", first_order_best),
        _summary_row("first_order_imm_confident_exact_core_best_events", first_order_confident),
        _summary_row("trajectory_capable_confident_vs_stationary_events", trajectory_confident),
    ]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def build_first_order_imm_mode_usage_comparison(event_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-order IMM mode usage by event class."""

    rows: list[dict[str, object]] = []
    group_columns = ["event_class", "selection_rule"]
    for (event_class, selection_rule), group in event_table.groupby(group_columns, sort=True, dropna=False):
        summary = build_first_order_imm_mode_usage_summary(group)
        for row in summary.to_dict("records"):
            row["event_class"] = event_class
            row["selection_rule"] = selection_rule
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def build_first_order_imm_mode_usage_rat_summary(
    event_table: pd.DataFrame,
    *,
    include_event_class: bool = False,
) -> pd.DataFrame:
    """Summarize first-order IMM best rows by rat."""

    first_order_best = event_table[
        event_table["exact_core_complete"].astype(bool)
        & event_table["first_order_imm_is_best_exact_core"].astype(bool)
    ].copy()
    rows = []
    group_columns = ["rat"]
    if include_event_class:
        group_columns = ["event_class", "selection_rule", "rat"]
    for key, group in first_order_best.groupby(group_columns, sort=True, dropna=False):
        if include_event_class:
            event_class, selection_rule, rat = key
            row = _summary_row(f"event_class={event_class};selection_rule={selection_rule};rat={rat}", group)
            row["event_class"] = event_class
            row["selection_rule"] = selection_rule
            row["rat"] = rat
        else:
            rat = key
            row = _summary_row(f"rat={rat}", group)
            row["rat"] = rat
        rows.append(row)
    if not rows:
        if include_event_class:
            return pd.DataFrame(columns=["event_class", "selection_rule", "rat", *SUMMARY_COLUMNS])
        return pd.DataFrame(columns=["rat", *SUMMARY_COLUMNS])
    if include_event_class:
        return pd.DataFrame(rows, columns=["event_class", "selection_rule", "rat", *SUMMARY_COLUMNS])
    return pd.DataFrame(rows, columns=["rat", *SUMMARY_COLUMNS])


def build_first_order_imm_mode_usage_gate_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return audit gates without conflating model-class and content claims."""

    rows: list[dict[str, object]] = []

    def add(
        gate: str,
        passed: bool,
        observed: object,
        criterion: str,
        claim_level: str,
        *,
        required: bool = True,
    ) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "claim_level": claim_level,
                "required_for_audit_overall": bool(required),
            }
        )

    complete = event_table[event_table["exact_core_complete"].astype(bool)].copy()
    first_order_best = complete[complete["first_order_imm_is_best_exact_core"].astype(bool)].copy()
    terminal_present = (
        first_order_best["terminal_mode_diagnostics_present"].astype(bool)
        if not first_order_best.empty
        else pd.Series(dtype=bool)
    )
    event_present = (
        first_order_best["event_mean_mode_diagnostics_present"].astype(bool)
        if not first_order_best.empty
        else pd.Series(dtype=bool)
    )
    trajectory_confident_count = int(complete["trajectory_capable_confident_vs_stationary"].astype(bool).sum())
    nonstationary_terminal = int(
        first_order_best["terminal_nonstationary_majority"].astype(bool).sum()
    ) if not first_order_best.empty else 0
    nonstationary_event = int(
        first_order_best["event_nonstationary_majority"].astype(bool).sum()
    ) if not first_order_best.empty else 0

    add("exact_core_events_present", not complete.empty, int(len(complete)), "complete exact-core events > 0", "audit")
    add(
        "trajectory_capable_model_class_claim_supported",
        not complete.empty and trajectory_confident_count > len(complete) / 2,
        f"{trajectory_confident_count}/{len(complete)}",
        f"trajectory-capable exact model beats stationary by >= {margin_threshold} on a majority of complete events",
        "model_class",
    )
    add(
        "first_order_imm_best_rows_present",
        not first_order_best.empty,
        int(len(first_order_best)),
        "first-order IMM is exact-core best on at least one event",
        "audit",
    )
    add(
        "first_order_imm_terminal_mode_diagnostics_present",
        not first_order_best.empty and int(terminal_present.sum()) == len(first_order_best),
        f"{int(terminal_present.sum())}/{len(first_order_best)}",
        "terminal stationary/diffusion/fragmented probabilities are present for first-order IMM best rows",
        "terminal_posterior",
    )
    add(
        "terminal_nonstationary_majority_among_first_order_imm_best",
        not first_order_best.empty
        and int(terminal_present.sum()) == len(first_order_best)
        and nonstationary_terminal > len(first_order_best) / 2,
        f"{nonstationary_terminal}/{len(first_order_best)}",
        "nonstationary terminal mode probability > 0.5 on a majority of first-order IMM best rows",
        "terminal_posterior",
        required=False,
    )
    add(
        "first_order_imm_event_mean_mode_diagnostics_present",
        not first_order_best.empty and int(event_present.sum()) == len(first_order_best),
        f"{int(event_present.sum())}/{len(first_order_best)}",
        "event-mean stationary/diffusion/fragmented probabilities are present for first-order IMM best rows",
        "posterior_content",
        required=False,
    )
    add(
        "event_mean_nonstationary_majority_among_first_order_imm_best",
        not first_order_best.empty
        and int(event_present.sum()) == len(first_order_best)
        and nonstationary_event > len(first_order_best) / 2,
        f"{nonstationary_event}/{len(first_order_best)}",
        "nonstationary event-mean mode probability > 0.5 on a majority of first-order IMM best rows",
        "posterior_content",
        required=False,
    )
    add(
        "posterior_content_claim_supported",
        not first_order_best.empty
        and int(event_present.sum()) == len(first_order_best)
        and nonstationary_event > len(first_order_best) / 2,
        f"event_mean_nonstationary_majority={nonstationary_event}/{len(first_order_best)}",
        "strong content claim requires event-mean mode mass, not terminal-only diagnostics",
        "posterior_content",
        required=False,
    )
    required_rows = [row for row in rows if row["required_for_audit_overall"]]
    add(
        "overall",
        all(row["passed"] for row in required_rows),
        f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required audit gates passed",
        "the audit can distinguish model-class support from posterior-content support",
        "audit",
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def build_first_order_imm_mode_usage_class_gate_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return gate summaries separately for detected replay and off-SWR scopes."""

    rows: list[dict[str, object]] = []
    for (event_class, selection_rule), group in event_table.groupby(
        ["event_class", "selection_rule"],
        sort=True,
        dropna=False,
    ):
        gates = build_first_order_imm_mode_usage_gate_summary(
            group,
            margin_threshold=margin_threshold,
        )
        for record in gates.to_dict("records"):
            record["event_class"] = event_class
            record["selection_rule"] = selection_rule
            rows.append(record)
    if not rows:
        return pd.DataFrame(columns=list(CLASS_GATE_COLUMNS))
    return pd.DataFrame(rows, columns=list(CLASS_GATE_COLUMNS))


def _safe_fraction(value: int, denominator: int) -> float:
    return float(value) / float(denominator) if denominator else float("nan")


def _median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.median()) if not values.empty else float("nan")


def _with_event_scope(
    frame: pd.DataFrame,
    *,
    event_class: str,
    selection_rule: str = "",
) -> pd.DataFrame:
    scoped = frame.copy()
    scoped["event_class"] = event_class
    scoped["selection_rule"] = selection_rule
    return scoped


def _selected_one_per_source_evidence(
    promoted_event_model_evidence: pd.DataFrame,
    one_per_source_decisions: pd.DataFrame,
    *,
    selection_rule: str,
) -> pd.DataFrame:
    required = {"session", "event_index", "null_index", "selection_rule"}
    missing = sorted(required.difference(one_per_source_decisions.columns))
    if missing:
        raise ValueError(f"one-per-source decisions are missing required columns: {missing}")

    selected = one_per_source_decisions[
        one_per_source_decisions["selection_rule"].astype(str).eq(selection_rule)
    ].copy()
    if selected.empty:
        raise ValueError(f"no one-per-source decisions found for selection rule {selection_rule!r}")

    evidence = promoted_event_model_evidence.copy()
    for frame in (selected, evidence):
        frame["session"] = frame["session"].astype(str)
        frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
        frame["null_index"] = pd.to_numeric(frame["null_index"], errors="raise").astype(int)

    decision_columns = [
        column
        for column in (
            "session",
            "event_index",
            "null_index",
            "source_event_group_id",
            "selection_rule",
        )
        if column in selected
    ]
    return evidence.merge(
        selected[decision_columns],
        on=["session", "event_index", "null_index"],
        how="inner",
        suffixes=("", "_selected"),
    )


def write_first_order_imm_mode_usage_audit(
    event_model_evidence: pd.DataFrame,
    output: Path,
    *,
    margin_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    event_table = build_first_order_imm_mode_usage_event_table(
        event_model_evidence,
        margin_threshold=margin_threshold,
    )
    summary = build_first_order_imm_mode_usage_summary(event_table)
    rat = build_first_order_imm_mode_usage_rat_summary(event_table)
    gates = build_first_order_imm_mode_usage_gate_summary(
        event_table,
        margin_threshold=margin_threshold,
    )
    outputs = {
        "first_order_imm_mode_usage_event_table.csv": event_table,
        "first_order_imm_mode_usage_summary.csv": summary,
        "first_order_imm_mode_usage_rat_summary.csv": rat,
        "first_order_imm_mode_usage_gate_summary.csv": gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def write_first_order_imm_mode_usage_comparison_audit(
    detected_event_model_evidence: pd.DataFrame,
    output: Path,
    *,
    promoted_off_swr_event_model_evidence: pd.DataFrame | None = None,
    one_per_source_decisions: pd.DataFrame | None = None,
    one_per_source_selection_rule: str = "strongest_exact_margin",
    margin_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    """Write a detected replay/off-SWR first-order IMM posterior-content comparison."""

    output.mkdir(parents=True, exist_ok=True)
    event_tables = [
        build_first_order_imm_mode_usage_event_table(
            _with_event_scope(detected_event_model_evidence, event_class="detected_replay_or_swr"),
            event_class="detected_replay_or_swr",
            group_columns=DEFAULT_GROUP_COLUMNS,
            margin_threshold=margin_threshold,
        )
    ]

    if promoted_off_swr_event_model_evidence is not None:
        promoted = _with_event_scope(
            promoted_off_swr_event_model_evidence,
            event_class="promoted_off_swr",
        )
        event_tables.append(
            build_first_order_imm_mode_usage_event_table(
                promoted,
                event_class="promoted_off_swr",
                group_columns=OFF_SWR_GROUP_COLUMNS,
                margin_threshold=margin_threshold,
            )
        )
        if one_per_source_decisions is not None:
            selected = _selected_one_per_source_evidence(
                promoted_off_swr_event_model_evidence,
                one_per_source_decisions,
                selection_rule=one_per_source_selection_rule,
            )
            selected = _with_event_scope(
                selected,
                event_class="promoted_off_swr_one_per_source",
                selection_rule=one_per_source_selection_rule,
            )
            event_tables.append(
                build_first_order_imm_mode_usage_event_table(
                    selected,
                    event_class="promoted_off_swr_one_per_source",
                    selection_rule=one_per_source_selection_rule,
                    group_columns=OFF_SWR_GROUP_COLUMNS,
                    margin_threshold=margin_threshold,
                )
            )

    event_table = pd.concat(event_tables, ignore_index=True)
    summary = build_first_order_imm_mode_usage_summary(event_table)
    comparison = build_first_order_imm_mode_usage_comparison(event_table)
    rat = build_first_order_imm_mode_usage_rat_summary(event_table, include_event_class=True)
    gates = build_first_order_imm_mode_usage_class_gate_summary(
        event_table,
        margin_threshold=margin_threshold,
    )
    one_per = event_table[event_table["event_class"].eq("promoted_off_swr_one_per_source")].copy()
    one_per_summary = build_first_order_imm_mode_usage_summary(one_per)
    one_per_gates = build_first_order_imm_mode_usage_gate_summary(
        one_per,
        margin_threshold=margin_threshold,
    )

    outputs = {
        "first_order_imm_mode_usage_event_table.csv": event_table,
        "first_order_imm_mode_usage_event_summary.csv": event_table,
        "first_order_imm_mode_usage_summary.csv": summary,
        "swr_off_swr_first_order_imm_mode_usage_comparison.csv": comparison,
        "first_order_imm_mode_usage_rat_summary.csv": rat,
        "rat_first_order_imm_mode_usage_summary.csv": rat,
        "first_order_imm_mode_usage_gate_summary.csv": gates,
        "off_swr_one_per_source_group_mode_usage_summary.csv": one_per_summary,
        "off_swr_one_per_source_group_posterior_content_gate.csv": one_per_gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", required=True)
    parser.add_argument("--promoted-off-swr-event-model-evidence")
    parser.add_argument("--one-per-source-decisions")
    parser.add_argument("--one-per-source-selection-rule", default="strongest_exact_margin")
    parser.add_argument("--output", default="results/first-order-imm-mode-usage-audit")
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    event_model_evidence = _read_event_model_evidence(Path(args.event_model_evidence))
    promoted = (
        _read_event_model_evidence(Path(args.promoted_off_swr_event_model_evidence))
        if args.promoted_off_swr_event_model_evidence
        else None
    )
    one_per = pd.read_csv(args.one_per_source_decisions) if args.one_per_source_decisions else None
    if promoted is None:
        outputs = write_first_order_imm_mode_usage_audit(
            event_model_evidence,
            Path(args.output),
            margin_threshold=args.margin_threshold,
        )
    else:
        outputs = write_first_order_imm_mode_usage_comparison_audit(
            event_model_evidence,
            Path(args.output),
            promoted_off_swr_event_model_evidence=promoted,
            one_per_source_decisions=one_per,
            one_per_source_selection_rule=args.one_per_source_selection_rule,
            margin_threshold=args.margin_threshold,
        )
    print("First-order IMM mode usage summary:")
    print(outputs["first_order_imm_mode_usage_summary.csv"].to_string(index=False))
    print("\nFirst-order IMM mode usage gates:")
    print(outputs["first_order_imm_mode_usage_gate_summary.csv"].to_string(index=False))
    if "swr_off_swr_first_order_imm_mode_usage_comparison.csv" in outputs:
        print("\nSWR/off-SWR first-order IMM mode usage comparison:")
        print(outputs["swr_off_swr_first_order_imm_mode_usage_comparison.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
