#!/usr/bin/env python3
"""Audit event-mean first-order IMM posterior content.

This script tests the stronger claim that first-order IMM winners actually use
nonstationary trajectory modes during the event and express spatial
displacement. It intentionally fails that content claim on older artifacts that
only contain terminal mode probabilities.
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

DEFAULT_GROUP_COLUMNS = ("session", "event_index")
OFF_SWR_GROUP_COLUMNS = ("session", "event_index", "null_index")
LEGACY_SUCCESS_STATUS_VALUES = {"", "nan", "none", "null", "na", "n/a", "<na>"}

EVENT_COLUMNS = (
    "event_class",
    "session",
    "rat",
    "event_index",
    "null_index",
    "candidate_id",
    "source_event_group",
    "selection_rule",
    "first_order_imm_is_best_exact_core",
    "best_exact_core_model",
    "best_exact_core_margin_to_runner_up",
    "mean_stationary_mode_probability",
    "mean_nonstationary_mode_probability",
    "terminal_stationary_mode_probability",
    "terminal_nonstationary_mode_probability",
    "fraction_time_map_stationary",
    "fraction_time_map_nonstationary",
    "nonstationary_bout_count",
    "longest_nonstationary_bout_s",
    "posterior_expected_path_length_cm",
    "posterior_net_displacement_cm",
    "posterior_path_speed_cm_s",
    "event_mean_mode_diagnostics_present",
    "map_mode_diagnostics_present",
    "spatial_content_diagnostics_present",
    "moderate_mode_gate_passed",
    "moderate_spatial_gate_passed",
    "strong_mode_gate_passed",
    "trajectory_content_gate_passed",
    "strong_trajectory_content_gate_passed",
    "content_diagnostic_status",
)

SUMMARY_COLUMNS = (
    "scope",
    "event_class",
    "rat",
    "session",
    "events",
    "first_order_imm_best_events",
    "event_mean_mode_diagnostic_events",
    "map_mode_diagnostic_events",
    "spatial_content_diagnostic_events",
    "moderate_content_gate_events",
    "moderate_content_gate_fraction_of_first_order_best",
    "strong_content_gate_events",
    "strong_content_gate_fraction_of_first_order_best",
    "median_mean_nonstationary_mode_probability",
    "median_fraction_time_map_nonstationary",
    "median_posterior_expected_path_length_cm",
    "median_posterior_net_displacement_cm",
    "posterior_content_status",
)

GATE_COLUMNS = ("event_class", "selection_rule", "gate", "passed", "observed", "criterion", "required_for_overall")


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _status_is_success(value: object) -> bool:
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    status = str(value).strip().lower()
    return status == "success" or status in LEGACY_SUCCESS_STATUS_VALUES


def _successful_status_mask(status: pd.Series) -> pd.Series:
    return status.map(_status_is_success).astype(bool)


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_value(group: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in group:
        return default
    value = group.iloc[0][column]
    return "" if pd.isna(value) else value


def _read_event_model_evidence(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")
    if "status" in frame:
        frame = frame[_successful_status_mask(frame["status"])].copy()
    frame["session"] = frame["session"].astype(str)
    frame["rat"] = frame["session"].map(_rat_from_session)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    if "null_index" in frame:
        frame["null_index"] = pd.to_numeric(frame["null_index"], errors="coerce")
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


def _diag_value(group: pd.DataFrame, model: str, column: str) -> float:
    rows = group[group["model"].eq(model)]
    if rows.empty or column not in rows:
        return float("nan")
    value = pd.to_numeric(rows.iloc[-1][column], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _best_exact_core(group: pd.DataFrame) -> tuple[str, float]:
    exact = group[group["model"].isin(REQUIRED_EXACT_CORE_MODELS) & _bool_column(group, "evidence_comparable")].copy()
    if exact.empty:
        return "", float("nan")
    exact = exact.sort_values("log_evidence", ascending=False)
    winner = str(exact.iloc[0]["model"])
    margin = float(exact.iloc[0]["log_evidence"] - exact.iloc[1]["log_evidence"]) if len(exact) > 1 else float("nan")
    return winner, margin


def _candidate_id(group: pd.DataFrame, session: str, event_index: int) -> str:
    null_index = _first_value(group, "null_index", "")
    if null_index == "":
        return f"{session}|event={event_index}"
    try:
        null_label = int(float(null_index))
    except (TypeError, ValueError):
        null_label = str(null_index)
    return f"{session}|event={event_index}|null={null_label}"


def _source_event_group(group: pd.DataFrame, session: str, event_index: int) -> str:
    explicit = _first_value(group, "source_event_group_id", "")
    if explicit:
        return str(explicit)
    source_event = _first_value(group, "source_event_index", event_index)
    return f"{session}|event={int(float(source_event))}"


def _row_status(row: dict[str, object]) -> str:
    if row["trajectory_content_gate_passed"]:
        return "moderate_posterior_content_gate_passed"
    if row["event_mean_mode_diagnostics_present"] and row["spatial_content_diagnostics_present"]:
        return "posterior_content_diagnostics_present_gate_failed"
    if row["event_mean_mode_diagnostics_present"]:
        return "missing_spatial_content_diagnostics"
    if row["terminal_stationary_mode_probability"] == row["terminal_stationary_mode_probability"]:
        return "terminal_only_mode_audit"
    return "missing_mode_diagnostics"


def build_event_mean_mode_usage_event_summary(
    event_model_evidence: pd.DataFrame,
    *,
    event_class: str,
    selection_rule: str = "",
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
    path_threshold_cm: float = 10.0,
) -> pd.DataFrame:
    evidence = event_model_evidence.copy()
    group_columns = tuple(group_columns)
    missing = sorted(set(group_columns).difference(evidence.columns))
    if missing:
        raise ValueError(f"event evidence is missing group columns: {missing}")

    rows: list[dict[str, object]] = []
    for _, group in evidence.groupby(list(group_columns), sort=True, dropna=False):
        session = str(group.iloc[0]["session"])
        event_index = int(group.iloc[0]["event_index"])
        winner, margin = _best_exact_core(group)
        mean_stationary = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_mode_stationary_event_probability")
        mean_diffusion = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_mode_diffusion_event_probability")
        mean_fragmented = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_mode_fragmented_event_probability")
        terminal_stationary = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_mode_stationary_terminal_probability")
        terminal_diffusion = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_mode_diffusion_terminal_probability")
        terminal_fragmented = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_mode_fragmented_terminal_probability")
        fraction_map_stationary = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_fraction_time_map_stationary")
        fraction_map_nonstationary = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_fraction_time_map_nonstationary")
        path_length = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_posterior_expected_path_length_cm")
        net_displacement = _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_posterior_net_displacement_cm")
        event_mean_present = all(np.isfinite(value) for value in (mean_stationary, mean_diffusion, mean_fragmented))
        map_present = all(np.isfinite(value) for value in (fraction_map_stationary, fraction_map_nonstationary))
        spatial_present = np.isfinite(path_length) and np.isfinite(net_displacement)
        mean_nonstationary = mean_diffusion + mean_fragmented if event_mean_present else float("nan")
        terminal_nonstationary = terminal_diffusion + terminal_fragmented if all(np.isfinite(value) for value in (terminal_diffusion, terminal_fragmented)) else float("nan")
        moderate_mode = bool(
            (np.isfinite(mean_nonstationary) and mean_nonstationary >= 0.5)
            or (np.isfinite(fraction_map_nonstationary) and fraction_map_nonstationary >= 0.5)
        )
        moderate_spatial = bool(
            (np.isfinite(path_length) and path_length >= path_threshold_cm)
            or (np.isfinite(net_displacement) and net_displacement >= path_threshold_cm)
        )
        strong_mode = bool(
            np.isfinite(mean_nonstationary)
            and mean_nonstationary >= 0.5
            and np.isfinite(fraction_map_nonstationary)
            and fraction_map_nonstationary >= 0.5
        )
        row = {
            "event_class": event_class,
            "session": session,
            "rat": _rat_from_session(session),
            "event_index": event_index,
            "null_index": _first_value(group, "null_index", ""),
            "candidate_id": _candidate_id(group, session, event_index),
            "source_event_group": _source_event_group(group, session, event_index),
            "selection_rule": selection_rule or str(_first_value(group, "selection_rule", "")),
            "first_order_imm_is_best_exact_core": winner == FIRST_ORDER_IMM,
            "best_exact_core_model": winner,
            "best_exact_core_margin_to_runner_up": margin,
            "mean_stationary_mode_probability": mean_stationary,
            "mean_nonstationary_mode_probability": mean_nonstationary,
            "terminal_stationary_mode_probability": terminal_stationary,
            "terminal_nonstationary_mode_probability": terminal_nonstationary,
            "fraction_time_map_stationary": fraction_map_stationary,
            "fraction_time_map_nonstationary": fraction_map_nonstationary,
            "nonstationary_bout_count": _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_nonstationary_bout_count"),
            "longest_nonstationary_bout_s": _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_longest_nonstationary_bout_s"),
            "posterior_expected_path_length_cm": path_length,
            "posterior_net_displacement_cm": net_displacement,
            "posterior_path_speed_cm_s": _diag_value(group, FIRST_ORDER_IMM, "diagnostic_state_space_imm_posterior_path_speed_cm_s"),
            "event_mean_mode_diagnostics_present": bool(event_mean_present),
            "map_mode_diagnostics_present": bool(map_present),
            "spatial_content_diagnostics_present": bool(spatial_present),
            "moderate_mode_gate_passed": moderate_mode,
            "moderate_spatial_gate_passed": moderate_spatial,
            "strong_mode_gate_passed": strong_mode,
            "trajectory_content_gate_passed": bool(winner == FIRST_ORDER_IMM and moderate_mode and moderate_spatial),
            "strong_trajectory_content_gate_passed": bool(winner == FIRST_ORDER_IMM and strong_mode and np.isfinite(path_length) and path_length >= path_threshold_cm),
        }
        row["content_diagnostic_status"] = _row_status(row)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _summary_row(scope: str, frame: pd.DataFrame, *, event_class: str = "", rat: str = "", session: str = "") -> dict[str, object]:
    first_order = frame[_bool_column(frame, "first_order_imm_is_best_exact_core")].copy()
    denominator = len(first_order)
    moderate = int(_bool_column(first_order, "trajectory_content_gate_passed").sum()) if not first_order.empty else 0
    strong = int(_bool_column(first_order, "strong_trajectory_content_gate_passed").sum()) if not first_order.empty else 0
    status = (
        "posterior_content_supported"
        if denominator > 0 and moderate > denominator / 2
        else "posterior_content_not_supported"
        if denominator > 0
        else "no_first_order_imm_best_rows"
    )
    return {
        "scope": scope,
        "event_class": event_class,
        "rat": rat,
        "session": session,
        "events": int(len(frame)),
        "first_order_imm_best_events": int(denominator),
        "event_mean_mode_diagnostic_events": int(_bool_column(first_order, "event_mean_mode_diagnostics_present").sum()) if not first_order.empty else 0,
        "map_mode_diagnostic_events": int(_bool_column(first_order, "map_mode_diagnostics_present").sum()) if not first_order.empty else 0,
        "spatial_content_diagnostic_events": int(_bool_column(first_order, "spatial_content_diagnostics_present").sum()) if not first_order.empty else 0,
        "moderate_content_gate_events": moderate,
        "moderate_content_gate_fraction_of_first_order_best": moderate / denominator if denominator else np.nan,
        "strong_content_gate_events": strong,
        "strong_content_gate_fraction_of_first_order_best": strong / denominator if denominator else np.nan,
        "median_mean_nonstationary_mode_probability": _median(first_order, "mean_nonstationary_mode_probability"),
        "median_fraction_time_map_nonstationary": _median(first_order, "fraction_time_map_nonstationary"),
        "median_posterior_expected_path_length_cm": _median(first_order, "posterior_expected_path_length_cm"),
        "median_posterior_net_displacement_cm": _median(first_order, "posterior_net_displacement_cm"),
        "posterior_content_status": status,
    }


def _median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.median()) if not values.empty else float("nan")


def build_mode_usage_summary(event_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row("all_events", event_summary)]
    for event_class, group in event_summary.groupby("event_class", sort=True, dropna=False):
        rows.append(_summary_row("event_class", group, event_class=str(event_class)))
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def build_group_mode_usage_summary(event_summary: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    rows = []
    for keys, group in event_summary.groupby(["event_class", group_column], sort=True, dropna=False):
        event_class, group_value = keys
        rows.append(
            _summary_row(
                group_column,
                group,
                event_class=str(event_class),
                rat=str(group_value) if group_column == "rat" else "",
                session=str(group_value) if group_column == "session" else "",
            )
        )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def build_mode_usage_gate_summary(
    event_summary: pd.DataFrame,
    *,
    path_threshold_cm: float = 10.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(event_class: str, selection_rule: str, gate: str, passed: bool, observed: object, criterion: str, *, required: bool = True) -> None:
        rows.append(
            {
                "event_class": event_class,
                "selection_rule": selection_rule,
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required),
            }
        )

    for (event_class, selection_rule), group in event_summary.groupby(["event_class", "selection_rule"], sort=True, dropna=False):
        first_order = group[_bool_column(group, "first_order_imm_is_best_exact_core")].copy()
        n_first = len(first_order)
        event_diag = int(_bool_column(first_order, "event_mean_mode_diagnostics_present").sum()) if n_first else 0
        map_diag = int(_bool_column(first_order, "map_mode_diagnostics_present").sum()) if n_first else 0
        spatial_diag = int(_bool_column(first_order, "spatial_content_diagnostics_present").sum()) if n_first else 0
        moderate = int(_bool_column(first_order, "trajectory_content_gate_passed").sum()) if n_first else 0
        strong = int(_bool_column(first_order, "strong_trajectory_content_gate_passed").sum()) if n_first else 0
        add(str(event_class), str(selection_rule), "events_present", len(group) > 0, int(len(group)), "event rows are present")
        add(str(event_class), str(selection_rule), "first_order_imm_best_rows_present", n_first > 0, n_first, "first-order IMM is exact-core best on at least one event")
        add(str(event_class), str(selection_rule), "event_mean_mode_diagnostics_complete", n_first > 0 and event_diag == n_first, f"{event_diag}/{n_first}", "event-mean mode probabilities are present for all first-order IMM best rows")
        add(str(event_class), str(selection_rule), "map_mode_diagnostics_complete", n_first > 0 and map_diag == n_first, f"{map_diag}/{n_first}", "MAP mode fractions and bout diagnostics are present for all first-order IMM best rows")
        add(str(event_class), str(selection_rule), "spatial_content_diagnostics_complete", n_first > 0 and spatial_diag == n_first, f"{spatial_diag}/{n_first}", "posterior path length/net displacement diagnostics are present for all first-order IMM best rows")
        add(str(event_class), str(selection_rule), "moderate_content_majority", n_first > 0 and moderate > n_first / 2, f"{moderate}/{n_first}", f"majority pass moderate gate: mean/fraction nonstationary >= 0.5 and path/net displacement >= {path_threshold_cm:g} cm")
        add(str(event_class), str(selection_rule), "strong_content_majority", n_first > 0 and strong > n_first / 2, f"{strong}/{n_first}", f"majority pass strong gate: mean and fraction nonstationary >= 0.5 and path length >= {path_threshold_cm:g} cm", required=False)
    required_rows = [row for row in rows if row["required_for_overall"]]
    add("all", "", "overall", all(row["passed"] for row in required_rows), f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed", "all required event-mean posterior-content gates pass")
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def _selected_one_per_source_evidence(
    promoted_evidence: pd.DataFrame,
    one_per_source_decisions: pd.DataFrame,
    *,
    selection_rule: str,
) -> pd.DataFrame:
    selected = one_per_source_decisions[
        one_per_source_decisions["selection_rule"].astype(str).eq(selection_rule)
    ].copy()
    if selected.empty:
        raise ValueError(f"no one-per-source decisions found for selection rule {selection_rule!r}")
    evidence = promoted_evidence.copy()
    for frame in (selected, evidence):
        frame["session"] = frame["session"].astype(str)
        frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
        frame["null_index"] = pd.to_numeric(frame["null_index"], errors="raise").astype(int)
    selected_columns = [
        column
        for column in ("session", "event_index", "null_index", "source_event_group_id", "selection_rule")
        if column in selected
    ]
    return evidence.merge(selected[selected_columns], on=["session", "event_index", "null_index"], how="inner")


def write_event_mean_mode_usage_audit(
    *,
    event_model_evidence: pd.DataFrame,
    output: Path,
    promoted_off_swr_event_model_evidence: pd.DataFrame | None = None,
    one_per_source_decisions: pd.DataFrame | None = None,
    one_per_source_selection_rule: str = "strongest_exact_margin",
    path_threshold_cm: float = 10.0,
) -> dict[str, pd.DataFrame]:
    tables = [
        build_event_mean_mode_usage_event_summary(
            event_model_evidence,
            event_class="detected_replay_or_swr",
            group_columns=DEFAULT_GROUP_COLUMNS,
            path_threshold_cm=path_threshold_cm,
        )
    ]
    if promoted_off_swr_event_model_evidence is not None:
        tables.append(
            build_event_mean_mode_usage_event_summary(
                promoted_off_swr_event_model_evidence,
                event_class="promoted_off_swr",
                group_columns=OFF_SWR_GROUP_COLUMNS,
                path_threshold_cm=path_threshold_cm,
            )
        )
        if one_per_source_decisions is not None:
            selected = _selected_one_per_source_evidence(
                promoted_off_swr_event_model_evidence,
                one_per_source_decisions,
                selection_rule=one_per_source_selection_rule,
            )
            tables.append(
                build_event_mean_mode_usage_event_summary(
                    selected,
                    event_class="promoted_off_swr_one_per_source",
                    selection_rule=one_per_source_selection_rule,
                    group_columns=OFF_SWR_GROUP_COLUMNS,
                    path_threshold_cm=path_threshold_cm,
                )
            )

    event_summary = pd.concat(tables, ignore_index=True)
    summary = build_mode_usage_summary(event_summary)
    gates = build_mode_usage_gate_summary(event_summary, path_threshold_cm=path_threshold_cm)
    rat = build_group_mode_usage_summary(event_summary, group_column="rat")
    session = build_group_mode_usage_summary(event_summary, group_column="session")
    comparison = summary[summary["scope"].eq("event_class")].copy()
    one_per = event_summary[event_summary["event_class"].eq("promoted_off_swr_one_per_source")].copy()
    one_per_summary = build_mode_usage_summary(one_per)

    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "first_order_imm_mode_usage_event_summary.csv": event_summary,
        "first_order_imm_mode_usage_gate_summary.csv": gates,
        "rat_first_order_imm_mode_usage_summary.csv": rat,
        "session_first_order_imm_mode_usage_summary.csv": session,
        "swr_off_swr_first_order_imm_mode_usage_comparison.csv": comparison,
        "off_swr_one_per_source_group_mode_usage_summary.csv": one_per_summary,
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
    parser.add_argument("--output", default="results/first-order-imm-event-mean-mode-usage-audit")
    parser.add_argument("--path-threshold-cm", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = write_event_mean_mode_usage_audit(
        event_model_evidence=_read_event_model_evidence(Path(args.event_model_evidence)),
        promoted_off_swr_event_model_evidence=_read_event_model_evidence(Path(args.promoted_off_swr_event_model_evidence))
        if args.promoted_off_swr_event_model_evidence
        else None,
        one_per_source_decisions=pd.read_csv(args.one_per_source_decisions) if args.one_per_source_decisions else None,
        one_per_source_selection_rule=args.one_per_source_selection_rule,
        output=Path(args.output),
        path_threshold_cm=args.path_threshold_cm,
    )
    print("First-order IMM event-mean posterior-content comparison:")
    print(outputs["swr_off_swr_first_order_imm_mode_usage_comparison.csv"].to_string(index=False))
    print("\nFirst-order IMM event-mean posterior-content gates:")
    print(outputs["first_order_imm_mode_usage_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
