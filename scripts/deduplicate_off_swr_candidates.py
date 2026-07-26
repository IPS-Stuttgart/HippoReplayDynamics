#!/usr/bin/env python3
"""Collapse promoted off-SWR candidates by source event group.

The promoted off-SWR validation artifact is window-level. This audit checks
whether the result survives a one-candidate-per-source-event rule so duplicate
matched-null windows around a small set of source events cannot inflate the
claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ("session", "event_index", "null_index")
GROUP_COLUMNS = ("session", "event_index")
EXACT_MARGIN = "trajectory_minus_nontrajectory_log_evidence"
DISCOVERY_MARGIN = "trajectory_family_margin"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"
PROMOTION_READY_LABEL = "promotion_ready_high_specificity_candidate"

SOURCE_GROUP_COLUMNS = (
    "source_event_group_id",
    "session",
    "rat",
    "source_event_index",
    "candidate_windows",
    "high_specificity_windows",
    "promotion_ready_windows",
    "exact_validated_windows",
    "exact_trajectory_confident_windows",
    "exact_nontrajectory_confident_windows",
    "immobile_windows",
    "running_windows",
    "earliest_window_start_s",
    "latest_window_end_s",
    "median_discovery_margin",
    "max_discovery_margin",
    "median_exact_margin",
    "max_exact_margin",
    "best_exact_trajectory_model_distribution",
)

DECISION_COLUMNS = (
    "selection_rule",
    "source_event_group_id",
    "session",
    "rat",
    "source_event_index",
    "event_index",
    "null_index",
    "candidate_rank",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "n_spikes",
    "active_cell_count",
    "run_or_immobility_state",
    "animal_speed_mean",
    "distance_to_nearest_swr_s",
    "trajectory_family_margin",
    "trajectory_minus_nontrajectory_log_evidence",
    "best_trajectory_model",
    "trajectory_confident_claim",
    "nontrajectory_confident_claim",
    "margin_decision",
)

SUMMARY_COLUMNS = (
    "selection_rule",
    "source_event_groups",
    "selected_candidates",
    "candidate_sessions",
    "candidate_rats",
    "trajectory_confident_candidates",
    "nontrajectory_confident_candidates",
    "ambiguous_candidates",
    "trajectory_confident_fraction",
    "median_exact_margin",
    "min_exact_margin",
    "max_exact_margin",
    "median_discovery_margin",
    "min_discovery_margin",
    "immobile_candidates",
    "running_candidates",
    "median_mean_speed_cm_s",
    "min_nearest_known_swr_distance_s",
    "first_order_imm_best_candidates",
    "first_order_imm_best_fraction",
    "exact_sparse_momentum_best_candidates",
    "exact_sparse_momentum_best_fraction",
    "best_trajectory_model_distribution",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required input table is missing: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(numeric) and numeric != 0.0)


def _bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _integer_identifier_series(series: pd.Series, name: str) -> pd.Series:
    if series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{name} must contain integer identifiers")
    try:
        integer = pd.to_numeric(series, errors="raise").astype("Int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integer identifiers") from exc
    if integer.isna().any():
        raise ValueError(f"{name} must contain integer identifiers")
    return integer.astype(int)


def _source_group_id(session: object, event_index: object) -> str:
    return f"{session}|event={int(event_index)}"


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    missing = [column for column in GROUP_COLUMNS if column not in out]
    if missing:
        raise ValueError(f"candidate table is missing source-group columns: {missing}")
    out["session"] = out["session"].astype(str)
    if "rat" not in out:
        out["rat"] = out["session"].map(_rat_from_session)
    out["rat"] = out["rat"].astype(str)
    out["event_index"] = _integer_identifier_series(out["event_index"], "event_index")
    if "null_index" in out:
        out["null_index"] = pd.to_numeric(out["null_index"], errors="coerce").astype("Int64")
    out["source_event_index"] = out["event_index"].astype(int)
    out["source_event_group_id"] = [
        _source_group_id(session, event_index)
        for session, event_index in zip(out["session"], out["event_index"], strict=True)
    ]
    return out


def _promotion_ready_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if "passes_high_specificity_promotion_filter" in frame:
        return _bool_series(frame, "passes_high_specificity_promotion_filter")
    if "high_specificity_label" in frame:
        return frame["high_specificity_label"].astype(str).eq(PROMOTION_READY_LABEL)
    return pd.Series(False, index=frame.index)


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    if frame.empty or not set(KEY_COLUMNS).issubset(frame.columns):
        return set()
    normalized = frame[list(KEY_COLUMNS)].copy()
    normalized["session"] = normalized["session"].astype(str)
    normalized["event_index"] = _integer_identifier_series(
        normalized["event_index"],
        "event_index",
    )
    normalized["null_index"] = pd.to_numeric(normalized["null_index"], errors="coerce").astype("Int64")
    return set(map(tuple, normalized.astype(object).to_numpy()))


def _filter_keys(frame: pd.DataFrame, keys: set[tuple[object, ...]]) -> pd.DataFrame:
    if frame.empty or not keys or not set(KEY_COLUMNS).issubset(frame.columns):
        return frame.iloc[0:0].copy()
    normalized = frame[list(KEY_COLUMNS)].copy()
    normalized["session"] = normalized["session"].astype(str)
    normalized["event_index"] = _integer_identifier_series(
        normalized["event_index"],
        "event_index",
    )
    normalized["null_index"] = pd.to_numeric(normalized["null_index"], errors="coerce").astype("Int64")
    mask = [tuple(row) in keys for row in normalized.astype(object).to_numpy()]
    return frame[pd.Series(mask, index=frame.index)].copy()


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.median()) if not values.empty else np.nan


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.min()) if not values.empty else np.nan


def _safe_max(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.max()) if not values.empty else np.nan


def _safe_fraction(value: int | float, denominator: int | float) -> float:
    if denominator is None or float(denominator) == 0.0:
        return np.nan
    return float(value) / float(denominator)


def _count_state(frame: pd.DataFrame, state: str) -> int:
    if "run_or_immobility_state" not in frame:
        return 0
    return int(frame["run_or_immobility_state"].astype(str).eq(state).sum())


def _model_distribution(frame: pd.DataFrame) -> str:
    if frame.empty or "best_trajectory_model" not in frame:
        return ""
    counts = frame["best_trajectory_model"].dropna().astype(str).value_counts()
    total = int(counts.sum())
    if total == 0:
        return ""
    return "; ".join(f"{model}={count} ({count / total:.3f})" for model, count in counts.items())


def build_source_event_group_summary(
    *,
    candidate_table: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
) -> pd.DataFrame:
    candidates = _prepare(candidate_table)
    high = _prepare(high_specificity)
    validated = _prepare(validation_decisions)
    ready = high[_promotion_ready_mask(high)].copy()
    group_ids = sorted(
        set(candidates.get("source_event_group_id", pd.Series(dtype=str)).dropna().astype(str))
        | set(high.get("source_event_group_id", pd.Series(dtype=str)).dropna().astype(str))
        | set(validated.get("source_event_group_id", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, object]] = []
    for group_id in group_ids:
        frames = {
            "candidates": _source_group_rows(candidates, group_id),
            "high": _source_group_rows(high, group_id),
            "ready": _source_group_rows(ready, group_id),
            "validated": _source_group_rows(validated, group_id),
        }
        representative = next((frame for frame in frames.values() if not frame.empty), pd.DataFrame()).iloc[0]
        validated_group = frames["validated"]
        rows.append(
            {
                "source_event_group_id": group_id,
                "session": str(representative.get("session", "")),
                "rat": str(representative.get("rat", "")),
                "source_event_index": int(representative.get("source_event_index", representative.get("event_index", -1))),
                "candidate_windows": int(len(frames["candidates"])),
                "high_specificity_windows": int(len(frames["high"])),
                "promotion_ready_windows": int(len(frames["ready"])),
                "exact_validated_windows": int(len(validated_group)),
                "exact_trajectory_confident_windows": int(_bool_series(validated_group, "trajectory_confident_claim").sum()),
                "exact_nontrajectory_confident_windows": int(_bool_series(validated_group, "nontrajectory_confident_claim").sum()),
                "immobile_windows": _count_state(validated_group, "immobile"),
                "running_windows": _count_state(validated_group, "run"),
                "earliest_window_start_s": _safe_min(validated_group, "window_start_s"),
                "latest_window_end_s": _safe_max(validated_group, "window_end_s"),
                "median_discovery_margin": _safe_median(validated_group, DISCOVERY_MARGIN),
                "max_discovery_margin": _safe_max(validated_group, DISCOVERY_MARGIN),
                "median_exact_margin": _safe_median(validated_group, EXACT_MARGIN),
                "max_exact_margin": _safe_max(validated_group, EXACT_MARGIN),
                "best_exact_trajectory_model_distribution": _model_distribution(validated_group),
            }
        )
    return pd.DataFrame(rows, columns=list(SOURCE_GROUP_COLUMNS))


def _source_group_rows(frame: pd.DataFrame, group_id: str) -> pd.DataFrame:
    if frame.empty or "source_event_group_id" not in frame:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame["source_event_group_id"].astype(str).eq(group_id)].copy()


def _select_one_per_group(validation_decisions: pd.DataFrame, *, rule: str) -> pd.DataFrame:
    frame = _prepare(validation_decisions)
    if frame.empty:
        return frame.copy()
    if rule == "strongest_exact_margin":
        sort_cols = [EXACT_MARGIN, DISCOVERY_MARGIN, "window_start_s"]
        ascending = [False, False, True]
    elif rule == "strongest_discovery_margin":
        sort_cols = [DISCOVERY_MARGIN, EXACT_MARGIN, "window_start_s"]
        ascending = [False, False, True]
    elif rule == "earliest_window":
        sort_cols = ["window_start_s", EXACT_MARGIN]
        ascending = [True, False]
    else:
        raise ValueError(f"unknown selection rule: {rule}")
    for column in sort_cols:
        if column not in frame:
            frame[column] = np.nan
    selected = (
        frame.sort_values(sort_cols, ascending=ascending, na_position="last")
        .groupby(list(GROUP_COLUMNS), as_index=False, sort=True)
        .head(1)
        .copy()
    )
    selected["selection_rule"] = rule
    return selected


def build_one_per_source_group_decisions(validation_decisions: pd.DataFrame) -> pd.DataFrame:
    selections = [
        _select_one_per_group(validation_decisions, rule=rule)
        for rule in ("strongest_exact_margin", "strongest_discovery_margin", "earliest_window")
    ]
    out = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    for column in DECISION_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[list(DECISION_COLUMNS)].copy()


def _summary_for_rule(rule: str, frame: pd.DataFrame) -> dict[str, object]:
    trajectory = _bool_series(frame, "trajectory_confident_claim")
    nontrajectory = _bool_series(frame, "nontrajectory_confident_claim")
    first_order = frame["best_trajectory_model"].astype(str).eq(FIRST_ORDER_IMM) if "best_trajectory_model" in frame else pd.Series(False, index=frame.index)
    momentum = frame["best_trajectory_model"].astype(str).eq(MOMENTUM_EXACT) if "best_trajectory_model" in frame else pd.Series(False, index=frame.index)
    return {
        "selection_rule": rule,
        "source_event_groups": int(frame["source_event_group_id"].nunique()) if "source_event_group_id" in frame else 0,
        "selected_candidates": int(len(frame)),
        "candidate_sessions": int(frame["session"].nunique()) if "session" in frame else 0,
        "candidate_rats": int(frame["rat"].nunique()) if "rat" in frame else 0,
        "trajectory_confident_candidates": int(trajectory.sum()),
        "nontrajectory_confident_candidates": int(nontrajectory.sum()),
        "ambiguous_candidates": int((~trajectory & ~nontrajectory).sum()) if len(frame) else 0,
        "trajectory_confident_fraction": _safe_fraction(int(trajectory.sum()), int(len(frame))),
        "median_exact_margin": _safe_median(frame, EXACT_MARGIN),
        "min_exact_margin": _safe_min(frame, EXACT_MARGIN),
        "max_exact_margin": _safe_max(frame, EXACT_MARGIN),
        "median_discovery_margin": _safe_median(frame, DISCOVERY_MARGIN),
        "min_discovery_margin": _safe_min(frame, DISCOVERY_MARGIN),
        "immobile_candidates": _count_state(frame, "immobile"),
        "running_candidates": _count_state(frame, "run"),
        "median_mean_speed_cm_s": _safe_median(frame, "animal_speed_mean"),
        "min_nearest_known_swr_distance_s": _safe_min(frame, "distance_to_nearest_swr_s"),
        "first_order_imm_best_candidates": int(first_order.sum()),
        "first_order_imm_best_fraction": _safe_fraction(int(first_order.sum()), int(len(frame))),
        "exact_sparse_momentum_best_candidates": int(momentum.sum()),
        "exact_sparse_momentum_best_fraction": _safe_fraction(int(momentum.sum()), int(len(frame))),
        "best_trajectory_model_distribution": _model_distribution(frame),
    }


def build_one_per_source_group_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    rows = [
        _summary_for_rule(rule, group.copy())
        for rule, group in decisions.groupby("selection_rule", sort=True)
    ]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def build_cluster_robustness_gate_summary(
    *,
    validation_decisions: pd.DataFrame,
    source_groups: pd.DataFrame,
    one_per_summary: pd.DataFrame,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, *, required: bool = True) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required),
            }
        )

    validated = _prepare(validation_decisions)
    source_group_count = int(validated["source_event_group_id"].nunique()) if not validated.empty else 0
    duplicate_ratio = _safe_fraction(len(validated), source_group_count)
    strongest = (
        one_per_summary[one_per_summary["selection_rule"].eq("strongest_exact_margin")].iloc[0]
        if not one_per_summary.empty and one_per_summary["selection_rule"].eq("strongest_exact_margin").any()
        else pd.Series(dtype=object)
    )
    all_rules_candidates = int(one_per_summary["selected_candidates"].sum()) if not one_per_summary.empty else 0
    all_rules_trajectory = int(one_per_summary["trajectory_confident_candidates"].sum()) if not one_per_summary.empty else 0
    all_rules_nontrajectory = int(one_per_summary["nontrajectory_confident_candidates"].sum()) if not one_per_summary.empty else 0
    min_rule_margin = _safe_min(one_per_summary, "min_exact_margin")

    add("promoted_candidates_present", len(validated) > 0, int(len(validated)), "promoted exact validation rows are present")
    add("source_event_groups_present", source_group_count > 0, source_group_count, "source-event groups are identifiable")
    add(
        "duplicate_windows_audited",
        len(validated) > source_group_count > 0,
        f"{len(validated)} windows / {source_group_count} source groups; ratio={duplicate_ratio:.3f}",
        "window-level candidates collapse to fewer source-event groups",
        required=False,
    )
    add(
        "deduplicated_source_groups_nontrivial",
        source_group_count >= 5,
        source_group_count,
        "at least five promoted source-event groups remain after de-duplication",
    )
    add(
        "strongest_exact_rule_trajectory_confident",
        not strongest.empty and int(strongest["trajectory_confident_candidates"]) == int(strongest["selected_candidates"]),
        f"{int(strongest.get('trajectory_confident_candidates', 0))}/{int(strongest.get('selected_candidates', 0))}",
        "one strongest exact-margin candidate per source group remains trajectory-confident",
    )
    add(
        "strongest_exact_rule_no_nontrajectory",
        not strongest.empty and int(strongest["nontrajectory_confident_candidates"]) == 0,
        int(strongest.get("nontrajectory_confident_candidates", 0)),
        "one strongest exact-margin candidate per source group has zero nontrajectory claims",
    )
    add(
        "strongest_exact_rule_margin_above_threshold",
        not strongest.empty
        and float(strongest["median_exact_margin"]) >= margin_threshold
        and float(strongest["min_exact_margin"]) >= margin_threshold,
        f"median={strongest.get('median_exact_margin', np.nan)}; min={strongest.get('min_exact_margin', np.nan)}",
        f"median and minimum exact margins remain >= {margin_threshold}",
    )
    add(
        "strongest_exact_rule_first_order_imm_common",
        not strongest.empty and float(strongest["first_order_imm_best_fraction"]) >= 0.5,
        f"{int(strongest.get('first_order_imm_best_candidates', 0))}/{int(strongest.get('selected_candidates', 0))}",
        "first-order IMM remains at least half of one-per-source best trajectory rows",
    )
    add(
        "all_selection_rules_preserve_trajectory_claim",
        all_rules_candidates > 0 and all_rules_trajectory == all_rules_candidates and all_rules_nontrajectory == 0,
        f"trajectory={all_rules_trajectory}/{all_rules_candidates}; nontrajectory={all_rules_nontrajectory}",
        "strongest, discovery-margin, and earliest one-per-source selections all remain trajectory-confident",
    )
    add(
        "all_selection_rules_margin_above_threshold",
        np.isfinite(min_rule_margin) and min_rule_margin >= margin_threshold,
        f"min_rule_margin={min_rule_margin}",
        f"all one-per-source selection rules keep exact margins >= {margin_threshold}",
    )
    required_rows = [row for row in rows if row["required_for_overall"]]
    add(
        "overall",
        all(row["passed"] for row in required_rows),
        f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed",
        "all required source-event de-duplication gates pass",
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_off_swr_candidate_dedup_outputs(
    *,
    validation_decisions: pd.DataFrame,
    output: Path,
    candidate_table: pd.DataFrame | None = None,
    high_specificity: pd.DataFrame | None = None,
    margin_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    candidate_table = pd.DataFrame() if candidate_table is None else candidate_table
    high_specificity = pd.DataFrame() if high_specificity is None else high_specificity
    validation_decisions = _prepare(validation_decisions)
    candidate_table = _prepare(candidate_table) if not candidate_table.empty else pd.DataFrame()
    high_specificity = _prepare(high_specificity) if not high_specificity.empty else pd.DataFrame()

    if not candidate_table.empty:
        validated_keys = _key_set(validation_decisions)
        validation_candidates = _filter_keys(candidate_table, validated_keys)
        if not validation_candidates.empty:
            missing_cols = [column for column in candidate_table.columns if column not in validation_decisions.columns]
            validation_decisions = validation_decisions.merge(
                validation_candidates[list(KEY_COLUMNS) + missing_cols],
                on=list(KEY_COLUMNS),
                how="left",
                suffixes=("", "_candidate"),
            )
            validation_decisions = _prepare(validation_decisions)

    source_groups = build_source_event_group_summary(
        candidate_table=candidate_table,
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    )
    decisions = build_one_per_source_group_decisions(validation_decisions)
    summary = build_one_per_source_group_summary(decisions)
    gates = build_cluster_robustness_gate_summary(
        validation_decisions=validation_decisions,
        source_groups=source_groups,
        one_per_summary=summary,
        margin_threshold=margin_threshold,
    )
    outputs = {
        "off_swr_candidate_source_event_group_summary.csv": source_groups,
        "off_swr_candidate_one_per_source_group_decisions.csv": decisions,
        "off_swr_candidate_one_per_source_group_summary.csv": summary,
        "off_swr_candidate_cluster_robustness_gate_summary.csv": gates,
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-decisions", required=True)
    parser.add_argument("--candidate-table")
    parser.add_argument("--high-specificity-table")
    parser.add_argument("--output", default="results/off-swr-candidate-source-dedup")
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = write_off_swr_candidate_dedup_outputs(
        validation_decisions=_read_required_csv(Path(args.validation_decisions)),
        candidate_table=_read_optional_csv(Path(args.candidate_table)) if args.candidate_table else None,
        high_specificity=_read_optional_csv(Path(args.high_specificity_table)) if args.high_specificity_table else None,
        output=Path(args.output),
        margin_threshold=args.margin_threshold,
    )
    print("Off-SWR one-per-source summary:")
    print(outputs["off_swr_candidate_one_per_source_group_summary.csv"].to_string(index=False))
    print("\nOff-SWR source de-duplication gates:")
    print(outputs["off_swr_candidate_cluster_robustness_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
