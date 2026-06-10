#!/usr/bin/env python3
"""Build a denominator-backed funnel for promoted off-SWR candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FUNNEL_SUMMARY_COLUMNS = (
    "stage_order",
    "stage",
    "stage_label",
    "source_table",
    "windows",
    "fraction_of_screened_off_swr_windows",
    "fraction_of_previous_stage",
    "candidate_sessions",
    "candidate_rats",
    "immobile_windows",
    "running_windows",
    "unknown_speed_windows",
    "median_discovery_margin",
    "min_discovery_margin",
    "median_exact_margin",
    "min_exact_margin",
    "best_trajectory_model_distribution",
)

GROUP_SUMMARY_COLUMNS = (
    "group_type",
    "rat",
    "session",
    "off_swr_windows",
    "weak_candidates",
    "moderate_candidates",
    "strong_candidates",
    "extreme_candidates",
    "promotion_ready_candidates",
    "exact_validated_candidates",
    "exact_trajectory_confident_candidates",
    "exact_nontrajectory_confident_candidates",
    "median_exact_margin",
    "min_exact_margin",
    "max_exact_margin",
)

REJECTION_SUMMARY_COLUMNS = (
    "funnel_status",
    "candidate_windows",
    "fraction_of_candidate_windows",
    "candidate_sessions",
    "candidate_rats",
    "immobile_windows",
    "running_windows",
    "unknown_speed_windows",
    "movement_or_low_information_windows",
    "interesting_candidate_windows",
    "median_discovery_margin",
    "median_animal_speed_mean",
    "median_n_spikes",
    "median_active_cell_count",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")

TIER_THRESHOLDS = (
    ("weak", 5.5),
    ("moderate", 20.0),
    ("strong", 50.0),
    ("extreme", 100.0),
)

KEY_COLUMNS = ("session", "event_index", "null_index")
EXACT_MARGIN_COLUMN = "trajectory_minus_nontrajectory_log_evidence"
DISCOVERY_MARGIN_COLUMN = "trajectory_family_margin"
PROMOTION_READY_LABEL = "promotion_ready_high_specificity_candidate"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required artifact table is missing: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _safe_fraction(value: int | float, denominator: int | float) -> float:
    if denominator is None:
        return np.nan
    denominator_value = float(denominator)
    if not np.isfinite(denominator_value) or denominator_value == 0.0:
        return np.nan
    return float(value) / denominator_value


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.median()) if not values.empty else np.nan


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.min()) if not values.empty else np.nan


def _safe_max(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.max()) if not values.empty else np.nan


def _count_state(frame: pd.DataFrame, state: str) -> int:
    if "run_or_immobility_state" not in frame:
        return 0
    return int(frame["run_or_immobility_state"].astype(str).eq(state).sum())


def _unknown_speed_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "run_or_immobility_state" not in frame:
        return int(len(frame))
    state = frame["run_or_immobility_state"].astype(str)
    return int((state.eq("unknown_speed") | state.eq("") | state.eq("nan")).sum())


def _model_distribution(frame: pd.DataFrame) -> str:
    if frame.empty or "best_trajectory_model" not in frame:
        return ""
    counts = frame["best_trajectory_model"].dropna().astype(str).value_counts()
    total = int(counts.sum())
    if total == 0:
        return ""
    return "; ".join(f"{model}={count} ({count / total:.3f})" for model, count in counts.items())


def _has_key_columns(frame: pd.DataFrame) -> bool:
    return set(KEY_COLUMNS).issubset(frame.columns)


def _key_tuples(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    if frame.empty or not _has_key_columns(frame):
        return []
    key_frame = frame[list(KEY_COLUMNS)].astype(object).where(pd.notna(frame[list(KEY_COLUMNS)]), None)
    return list(map(tuple, key_frame.to_numpy()))


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(_key_tuples(frame))


def _mask_keys(frame: pd.DataFrame, keys: set[tuple[object, ...]]) -> pd.Series:
    if frame.empty or not keys or not _has_key_columns(frame):
        return pd.Series(False, index=frame.index)
    values = _key_tuples(frame)
    return pd.Series([value in keys for value in values], index=frame.index)


def _tier_subset(candidate_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if candidate_table.empty:
        return candidate_table.copy()
    return candidate_table[_numeric(candidate_table, DISCOVERY_MARGIN_COLUMN) >= float(threshold)].copy()


def _tier_summary_value(tier_summary: pd.DataFrame, tier: str, column: str) -> float:
    if tier_summary.empty or "candidate_tier" not in tier_summary or column not in tier_summary:
        return np.nan
    rows = tier_summary[tier_summary["candidate_tier"].astype(str).eq(tier)]
    if rows.empty:
        return np.nan
    value = pd.to_numeric(rows.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _off_swr_window_count(tier_summary: pd.DataFrame, run_state: pd.DataFrame, candidate_table: pd.DataFrame) -> int:
    value = _tier_summary_value(tier_summary, "weak", "off_swr_windows")
    if np.isfinite(value):
        return int(value)
    if not run_state.empty and "windows" in run_state:
        return int(_numeric(run_state, "windows").sum())
    return int(len(candidate_table))


def _run_state_denominator(tier_summary: pd.DataFrame, state_column: str) -> int:
    value = _tier_summary_value(tier_summary, "weak", state_column)
    return int(value) if np.isfinite(value) else 0


def _stage_row(
    *,
    order: int,
    stage: str,
    label: str,
    source_table: str,
    frame: pd.DataFrame,
    windows: int,
    screened: int,
    previous: int | None,
    median_margin_column: str = DISCOVERY_MARGIN_COLUMN,
    min_margin_column: str = DISCOVERY_MARGIN_COLUMN,
) -> dict[str, object]:
    return {
        "stage_order": int(order),
        "stage": stage,
        "stage_label": label,
        "source_table": source_table,
        "windows": int(windows),
        "fraction_of_screened_off_swr_windows": _safe_fraction(int(windows), screened),
        "fraction_of_previous_stage": np.nan if previous is None else _safe_fraction(int(windows), int(previous)),
        "candidate_sessions": int(frame["session"].nunique()) if not frame.empty and "session" in frame else 0,
        "candidate_rats": int(frame["rat"].nunique()) if not frame.empty and "rat" in frame else 0,
        "immobile_windows": _count_state(frame, "immobile"),
        "running_windows": _count_state(frame, "run"),
        "unknown_speed_windows": _unknown_speed_count(frame),
        "median_discovery_margin": _safe_median(frame, median_margin_column) if median_margin_column else np.nan,
        "min_discovery_margin": _safe_min(frame, min_margin_column) if min_margin_column else np.nan,
        "median_exact_margin": _safe_median(frame, EXACT_MARGIN_COLUMN),
        "min_exact_margin": _safe_min(frame, EXACT_MARGIN_COLUMN),
        "best_trajectory_model_distribution": _model_distribution(frame),
    }


def promotion_ready_mask(high_specificity: pd.DataFrame) -> pd.Series:
    if high_specificity.empty:
        return pd.Series(False, index=high_specificity.index, dtype=bool)
    if "passes_high_specificity_promotion_filter" in high_specificity:
        return high_specificity["passes_high_specificity_promotion_filter"].map(_as_bool)
    if "high_specificity_label" in high_specificity:
        return high_specificity["high_specificity_label"].astype(str).eq(PROMOTION_READY_LABEL)
    return pd.Series(False, index=high_specificity.index, dtype=bool)


def _complete_validation_rows(validation_decisions: pd.DataFrame) -> pd.DataFrame:
    if validation_decisions.empty:
        return validation_decisions.copy()
    if "required_models_complete" not in validation_decisions:
        return validation_decisions.copy()
    return validation_decisions[validation_decisions["required_models_complete"].map(_as_bool)].copy()


def _attach_reference_columns(frame: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    if "rat" in columns and "rat" not in output and "session" in output:
        output["rat"] = output["session"].astype(str).str.split("/", n=1).str[0]
    missing = [column for column in columns if column not in output.columns and column in reference.columns]
    if missing and not output.empty and not reference.empty and _has_key_columns(output) and _has_key_columns(reference):
        metadata = reference[list(KEY_COLUMNS) + missing].drop_duplicates(subset=list(KEY_COLUMNS))
        output = output.merge(metadata, on=list(KEY_COLUMNS), how="left", validate="many_to_one")
    return output


def _validation_key_match_observation(validation_decisions: pd.DataFrame, ready: pd.DataFrame) -> tuple[bool, str]:
    if _has_key_columns(validation_decisions) and _has_key_columns(ready):
        validation_keys = _key_tuples(validation_decisions)
        ready_keys = _key_tuples(ready)
        validation_key_set = set(validation_keys)
        ready_key_set = set(ready_keys)
        duplicate_keys = (len(validation_keys) - len(validation_key_set)) + (len(ready_keys) - len(ready_key_set))
        matched_keys = len(validation_key_set & ready_key_set)
        return (
            validation_key_set == ready_key_set and duplicate_keys == 0,
            (
                f"matched_keys={matched_keys}/{len(ready_key_set)}; "
                f"validation_rows={len(validation_keys)}; "
                f"promotion_ready_rows={len(ready_keys)}; "
                f"duplicate_keys={duplicate_keys}"
            ),
        )
    return len(validation_decisions) == len(ready), f"{len(validation_decisions)}/{len(ready)}"


def build_funnel_summary(
    *,
    candidate_table: pd.DataFrame,
    tier_summary: pd.DataFrame,
    run_state_summary: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
) -> pd.DataFrame:
    screened = _off_swr_window_count(tier_summary, run_state_summary, candidate_table)
    rows: list[dict[str, object]] = []
    rows.append(
        _stage_row(
            order=1,
            stage="screened_off_swr_windows",
            label="All scored off-SWR windows",
            source_table="off_swr_candidate_tier_threshold_summary.csv",
            frame=pd.DataFrame(),
            windows=screened,
            screened=screened,
            previous=None,
            median_margin_column="",
            min_margin_column="",
        )
        | {
            "immobile_windows": _run_state_denominator(tier_summary, "immobile_windows"),
            "running_windows": _run_state_denominator(tier_summary, "running_windows"),
            "unknown_speed_windows": _run_state_denominator(tier_summary, "unknown_speed_windows"),
        }
    )
    previous = screened
    for order, (tier, threshold) in enumerate(TIER_THRESHOLDS, start=2):
        subset = _tier_subset(candidate_table, threshold)
        rows.append(
            _stage_row(
                order=order,
                stage=f"{tier}_trajectory_candidates",
                label=f"{tier.title()} trajectory-family candidates (margin >= {threshold:g})",
                source_table="off_swr_candidate_table.csv",
                frame=subset,
                windows=int(len(subset)),
                screened=screened,
                previous=previous,
            )
        )
        previous = int(len(subset))

    strong_after_1s_count = _tier_summary_value(tier_summary, "strong", "candidate_windows_after_1s_swr_exclusion")
    strong_after_1s_keys = _key_set(high_specificity)
    strong_after_1s = candidate_table[_mask_keys(candidate_table, strong_after_1s_keys)].copy()
    if strong_after_1s.empty and np.isfinite(strong_after_1s_count):
        strong_after_1s = _tier_subset(candidate_table, 50.0)
    rows.append(
        _stage_row(
            order=6,
            stage="strong_candidates_after_1s_swr_exclusion",
            label="Strong candidates after 1 s nearest-SWR exclusion",
            source_table="off_swr_high_specificity_candidate_table.csv",
            frame=strong_after_1s,
            windows=int(strong_after_1s_count) if np.isfinite(strong_after_1s_count) else int(len(strong_after_1s)),
            screened=screened,
            previous=int(len(_tier_subset(candidate_table, 50.0))),
        )
    )

    ready = high_specificity[promotion_ready_mask(high_specificity)].copy()
    rows.append(
        _stage_row(
            order=7,
            stage="promotion_ready_candidates",
            label="Promotion-ready high-specificity candidates",
            source_table="off_swr_high_specificity_candidate_table.csv",
            frame=ready,
            windows=int(len(ready)),
            screened=screened,
            previous=int(len(strong_after_1s)),
        )
    )

    validated = _complete_validation_rows(validation_decisions)
    rows.append(
        _stage_row(
            order=8,
            stage="exact_core_validated_candidates",
            label="Promotion-ready candidates with complete exact-core validation",
            source_table="promoted_off_swr_candidate_exact_core_decisions.csv",
            frame=validated,
            windows=int(len(validated)),
            screened=screened,
            previous=int(len(ready)),
            median_margin_column=EXACT_MARGIN_COLUMN,
            min_margin_column=EXACT_MARGIN_COLUMN,
        )
    )
    trajectory = (
        validated[validated["trajectory_confident_claim"].map(_as_bool)].copy()
        if not validated.empty and "trajectory_confident_claim" in validated
        else pd.DataFrame()
    )
    rows.append(
        _stage_row(
            order=9,
            stage="exact_core_trajectory_confident_candidates",
            label="Exact-core trajectory-confident promoted candidates",
            source_table="promoted_off_swr_candidate_exact_core_decisions.csv",
            frame=trajectory,
            windows=int(len(trajectory)),
            screened=screened,
            previous=int(len(validated)),
            median_margin_column=EXACT_MARGIN_COLUMN,
            min_margin_column=EXACT_MARGIN_COLUMN,
        )
    )
    return pd.DataFrame(rows, columns=list(FUNNEL_SUMMARY_COLUMNS))


def build_group_summary(
    *,
    tier_session_summary: pd.DataFrame,
    tier_rat_summary: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        _group_rows(
            tier_frame=tier_rat_summary,
            high_specificity=high_specificity,
            validation_decisions=validation_decisions,
            group_type="rat",
            group_cols=("rat",),
        )
    )
    rows.extend(
        _group_rows(
            tier_frame=tier_session_summary,
            high_specificity=high_specificity,
            validation_decisions=validation_decisions,
            group_type="session",
            group_cols=("rat", "session"),
        )
    )
    return pd.DataFrame(rows, columns=list(GROUP_SUMMARY_COLUMNS))


def _group_rows(
    *,
    tier_frame: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
    group_type: str,
    group_cols: tuple[str, ...],
) -> Iterable[dict[str, object]]:
    if tier_frame.empty:
        return []
    rows = []
    ready = _attach_reference_columns(high_specificity[promotion_ready_mask(high_specificity)].copy(), high_specificity, group_cols)
    validation_decisions = _attach_reference_columns(
        _complete_validation_rows(validation_decisions),
        ready if not ready.empty else high_specificity,
        group_cols,
    )
    tier_groups = tier_frame.groupby(list(group_cols), sort=True)
    for key, group in tier_groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {column: str(value) for column, value in zip(group_cols, key_tuple, strict=True)}
        rat = key_map.get("rat", "")
        session = key_map.get("session", "all")

        def tier_count(tier: str, column: str = "candidate_windows") -> int:
            subset = group[group["candidate_tier"].astype(str).eq(tier)]
            if subset.empty or column not in subset:
                return 0
            value = pd.to_numeric(subset.iloc[0][column], errors="coerce")
            return int(value) if pd.notna(value) else 0

        off_swr = tier_count("weak", "off_swr_windows")
        ready_group = _filter_group(ready, key_map)
        validated_group = _filter_group(validation_decisions, key_map)
        exact_margin = _numeric(validated_group, EXACT_MARGIN_COLUMN)
        rows.append(
            {
                "group_type": group_type,
                "rat": rat,
                "session": session,
                "off_swr_windows": off_swr,
                "weak_candidates": tier_count("weak"),
                "moderate_candidates": tier_count("moderate"),
                "strong_candidates": tier_count("strong"),
                "extreme_candidates": tier_count("extreme"),
                "promotion_ready_candidates": int(len(ready_group)),
                "exact_validated_candidates": int(len(validated_group)),
                "exact_trajectory_confident_candidates": int(validated_group["trajectory_confident_claim"].map(_as_bool).sum())
                if not validated_group.empty and "trajectory_confident_claim" in validated_group
                else 0,
                "exact_nontrajectory_confident_candidates": int(validated_group["nontrajectory_confident_claim"].map(_as_bool).sum())
                if not validated_group.empty and "nontrajectory_confident_claim" in validated_group
                else 0,
                "median_exact_margin": float(exact_margin.median()) if exact_margin.notna().any() else np.nan,
                "min_exact_margin": float(exact_margin.min()) if exact_margin.notna().any() else np.nan,
                "max_exact_margin": float(exact_margin.max()) if exact_margin.notna().any() else np.nan,
            }
        )
    return rows


def _filter_group(frame: pd.DataFrame, key_map: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    for column, value in key_map.items():
        if column not in frame:
            return frame.iloc[0:0].copy()
        mask &= frame[column].astype(str).eq(str(value))
    return frame[mask].copy()


def build_rejection_summary(
    *,
    candidate_table: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_table.empty:
        return pd.DataFrame(columns=list(REJECTION_SUMMARY_COLUMNS))
    frame = candidate_table.copy()
    high_keys = _key_set(high_specificity)
    ready_keys = _key_set(high_specificity[promotion_ready_mask(high_specificity)].copy())
    validated_keys = _key_set(_complete_validation_rows(validation_decisions))
    high_mask = _mask_keys(frame, high_keys)
    ready_mask = _mask_keys(frame, ready_keys)
    validated_mask = _mask_keys(frame, validated_keys)
    strong_mask = _numeric(frame, DISCOVERY_MARGIN_COLUMN) >= 50.0
    labels = np.full(len(frame), "below_strong_threshold", dtype=object)
    labels[strong_mask.to_numpy() & ~high_mask.to_numpy()] = "strong_not_high_specificity_distance"
    labels[high_mask.to_numpy() & ~ready_mask.to_numpy()] = "high_specificity_filter_rejected"
    labels[ready_mask.to_numpy() & ~validated_mask.to_numpy()] = "promotion_ready_not_exact_validated"
    labels[validated_mask.to_numpy()] = "exact_validated_promotion_ready"
    frame["funnel_status"] = labels
    rows = []
    for status, group in frame.groupby("funnel_status", sort=True):
        label = group["candidate_specificity_label"].astype(str) if "candidate_specificity_label" in group else pd.Series("", index=group.index)
        rows.append(
            {
                "funnel_status": str(status),
                "candidate_windows": int(len(group)),
                "fraction_of_candidate_windows": _safe_fraction(int(len(group)), len(frame)),
                "candidate_sessions": int(group["session"].nunique()) if "session" in group else 0,
                "candidate_rats": int(group["rat"].nunique()) if "rat" in group else 0,
                "immobile_windows": _count_state(group, "immobile"),
                "running_windows": _count_state(group, "run"),
                "unknown_speed_windows": _unknown_speed_count(group),
                "movement_or_low_information_windows": int(label.ne("interesting_off_swr_trajectory_candidate").sum()),
                "interesting_candidate_windows": int(label.eq("interesting_off_swr_trajectory_candidate").sum()),
                "median_discovery_margin": _safe_median(group, DISCOVERY_MARGIN_COLUMN),
                "median_animal_speed_mean": _safe_median(group, "animal_speed_mean"),
                "median_n_spikes": _safe_median(group, "n_spikes"),
                "median_active_cell_count": _safe_median(group, "active_cell_count"),
            }
        )
    return pd.DataFrame(rows, columns=list(REJECTION_SUMMARY_COLUMNS))


def build_gate_summary(
    *,
    candidate_table: pd.DataFrame,
    tier_summary: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
    funnel: pd.DataFrame,
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

    screened = int(funnel[funnel["stage"].eq("screened_off_swr_windows")]["windows"].iloc[0]) if not funnel.empty else 0
    ready = high_specificity[promotion_ready_mask(high_specificity)].copy()
    validation_matches_ready, validation_match_observed = _validation_key_match_observation(validation_decisions, ready)
    complete = validation_decisions["required_models_complete"].map(_as_bool) if "required_models_complete" in validation_decisions else pd.Series(dtype=bool)
    trajectory = validation_decisions["trajectory_confident_claim"].map(_as_bool) if "trajectory_confident_claim" in validation_decisions else pd.Series(dtype=bool)
    nontrajectory = (
        validation_decisions["nontrajectory_confident_claim"].map(_as_bool)
        if "nontrajectory_confident_claim" in validation_decisions
        else pd.Series(dtype=bool)
    )
    add("screened_denominator_present", screened > 0, screened, "total screened off-SWR windows is available")
    add("candidate_table_present", not candidate_table.empty, int(len(candidate_table)), "ranked off-SWR candidate table is available")
    add(
        "tier_threshold_summary_present",
        not tier_summary.empty,
        int(len(tier_summary)),
        "weak/moderate/strong/extreme tier threshold summary is available",
    )
    add("promotion_ready_candidates_present", not ready.empty, int(len(ready)), "promotion-ready high-specificity candidates are present")
    add(
        "exact_validation_matches_promotion_ready",
        validation_matches_ready,
        validation_match_observed,
        "exact validation decision rows match promotion-ready candidates",
    )
    add(
        "exact_required_core_complete",
        not validation_decisions.empty and int(complete.sum()) == len(validation_decisions),
        f"{int(complete.sum())}/{len(validation_decisions)}",
        "all exact validation rows have complete required model evidence",
    )
    add(
        "exact_trajectory_supports_all_validated",
        not validation_decisions.empty and int(trajectory.sum()) == len(validation_decisions) and int(nontrajectory.sum()) == 0,
        f"trajectory={int(trajectory.sum())}/{len(validation_decisions)}; nontrajectory={int(nontrajectory.sum())}",
        "all exact-validated promotion-ready candidates are trajectory-confident and none are nontrajectory-confident",
    )
    add(
        "movement_skew_reported",
        "candidate_specificity_label" in candidate_table and "run_or_immobility_state" in candidate_table,
        "candidate_specificity_label/run_or_immobility_state",
        "rejected-candidate movement/specificity fields are available",
        required=False,
    )
    required_rows = [row for row in rows if row["required_for_overall"]]
    add(
        "overall",
        all(row["passed"] for row in required_rows),
        f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed",
        "all required promotion-funnel gates pass",
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_off_swr_promotion_funnel_outputs(
    *,
    discovery_dir: Path,
    validation_dir: Path,
    output: Path,
) -> dict[str, pd.DataFrame]:
    candidate_table = _read_required_csv(discovery_dir / "off_swr_candidate_table.csv")
    tier_summary = _read_required_csv(discovery_dir / "off_swr_candidate_tier_threshold_summary.csv")
    tier_session = _read_required_csv(discovery_dir / "off_swr_candidate_tier_session_summary.csv")
    tier_rat = _read_required_csv(discovery_dir / "off_swr_candidate_tier_rat_summary.csv")
    high_specificity = _read_required_csv(discovery_dir / "off_swr_high_specificity_candidate_table.csv")
    run_state = _read_optional_csv(discovery_dir / "off_swr_run_state_stratified_summary.csv")
    validation_decisions = _read_required_csv(validation_dir / "promoted_off_swr_candidate_exact_core_decisions.csv")

    output.mkdir(parents=True, exist_ok=True)
    funnel = build_funnel_summary(
        candidate_table=candidate_table,
        tier_summary=tier_summary,
        run_state_summary=run_state,
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    )
    group = build_group_summary(
        tier_session_summary=tier_session,
        tier_rat_summary=tier_rat,
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    )
    rejection = build_rejection_summary(
        candidate_table=candidate_table,
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    )
    gates = build_gate_summary(
        candidate_table=candidate_table,
        tier_summary=tier_summary,
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
        funnel=funnel,
    )
    outputs = {
        "off_swr_promotion_funnel_summary.csv": funnel,
        "off_swr_promotion_funnel_group_summary.csv": group,
        "off_swr_promotion_funnel_rejection_summary.csv": rejection,
        "off_swr_promotion_funnel_gate_summary.csv": gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-dir", required=True, help="Directory containing off-SWR discovery CSV outputs.")
    parser.add_argument("--validation-dir", required=True, help="Directory containing promoted candidate exact-core validation CSV outputs.")
    parser.add_argument("--output", default="results/off-swr-promotion-funnel")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = write_off_swr_promotion_funnel_outputs(
        discovery_dir=Path(args.discovery_dir),
        validation_dir=Path(args.validation_dir),
        output=Path(args.output),
    )
    print("Off-SWR promotion funnel:")
    print(outputs["off_swr_promotion_funnel_summary.csv"].to_string(index=False))
    print("\nOff-SWR promotion funnel gates:")
    print(outputs["off_swr_promotion_funnel_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
