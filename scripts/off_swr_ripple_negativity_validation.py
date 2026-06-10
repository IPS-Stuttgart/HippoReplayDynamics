#!/usr/bin/env python3
"""Validate whether promoted off-SWR candidates are ripple-band negative."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ("session", "event_index", "null_index")
DEFAULT_RIPPLE_Z_THRESHOLD = 3.0

PEAK_RIPPLE_POWER_ALIASES = (
    "peak_ripple_band_power_z",
    "max_ripple_band_power_z",
    "ripple_band_peak_z",
    "peak_ripple_power_z",
    "ripple_power_z",
    "ripple_power",
)
MEAN_RIPPLE_POWER_ALIASES = (
    "mean_ripple_band_power_z",
    "ripple_band_mean_z",
    "mean_ripple_power_z",
    "ripple_band_power",
    "lfp_power",
)
SHARP_WAVE_POWER_ALIASES = (
    "peak_sharp_wave_power_z",
    "sharp_wave_power_z",
    "sharp_wave_band_power",
)
WINDOW_CROSSING_ALIASES = (
    "ripple_threshold_crossing",
    "ripple_threshold_crossing_window",
    "has_ripple_threshold_crossing",
    "ripple_band_threshold_crossing",
    "ripple_crossing",
)
BUFFER_CROSSING_ALIASES = {
    "50ms": (
        "ripple_threshold_crossing_pm50ms",
        "ripple_threshold_crossing_50ms_buffer",
        "ripple_threshold_crossing_plusminus_50ms",
        "ripple_threshold_crossing_buffer_0p05s",
    ),
    "100ms": (
        "ripple_threshold_crossing_pm100ms",
        "ripple_threshold_crossing_100ms_buffer",
        "ripple_threshold_crossing_plusminus_100ms",
        "ripple_threshold_crossing_buffer_0p1s",
    ),
    "250ms": (
        "ripple_threshold_crossing_pm250ms",
        "ripple_threshold_crossing_250ms_buffer",
        "ripple_threshold_crossing_plusminus_250ms",
        "ripple_threshold_crossing_buffer_0p25s",
    ),
}

PROMOTED_LFP_COLUMNS = (
    "session",
    "rat",
    "event_index",
    "null_index",
    "candidate_id",
    "window_start_s",
    "window_end_s",
    "duration_s",
    "run_or_immobility_state",
    "animal_speed_mean",
    "nearest_known_swr_distance_s",
    "trajectory_family_margin",
    "exact_family_margin",
    "trajectory_confident_claim",
    "nontrajectory_confident_claim",
    "peak_ripple_band_power_z",
    "mean_ripple_band_power_z",
    "sharp_wave_power_z",
    "ripple_threshold_crossing_window",
    "ripple_threshold_crossing_window_inferred_from_peak_z",
    "ripple_threshold_crossing_pm50ms",
    "ripple_threshold_crossing_pm100ms",
    "ripple_threshold_crossing_pm250ms",
    "lfp_power_available",
    "buffer_crossing_fields_available",
    "ripple_negative_window",
    "ripple_negative_with_buffers",
    "lfp_validation_status",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")

RIPPLE_POWER_SUMMARY_COLUMNS = (
    "comparison_group",
    "windows",
    "lfp_power_available_windows",
    "window_ripple_negative_windows",
    "buffer_ripple_negative_windows",
    "median_peak_ripple_band_power_z",
    "max_peak_ripple_band_power_z",
    "median_mean_ripple_band_power_z",
    "window_threshold_crossing_windows",
    "pm50ms_threshold_crossing_windows",
    "pm100ms_threshold_crossing_windows",
    "pm250ms_threshold_crossing_windows",
)

JOINT_COMPARISON_COLUMNS = (
    "event_class",
    "events",
    "trajectory_confident_claims",
    "first_order_imm_best_events",
    "median_trajectory_family_margin",
    "lfp_power_available_events",
    "median_peak_ripple_band_power_z",
    "max_peak_ripple_band_power_z",
    "window_threshold_crossing_events",
    "buffer_threshold_crossing_events",
    "ripple_negative_events",
)


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


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    values = frame[column]
    if values.dtype == bool:
        return values.astype("boolean")
    normalized = values.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "", "nan", "none"}
    return normalized.map(
        lambda value: True if value in true_values else False if value in false_values else pd.NA,
    ).astype("boolean")


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


def _first_existing_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for column in aliases:
        if column in frame.columns:
            return column
    return None


def _copy_first_numeric(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    column = _first_existing_column(frame, aliases)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return _numeric(frame, column)


def _copy_first_bool(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    column = _first_existing_column(frame, aliases)
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    return _bool_series(frame, column)


def _normalize_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "session" in out:
        out["session"] = out["session"].astype(str)
    if "event_index" in out:
        out["event_index"] = pd.to_numeric(out["event_index"], errors="coerce").astype("Int64")
    if "null_index" in out:
        out["null_index"] = pd.to_numeric(out["null_index"], errors="coerce").astype("Int64")
    return out


def _candidate_id(row: pd.Series) -> str:
    return f"{row.get('session', '')}|event={row.get('event_index', '')}|null={row.get('null_index', '')}"


def _rat_from_session(value: object) -> str:
    return str(value).split("/", 1)[0]


def _coalesce_columns(frame: pd.DataFrame, column: str, fallback: str) -> pd.DataFrame:
    if fallback not in frame:
        return frame
    if column in frame:
        frame[column] = frame[column].combine_first(frame[fallback])
    else:
        frame[column] = frame[fallback]
    return frame.drop(columns=[fallback])


def _merge_covariates(promoted: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    promoted = _normalize_key_columns(promoted)
    if covariates.empty:
        return promoted
    covariates = _normalize_key_columns(covariates)
    if not set(KEY_COLUMNS).issubset(covariates.columns):
        return promoted
    covariates = covariates.drop_duplicates(list(KEY_COLUMNS), keep="first")
    merged = promoted.merge(covariates, on=list(KEY_COLUMNS), how="left", suffixes=("", "_covariate"))
    for column in covariates.columns:
        fallback = f"{column}_covariate"
        if fallback in merged:
            merged = _coalesce_columns(merged, column, fallback)
    return merged


def promoted_candidate_lfp_table(
    promoted_decisions: pd.DataFrame,
    *,
    covariate_table: pd.DataFrame | None = None,
    ripple_z_threshold: float = DEFAULT_RIPPLE_Z_THRESHOLD,
) -> pd.DataFrame:
    """Return one row per promoted candidate with ripple-power availability."""

    if promoted_decisions.empty:
        return pd.DataFrame(columns=list(PROMOTED_LFP_COLUMNS))
    frame = _merge_covariates(
        promoted_decisions,
        covariate_table if covariate_table is not None else pd.DataFrame(),
    )
    frame = _normalize_key_columns(frame)
    out = pd.DataFrame(index=frame.index)
    out["session"] = frame["session"].astype(str)
    out["rat"] = frame["rat"].astype(str) if "rat" in frame else out["session"].map(_rat_from_session)
    out["event_index"] = pd.to_numeric(frame["event_index"], errors="coerce").astype("Int64")
    out["null_index"] = pd.to_numeric(frame["null_index"], errors="coerce").astype("Int64")
    out["candidate_id"] = out.apply(_candidate_id, axis=1)
    out["window_start_s"] = _copy_first_numeric(frame, ("window_start_s",))
    out["window_end_s"] = _copy_first_numeric(frame, ("window_end_s",))
    out["duration_s"] = _copy_first_numeric(frame, ("duration_s", "window_duration_s"))
    out["run_or_immobility_state"] = (
        frame["run_or_immobility_state"].astype(str) if "run_or_immobility_state" in frame else ""
    )
    out["animal_speed_mean"] = _copy_first_numeric(frame, ("animal_speed_mean", "mean_speed_cm_s"))
    out["nearest_known_swr_distance_s"] = _copy_first_numeric(
        frame,
        ("nearest_known_swr_distance_s", "distance_to_nearest_swr_s"),
    )
    out["trajectory_family_margin"] = _copy_first_numeric(frame, ("trajectory_family_margin",))
    out["exact_family_margin"] = _copy_first_numeric(
        frame,
        ("trajectory_minus_nontrajectory_log_evidence", "trajectory_minus_nontrajectory_margin"),
    )
    out["trajectory_confident_claim"] = _copy_first_bool(frame, ("trajectory_confident_claim",))
    out["nontrajectory_confident_claim"] = _copy_first_bool(frame, ("nontrajectory_confident_claim",))
    out["peak_ripple_band_power_z"] = _copy_first_numeric(frame, PEAK_RIPPLE_POWER_ALIASES)
    out["mean_ripple_band_power_z"] = _copy_first_numeric(frame, MEAN_RIPPLE_POWER_ALIASES)
    out["sharp_wave_power_z"] = _copy_first_numeric(frame, SHARP_WAVE_POWER_ALIASES)
    explicit_crossing = _copy_first_bool(frame, WINDOW_CROSSING_ALIASES)
    inferred_crossing = out["peak_ripple_band_power_z"].ge(float(ripple_z_threshold))
    inferred_crossing = inferred_crossing.where(out["peak_ripple_band_power_z"].notna(), pd.NA).astype("boolean")
    out["ripple_threshold_crossing_window"] = explicit_crossing.combine_first(inferred_crossing)
    out["ripple_threshold_crossing_window_inferred_from_peak_z"] = explicit_crossing.isna() & inferred_crossing.notna()
    for label, aliases in BUFFER_CROSSING_ALIASES.items():
        out[f"ripple_threshold_crossing_pm{label}"] = _copy_first_bool(frame, aliases)
    lfp_columns = [
        "peak_ripple_band_power_z",
        "mean_ripple_band_power_z",
        "sharp_wave_power_z",
        "ripple_threshold_crossing_window",
        "ripple_threshold_crossing_pm50ms",
        "ripple_threshold_crossing_pm100ms",
        "ripple_threshold_crossing_pm250ms",
    ]
    out["lfp_power_available"] = out[lfp_columns].notna().any(axis=1)
    buffer_columns = [
        "ripple_threshold_crossing_pm50ms",
        "ripple_threshold_crossing_pm100ms",
        "ripple_threshold_crossing_pm250ms",
    ]
    out["buffer_crossing_fields_available"] = out[buffer_columns].notna().all(axis=1)
    no_window_crossing = out["ripple_threshold_crossing_window"].eq(False)
    peak_ok = out["peak_ripple_band_power_z"].lt(float(ripple_z_threshold)) | out["peak_ripple_band_power_z"].isna()
    out["ripple_negative_window"] = out["lfp_power_available"] & no_window_crossing.fillna(False) & peak_ok
    no_buffer_crossing = out[buffer_columns].eq(False).all(axis=1)
    out["ripple_negative_with_buffers"] = (
        out["ripple_negative_window"] & out["buffer_crossing_fields_available"] & no_buffer_crossing
    )
    status = np.full(len(out), "ripple_negative_with_buffers", dtype=object)
    status[~out["lfp_power_available"].to_numpy(dtype=bool)] = "unsupported_no_lfp_power_columns"
    status[out["lfp_power_available"].to_numpy(dtype=bool) & ~out["buffer_crossing_fields_available"].to_numpy(dtype=bool)] = (
        "incomplete_buffer_crossing_fields"
    )
    status[out["lfp_power_available"].to_numpy(dtype=bool) & ~out["ripple_negative_window"].to_numpy(dtype=bool)] = (
        "window_ripple_positive_or_indeterminate"
    )
    status[
        out["ripple_negative_window"].to_numpy(dtype=bool)
        & out["buffer_crossing_fields_available"].to_numpy(dtype=bool)
        & ~out["ripple_negative_with_buffers"].to_numpy(dtype=bool)
    ] = "buffer_ripple_positive_or_indeterminate"
    out["lfp_validation_status"] = status
    return out[list(PROMOTED_LFP_COLUMNS)]


def _summary_row(group_name: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "comparison_group": group_name,
            "windows": 0,
            "lfp_power_available_windows": 0,
            "window_ripple_negative_windows": 0,
            "buffer_ripple_negative_windows": 0,
            "median_peak_ripple_band_power_z": np.nan,
            "max_peak_ripple_band_power_z": np.nan,
            "median_mean_ripple_band_power_z": np.nan,
            "window_threshold_crossing_windows": 0,
            "pm50ms_threshold_crossing_windows": 0,
            "pm100ms_threshold_crossing_windows": 0,
            "pm250ms_threshold_crossing_windows": 0,
        }
    peak = _numeric(frame, "peak_ripple_band_power_z")
    mean = _numeric(frame, "mean_ripple_band_power_z")
    return {
        "comparison_group": group_name,
        "windows": int(len(frame)),
        "lfp_power_available_windows": int(_as_bool_series(frame["lfp_power_available"]).sum())
        if "lfp_power_available" in frame
        else 0,
        "window_ripple_negative_windows": int(_as_bool_series(frame["ripple_negative_window"]).sum())
        if "ripple_negative_window" in frame
        else 0,
        "buffer_ripple_negative_windows": int(_as_bool_series(frame["ripple_negative_with_buffers"]).sum())
        if "ripple_negative_with_buffers" in frame
        else 0,
        "median_peak_ripple_band_power_z": float(peak.median()) if peak.notna().any() else np.nan,
        "max_peak_ripple_band_power_z": float(peak.max()) if peak.notna().any() else np.nan,
        "median_mean_ripple_band_power_z": float(mean.median()) if mean.notna().any() else np.nan,
        "window_threshold_crossing_windows": int(_as_bool_series(frame["ripple_threshold_crossing_window"]).sum())
        if "ripple_threshold_crossing_window" in frame
        else 0,
        "pm50ms_threshold_crossing_windows": int(_as_bool_series(frame["ripple_threshold_crossing_pm50ms"]).sum())
        if "ripple_threshold_crossing_pm50ms" in frame
        else 0,
        "pm100ms_threshold_crossing_windows": int(_as_bool_series(frame["ripple_threshold_crossing_pm100ms"]).sum())
        if "ripple_threshold_crossing_pm100ms" in frame
        else 0,
        "pm250ms_threshold_crossing_windows": int(_as_bool_series(frame["ripple_threshold_crossing_pm250ms"]).sum())
        if "ripple_threshold_crossing_pm250ms" in frame
        else 0,
    }


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.map(_as_bool).fillna(False).astype(bool)


def ripple_power_matched_null_summary(
    promoted_lfp: pd.DataFrame,
    *,
    off_swr_window_table: pd.DataFrame | None = None,
    ripple_z_threshold: float = DEFAULT_RIPPLE_Z_THRESHOLD,
) -> pd.DataFrame:
    rows = [_summary_row("promoted_off_swr_candidates", promoted_lfp)]
    if off_swr_window_table is not None and not off_swr_window_table.empty:
        all_windows = promoted_candidate_lfp_table(
            off_swr_window_table,
            covariate_table=None,
            ripple_z_threshold=ripple_z_threshold,
        )
        promoted_keys = set(map(tuple, promoted_lfp[list(KEY_COLUMNS)].astype(object).to_numpy()))
        window_keys = list(map(tuple, all_windows[list(KEY_COLUMNS)].astype(object).to_numpy()))
        is_promoted = pd.Series([key in promoted_keys for key in window_keys], index=all_windows.index)
        rows.append(_summary_row("all_available_off_swr_windows", all_windows))
        rows.append(_summary_row("non_promoted_off_swr_windows", all_windows[~is_promoted].copy()))
    return pd.DataFrame(rows, columns=list(RIPPLE_POWER_SUMMARY_COLUMNS))


def _joint_lfp_frame(dynamics: pd.DataFrame, promoted_lfp: pd.DataFrame) -> pd.DataFrame:
    if dynamics.empty:
        return pd.DataFrame()
    out = _normalize_key_columns(dynamics)
    promoted = promoted_lfp[list(KEY_COLUMNS) + [
        "peak_ripple_band_power_z",
        "mean_ripple_band_power_z",
        "ripple_threshold_crossing_window",
        "ripple_threshold_crossing_pm50ms",
        "ripple_threshold_crossing_pm100ms",
        "ripple_threshold_crossing_pm250ms",
        "lfp_power_available",
        "ripple_negative_with_buffers",
    ]].copy()
    out = out.merge(promoted, on=list(KEY_COLUMNS), how="left", suffixes=("", "_promoted_lfp"))
    for column in (
        "peak_ripple_band_power_z",
        "mean_ripple_band_power_z",
        "ripple_threshold_crossing_window",
        "ripple_threshold_crossing_pm50ms",
        "ripple_threshold_crossing_pm100ms",
        "ripple_threshold_crossing_pm250ms",
        "lfp_power_available",
        "ripple_negative_with_buffers",
    ):
        fallback = f"{column}_promoted_lfp"
        if fallback in out:
            out = _coalesce_columns(out, column, fallback)
    return out


def swr_off_swr_lfp_dynamics_comparison(
    dynamics_comparison: pd.DataFrame,
    promoted_lfp: pd.DataFrame,
) -> pd.DataFrame:
    frame = _joint_lfp_frame(dynamics_comparison, promoted_lfp)
    if frame.empty or "event_class" not in frame:
        return pd.DataFrame(columns=list(JOINT_COMPARISON_COLUMNS))
    rows: list[dict[str, object]] = []
    for event_class, group in frame.groupby("event_class", sort=True):
        peak = _numeric(group, "peak_ripple_band_power_z")
        trajectory_claim = _as_bool_series(group.get("trajectory_confident_claim", pd.Series(False, index=group.index)))
        first_order = group.get("best_exact_trajectory_model", pd.Series("", index=group.index)).astype(str).eq(
            "sorted-spike-state-space-first-order-imm",
        )
        buffer_crossing = (
            _as_bool_series(group.get("ripple_threshold_crossing_pm50ms", pd.Series(False, index=group.index)))
            | _as_bool_series(group.get("ripple_threshold_crossing_pm100ms", pd.Series(False, index=group.index)))
            | _as_bool_series(group.get("ripple_threshold_crossing_pm250ms", pd.Series(False, index=group.index)))
        )
        rows.append(
            {
                "event_class": str(event_class),
                "events": int(len(group)),
                "trajectory_confident_claims": int(trajectory_claim.sum()),
                "first_order_imm_best_events": int(first_order.sum()),
                "median_trajectory_family_margin": float(
                    pd.to_numeric(group.get("trajectory_minus_nontrajectory_margin", pd.Series(dtype=float)), errors="coerce").median(),
                ),
                "lfp_power_available_events": int(_as_bool_series(group.get("lfp_power_available", pd.Series(False, index=group.index))).sum()),
                "median_peak_ripple_band_power_z": float(peak.median()) if peak.notna().any() else np.nan,
                "max_peak_ripple_band_power_z": float(peak.max()) if peak.notna().any() else np.nan,
                "window_threshold_crossing_events": int(
                    _as_bool_series(group.get("ripple_threshold_crossing_window", pd.Series(False, index=group.index))).sum(),
                ),
                "buffer_threshold_crossing_events": int(buffer_crossing.sum()),
                "ripple_negative_events": int(
                    _as_bool_series(group.get("ripple_negative_with_buffers", pd.Series(False, index=group.index))).sum(),
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(JOINT_COMPARISON_COLUMNS))


def lfp_gate_summary(
    promoted_lfp: pd.DataFrame,
    *,
    ripple_z_threshold: float = DEFAULT_RIPPLE_Z_THRESHOLD,
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

    total = int(len(promoted_lfp))
    lfp_available = int(_as_bool_series(promoted_lfp.get("lfp_power_available", pd.Series(dtype=bool))).sum())
    buffer_available = int(_as_bool_series(promoted_lfp.get("buffer_crossing_fields_available", pd.Series(dtype=bool))).sum())
    window_negative = int(_as_bool_series(promoted_lfp.get("ripple_negative_window", pd.Series(dtype=bool))).sum())
    buffer_negative = int(_as_bool_series(promoted_lfp.get("ripple_negative_with_buffers", pd.Series(dtype=bool))).sum())
    trajectory_claims = int(_as_bool_series(promoted_lfp.get("trajectory_confident_claim", pd.Series(dtype=bool))).sum())
    negative = promoted_lfp[_as_bool_series(promoted_lfp.get("ripple_negative_with_buffers", pd.Series(dtype=bool)))].copy()
    negative_trajectory = int(_as_bool_series(negative.get("trajectory_confident_claim", pd.Series(dtype=bool))).sum())
    peak = _numeric(promoted_lfp, "peak_ripple_band_power_z")

    add("promoted_candidates_present", total > 0, total, "promoted off-SWR candidates > 0")
    add(
        "trajectory_evidence_present_before_lfp_filter",
        total > 0 and trajectory_claims == total,
        f"{trajectory_claims}/{total}",
        "all promoted candidates are trajectory-confident before LFP filtering",
    )
    add(
        "lfp_power_or_crossing_fields_available",
        lfp_available == total and total > 0,
        f"{lfp_available}/{total}",
        "ripple/LFP power or crossing fields are present for all promoted candidates",
    )
    add(
        "buffer_crossing_fields_available",
        buffer_available == total and total > 0,
        f"{buffer_available}/{total}",
        "threshold-crossing fields are present for +/-50, +/-100, and +/-250 ms buffers",
    )
    add(
        "candidate_windows_ripple_negative",
        window_negative == total and total > 0,
        f"{window_negative}/{total}; threshold={ripple_z_threshold}",
        "all promoted candidates are below ripple threshold in the candidate window",
    )
    add(
        "candidate_buffers_ripple_negative",
        buffer_negative == total and total > 0,
        f"{buffer_negative}/{total}",
        "all promoted candidates have no ripple-threshold crossing in requested buffers",
    )
    add(
        "trajectory_evidence_preserved_after_lfp_filter",
        buffer_negative > 0 and negative_trajectory == buffer_negative,
        f"{negative_trajectory}/{buffer_negative}",
        "all ripple-negative promoted candidates remain trajectory-confident",
    )
    add(
        "peak_ripple_power_below_threshold",
        peak.notna().all() and bool(peak.lt(float(ripple_z_threshold)).all()) if total else False,
        f"max={float(peak.max()) if peak.notna().any() else np.nan}; threshold={ripple_z_threshold}",
        "all promoted candidates have peak ripple-band power below threshold",
    )
    required_rows = [row for row in rows if row["required_for_overall"]]
    add(
        "overall",
        all(row["passed"] for row in required_rows),
        f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed",
        "all required gates pass",
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_outputs(
    *,
    promoted_decisions: Path,
    output: Path,
    off_swr_window_table: Path | None = None,
    promoted_covariates: Path | None = None,
    swr_off_swr_dynamics: Path | None = None,
    ripple_z_threshold: float = DEFAULT_RIPPLE_Z_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    decisions = _read_required_csv(promoted_decisions)
    off_swr_windows = _read_optional_csv(off_swr_window_table)
    covariates = _read_optional_csv(promoted_covariates)
    if covariates.empty and not off_swr_windows.empty:
        covariates = off_swr_windows
    dynamics = _read_optional_csv(swr_off_swr_dynamics)

    output.mkdir(parents=True, exist_ok=True)
    promoted_lfp = promoted_candidate_lfp_table(
        decisions,
        covariate_table=covariates,
        ripple_z_threshold=ripple_z_threshold,
    )
    outputs = {
        "off_swr_candidate_lfp_ripple_power.csv": promoted_lfp,
        "off_swr_candidate_lfp_gate_summary.csv": lfp_gate_summary(
            promoted_lfp,
            ripple_z_threshold=ripple_z_threshold,
        ),
        "off_swr_candidate_ripple_power_matched_null.csv": ripple_power_matched_null_summary(
            promoted_lfp,
            off_swr_window_table=off_swr_windows,
            ripple_z_threshold=ripple_z_threshold,
        ),
        "swr_off_swr_lfp_dynamics_comparison.csv": swr_off_swr_lfp_dynamics_comparison(
            dynamics,
            promoted_lfp,
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-decisions", required=True, type=Path)
    parser.add_argument("--off-swr-window-table", type=Path)
    parser.add_argument("--promoted-covariates", type=Path)
    parser.add_argument("--swr-off-swr-dynamics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ripple-z-threshold", type=float, default=DEFAULT_RIPPLE_Z_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = write_outputs(
        promoted_decisions=args.promoted_decisions,
        off_swr_window_table=args.off_swr_window_table,
        promoted_covariates=args.promoted_covariates,
        swr_off_swr_dynamics=args.swr_off_swr_dynamics,
        output=args.output,
        ripple_z_threshold=args.ripple_z_threshold,
    )
    print(outputs["off_swr_candidate_lfp_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
