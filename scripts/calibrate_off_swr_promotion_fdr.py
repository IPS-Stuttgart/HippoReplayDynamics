#!/usr/bin/env python3
"""Calibrate empirical false-promotion controls for off-SWR candidates.

The promotion funnel establishes a strict denominator-backed subset. This script
adds a calibration layer: observed running/ordinary controls should not pass the
promotion rule, and shuffled label/immobility nulls should produce fewer
promotion-ready rows than the real alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ("session", "event_index", "null_index")
DISCOVERY_MARGIN_COLUMN = "trajectory_family_margin"
EXACT_MARGIN_COLUMN = "trajectory_minus_nontrajectory_log_evidence"
PROMOTION_READY_LABEL = "promotion_ready_high_specificity_candidate"
INTERESTING_LABEL = "interesting_off_swr_trajectory_candidate"

THRESHOLDS = (
    ("weak", 5.5),
    ("moderate", 20.0),
    ("strong", 50.0),
    ("extreme", 100.0),
)

CALIBRATION_COLUMNS = (
    "control_source",
    "null_type",
    "eligible_windows",
    "observed_promotion_ready_windows",
    "control_promotion_ready_windows",
    "control_promotion_ready_fraction",
    "mean_null_promotion_ready_windows",
    "median_null_promotion_ready_windows",
    "p95_null_promotion_ready_windows",
    "p99_null_promotion_ready_windows",
    "max_null_promotion_ready_windows",
    "observed_minus_p95_null_windows",
    "observed_exceeds_null_p95",
    "observed_exceeds_null_p99",
    "empirical_p_value_ge_observed",
    "permutations",
    "calibration_interpretation",
)

FDR_SUMMARY_COLUMNS = (
    "screened_off_swr_windows",
    "candidate_windows",
    "high_specificity_windows",
    "promotion_ready_windows",
    "exact_validated_windows",
    "exact_trajectory_confident_windows",
    "direct_control_windows",
    "direct_control_promotion_ready_windows",
    "direct_control_false_promotion_fraction",
    "max_mean_permutation_null_promotions",
    "max_p95_permutation_null_promotions",
    "max_p99_permutation_null_promotions",
    "min_permutation_empirical_p_value",
    "mean_permutation_fdr_estimate",
    "p95_permutation_fdr_bound",
    "fdr_calibration_status",
)

THRESHOLD_COLUMNS = (
    "candidate_tier",
    "margin_threshold",
    "screened_off_swr_windows",
    "candidate_windows",
    "candidate_fraction",
    "high_specificity_windows",
    "promotion_ready_windows",
    "promotion_ready_fraction_of_screened",
    "promotion_ready_fraction_of_candidates",
    "running_control_windows",
    "running_control_promotion_ready_windows",
    "ordinary_control_windows",
    "ordinary_control_promotion_ready_windows",
    "joint_shuffle_mean_promotions",
    "joint_shuffle_p95_promotions",
    "joint_shuffle_p99_promotions",
    "observed_exceeds_joint_shuffle_p95",
    "observed_exceeds_joint_shuffle_p99",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")


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
    if denominator is None or float(denominator) == 0.0:
        return np.nan
    return float(value) / float(denominator)


def _safe_int(value: object, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else default


def _screened_count(candidate_table: pd.DataFrame, tier_summary: pd.DataFrame) -> int:
    if not tier_summary.empty and "off_swr_windows" in tier_summary:
        values = _numeric(tier_summary, "off_swr_windows").dropna()
        if not values.empty:
            return int(values.max())
    return int(len(candidate_table))


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    if frame.empty or not set(KEY_COLUMNS).issubset(frame.columns):
        return set()
    return set(map(tuple, frame[list(KEY_COLUMNS)].astype(object).to_numpy()))


def _mask_keys(frame: pd.DataFrame, keys: set[tuple[object, ...]]) -> pd.Series:
    if frame.empty or not keys or not set(KEY_COLUMNS).issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    values = list(map(tuple, frame[list(KEY_COLUMNS)].astype(object).to_numpy()))
    return pd.Series([value in keys for value in values], index=frame.index)


def promotion_ready_mask(high_specificity: pd.DataFrame) -> pd.Series:
    if high_specificity.empty:
        return pd.Series(dtype=bool)
    if "passes_high_specificity_promotion_filter" in high_specificity:
        return high_specificity["passes_high_specificity_promotion_filter"].map(_as_bool)
    if "high_specificity_label" in high_specificity:
        return high_specificity["high_specificity_label"].astype(str).eq(PROMOTION_READY_LABEL)
    return pd.Series(False, index=high_specificity.index)


def _component_mask(high_specificity: pd.DataFrame, component: str) -> pd.Series:
    if high_specificity.empty:
        return pd.Series(dtype=bool)
    if component in high_specificity:
        return high_specificity[component].map(_as_bool)
    if component == "passes_strong_tier":
        return _numeric(high_specificity, DISCOVERY_MARGIN_COLUMN) >= 50.0
    if component == "passes_1s_swr_exclusion":
        return _numeric(high_specificity, "distance_to_nearest_swr_s") >= 1.0
    if component == "speed_available":
        return _numeric(high_specificity, "animal_speed_mean").notna()
    if component == "passes_immobility_filter":
        return high_specificity.get("run_or_immobility_state", pd.Series("", index=high_specificity.index)).astype(str).eq("immobile")
    if component == "passes_specificity_label_filter":
        return high_specificity.get("candidate_specificity_label", pd.Series("", index=high_specificity.index)).astype(str).eq(INTERESTING_LABEL)
    return pd.Series(False, index=high_specificity.index)


def _base_rule_mask(high_specificity: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=high_specificity.index)
    for component in ("passes_strong_tier", "passes_1s_swr_exclusion", "speed_available"):
        mask &= _component_mask(high_specificity, component)
    return mask


def _interesting_mask(high_specificity: pd.DataFrame) -> pd.Series:
    return _component_mask(high_specificity, "passes_specificity_label_filter")


def _immobile_mask(high_specificity: pd.DataFrame) -> pd.Series:
    return _component_mask(high_specificity, "passes_immobility_filter")


def _direct_control_row(
    *,
    control_source: str,
    frame: pd.DataFrame,
    ready: pd.Series,
    observed_ready: int,
    interpretation: str,
) -> dict[str, object]:
    control_ready = int(ready.loc[frame.index].sum()) if not frame.empty else 0
    return {
        "control_source": control_source,
        "null_type": "observed_control",
        "eligible_windows": int(len(frame)),
        "observed_promotion_ready_windows": int(observed_ready),
        "control_promotion_ready_windows": control_ready,
        "control_promotion_ready_fraction": _safe_fraction(control_ready, len(frame)),
        "mean_null_promotion_ready_windows": np.nan,
        "median_null_promotion_ready_windows": np.nan,
        "p95_null_promotion_ready_windows": np.nan,
        "p99_null_promotion_ready_windows": np.nan,
        "max_null_promotion_ready_windows": np.nan,
        "observed_minus_p95_null_windows": np.nan,
        "observed_exceeds_null_p95": np.nan,
        "observed_exceeds_null_p99": np.nan,
        "empirical_p_value_ge_observed": np.nan,
        "permutations": 0,
        "calibration_interpretation": interpretation,
    }


def _permutation_row(
    *,
    control_source: str,
    counts: np.ndarray,
    observed_ready: int,
    eligible_windows: int,
    permutations: int,
) -> dict[str, object]:
    p95 = float(np.quantile(counts, 0.95)) if len(counts) else np.nan
    p99 = float(np.quantile(counts, 0.99)) if len(counts) else np.nan
    empirical_p = float((np.count_nonzero(counts >= observed_ready) + 1) / (len(counts) + 1)) if len(counts) else np.nan
    interpretation = (
        "observed_promotions_exceed_shuffled_null"
        if np.isfinite(p95) and observed_ready > p95
        else "observed_promotions_do_not_exceed_shuffled_null"
    )
    return {
        "control_source": control_source,
        "null_type": "permutation_null",
        "eligible_windows": int(eligible_windows),
        "observed_promotion_ready_windows": int(observed_ready),
        "control_promotion_ready_windows": np.nan,
        "control_promotion_ready_fraction": np.nan,
        "mean_null_promotion_ready_windows": float(np.mean(counts)) if len(counts) else np.nan,
        "median_null_promotion_ready_windows": float(np.median(counts)) if len(counts) else np.nan,
        "p95_null_promotion_ready_windows": p95,
        "p99_null_promotion_ready_windows": p99,
        "max_null_promotion_ready_windows": float(np.max(counts)) if len(counts) else np.nan,
        "observed_minus_p95_null_windows": float(observed_ready - p95) if np.isfinite(p95) else np.nan,
        "observed_exceeds_null_p95": bool(np.isfinite(p95) and observed_ready > p95),
        "observed_exceeds_null_p99": bool(np.isfinite(p99) and observed_ready > p99),
        "empirical_p_value_ge_observed": empirical_p,
        "permutations": int(permutations),
        "calibration_interpretation": interpretation,
    }


def _permutation_counts(
    high_specificity: pd.DataFrame,
    *,
    mode: str,
    threshold: float,
    permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if high_specificity.empty or permutations <= 0:
        return np.array([], dtype=int)
    threshold_mask = _numeric(high_specificity, DISCOVERY_MARGIN_COLUMN).to_numpy() >= float(threshold)
    base = (_base_rule_mask(high_specificity).to_numpy() & threshold_mask).astype(bool)
    immobile = _immobile_mask(high_specificity).to_numpy().astype(bool)
    interesting = _interesting_mask(high_specificity).to_numpy().astype(bool)
    counts = np.zeros(permutations, dtype=int)
    for idx in range(permutations):
        if mode == "label_shuffle_null":
            trial = base & immobile & rng.permutation(interesting)
        elif mode == "immobility_shuffle_null":
            trial = base & rng.permutation(immobile) & interesting
        elif mode == "joint_label_immobility_shuffle_null":
            trial = base & rng.permutation(immobile) & rng.permutation(interesting)
        else:
            raise ValueError(f"unknown permutation mode: {mode}")
        counts[idx] = int(trial.sum())
    return counts


def build_null_calibration(
    *,
    high_specificity: pd.DataFrame,
    n_permutations: int = 10_000,
    random_seed: int = 1,
) -> pd.DataFrame:
    ready = promotion_ready_mask(high_specificity)
    observed_ready = int(ready.sum())
    running = high_specificity[
        high_specificity.get("run_or_immobility_state", pd.Series("", index=high_specificity.index)).astype(str).eq("run")
    ].copy()
    ordinary = high_specificity[~_interesting_mask(high_specificity)].copy()
    rejected = high_specificity[~ready].copy()

    rows = [
        _direct_control_row(
            control_source="running_high_specificity_controls",
            frame=running,
            ready=ready,
            observed_ready=observed_ready,
            interpretation="running controls should not pass the immobility promotion filter",
        ),
        _direct_control_row(
            control_source="ordinary_movement_spiking_high_specificity_controls",
            frame=ordinary,
            ready=ready,
            observed_ready=observed_ready,
            interpretation="ordinary movement/spiking controls should not pass the specificity-label filter",
        ),
        _direct_control_row(
            control_source="rejected_high_specificity_audit_controls",
            frame=rejected,
            ready=ready,
            observed_ready=observed_ready,
            interpretation="audit row: high-specificity rows rejected by the promotion rule",
        ),
    ]

    rng = np.random.default_rng(random_seed)
    for mode in (
        "label_shuffle_null",
        "immobility_shuffle_null",
        "joint_label_immobility_shuffle_null",
    ):
        counts = _permutation_counts(
            high_specificity,
            mode=mode,
            threshold=50.0,
            permutations=n_permutations,
            rng=rng,
        )
        rows.append(
            _permutation_row(
                control_source=mode,
                counts=counts,
                observed_ready=observed_ready,
                eligible_windows=len(high_specificity),
                permutations=n_permutations,
            )
        )
    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))


def build_threshold_sensitivity(
    *,
    candidate_table: pd.DataFrame,
    high_specificity: pd.DataFrame,
    tier_summary: pd.DataFrame,
    screened: int,
    n_permutations: int = 10_000,
    random_seed: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ready = promotion_ready_mask(high_specificity)
    rng = np.random.default_rng(random_seed + 17)
    for tier, threshold in THRESHOLDS:
        candidates = candidate_table[_numeric(candidate_table, DISCOVERY_MARGIN_COLUMN) >= threshold].copy()
        high = high_specificity[_numeric(high_specificity, DISCOVERY_MARGIN_COLUMN) >= threshold].copy()
        high_ready = ready.loc[high.index] if not high.empty else pd.Series(dtype=bool)
        running = high[
            high.get("run_or_immobility_state", pd.Series("", index=high.index)).astype(str).eq("run")
        ].copy()
        ordinary = high[~_interesting_mask(high)].copy()
        counts = _permutation_counts(
            high_specificity,
            mode="joint_label_immobility_shuffle_null",
            threshold=threshold,
            permutations=n_permutations,
            rng=rng,
        )
        tier_rows = tier_summary[tier_summary["candidate_tier"].astype(str).eq(tier)] if not tier_summary.empty else pd.DataFrame()
        candidate_count = (
            _safe_int(tier_rows.iloc[0]["candidate_windows"])
            if not tier_rows.empty and "candidate_windows" in tier_rows
            else int(len(candidates))
        )
        p95 = float(np.quantile(counts, 0.95)) if len(counts) else np.nan
        p99 = float(np.quantile(counts, 0.99)) if len(counts) else np.nan
        observed_ready = int(high_ready.sum())
        rows.append(
            {
                "candidate_tier": tier,
                "margin_threshold": float(threshold),
                "screened_off_swr_windows": int(screened),
                "candidate_windows": candidate_count,
                "candidate_fraction": _safe_fraction(candidate_count, screened),
                "high_specificity_windows": int(len(high)),
                "promotion_ready_windows": observed_ready,
                "promotion_ready_fraction_of_screened": _safe_fraction(observed_ready, screened),
                "promotion_ready_fraction_of_candidates": _safe_fraction(observed_ready, candidate_count),
                "running_control_windows": int(len(running)),
                "running_control_promotion_ready_windows": int(high_ready.loc[running.index].sum()) if not running.empty else 0,
                "ordinary_control_windows": int(len(ordinary)),
                "ordinary_control_promotion_ready_windows": int(high_ready.loc[ordinary.index].sum()) if not ordinary.empty else 0,
                "joint_shuffle_mean_promotions": float(np.mean(counts)) if len(counts) else np.nan,
                "joint_shuffle_p95_promotions": p95,
                "joint_shuffle_p99_promotions": p99,
                "observed_exceeds_joint_shuffle_p95": bool(np.isfinite(p95) and observed_ready > p95),
                "observed_exceeds_joint_shuffle_p99": bool(np.isfinite(p99) and observed_ready > p99),
            }
        )
    return pd.DataFrame(rows, columns=list(THRESHOLD_COLUMNS))


def build_empirical_fdr_summary(
    *,
    candidate_table: pd.DataFrame,
    high_specificity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
    calibration: pd.DataFrame,
    screened: int,
) -> pd.DataFrame:
    ready = promotion_ready_mask(high_specificity)
    observed_ready = int(ready.sum())
    permutation = calibration[calibration["null_type"].eq("permutation_null")].copy()
    running_control = high_specificity.get(
        "run_or_immobility_state",
        pd.Series("", index=high_specificity.index),
    ).astype(str).eq("run")
    ordinary_control = ~_interesting_mask(high_specificity)
    direct_control = running_control | ordinary_control
    direct_controls = int(direct_control.sum())
    direct_promotions = int((ready & direct_control).sum())
    complete = (
        validation_decisions["required_models_complete"].map(_as_bool)
        if "required_models_complete" in validation_decisions
        else pd.Series(dtype=bool)
    )
    trajectory = (
        validation_decisions["trajectory_confident_claim"].map(_as_bool)
        if "trajectory_confident_claim" in validation_decisions
        else pd.Series(dtype=bool)
    )
    max_mean = _numeric(permutation, "mean_null_promotion_ready_windows").max()
    max_p95 = _numeric(permutation, "p95_null_promotion_ready_windows").max()
    max_p99 = _numeric(permutation, "p99_null_promotion_ready_windows").max()
    min_p = _numeric(permutation, "empirical_p_value_ge_observed").min()
    status = (
        "empirical_controls_support_promotion_specificity"
        if observed_ready > 0
        and direct_promotions == 0
        and pd.notna(max_p95)
        and observed_ready > max_p95
        else "empirical_fdr_not_supported"
    )
    row = {
        "screened_off_swr_windows": int(screened),
        "candidate_windows": int(len(candidate_table)),
        "high_specificity_windows": int(len(high_specificity)),
        "promotion_ready_windows": observed_ready,
        "exact_validated_windows": int(complete.sum()) if not validation_decisions.empty else 0,
        "exact_trajectory_confident_windows": int(trajectory.sum()) if not validation_decisions.empty else 0,
        "direct_control_windows": direct_controls,
        "direct_control_promotion_ready_windows": direct_promotions,
        "direct_control_false_promotion_fraction": _safe_fraction(direct_promotions, direct_controls),
        "max_mean_permutation_null_promotions": float(max_mean) if pd.notna(max_mean) else np.nan,
        "max_p95_permutation_null_promotions": float(max_p95) if pd.notna(max_p95) else np.nan,
        "max_p99_permutation_null_promotions": float(max_p99) if pd.notna(max_p99) else np.nan,
        "min_permutation_empirical_p_value": float(min_p) if pd.notna(min_p) else np.nan,
        "mean_permutation_fdr_estimate": _safe_fraction(float(max_mean), observed_ready) if pd.notna(max_mean) else np.nan,
        "p95_permutation_fdr_bound": _safe_fraction(float(max_p95), observed_ready) if pd.notna(max_p95) else np.nan,
        "fdr_calibration_status": status,
    }
    return pd.DataFrame([row], columns=list(FDR_SUMMARY_COLUMNS))


def build_gate_summary(
    *,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    validation_decisions: pd.DataFrame,
    high_specificity: pd.DataFrame,
    n_permutations: int,
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

    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    permutation = calibration[calibration["null_type"].eq("permutation_null")].copy()
    direct_promotions = _safe_int(row.get("direct_control_promotion_ready_windows", 0))
    direct_controls = _safe_int(row.get("direct_control_windows", 0))
    ready = _safe_int(row.get("promotion_ready_windows", 0))
    exact_validated = _safe_int(row.get("exact_validated_windows", 0))
    exact_trajectory = _safe_int(row.get("exact_trajectory_confident_windows", 0))
    p95_bound = pd.to_numeric(row.get("p95_permutation_fdr_bound", np.nan), errors="coerce")
    min_p = pd.to_numeric(row.get("min_permutation_empirical_p_value", np.nan), errors="coerce")

    add("screened_denominator_present", _safe_int(row.get("screened_off_swr_windows", 0)) > 0, row.get("screened_off_swr_windows", 0), "screened off-SWR denominator is present")
    add("promotion_ready_candidates_present", ready > 0, ready, "promotion-ready candidates are present")
    add("exact_validation_matches_promotion_ready", exact_validated == ready, f"{exact_validated}/{ready}", "exact validation rows match promotion-ready candidates")
    add("exact_trajectory_supports_promoted_candidates", exact_trajectory == ready and ready > 0, f"{exact_trajectory}/{ready}", "all promotion-ready rows validate as trajectory-confident exact-core candidates")
    add("direct_control_pools_present", direct_controls > 0, direct_controls, "running or ordinary movement/spiking control windows are present")
    add("direct_control_promotions_zero", direct_promotions == 0, direct_promotions, "observed running/ordinary controls have zero promotion-ready rows")
    add("permutation_nulls_evaluated", not permutation.empty and int(_numeric(permutation, "permutations").min()) >= n_permutations, f"{len(permutation)} nulls; {n_permutations} permutations", "label/immobility permutation nulls were evaluated")
    add("observed_exceeds_permutation_p95", bool(permutation["observed_exceeds_null_p95"].map(_as_bool).all()), f"{int(permutation['observed_exceeds_null_p95'].map(_as_bool).sum())}/{len(permutation)}", "observed promotions exceed every permutation-null 95th percentile")
    add(
        "observed_exceeds_permutation_p99",
        bool(permutation["observed_exceeds_null_p99"].map(_as_bool).all()),
        f"{int(permutation['observed_exceeds_null_p99'].map(_as_bool).sum())}/{len(permutation)}",
        "observed promotions exceed every permutation-null 99th percentile",
        required=False,
    )
    add(
        "permutation_empirical_p_value_small",
        pd.notna(min_p) and float(min_p) <= 0.01,
        min_p,
        "minimum empirical p-value across permutation nulls is <= 0.01",
        required=False,
    )
    add("permutation_fdr_bound_below_one", pd.notna(p95_bound) and float(p95_bound) < 1.0, p95_bound, "95th-percentile permutation false-promotion bound is below observed promotions")
    strong = threshold_sensitivity[threshold_sensitivity["candidate_tier"].eq("strong")]
    extreme = threshold_sensitivity[threshold_sensitivity["candidate_tier"].eq("extreme")]
    stable = (
        not strong.empty
        and not extreme.empty
        and _safe_int(strong.iloc[0]["promotion_ready_windows"]) > 0
        and _safe_int(extreme.iloc[0]["promotion_ready_windows"]) > 0
        and bool(strong.iloc[0]["observed_exceeds_joint_shuffle_p95"])
        and bool(extreme.iloc[0]["observed_exceeds_joint_shuffle_p95"])
    )
    add("threshold_curve_stable_high_specificity_region", stable, f"strong={_safe_int(strong.iloc[0]['promotion_ready_windows']) if not strong.empty else 0}; extreme={_safe_int(extreme.iloc[0]['promotion_ready_windows']) if not extreme.empty else 0}", "strong and extreme tiers retain promoted candidates above joint-shuffle p95")
    add("optional_time_shifted_or_wrong_map_controls_missing", True, "not provided", "time-shifted/wrong-map tables are optional external controls in this implementation", required=False)

    required_rows = [item for item in rows if item["required_for_overall"]]
    add(
        "overall",
        all(item["passed"] for item in required_rows),
        f"{sum(item['passed'] for item in required_rows)}/{len(required_rows)} required gates passed",
        "all required empirical FDR calibration gates pass",
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_off_swr_promotion_fdr_outputs(
    *,
    discovery_dir: Path,
    validation_dir: Path,
    output: Path,
    n_permutations: int = 10_000,
    random_seed: int = 1,
) -> dict[str, pd.DataFrame]:
    candidate_table = _read_required_csv(discovery_dir / "off_swr_candidate_table.csv")
    high_specificity = _read_required_csv(discovery_dir / "off_swr_high_specificity_candidate_table.csv")
    tier_summary = _read_optional_csv(discovery_dir / "off_swr_candidate_tier_threshold_summary.csv")
    validation_decisions = _read_required_csv(validation_dir / "promoted_off_swr_candidate_exact_core_decisions.csv")

    screened = _screened_count(candidate_table, tier_summary)
    calibration = build_null_calibration(
        high_specificity=high_specificity,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    threshold = build_threshold_sensitivity(
        candidate_table=candidate_table,
        high_specificity=high_specificity,
        tier_summary=tier_summary,
        screened=screened,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    summary = build_empirical_fdr_summary(
        candidate_table=candidate_table,
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
        calibration=calibration,
        screened=screened,
    )
    gates = build_gate_summary(
        summary=summary,
        calibration=calibration,
        threshold_sensitivity=threshold,
        validation_decisions=validation_decisions,
        high_specificity=high_specificity,
        n_permutations=n_permutations,
    )

    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "off_swr_promotion_null_calibration.csv": calibration,
        "off_swr_promotion_empirical_fdr_summary.csv": summary,
        "off_swr_promotion_threshold_sensitivity.csv": threshold,
        "off_swr_promotion_null_gate_summary.csv": gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-dir", required=True, help="Directory containing off-SWR discovery CSV outputs.")
    parser.add_argument("--validation-dir", required=True, help="Directory containing promoted candidate exact-core validation CSV outputs.")
    parser.add_argument("--output", default="results/off-swr-promotion-fdr-calibration")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = write_off_swr_promotion_fdr_outputs(
        discovery_dir=Path(args.discovery_dir),
        validation_dir=Path(args.validation_dir),
        output=Path(args.output),
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
    )
    print("Off-SWR promotion empirical FDR summary:")
    print(outputs["off_swr_promotion_empirical_fdr_summary.csv"].to_string(index=False))
    print("\nOff-SWR promotion FDR gates:")
    print(outputs["off_swr_promotion_null_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
