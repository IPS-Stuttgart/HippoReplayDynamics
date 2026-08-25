#!/usr/bin/env python3
"""Test predeclared replay commitment/composition hypotheses."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402


PRIMARY_OUTPUT = "replay_dynamics_behavior_primary_tests.csv"
BY_RAT_OUTPUT = "replay_dynamics_behavior_by_rat.csv"
LOO_OUTPUT = "replay_dynamics_behavior_leave_one_rat_out.csv"
MATCHED_OUTPUT = "replay_dynamics_behavior_matched_sensitivity.csv"
MATCHED_SUMMARY_OUTPUT = "replay_dynamics_behavior_matched_sensitivity_summary.csv"
GATE_OUTPUT = "replay_dynamics_behavior_primary_gate_summary.csv"
MANIFEST_OUTPUT = "replay_dynamics_behavior_primary_manifest.json"
SUMMARY_OUTPUT = "replay_dynamics_behavior_primary_summary.md"

PRIMARY_PREDICTOR = "delta_momentum_minus_imm"
NUMERIC_CONTROLS = (
    "event_duration_ms",
    "log_n_spikes",
    "active_cell_count",
    "posterior_entropy",
    "trajectory_minus_stationary_log_evidence",
    "log_posterior_path_length_cm",
    "current_animal_x_cm",
    "current_animal_y_cm",
    "next_well_x_cm",
    "next_well_y_cm",
    "log_route_frequency",
    "time_to_departure_s",
    "elapsed_time_since_reward_s",
    "run_decoder_error_cm",
)
CATEGORICAL_CONTROLS = ("session",)


def _analysis_frame(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["log_n_spikes"] = np.log1p(pd.to_numeric(out["n_spikes"], errors="coerce"))
    out["log_posterior_path_length_cm"] = np.log1p(
        pd.to_numeric(out["posterior_path_length_cm"], errors="coerce").clip(lower=0.0)
    )
    out["log_route_frequency"] = np.log1p(
        pd.to_numeric(out["route_frequency"], errors="coerce").clip(lower=0.0)
    )
    return out


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    scale = float(np.std(array, ddof=0))
    return (array - mean) / scale if scale > 0.0 else np.zeros_like(array)


def _design_matrix(
    frame: pd.DataFrame,
    *,
    predictor: str,
    numeric_controls: Sequence[str],
    categorical_controls: Sequence[str],
) -> tuple[np.ndarray, int, list[str]]:
    columns = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    predictor_index = 1
    columns.append(_standardize(pd.to_numeric(frame[predictor], errors="raise").to_numpy()))
    names.append(predictor)
    for control in numeric_controls:
        values = pd.to_numeric(frame[control], errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        median = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
        missing = ~np.isfinite(values)
        values[missing] = median
        standardized = _standardize(values)
        if np.std(standardized) > 0.0:
            columns.append(standardized)
            names.append(control)
        if np.any(missing) and not np.all(missing):
            columns.append(missing.astype(float))
            names.append(f"{control}_missing")
    for control in categorical_controls:
        values = frame[control].astype(str)
        dummies = pd.get_dummies(values, prefix=control, drop_first=True, dtype=float)
        for name in dummies.columns:
            columns.append(dummies[name].to_numpy(dtype=float))
            names.append(str(name))
    return np.column_stack(columns), predictor_index, names


def adjusted_coefficient(
    frame: pd.DataFrame,
    *,
    outcome: str,
    predictor: str = PRIMARY_PREDICTOR,
    numeric_controls: Sequence[str] = NUMERIC_CONTROLS,
    categorical_controls: Sequence[str] = CATEGORICAL_CONTROLS,
) -> tuple[float, int, int]:
    required = [outcome, predictor, *numeric_controls, *categorical_controls]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"analysis table is missing required columns: {missing}")
    selected = frame[np.isfinite(pd.to_numeric(frame[outcome], errors="coerce"))].copy()
    selected = selected[np.isfinite(pd.to_numeric(selected[predictor], errors="coerce"))]
    if len(selected) < 8:
        return np.nan, int(len(selected)), 0
    y = _standardize(pd.to_numeric(selected[outcome], errors="raise").to_numpy(dtype=float))
    design, predictor_index, _ = _design_matrix(
        selected,
        predictor=predictor,
        numeric_controls=numeric_controls,
        categorical_controls=categorical_controls,
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank <= predictor_index or len(selected) - rank < 2:
        return np.nan, int(len(selected)), rank
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficient[predictor_index]), int(len(selected)), rank


def _raw_spearman(frame: pd.DataFrame, outcome: str) -> float:
    selected = frame[[PRIMARY_PREDICTOR, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) < 3:
        return np.nan
    return float(spearmanr(selected[PRIMARY_PREDICTOR], selected[outcome]).statistic)


def rat_cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    rats = sorted(frame["rat"].astype(str).unique())
    if len(rats) < 2:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    groups = {rat: frame[frame["rat"].astype(str).eq(rat)] for rat in rats}
    for _ in range(int(replicates)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        parts = []
        for draw, rat in enumerate(sampled):
            part = groups[str(rat)].copy()
            part["rat"] = f"bootstrap_{draw}_{rat}"
            part["session"] = part["session"].astype(str).map(
                lambda value, draw_index=draw: f"bootstrap_{draw_index}_{value}"
            )
            parts.append(part)
        value = float(statistic(pd.concat(parts, ignore_index=True)))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        int(len(values)),
    )


def _status(estimate: float, low: float, high: float, expected_sign: int) -> str:
    if not all(np.isfinite(value) for value in (estimate, low, high)):
        return "insufficient"
    if expected_sign > 0 and low > 0.0:
        return "supported"
    if expected_sign < 0 and high < 0.0:
        return "supported"
    if expected_sign > 0 and high < 0.0:
        return "contradicted"
    if expected_sign < 0 and low > 0.0:
        return "contradicted"
    return "inconclusive"


def _primary_specifications() -> tuple[dict[str, object], ...]:
    return (
        {
            "test": "composition_decreases_with_momentum_axis",
            "outcome": "composition_index",
            "cohort_column": "composition_evaluable",
            "expected_sign": -1,
        },
        {
            "test": "future_commitment_increases_with_momentum_axis",
            "outcome": "future_commitment_index",
            "cohort_column": "future_commitment_index",
            "expected_sign": 1,
        },
        {
            "test": "momentum_axis_incrementally_predicts_actual_future_route",
            "outcome": "actual_future_closer_than_alternatives",
            "cohort_column": "future_commitment_index",
            "expected_sign": 1,
        },
    )


def run_primary_tests(
    events: pd.DataFrame,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _analysis_frame(events)
    frame["actual_future_closer_than_alternatives"] = (
        frame["future_commitment_index"] > 0.0
    ).astype(float)
    frame.loc[frame["future_commitment_index"].isna(), "actual_future_closer_than_alternatives"] = np.nan
    primary_rows: list[dict[str, object]] = []
    rat_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    for index, specification in enumerate(_primary_specifications()):
        outcome = str(specification["outcome"])
        expected_sign = int(specification["expected_sign"])
        selected = frame[np.isfinite(pd.to_numeric(frame[outcome], errors="coerce"))].copy()
        estimate, n_events, rank = adjusted_coefficient(selected, outcome=outcome)

        def statistic(data: pd.DataFrame, outcome_name: str = outcome) -> float:
            return adjusted_coefficient(data, outcome=outcome_name)[0]

        low, high, completed = rat_cluster_bootstrap(
            selected,
            statistic,
            replicates=int(bootstrap_replicates),
            seed=int(seed + index),
        )
        primary_rows.append(
            {
                "test": specification["test"],
                "outcome": outcome,
                "primary_predictor": PRIMARY_PREDICTOR,
                "expected_sign": "positive" if expected_sign > 0 else "negative",
                "events": n_events,
                "rats": int(selected["rat"].nunique()),
                "sessions": int(selected["session"].nunique()),
                "raw_spearman_r": _raw_spearman(selected, outcome),
                "adjusted_standardized_coefficient": estimate,
                "rat_cluster_bootstrap_ci_low": low,
                "rat_cluster_bootstrap_ci_high": high,
                "bootstrap_replicates_completed": completed,
                "design_rank": rank,
                "residual_degrees_of_freedom": int(n_events - rank),
                "status": _status(estimate, low, high, expected_sign),
                "numeric_controls": ";".join(NUMERIC_CONTROLS),
                "categorical_controls": ";".join(CATEGORICAL_CONTROLS),
            }
        )
        for rat, group in selected.groupby("rat", sort=True):
            coefficient, count, rat_rank = adjusted_coefficient(
                group,
                outcome=outcome,
                categorical_controls=("session",),
            )
            rat_rows.append(
                {
                    "test": specification["test"],
                    "rat": rat,
                    "events": count,
                    "raw_spearman_r": _raw_spearman(group, outcome),
                    "adjusted_standardized_coefficient": coefficient,
                    "design_rank": rat_rank,
                    "residual_degrees_of_freedom": int(count - rat_rank),
                }
            )
        for omitted_rat in sorted(selected["rat"].astype(str).unique()):
            retained = selected[~selected["rat"].astype(str).eq(omitted_rat)]
            coefficient, count, retained_rank = adjusted_coefficient(
                retained,
                outcome=outcome,
            )
            loo_rows.append(
                {
                    "test": specification["test"],
                    "omitted_rat": omitted_rat,
                    "events": count,
                    "rats_retained": int(retained["rat"].nunique()),
                    "adjusted_standardized_coefficient": coefficient,
                    "expected_direction_retained": bool(
                        np.isfinite(coefficient) and coefficient * expected_sign > 0.0
                    ),
                    "design_rank": retained_rank,
                    "residual_degrees_of_freedom": int(count - retained_rank),
                }
            )
    return pd.DataFrame(primary_rows), pd.DataFrame(rat_rows), pd.DataFrame(loo_rows)


def _quality_match(
    events: pd.DataFrame,
    *,
    treated_column: str,
    control_column: str,
) -> pd.DataFrame:
    frame = _analysis_frame(events)
    quality = [
        "event_duration_ms",
        "log_n_spikes",
        "active_cell_count",
        "posterior_entropy",
        "trajectory_minus_stationary_log_evidence",
        "log_posterior_path_length_cm",
    ]
    treated = frame[frame[treated_column].astype(bool)].copy()
    controls = frame[frame[control_column].astype(bool)].copy()
    if treated.empty or controls.empty:
        return pd.DataFrame()
    combined = pd.concat([treated[quality], controls[quality]], ignore_index=True)
    median = combined.median(numeric_only=True)
    scale = combined.std(numeric_only=True).replace(0.0, 1.0)
    treated_values = ((treated[quality].fillna(median) - median) / scale).to_numpy(dtype=float)
    control_values = ((controls[quality].fillna(median) - median) / scale).to_numpy(dtype=float)
    available = set(range(len(controls)))
    rows: list[dict[str, object]] = []
    for treated_index in range(len(treated)):
        if not available:
            break
        same_session = {
            control_index
            for control_index in available
            if str(controls.iloc[control_index]["session"])
            == str(treated.iloc[treated_index]["session"])
        }
        same_rat = {
            control_index
            for control_index in available
            if str(controls.iloc[control_index]["rat"])
            == str(treated.iloc[treated_index]["rat"])
        }
        candidates = same_session or same_rat
        if not candidates:
            continue
        distances = {
            control_index: float(
                np.linalg.norm(treated_values[treated_index] - control_values[control_index])
            )
            for control_index in candidates
        }
        matched_index = min(distances, key=lambda value: (distances[value], value))
        available.remove(matched_index)
        treated_row = treated.iloc[treated_index]
        control_row = controls.iloc[matched_index]
        rows.append(
            {
                "pair_id": len(rows),
                "treated_session": treated_row["session"],
                "treated_event_index": int(treated_row["event_index"]),
                "control_session": control_row["session"],
                "control_event_index": int(control_row["event_index"]),
                "quality_distance": distances[matched_index],
                "match_scope": (
                    "same_session" if matched_index in same_session else "same_rat"
                ),
                "composition_difference_treated_minus_control": float(
                    treated_row["composition_index"] - control_row["composition_index"]
                ) if np.isfinite(treated_row["composition_index"])
                and np.isfinite(control_row["composition_index"]) else np.nan,
                "commitment_difference_treated_minus_control": float(
                    treated_row["future_commitment_index"]
                    - control_row["future_commitment_index"]
                ) if np.isfinite(treated_row["future_commitment_index"])
                and np.isfinite(control_row["future_commitment_index"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_matched_sensitivity(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        (
            "composition_momentum_minus_clean_imm",
            "composition_difference_treated_minus_control",
            "negative",
        ),
        (
            "commitment_momentum_minus_clean_imm",
            "commitment_difference_treated_minus_control",
            "positive",
        ),
    )
    for test, column, expected_direction in specifications:
        values = pd.to_numeric(
            matched.get(column, pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        rows.append(
            {
                "test": test,
                "expected_direction": expected_direction,
                "matched_pairs": int(len(values)),
                "median_treated_minus_control": float(values.median())
                if len(values)
                else np.nan,
                "mean_treated_minus_control": float(values.mean())
                if len(values)
                else np.nan,
                "positive_fraction": float(np.mean(values > 0.0))
                if len(values)
                else np.nan,
                "role": "secondary_descriptive_only",
            }
        )
    return pd.DataFrame(rows)


def build_gate_summary(
    primary: pd.DataFrame,
    by_rat: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    minimum_events: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for test in (
        "composition_decreases_with_momentum_axis",
        "future_commitment_increases_with_momentum_axis",
        "momentum_axis_incrementally_predicts_actual_future_route",
    ):
        match = primary[primary["test"].eq(test)]
        row = match.iloc[0] if not match.empty else None
        count = int(row["events"]) if row is not None else 0
        status = str(row["status"]) if row is not None else "missing"
        rat_effects = by_rat[by_rat["test"].eq(test)]
        loo = leave_one_out[leave_one_out["test"].eq(test)]
        rows.extend(
            [
                {
                    "gate": f"{test}_cohort_size",
                    "passed": count >= int(minimum_events),
                    "value": count,
                    "required": f">={int(minimum_events)}",
                },
                {
                    "gate": f"{test}_primary_support",
                    "passed": status == "supported",
                    "value": status,
                    "required": "supported",
                },
                {
                    "gate": f"{test}_all_rats_estimable",
                    "passed": len(rat_effects) == 4
                    and rat_effects["adjusted_standardized_coefficient"].notna().all(),
                    "value": int(rat_effects["adjusted_standardized_coefficient"].notna().sum()),
                    "required": 4,
                },
                {
                    "gate": f"{test}_leave_one_rat_out_direction",
                    "passed": len(loo) == 4 and loo["expected_direction_retained"].all(),
                    "value": int(loo["expected_direction_retained"].sum()),
                    "required": 4,
                },
            ]
        )
    decoder_values = (
        pd.to_numeric(events["run_decoder_error_cm"], errors="coerce")
        if events is not None and "run_decoder_error_cm" in events
        else pd.Series(dtype=float)
    )
    decoder_sessions = (
        int(events.loc[decoder_values.notna(), "session"].nunique())
        if events is not None and "session" in events
        else 0
    )
    required_sessions = int(events["session"].nunique()) if events is not None else 0
    rows.append(
        {
            "gate": "run_decoder_error_control_available",
            "passed": bool(required_sessions > 0 and decoder_sessions == required_sessions),
            "value": decoder_sessions,
            "required": required_sessions,
        }
    )
    rows.append(
        {
            "gate": "overall_strong_primary_support",
            "passed": bool(all(row["passed"] for row in rows)),
            "value": int(sum(bool(row["passed"]) for row in rows)),
            "required": len(rows),
        }
    )
    return pd.DataFrame(rows)


def run_analysis(
    *,
    event_metrics_csv: str | Path,
    output_dir: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 1,
) -> dict[str, Path]:
    events = pd.read_csv(event_metrics_csv)
    primary, by_rat, leave_one_out = run_primary_tests(
        events,
        bootstrap_replicates=int(bootstrap_replicates),
        seed=int(seed),
    )
    matched = _quality_match(
        events,
        treated_column="raw_momentum_win",
        control_column="clean_imm",
    )
    matched_summary = summarize_matched_sensitivity(matched)
    gates = build_gate_summary(primary, by_rat, leave_one_out, events=events)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in {
        PRIMARY_OUTPUT: primary,
        BY_RAT_OUTPUT: by_rat,
        LOO_OUTPUT: leave_one_out,
        MATCHED_OUTPUT: matched,
        MATCHED_SUMMARY_OUTPUT: matched_summary,
        GATE_OUTPUT: gates,
    }.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    provenance = build_script_provenance(
        input_paths={"event_metrics_csv": event_metrics_csv},
        cwd=ROOT,
    )
    manifest = {
        "analysis": "replay_dynamics_behavior_primary_tests",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_predictor": PRIMARY_PREDICTOR,
        "categorical_model_classes_role": "secondary_descriptive_only",
        "bootstrap_replicates": int(bootstrap_replicates),
        "seed": int(seed),
        "numeric_controls": list(NUMERIC_CONTROLS),
        "categorical_controls": list(CATEGORICAL_CONTROLS),
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    strong = bool(
        gates.loc[gates["gate"].eq("overall_strong_primary_support"), "passed"].iloc[0]
    )
    summary = [
        "# Replay dynamics/behavior primary tests",
        "",
        f"- Primary predictor: `{PRIMARY_PREDICTOR}`",
        "- Categorical clean-IMM and momentum groups: secondary descriptive sensitivity only",
        f"- Strong joint primary support: {'PASS' if strong else 'NOT ESTABLISHED'}",
        "",
    ]
    for row in primary.itertuples(index=False):
        summary.append(
            f"- {row.test}: beta={row.adjusted_standardized_coefficient:.3f}, "
            f"95% rat-bootstrap CI [{row.rat_cluster_bootstrap_ci_low:.3f}, "
            f"{row.rat_cluster_bootstrap_ci_high:.3f}], status={row.status}."
        )
    summary_path = output / SUMMARY_OUTPUT
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    paths[SUMMARY_OUTPUT] = summary_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        event_metrics_csv=args.event_metrics,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
