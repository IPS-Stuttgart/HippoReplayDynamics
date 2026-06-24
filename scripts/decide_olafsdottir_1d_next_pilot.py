#!/usr/bin/env python3
"""Decide the next Olafsdottir 1D pilot tier from existing debug reports.

This helper is intentionally non-scoring. It reads the pilot20 debug report
tables, optional pre-evidence event QC, decoder QC, and pilot-selection tables,
then recommends whether the next frozen pilot should emphasize event strength,
decoder quality, pair-local debugging, or stop for model/decoder validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


QUALITY_INPUT = "olafsdottir_1d_sleep_debug_quality_table.csv"
PAIR_INPUT = "olafsdottir_1d_sleep_by_pair_debug_summary.csv"
ANIMAL_INPUT = "olafsdottir_1d_sleep_by_animal_debug_summary.csv"

CORRELATION_OUTPUT = "olafsdottir_1d_margin_predictor_correlations.csv"
EVENT_STRENGTH_OUTPUT = "olafsdottir_1d_margin_by_event_strength_quantile.csv"
DECODER_QUALITY_OUTPUT = "olafsdottir_1d_margin_by_decoder_quality_quantile.csv"
PAIR_DECISION_OUTPUT = "olafsdottir_1d_margin_by_pair_decision.csv"
RECOMMENDATION_OUTPUT = "olafsdottir_1d_next_pilot_recommendation.csv"
SUMMARY_OUTPUT = "olafsdottir_1d_next_pilot_decision.md"

PAIR_KEYS = ["animal", "date", "track1_session", "sleeppost_session"]
TARGET_COLUMNS = [
    "delta_best_trajectory_minus_stationary",
    "delta_imm_minus_fragmented",
]
EVENT_STRENGTH_PREDICTORS = [
    "n_spikes",
    "n_active_units",
    "duration_ms",
    "mean_mua_rate_hz",
    "peak_mua_rate_hz",
    "event_detection_score",
]
DECODER_QUALITY_PREDICTORS = [
    "encoding_units_passing_qc",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
]
LOWER_IS_BETTER = {"posterior_mean_error_cm_median", "map_error_cm_median", "mean_speed_cm_s"}
RECOMMENDATIONS = {
    "run_high_information_pilot20_debug",
    "run_decoder_strong_pilot20_debug",
    "run_pair_targeted_debug",
    "stop_no_signal",
}


def run_next_pilot_decision(
    *,
    report_dir: str | Path,
    event_qc: str | Path | None = None,
    decoder_qc: str | Path | None = None,
    pilot_selection: str | Path | None = None,
    output_dir: str | Path,
    margin_threshold: float = 5.5,
    correlation_threshold: float = 0.35,
    dominance_fraction_threshold: float = 0.60,
) -> dict[str, pd.DataFrame]:
    report_root = Path(report_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quality = load_quality_table(report_root / QUALITY_INPUT)
    quality = enrich_quality_table(
        quality,
        event_qc=read_optional_csv(event_qc),
        decoder_qc=read_optional_csv(decoder_qc),
        pilot_selection=read_optional_csv(pilot_selection),
    )
    correlations = build_correlation_table(quality)
    event_strength = build_quantile_table(
        quality,
        predictors=EVENT_STRENGTH_PREDICTORS,
        predictor_group="event_strength",
        margin_threshold=margin_threshold,
    )
    decoder_quality = build_quantile_table(
        quality,
        predictors=DECODER_QUALITY_PREDICTORS,
        predictor_group="decoder_quality",
        margin_threshold=margin_threshold,
    )
    pair_decision = build_pair_decision_table(quality, margin_threshold=margin_threshold)
    recommendation = build_recommendation_table(
        quality,
        correlations,
        pair_decision,
        margin_threshold=margin_threshold,
        correlation_threshold=correlation_threshold,
        dominance_fraction_threshold=dominance_fraction_threshold,
    )

    correlations.to_csv(out / CORRELATION_OUTPUT, index=False)
    event_strength.to_csv(out / EVENT_STRENGTH_OUTPUT, index=False)
    decoder_quality.to_csv(out / DECODER_QUALITY_OUTPUT, index=False)
    pair_decision.to_csv(out / PAIR_DECISION_OUTPUT, index=False)
    recommendation.to_csv(out / RECOMMENDATION_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(
        build_markdown_summary(
            quality=quality,
            correlations=correlations,
            event_strength=event_strength,
            decoder_quality=decoder_quality,
            pair_decision=pair_decision,
            recommendation=recommendation,
            margin_threshold=margin_threshold,
        ),
        encoding="utf-8",
    )
    return {
        "quality": quality,
        "correlations": correlations,
        "event_strength_quantiles": event_strength,
        "decoder_quality_quantiles": decoder_quality,
        "pair_decision": pair_decision,
        "recommendation": recommendation,
    }


def load_quality_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    quality = pd.read_csv(path)
    missing = sorted({"animal", "event_id", *TARGET_COLUMNS}.difference(quality.columns))
    if missing:
        raise ValueError(f"quality table missing required columns: {missing}")
    return quality


def read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path in (None, ""):
        return pd.DataFrame()
    file_path = Path(path)
    return pd.read_csv(file_path) if file_path.is_file() else pd.DataFrame()


def enrich_quality_table(
    quality: pd.DataFrame,
    *,
    event_qc: pd.DataFrame,
    decoder_qc: pd.DataFrame,
    pilot_selection: pd.DataFrame,
) -> pd.DataFrame:
    out = quality.copy()
    if not decoder_qc.empty and set(PAIR_KEYS).issubset(decoder_qc.columns):
        decoder_cols = [column for column in decoder_qc.columns if column in PAIR_KEYS or column not in out.columns]
        out = out.merge(decoder_qc[dedupe(decoder_cols)].drop_duplicates(PAIR_KEYS), on=PAIR_KEYS, how="left")
    event_keys = ["animal", "date", "track1_session", "sleeppost_session", "event_id"]
    if not pilot_selection.empty and set(event_keys).issubset(pilot_selection.columns):
        pilot = filter_to_scored_tier(pilot_selection, out)
        pilot_cols = [column for column in pilot.columns if column in event_keys or column not in out.columns]
        out = out.merge(pilot[dedupe(pilot_cols)].drop_duplicates(event_keys), on=event_keys, how="left")
    if not event_qc.empty and {"animal", "date", "event_id"}.issubset(event_qc.columns):
        event_cols = [column for column in event_qc.columns if column in {"animal", "date", "event_id"} or column not in out.columns]
        out = out.merge(event_qc[dedupe(event_cols)].drop_duplicates(["animal", "date", "event_id"]), on=["animal", "date", "event_id"], how="left")
    return out


def filter_to_scored_tier(selection: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if "selection_tier" not in selection.columns or "pilot_tier" not in quality.columns:
        return selection
    tiers = {str(value) for value in quality["pilot_tier"].dropna().unique()}
    filtered = selection[selection["selection_tier"].astype(str).isin(tiers)].copy()
    return filtered if not filtered.empty else selection


def build_correlation_table(quality: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        for predictor in dedupe([*EVENT_STRENGTH_PREDICTORS, *DECODER_QUALITY_PREDICTORS, "mean_speed_cm_s"]):
            if target not in quality.columns or predictor not in quality.columns:
                continue
            target_values = pd.to_numeric(quality[target], errors="coerce")
            predictor_values = pd.to_numeric(quality[predictor], errors="coerce")
            frame = pd.DataFrame({"target": target_values, "predictor": predictor_values}).dropna()
            r = pearson(frame["target"], frame["predictor"])
            expected_r = expected_direction_adjusted_r(predictor, r)
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "predictor_group": predictor_group(predictor),
                    "n": int(len(frame)),
                    "pearson_r": r,
                    "expected_direction_adjusted_r": expected_r,
                    "abs_pearson_r": abs(r) if np.isfinite(r) else np.nan,
                    "expected_direction_support": bool(np.isfinite(expected_r) and expected_r > 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_quantile_table(
    quality: pd.DataFrame,
    *,
    predictors: Sequence[str],
    predictor_group: str,
    margin_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for predictor in predictors:
        if predictor not in quality.columns:
            continue
        predictor_values = pd.to_numeric(quality[predictor], errors="coerce")
        if predictor_values.dropna().nunique() < 2:
            continue
        quantiles = assign_quantile_labels(predictor_values, lower_is_better=predictor in LOWER_IS_BETTER)
        for target in TARGET_COLUMNS:
            if target not in quality.columns:
                continue
            data = quality.assign(_predictor=predictor_values, _target=pd.to_numeric(quality[target], errors="coerce"), _quantile=quantiles)
            for label, group in data.dropna(subset=["_predictor", "_target", "_quantile"]).groupby("_quantile", sort=False):
                values = group["_target"]
                rows.append(
                    {
                        "predictor_group": predictor_group,
                        "predictor": predictor,
                        "target": target,
                        "quality_quantile": label,
                        "events": int(len(group)),
                        "median_predictor": finite_median(group["_predictor"]),
                        "mean_margin": finite_mean(values),
                        "median_margin": finite_median(values),
                        "positive_margin_events": int((values > 0.0).sum()),
                        "positive_margin_fraction": safe_fraction(int((values > 0.0).sum()), len(group)),
                        "confident_positive_events": int((values >= float(margin_threshold)).sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_pair_decision_table(quality: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in quality.groupby(PAIR_KEYS, sort=True, dropna=False):
        traj = pd.to_numeric(group["delta_best_trajectory_minus_stationary"], errors="coerce")
        imm = pd.to_numeric(group["delta_imm_minus_fragmented"], errors="coerce")
        row = {column: value for column, value in zip(PAIR_KEYS, key)}
        row.update(
            {
                "events": int(len(group)),
                "trajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()) if "trajectory_family_claim" in group else int((traj >= margin_threshold).sum()),
                "positive_trajectory_margin_events": int((traj > 0.0).sum()),
                "median_delta_best_trajectory_minus_stationary": finite_median(traj),
                "max_delta_best_trajectory_minus_stationary": finite_max(traj),
                "imm_confident_wins": int((imm >= float(margin_threshold)).sum()),
                "positive_imm_margin_events": int((imm > 0.0).sum()),
                "median_delta_imm_minus_fragmented": finite_median(imm),
                "max_delta_imm_minus_fragmented": finite_max(imm),
                "mean_n_spikes": finite_mean(pd.to_numeric(group.get("n_spikes", pd.Series(dtype=float)), errors="coerce")),
                "mean_n_active_units": finite_mean(pd.to_numeric(group.get("n_active_units", pd.Series(dtype=float)), errors="coerce")),
                "median_duration_ms": finite_median(pd.to_numeric(group.get("duration_ms", pd.Series(dtype=float)), errors="coerce")),
                "median_decoder_posterior_error_cm": finite_median(pd.to_numeric(group.get("posterior_mean_error_cm_median", pd.Series(dtype=float)), errors="coerce")),
            }
        )
        row["pair_debug_status"] = pair_status(row, margin_threshold=margin_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_status(row: dict[str, object], *, margin_threshold: float) -> str:
    del margin_threshold
    if int(row.get("trajectory_confident_claims", 0)) > 0 or int(row.get("imm_confident_wins", 0)) > 0:
        return "localized_positive_signal"
    if float_or_nan(row.get("median_delta_best_trajectory_minus_stationary")) > 0.0:
        return "weak_positive_trajectory_trend"
    if float_or_nan(row.get("median_delta_best_trajectory_minus_stationary")) <= 0.0:
        return "weak_or_negative_trajectory_trend"
    return "insufficient_metrics"


def build_recommendation_table(
    quality: pd.DataFrame,
    correlations: pd.DataFrame,
    pair_decision: pd.DataFrame,
    *,
    margin_threshold: float,
    correlation_threshold: float,
    dominance_fraction_threshold: float,
) -> pd.DataFrame:
    events = int(len(quality))
    traj = pd.to_numeric(quality.get("delta_best_trajectory_minus_stationary", pd.Series(dtype=float)), errors="coerce")
    imm = pd.to_numeric(quality.get("delta_imm_minus_fragmented", pd.Series(dtype=float)), errors="coerce")
    trajectory_confident = int((traj >= float(margin_threshold)).sum())
    imm_confident = int((imm >= float(margin_threshold)).sum())
    event_score = best_expected_correlation(correlations, group="event_strength", target="delta_best_trajectory_minus_stationary")
    decoder_score = best_expected_correlation(correlations, group="decoder_quality", target="delta_best_trajectory_minus_stationary")
    event_score_value = decision_score(event_score["score"])
    decoder_score_value = decision_score(decoder_score["score"])
    pair_signal = pair_signal_strength(
        pair_decision,
        quality,
        margin_threshold=margin_threshold,
        dominance_fraction_threshold=dominance_fraction_threshold,
    )

    recommendation = "stop_no_signal"
    reason = "No event-strength, decoder-quality, or pair-local pattern is strong enough to justify immediate resampling."
    if event_score_value >= correlation_threshold and event_score_value >= decoder_score_value + 0.10:
        recommendation = "run_high_information_pilot20_debug"
        reason = f"Trajectory margins track event-strength predictor {event_score['predictor']} with adjusted r={event_score_value:.3g}."
    elif decoder_score_value >= correlation_threshold and decoder_score_value >= event_score_value + 0.10:
        recommendation = "run_decoder_strong_pilot20_debug"
        reason = f"Trajectory margins track decoder-quality predictor {decoder_score['predictor']} with adjusted r={decoder_score_value:.3g}."
    elif pair_signal["localized"]:
        recommendation = "run_pair_targeted_debug"
        reason = pair_signal["reason"]
    assert recommendation in RECOMMENDATIONS
    row = {
        "recommendation": recommendation,
        "primary_reason": reason,
        "events": events,
        "trajectory_confident_events": trajectory_confident,
        "imm_confident_events": imm_confident,
        "median_delta_best_trajectory_minus_stationary": finite_median(traj),
        "median_delta_imm_minus_fragmented": finite_median(imm),
        "best_event_strength_predictor": event_score["predictor"],
        "best_event_strength_expected_r": event_score["score"],
        "best_decoder_quality_predictor": decoder_score["predictor"],
        "best_decoder_quality_expected_r": decoder_score["score"],
        "localized_signal": pair_signal["localized"],
        "localized_signal_detail": pair_signal["reason"],
        "margin_threshold": float(margin_threshold),
        "correlation_threshold": float(correlation_threshold),
        "dominance_fraction_threshold": float(dominance_fraction_threshold),
    }
    return pd.DataFrame([row])


def best_expected_correlation(correlations: pd.DataFrame, *, group: str, target: str) -> dict[str, object]:
    empty = {"predictor": "", "score": float("nan")}
    if correlations.empty:
        return empty
    subset = correlations[(correlations["predictor_group"].eq(group)) & (correlations["target"].eq(target))].copy()
    subset = subset[pd.to_numeric(subset["expected_direction_adjusted_r"], errors="coerce").notna()]
    if subset.empty:
        return empty
    subset = subset.sort_values("expected_direction_adjusted_r", ascending=False)
    top = subset.iloc[0]
    return {"predictor": str(top["predictor"]), "score": float(top["expected_direction_adjusted_r"])}


def decision_score(value: object) -> float:
    score = float_or_nan(value)
    return score if np.isfinite(score) else float("-inf")


def pair_signal_strength(
    pair_decision: pd.DataFrame,
    quality: pd.DataFrame,
    *,
    margin_threshold: float,
    dominance_fraction_threshold: float,
) -> dict[str, object]:
    if pair_decision.empty or quality.empty:
        return {"localized": False, "reason": "No pair decision rows available."}
    positive_pairs = pair_decision[
        (pd.to_numeric(pair_decision.get("trajectory_confident_claims", 0), errors="coerce") > 0)
        | (pd.to_numeric(pair_decision.get("imm_confident_wins", 0), errors="coerce") > 0)
        | (pd.to_numeric(pair_decision.get("median_delta_best_trajectory_minus_stationary", np.nan), errors="coerce") > 0.0)
    ]
    if positive_pairs.empty:
        return {"localized": False, "reason": "No pair has confident or positive trajectory-family signal."}
    total_trajectory_confident = int(pd.to_numeric(pair_decision.get("trajectory_confident_claims", 0), errors="coerce").sum())
    total_imm_confident = int(pd.to_numeric(pair_decision.get("imm_confident_wins", 0), errors="coerce").sum())
    top_animal = animal_signal_concentration(quality, margin_threshold=margin_threshold)
    if total_trajectory_confident > 0 and top_animal["trajectory_fraction"] >= dominance_fraction_threshold:
        return {
            "localized": True,
            "reason": f"Trajectory-confident events are concentrated in animal {top_animal['animal']} ({top_animal['trajectory_count']}/{total_trajectory_confident}).",
        }
    if total_imm_confident > 0 and top_animal["imm_fraction"] >= dominance_fraction_threshold:
        return {
            "localized": True,
            "reason": f"IMM-confident events are concentrated in animal {top_animal['animal']} ({top_animal['imm_count']}/{total_imm_confident}).",
        }
    if len(positive_pairs) <= max(2, int(np.ceil(0.25 * len(pair_decision)))):
        pairs = "; ".join(
            f"{row.animal}/{row.date}" for row in positive_pairs.sort_values("max_delta_best_trajectory_minus_stationary", ascending=False).itertuples(index=False)
        )
        return {"localized": True, "reason": f"Only {len(positive_pairs)}/{len(pair_decision)} pairs show positive/confident signal: {pairs}."}
    return {"localized": False, "reason": "Positive signal is not strongly localized by pair or animal."}


def animal_signal_concentration(quality: pd.DataFrame, *, margin_threshold: float) -> dict[str, object]:
    if quality.empty or "animal" not in quality.columns:
        return {"animal": "", "trajectory_count": 0, "imm_count": 0, "trajectory_fraction": 0.0, "imm_fraction": 0.0}
    traj = quality.assign(
        _trajectory_conf=pd.to_numeric(quality["delta_best_trajectory_minus_stationary"], errors="coerce") >= float(margin_threshold),
        _imm_conf=pd.to_numeric(quality["delta_imm_minus_fragmented"], errors="coerce") >= float(margin_threshold),
    )
    counts = traj.groupby("animal", sort=True)[["_trajectory_conf", "_imm_conf"]].sum().reset_index()
    if counts.empty:
        return {"animal": "", "trajectory_count": 0, "imm_count": 0, "trajectory_fraction": 0.0, "imm_fraction": 0.0}
    counts["combined"] = counts["_trajectory_conf"] + counts["_imm_conf"]
    top = counts.sort_values(["combined", "_trajectory_conf", "_imm_conf"], ascending=False).iloc[0]
    total_traj = int(counts["_trajectory_conf"].sum())
    total_imm = int(counts["_imm_conf"].sum())
    return {
        "animal": str(top["animal"]),
        "trajectory_count": int(top["_trajectory_conf"]),
        "imm_count": int(top["_imm_conf"]),
        "trajectory_fraction": safe_fraction(int(top["_trajectory_conf"]), total_traj),
        "imm_fraction": safe_fraction(int(top["_imm_conf"]), total_imm),
    }


def build_markdown_summary(
    *,
    quality: pd.DataFrame,
    correlations: pd.DataFrame,
    event_strength: pd.DataFrame,
    decoder_quality: pd.DataFrame,
    pair_decision: pd.DataFrame,
    recommendation: pd.DataFrame,
    margin_threshold: float,
) -> str:
    del event_strength, decoder_quality
    row = recommendation.iloc[0].to_dict() if not recommendation.empty else {"recommendation": "", "primary_reason": ""}
    top_corr = correlations.sort_values("expected_direction_adjusted_r", ascending=False).head(8) if not correlations.empty else pd.DataFrame()
    pair_focus = pair_decision.sort_values("max_delta_best_trajectory_minus_stationary", ascending=False).head(8) if not pair_decision.empty else pd.DataFrame()
    lines = [
        "# Olafsdottir 1D Next-Pilot Decision",
        "",
        "This decision helper only reads existing debug report and QC tables; it does not rescore events.",
        "",
        "## Recommendation",
        "",
        f"- recommendation: {row.get('recommendation', '')}",
        f"- reason: {row.get('primary_reason', '')}",
        f"- events: {len(quality)}",
        f"- median trajectory-minus-stationary: {float_or_nan(row.get('median_delta_best_trajectory_minus_stationary')):.6g}",
        f"- median IMM-minus-fragmented: {float_or_nan(row.get('median_delta_imm_minus_fragmented')):.6g}",
        f"- margin threshold: {margin_threshold:g}",
        "",
        "## Strongest Directional Correlations",
        "",
        dataframe_to_markdown(top_corr[["target", "predictor", "predictor_group", "n", "pearson_r", "expected_direction_adjusted_r"]] if not top_corr.empty else top_corr),
        "",
        "## Pair-Level Readout",
        "",
        dataframe_to_markdown(
            pair_focus[
                [
                    "animal",
                    "date",
                    "events",
                    "trajectory_confident_claims",
                    "imm_confident_wins",
                    "median_delta_best_trajectory_minus_stationary",
                    "median_delta_imm_minus_fragmented",
                    "pair_debug_status",
                ]
            ]
            if not pair_focus.empty
            else pair_focus
        ),
        "",
        "## Claim Boundary",
        "",
        "This output chooses the next debug action only. It does not support pilot_50 biology, 1D-vs-2D comparison, or cross-dataset generalization.",
        "",
    ]
    return "\n".join(lines) + "\n"


def predictor_group(predictor: str) -> str:
    if predictor in EVENT_STRENGTH_PREDICTORS:
        return "event_strength"
    if predictor in DECODER_QUALITY_PREDICTORS:
        return "decoder_quality"
    return "other"


def expected_direction_adjusted_r(predictor: str, r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    return -float(r) if predictor in LOWER_IS_BETTER else float(r)


def assign_quantile_labels(values: pd.Series, *, lower_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        quantiles = pd.qcut(numeric.rank(method="first"), q=min(3, int(numeric.notna().sum())), labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=values.index, dtype=object)
    labels = []
    for value in quantiles:
        if pd.isna(value):
            labels.append(np.nan)
            continue
        int_value = int(value)
        if lower_is_better:
            label = ["high_quality", "middle", "low_quality"][min(int_value, 2)]
        else:
            label = ["low", "middle", "high"][min(int_value, 2)]
        labels.append(label)
    return pd.Series(labels, index=values.index, dtype=object)


def pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])


def finite_values(values: Iterable[object]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.mean(finite)) if finite.size else np.nan


def finite_median(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.median(finite)) if finite.size else np.nan


def finite_max(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.max(finite)) if finite.size else np.nan


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def dedupe(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            out.append(column)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, help="Existing Olafsdottir evidence debug report directory")
    parser.add_argument("--event-qc", help="Optional SleepPOST candidate event QC CSV")
    parser.add_argument("--decoder-qc", help="Optional Track1 decoder QC CSV")
    parser.add_argument("--pilot-selection", help="Optional frozen pilot selection CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for decision outputs")
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--correlation-threshold", type=float, default=0.35)
    parser.add_argument("--dominance-fraction-threshold", type=float, default=0.60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_next_pilot_decision(
        report_dir=args.report_dir,
        event_qc=args.event_qc,
        decoder_qc=args.decoder_qc,
        pilot_selection=args.pilot_selection,
        output_dir=args.output_dir,
        margin_threshold=args.margin_threshold,
        correlation_threshold=args.correlation_threshold,
        dominance_fraction_threshold=args.dominance_fraction_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
