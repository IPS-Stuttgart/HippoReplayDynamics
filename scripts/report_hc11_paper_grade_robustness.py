#!/usr/bin/env python3
"""Build hc-11 robustness tables from an existing model-evidence smoke.

This script is intentionally non-rescoring. It consumes an hc-11 event-model
evidence CSV plus an optional clean-IMM time-order shuffle artifact and writes
the animal/session, leave-one-animal-out, bootstrap, IMM-vs-fragmented, and gate
tables needed to decide whether hc-11 can move beyond external smoke status.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_MODELS = ("stationary", "diffusion", "fragmented", "first_order_imm", "momentum")
MINIMAL_CORE_MODELS = ("stationary", "diffusion", "fragmented", "first_order_imm")
TRAJECTORY_MODELS = ("diffusion", "fragmented", "first_order_imm", "momentum")

EVENT_COLUMNS = [
    "animal",
    "session",
    "event_index",
    "logZ_stationary",
    "logZ_diffusion",
    "logZ_fragmented",
    "logZ_first_order_imm",
    "logZ_momentum",
    "minimal_core_complete",
    "missing_minimal_core_models",
    "models_scored",
    "best_model",
    "best_log_evidence",
    "best_trajectory_model",
    "best_trajectory_log_evidence",
    "delta_trajectory_minus_stationary",
    "trajectory_confident_claim",
    "stationary_confident_claim",
    "delta_imm_minus_fragmented",
    "imm_raw_win",
    "imm_confident_win_at_threshold",
    "fragmented_confident_win_at_threshold",
    "delta_momentum_minus_diffusion",
    "momentum_raw_win_vs_diffusion",
    "momentum_confident_win_vs_diffusion",
    "duration_ms",
    "n_spikes",
    "n_active_units",
]


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
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"", "0", "0.0", "false", "f", "no", "n", "nan", "none", "null", "off"}:
        return False
    return False


def _status_success(value: object) -> bool:
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    text = str(value).strip().lower()
    return text in {"", "success", "nan", "none", "null", "na", "n/a", "<na>"}


def _canonical_model(model: object) -> str:
    text = str(model).strip().lower()
    norm = text.replace("-", "_").replace(" ", "_")
    if "stationary" in norm:
        return "stationary"
    if "fragmented" in norm or "fragment" in norm:
        return "fragmented"
    if "first_order_imm" in norm or "firstorderimm" in norm:
        return "first_order_imm"
    if "diffusion" in norm or "brownian" in norm:
        return "diffusion"
    if "momentum" in norm or "constant_velocity" in norm:
        return "momentum"
    return norm


def _animal_from_row(row: pd.Series) -> str:
    for column in ("animal", "rat", "subject"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    session = str(row.get("session", ""))
    if "/" in session:
        return session.split("/", 1)[0]
    if "_" in session:
        return session.split("_", 1)[0]
    return session or "unknown"


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _finite_delta(left: object, right: object) -> float:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float("nan")
    return float(left_value - right_value)


def _finite_max(values: list[object]) -> tuple[str, float]:
    best_model = ""
    best_value = float("nan")
    for model, value in values:
        numeric = _safe_float(value)
        if not np.isfinite(numeric):
            continue
        if not best_model or numeric > best_value:
            best_model = str(model)
            best_value = numeric
    return best_model, best_value


def read_event_model_evidence(path: str | Path) -> pd.DataFrame:
    return normalize_event_model_evidence(pd.read_csv(path))


def normalize_event_model_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")
    frame = frame.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].map(_status_success)].copy()
    if "evidence_comparable" in frame.columns:
        frame = frame[frame["evidence_comparable"].map(_as_bool)].copy()
    frame["session"] = frame["session"].astype(str)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    frame["model"] = frame["model"].astype(str)
    frame["canonical_model"] = frame["model"].map(_canonical_model)
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    frame = frame.dropna(subset=["log_evidence"]).copy()
    if "animal" not in frame.columns:
        frame["animal"] = frame.apply(_animal_from_row, axis=1)
    else:
        frame["animal"] = frame.apply(_animal_from_row, axis=1)
    return frame


def build_event_table(evidence: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    if not {"animal", "canonical_model"}.issubset(evidence.columns):
        evidence = normalize_event_model_evidence(evidence)
    if evidence.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows: list[dict[str, object]] = []
    for (animal, session, event_index), group in evidence.groupby(
        ["animal", "session", "event_index"],
        sort=True,
    ):
        model_values: dict[str, float] = {}
        for model, model_group in group.groupby("canonical_model"):
            if model in CANONICAL_MODELS:
                model_values[model] = float(model_group["log_evidence"].max())
        missing_minimal = [model for model in MINIMAL_CORE_MODELS if model not in model_values]
        all_values = [(model, model_values.get(model, np.nan)) for model in CANONICAL_MODELS]
        best_model, best_logz = _finite_max(all_values)
        trajectory_values = [(model, model_values.get(model, np.nan)) for model in TRAJECTORY_MODELS]
        best_trajectory_model, best_trajectory_logz = _finite_max(trajectory_values)
        row = {
            "animal": str(animal),
            "session": str(session),
            "event_index": int(event_index),
            "logZ_stationary": model_values.get("stationary", np.nan),
            "logZ_diffusion": model_values.get("diffusion", np.nan),
            "logZ_fragmented": model_values.get("fragmented", np.nan),
            "logZ_first_order_imm": model_values.get("first_order_imm", np.nan),
            "logZ_momentum": model_values.get("momentum", np.nan),
            "minimal_core_complete": not missing_minimal,
            "missing_minimal_core_models": " ".join(missing_minimal),
            "models_scored": ";".join(model for model in CANONICAL_MODELS if model in model_values),
            "best_model": best_model,
            "best_log_evidence": best_logz,
            "best_trajectory_model": best_trajectory_model,
            "best_trajectory_log_evidence": best_trajectory_logz,
            "duration_ms": _first_present(group, ["duration_ms", "event_duration_ms", "duration_s"], multiplier={"duration_s": 1000.0}),
            "n_spikes": _first_present(group, ["n_spikes", "total_spikes", "event_spikes"]),
            "n_active_units": _first_present(group, ["n_active_units", "active_units", "n_active_cells"]),
        }
        row["delta_trajectory_minus_stationary"] = _finite_delta(
            row["best_trajectory_log_evidence"],
            row["logZ_stationary"],
        )
        row["trajectory_confident_claim"] = bool(row["delta_trajectory_minus_stationary"] >= margin_threshold)
        row["stationary_confident_claim"] = bool(row["delta_trajectory_minus_stationary"] <= -margin_threshold)
        row["delta_imm_minus_fragmented"] = _finite_delta(row["logZ_first_order_imm"], row["logZ_fragmented"])
        row["imm_raw_win"] = bool(row["delta_imm_minus_fragmented"] > 0.0) if np.isfinite(row["delta_imm_minus_fragmented"]) else False
        row["imm_confident_win_at_threshold"] = bool(row["delta_imm_minus_fragmented"] >= margin_threshold)
        row["fragmented_confident_win_at_threshold"] = bool(row["delta_imm_minus_fragmented"] <= -margin_threshold)
        row["delta_momentum_minus_diffusion"] = _finite_delta(row["logZ_momentum"], row["logZ_diffusion"])
        row["momentum_raw_win_vs_diffusion"] = bool(row["delta_momentum_minus_diffusion"] > 0.0) if np.isfinite(row["delta_momentum_minus_diffusion"]) else False
        row["momentum_confident_win_vs_diffusion"] = bool(row["delta_momentum_minus_diffusion"] >= margin_threshold)
        rows.append(row)

    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values(["animal", "session", "event_index"]).reset_index(drop=True)


def _first_present(group: pd.DataFrame, columns: list[str], multiplier: dict[str, float] | None = None) -> float:
    multiplier = multiplier or {}
    for column in columns:
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0]) * multiplier.get(column, 1.0)
    return float("nan")


def summarize_events(events: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    columns = (group_cols or []) + [
        "events",
        "trajectory_confident_count",
        "trajectory_confident_fraction",
        "stationary_confident_count",
        "mean_trajectory_minus_stationary",
        "median_trajectory_minus_stationary",
        "stationary_best_count",
        "diffusion_best_count",
        "fragmented_best_count",
        "momentum_best_count",
        "first_order_imm_best_count",
        "imm_raw_wins",
        "imm_confident_wins",
        "fragmented_confident_wins",
        "median_delta_imm_minus_fragmented",
        "momentum_raw_wins_vs_diffusion",
        "momentum_confident_wins_vs_diffusion",
        "median_delta_momentum_minus_diffusion",
        "minimal_core_complete_events",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    if group_cols:
        iterator = events.groupby(group_cols, sort=True)
    else:
        iterator = [((), events)]
    rows: list[dict[str, object]] = []
    for keys, group in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols or [], keys, strict=True))
        row.update(_summary_values(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _summary_values(group: pd.DataFrame) -> dict[str, object]:
    events = len(group)
    family_margin = pd.to_numeric(group["delta_trajectory_minus_stationary"], errors="coerce")
    imm_delta = pd.to_numeric(group["delta_imm_minus_fragmented"], errors="coerce")
    mom_delta = pd.to_numeric(group["delta_momentum_minus_diffusion"], errors="coerce")
    best_counts = group["best_model"].value_counts()
    return {
        "events": events,
        "trajectory_confident_count": int(group["trajectory_confident_claim"].sum()),
        "trajectory_confident_fraction": float(group["trajectory_confident_claim"].mean()) if events else np.nan,
        "stationary_confident_count": int(group["stationary_confident_claim"].sum()),
        "mean_trajectory_minus_stationary": float(family_margin.mean()) if family_margin.notna().any() else np.nan,
        "median_trajectory_minus_stationary": float(family_margin.median()) if family_margin.notna().any() else np.nan,
        "stationary_best_count": int(best_counts.get("stationary", 0)),
        "diffusion_best_count": int(best_counts.get("diffusion", 0)),
        "fragmented_best_count": int(best_counts.get("fragmented", 0)),
        "momentum_best_count": int(best_counts.get("momentum", 0)),
        "first_order_imm_best_count": int(best_counts.get("first_order_imm", 0)),
        "imm_raw_wins": int(group["imm_raw_win"].sum()),
        "imm_confident_wins": int(group["imm_confident_win_at_threshold"].sum()),
        "fragmented_confident_wins": int(group["fragmented_confident_win_at_threshold"].sum()),
        "median_delta_imm_minus_fragmented": float(imm_delta.median()) if imm_delta.notna().any() else np.nan,
        "momentum_raw_wins_vs_diffusion": int(group["momentum_raw_win_vs_diffusion"].sum()),
        "momentum_confident_wins_vs_diffusion": int(group["momentum_confident_win_vs_diffusion"].sum()),
        "median_delta_momentum_minus_diffusion": float(mom_delta.median()) if mom_delta.notna().any() else np.nan,
        "minimal_core_complete_events": int(group["minimal_core_complete"].sum()),
    }


def build_leave_one_animal_out(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "held_out_animal",
        "retained_animals",
        "retained_sessions",
        *[column for column in summarize_events(events).columns if column not in {"animal"}],
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for animal in sorted(events["animal"].unique()):
        retained = events[~events["animal"].eq(animal)].copy()
        row = {
            "held_out_animal": animal,
            "retained_animals": retained["animal"].nunique(),
            "retained_sessions": retained["session"].nunique(),
        }
        row.update(_summary_values(retained) if not retained.empty else _empty_summary_values())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _empty_summary_values() -> dict[str, object]:
    return _summary_values(pd.DataFrame(columns=EVENT_COLUMNS))


def build_animal_cluster_bootstrap(
    events: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    columns = ["metric", "observed", "ci95_low", "ci95_high", "bootstrap_mean", "probability_positive_or_majority", "n_bootstrap", "animals"]
    if events.empty or events["animal"].nunique() == 0:
        return pd.DataFrame(columns=columns)
    animals = sorted(events["animal"].unique())
    by_animal = {animal: events[events["animal"].eq(animal)].copy() for animal in animals}
    rng = np.random.default_rng(seed)

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        margin = pd.to_numeric(frame["delta_trajectory_minus_stationary"], errors="coerce").dropna()
        return {
            "trajectory_confident_fraction": float(frame["trajectory_confident_claim"].mean()) if len(frame) else np.nan,
            "mean_trajectory_minus_stationary": float(margin.mean()) if not margin.empty else np.nan,
            "median_trajectory_minus_stationary": float(margin.median()) if not margin.empty else np.nan,
            "imm_confident_fraction": float(frame["imm_confident_win_at_threshold"].mean()) if len(frame) else np.nan,
        }

    observed = metrics(events)
    boot: dict[str, list[float]] = {metric: [] for metric in observed}
    for _ in range(n_bootstrap):
        sampled = rng.choice(animals, size=len(animals), replace=True)
        sample = pd.concat([by_animal[animal] for animal in sampled], ignore_index=True)
        values = metrics(sample)
        for metric, value in values.items():
            if np.isfinite(value):
                boot[metric].append(float(value))

    rows = []
    for metric, values in boot.items():
        arr = np.asarray(values, dtype=float)
        if arr.size:
            low = float(np.quantile(arr, 0.025))
            high = float(np.quantile(arr, 0.975))
            mean = float(arr.mean())
            if metric.endswith("_fraction"):
                probability = float(np.mean(arr > 0.5))
            else:
                probability = float(np.mean(arr > 0.0))
        else:
            low = high = mean = probability = np.nan
        rows.append(
            {
                "metric": metric,
                "observed": observed[metric],
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_mean": mean,
                "probability_positive_or_majority": probability,
                "n_bootstrap": n_bootstrap,
                "animals": len(animals),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_imm_vs_fragmented_audit(events: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "session",
                "event_index",
                "logZ_first_order_imm",
                "logZ_fragmented",
                "delta_imm_minus_fragmented",
                "imm_raw_win",
                "imm_confident_win_at_threshold",
                "fragmented_confident_win_at_threshold",
                "best_model",
                "within_family_classification",
                "n_spikes",
                "duration_ms",
            ]
        )
    out = events[
        [
            "animal",
            "session",
            "event_index",
            "logZ_first_order_imm",
            "logZ_fragmented",
            "delta_imm_minus_fragmented",
            "imm_raw_win",
            "imm_confident_win_at_threshold",
            "fragmented_confident_win_at_threshold",
            "best_model",
            "n_spikes",
            "duration_ms",
        ]
    ].copy()
    out["within_family_classification"] = out.apply(
        lambda row: _within_family_classification(row, margin_threshold),
        axis=1,
    )
    return out[
        [
            "animal",
            "session",
            "event_index",
            "logZ_first_order_imm",
            "logZ_fragmented",
            "delta_imm_minus_fragmented",
            "imm_raw_win",
            "imm_confident_win_at_threshold",
            "fragmented_confident_win_at_threshold",
            "best_model",
            "within_family_classification",
            "n_spikes",
            "duration_ms",
        ]
    ]


def _within_family_classification(row: pd.Series, threshold: float) -> str:
    delta = _safe_float(row.get("delta_imm_minus_fragmented"))
    if str(row.get("best_model")) == "first_order_imm" and np.isfinite(delta) and delta >= threshold:
        return "clean_imm_candidate"
    if str(row.get("best_model")) == "fragmented" and np.isfinite(delta) and delta <= -threshold:
        return "fragmented_candidate"
    if np.isfinite(delta) and abs(delta) < threshold:
        return "imm_fragmented_ambiguous"
    return "trajectory_family_other"


def build_time_order_summary(path: str | Path | None) -> pd.DataFrame:
    columns = [
        "status",
        "events",
        "clean_imm_events",
        "median_time_order_advantage",
        "clean_imm_original_above_shuffle_median_fraction",
        "clean_imm_original_above_shuffle_p95_count",
        "time_order_gate_passed",
        "source",
        "note",
    ]
    if not path:
        return pd.DataFrame(
            [
                {
                    "status": "not_run",
                    "events": 0,
                    "clean_imm_events": 0,
                    "median_time_order_advantage": np.nan,
                    "clean_imm_original_above_shuffle_median_fraction": np.nan,
                    "clean_imm_original_above_shuffle_p95_count": 0,
                    "time_order_gate_passed": False,
                    "source": "",
                    "note": "No --time-order-shuffle-decisions artifact was provided.",
                }
            ],
            columns=columns,
        )
    decisions = pd.read_csv(path)
    required = {"event_group", "time_order_advantage", "original_above_shuffle_median", "original_above_shuffle_p95"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"time-order shuffle decisions are missing required columns: {missing}")
    clean = decisions[decisions["event_group"].astype(str).eq("clean_imm")].copy()
    clean_fraction = float(clean["original_above_shuffle_median"].map(_as_bool).mean()) if len(clean) else np.nan
    clean_p95 = int(clean["original_above_shuffle_p95"].map(_as_bool).sum()) if len(clean) else 0
    median_advantage = _safe_float(pd.to_numeric(clean["time_order_advantage"], errors="coerce").median()) if len(clean) else np.nan
    gate_passed = bool(np.isfinite(median_advantage) and median_advantage > 0.0 and np.isfinite(clean_fraction) and clean_fraction > 0.60 and clean_p95 > 0)
    return pd.DataFrame(
        [
            {
                "status": "provided",
                "events": len(decisions),
                "clean_imm_events": len(clean),
                "median_time_order_advantage": median_advantage,
                "clean_imm_original_above_shuffle_median_fraction": clean_fraction,
                "clean_imm_original_above_shuffle_p95_count": clean_p95,
                "time_order_gate_passed": gate_passed,
                "source": str(path),
                "note": "Summarized from provided clean-IMM time-order shuffle decisions.",
            }
        ],
        columns=columns,
    )


def build_gate_summary(
    events: pd.DataFrame,
    by_animal: pd.DataFrame,
    by_session: pd.DataFrame,
    leave_one: pd.DataFrame,
    bootstrap: pd.DataFrame,
    imm_audit: pd.DataFrame,
    time_order: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, gate_type: str = "technical") -> None:
        rows.append(
            {
                "gate": gate,
                "gate_type": gate_type,
                "passed": bool(passed),
                "status": "pass" if passed else "fail",
                "observed": observed,
                "criterion": criterion,
            }
        )

    add("event_rows_present", len(events) > 0, len(events), "at least one event")
    add("minimal_core_complete_all_events", bool(events["minimal_core_complete"].all()) if len(events) else False, f"{int(events['minimal_core_complete'].sum()) if len(events) else 0}/{len(events)}", "stationary/diffusion/fragmented/first-order IMM present for every event")
    add("stationary_comparator_present", bool(events["logZ_stationary"].notna().all()) if len(events) else False, int(events["logZ_stationary"].notna().sum()) if len(events) else 0, "stationary rows present for every event")
    add("fragmented_comparator_present", bool(events["logZ_fragmented"].notna().all()) if len(events) else False, int(events["logZ_fragmented"].notna().sum()) if len(events) else 0, "fragmented rows present for every event")
    add("first_order_imm_present", bool(events["logZ_first_order_imm"].notna().all()) if len(events) else False, int(events["logZ_first_order_imm"].notna().sum()) if len(events) else 0, "first-order IMM rows present for every event")
    add("momentum_axis_reported", bool(events["logZ_momentum"].notna().any()) if len(events) else False, int(events["logZ_momentum"].notna().sum()) if len(events) else 0, "momentum row is present for at least some events", gate_type="diagnostic")

    add("multiple_animals_represented", events["animal"].nunique() >= 2 if len(events) else False, events["animal"].nunique() if len(events) else 0, "at least two animals represented", gate_type="robustness")
    add("multiple_sessions_represented", events["session"].nunique() >= 2 if len(events) else False, events["session"].nunique() if len(events) else 0, "at least two sessions represented", gate_type="robustness")
    total_claims = int(events["trajectory_confident_claim"].sum()) if len(events) else 0
    if total_claims:
        max_animal_claim_share = float(by_animal["trajectory_confident_count"].max() / total_claims)
    else:
        max_animal_claim_share = np.nan
    add("trajectory_signal_not_single_animal_dominated", np.isfinite(max_animal_claim_share) and max_animal_claim_share <= 0.75, max_animal_claim_share, "no one animal has >75% of trajectory-confident claims", gate_type="robustness")
    add("overall_trajectory_confident_majority", float(events["trajectory_confident_claim"].mean()) > 0.5 if len(events) else False, float(events["trajectory_confident_claim"].mean()) if len(events) else np.nan, "trajectory-confident fraction > 0.5", gate_type="robustness")
    add("leave_one_animal_out_claim_majority", bool(leave_one["trajectory_confident_fraction"].gt(0.5).all()) if len(leave_one) else False, f"{int(leave_one['trajectory_confident_fraction'].gt(0.5).sum()) if len(leave_one) else 0}/{len(leave_one)}", "each leave-one-animal-out subset has trajectory-confident majority", gate_type="robustness")
    add("leave_one_animal_out_median_margin_positive", bool(leave_one["median_trajectory_minus_stationary"].gt(0).all()) if len(leave_one) else False, f"{int(leave_one['median_trajectory_minus_stationary'].gt(0).sum()) if len(leave_one) else 0}/{len(leave_one)}", "each leave-one-animal-out subset has positive median family margin", gate_type="robustness")

    claim_boot = _bootstrap_metric(bootstrap, "trajectory_confident_fraction")
    mean_boot = _bootstrap_metric(bootstrap, "mean_trajectory_minus_stationary")
    median_boot = _bootstrap_metric(bootstrap, "median_trajectory_minus_stationary")
    add("animal_bootstrap_claim_fraction_lower_above_half", bool(claim_boot.get("ci95_low", np.nan) > 0.5), claim_boot.get("ci95_low", np.nan), "animal-cluster bootstrap lower CI for claim fraction > 0.5", gate_type="robustness")
    add("animal_bootstrap_mean_margin_lower_positive", bool(mean_boot.get("ci95_low", np.nan) > 0.0), mean_boot.get("ci95_low", np.nan), "animal-cluster bootstrap mean-margin lower CI > 0", gate_type="robustness")
    add("animal_bootstrap_median_margin_lower_positive", bool(median_boot.get("ci95_low", np.nan) > 0.0), median_boot.get("ci95_low", np.nan), "animal-cluster bootstrap median-margin lower CI > 0", gate_type="robustness")

    add("imm_vs_fragmented_axis_present", bool(imm_audit["delta_imm_minus_fragmented"].notna().any()) if len(imm_audit) else False, int(imm_audit["delta_imm_minus_fragmented"].notna().sum()) if len(imm_audit) else 0, "IMM-vs-fragmented paired margins are available", gate_type="imm_audit")
    add("clean_imm_subset_present", bool(imm_audit["within_family_classification"].eq("clean_imm_candidate").any()) if len(imm_audit) else False, int(imm_audit["within_family_classification"].eq("clean_imm_candidate").sum()) if len(imm_audit) else 0, "at least one clean IMM candidate", gate_type="imm_audit")

    time_order_row = time_order.iloc[0] if len(time_order) else pd.Series(dtype=object)
    add("time_order_shuffle_artifact_present", str(time_order_row.get("status", "")) == "provided", time_order_row.get("status", ""), "hc-11 clean-IMM time-order shuffle decisions provided", gate_type="time_order")
    add("time_order_shuffle_clean_imm_gate_passed", _as_bool(time_order_row.get("time_order_gate_passed", False)), time_order_row.get("time_order_gate_passed", False), "clean IMM time-order gate passes", gate_type="time_order")

    technical_pass = all(row["passed"] for row in rows if row["gate_type"] == "technical")
    robustness_pass = all(row["passed"] for row in rows if row["gate_type"] == "robustness")
    imm_pass = all(row["passed"] for row in rows if row["gate_type"] == "imm_audit")
    time_order_pass = all(row["passed"] for row in rows if row["gate_type"] == "time_order")
    paper_grade = technical_pass and robustness_pass and imm_pass and time_order_pass
    rows.append({"gate": "technical_overall", "gate_type": "overall", "passed": technical_pass, "status": "pass" if technical_pass else "fail", "observed": technical_pass, "criterion": "all technical gates pass"})
    rows.append({"gate": "robustness_overall", "gate_type": "overall", "passed": robustness_pass, "status": "pass" if robustness_pass else "fail", "observed": robustness_pass, "criterion": "all spread/bootstrap robustness gates pass"})
    rows.append({"gate": "paper_grade_overall", "gate_type": "overall", "passed": paper_grade, "status": "pass" if paper_grade else "fail", "observed": f"technical={technical_pass}; robustness={robustness_pass}; imm={imm_pass}; time_order={time_order_pass}", "criterion": "technical, robustness, IMM, and time-order gates all pass"})
    return pd.DataFrame(rows)


def _bootstrap_metric(bootstrap: pd.DataFrame, metric: str) -> dict[str, float]:
    if bootstrap.empty:
        return {}
    rows = bootstrap[bootstrap["metric"].eq(metric)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {column: _safe_float(row[column]) for column in bootstrap.columns if column not in {"metric"}}


def write_outputs(
    evidence: pd.DataFrame,
    output: str | Path,
    *,
    margin_threshold: float,
    n_bootstrap: int,
    seed: int,
    time_order_shuffle_decisions: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    events = build_event_table(evidence, margin_threshold=margin_threshold)
    by_animal = summarize_events(events, ["animal"])
    by_session = summarize_events(events, ["animal", "session"])
    leave_one = build_leave_one_animal_out(events)
    bootstrap = build_animal_cluster_bootstrap(events, n_bootstrap=n_bootstrap, seed=seed)
    imm_audit = build_imm_vs_fragmented_audit(events, margin_threshold=margin_threshold)
    time_order = build_time_order_summary(time_order_shuffle_decisions)
    gates = build_gate_summary(events, by_animal, by_session, leave_one, bootstrap, imm_audit, time_order)

    outputs = {
        "hc11_event_claim_table.csv": events,
        "hc11_by_animal_summary.csv": by_animal,
        "hc11_by_session_summary.csv": by_session,
        "hc11_leave_one_animal_out_summary.csv": leave_one,
        "hc11_animal_cluster_bootstrap.csv": bootstrap,
        "hc11_imm_vs_fragmented_audit.csv": imm_audit,
        "hc11_time_order_shuffle_clean_imm.csv": time_order,
        "hc11_gate_summary.csv": gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    rat_dir = out / "hc11_rat"
    rat_dir.mkdir(exist_ok=True)
    bootstrap.to_csv(rat_dir / "animal_cluster_bootstrap.csv", index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", required=True, help="Existing hc-11 event-model evidence CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--time-order-shuffle-decisions",
        default="",
        help="Optional clean_imm_time_order_shuffle_decisions.csv for hc-11 clean IMM events",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = read_event_model_evidence(args.event_model_evidence)
    write_outputs(
        evidence,
        args.output_dir,
        margin_threshold=args.margin_threshold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        time_order_shuffle_decisions=args.time_order_shuffle_decisions or None,
    )
    print(f"Wrote hc-11 robustness report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
