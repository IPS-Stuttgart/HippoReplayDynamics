#!/usr/bin/env python3
"""Trajectory-IMM model-superiority and interpretability diagnostics.

This script is intentionally a referee, not a new model implementation.  It
turns "Better claim #4" into explicit promotion criteria for any future
trajectory-IMM variant:

* beat first-order IMM on paired full-core evidence;
* beat exact-sparse momentum as an interpretable trajectory-family extension;
* preserve rat-level and leave-one-rat-out robustness;
* expose mode posterior diagnostics that make the switching model interpretable.

The current trajectory-IMM row is expected to fail the first-order-IMM promotion
gate.  That failure is useful: it prevents paper claims from moving ahead of the
evidence while giving model-development PRs a stable target to optimize.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TRAJECTORY_IMM_MODEL = "sorted-spike-state-space-trajectory-imm-exact-sparse"
DEFAULT_FIRST_ORDER_IMM_MODEL = "sorted-spike-state-space-first-order-imm"
DEFAULT_MOMENTUM_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
DEFAULT_STATIONARY_MODEL = "sorted-spike-state-space-stationary"
DEFAULT_DIFFUSION_MODEL = "sorted-spike-state-space-diffusion"
DEFAULT_FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"
DEFAULT_MARGIN_THRESHOLD = 5.5
DEFAULT_RAT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED = 1
EXACT_EVIDENCE_SUPPORT = "exact_full_grid"
DEFAULT_REQUIRED_AUGMENTED_CORE_MODELS = (
    DEFAULT_STATIONARY_MODEL,
    DEFAULT_DIFFUSION_MODEL,
    DEFAULT_FRAGMENTED_MODEL,
    DEFAULT_FIRST_ORDER_IMM_MODEL,
    DEFAULT_MOMENTUM_MODEL,
    DEFAULT_TRAJECTORY_IMM_MODEL,
)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _as_models(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_REQUIRED_AUGMENTED_CORE_MODELS
    if isinstance(value, str):
        return tuple(part for part in re.split(r"[,\s]+", value.strip()) if part)
    return tuple(str(part) for part in value if str(part))


def _as_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "", "nan", "none"}:
        return False
    return default


def _bool_column(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(lambda value: _as_bool(value, default=default)).astype(bool)


def _success_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return successful model-evidence rows with normalized core columns."""

    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"model-evidence table is missing required columns: {missing}")

    out = frame.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).eq("success")].copy()
    out["session"] = out["session"].astype(str)
    out["rat"] = out["session"].map(_rat_from_session)
    out["event_index"] = out["event_index"].astype(int)
    out["model"] = out["model"].astype(str)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    out = out.dropna(subset=["log_evidence"]).copy()
    if "evidence_comparable" in out.columns:
        out["evidence_comparable"] = _bool_column(out, "evidence_comparable")
    elif "evidence_support" in out.columns:
        out["evidence_comparable"] = out["evidence_support"].astype(str).eq(EXACT_EVIDENCE_SUPPORT)
    else:
        out["evidence_comparable"] = True
    return out


def _value_for_model(group: pd.DataFrame, model: str) -> float:
    row = group[group["model"].astype(str).eq(str(model))]
    if row.empty:
        return float("nan")
    return float(row.sort_index().iloc[-1]["log_evidence"])


def _best_model(group: pd.DataFrame, models: set[str]) -> tuple[str, float]:
    subset = group[group["model"].isin(models)].dropna(subset=["log_evidence"])
    if subset.empty:
        return "", float("nan")
    row = subset.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0]
    return str(row["model"]), float(row["log_evidence"])


def _rank_among_models(group: pd.DataFrame, model: str, models: set[str]) -> int | float:
    subset = group[group["model"].isin(models)].dropna(subset=["log_evidence"]).copy()
    if subset.empty or model not in set(subset["model"]):
        return np.nan
    target = float(subset.loc[subset["model"].eq(model), "log_evidence"].iloc[-1])
    return int(1 + (subset["log_evidence"].astype(float) > target).sum())


def trajectory_imm_event_pairs(
    scores: pd.DataFrame,
    *,
    trajectory_imm_model: str = DEFAULT_TRAJECTORY_IMM_MODEL,
    first_order_imm_model: str = DEFAULT_FIRST_ORDER_IMM_MODEL,
    momentum_model: str = DEFAULT_MOMENTUM_MODEL,
    required_core_models: tuple[str, ...] = DEFAULT_REQUIRED_AUGMENTED_CORE_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Return paired event-level evidence comparisons for trajectory-IMM."""

    ok = _success_rows(scores)
    comparable = ok[_bool_column(ok, "evidence_comparable")].copy()
    required_core_models = _as_models(required_core_models)
    core_set = set(required_core_models)
    columns = [
        "rat",
        "session",
        "event_index",
        "required_core_models_present",
        "required_core_models_total",
        "required_core_complete",
        "missing_required_core_models",
        "trajectory_imm_model",
        "first_order_imm_model",
        "momentum_model",
        "trajectory_imm_log_evidence",
        "first_order_imm_log_evidence",
        "momentum_log_evidence",
        "stationary_log_evidence",
        "delta_trajectory_imm_minus_first_order_imm",
        "delta_trajectory_imm_minus_momentum",
        "delta_trajectory_imm_minus_stationary",
        "trajectory_imm_beats_first_order_imm",
        "trajectory_imm_beats_momentum",
        "trajectory_imm_confident_vs_first_order_imm",
        "first_order_imm_confident_vs_trajectory_imm",
        "trajectory_imm_ambiguous_vs_first_order_imm",
        "trajectory_imm_confident_vs_momentum",
        "momentum_confident_vs_trajectory_imm",
        "trajectory_imm_ambiguous_vs_momentum",
        "best_exact_core_model",
        "best_exact_core_log_evidence",
        "trajectory_imm_rank_in_required_core",
        "trajectory_imm_raw_best_exact_core",
        "trajectory_imm_confident_exact_core_claim",
        "first_order_imm_raw_best_exact_core",
        "momentum_raw_best_exact_core",
    ]
    rows: list[dict[str, object]] = []
    event_keys = ok[["session", "event_index", "rat"]].drop_duplicates().sort_values(["session", "event_index"])
    for _, event in event_keys.iterrows():
        session = str(event["session"])
        event_index = int(event["event_index"])
        group = comparable[
            comparable["session"].eq(session)
            & comparable["event_index"].eq(event_index)
        ].copy()
        present = set(group["model"].astype(str))
        missing = tuple(model for model in required_core_models if model not in present)
        trajectory_value = _value_for_model(group, trajectory_imm_model)
        first_order_value = _value_for_model(group, first_order_imm_model)
        momentum_value = _value_for_model(group, momentum_model)
        stationary_value = _value_for_model(group, DEFAULT_STATIONARY_MODEL)
        best_core_model, best_core_value = _best_model(group, core_set)
        rank = _rank_among_models(group, trajectory_imm_model, core_set)
        delta_first = trajectory_value - first_order_value
        delta_momentum = trajectory_value - momentum_value
        delta_stationary = trajectory_value - stationary_value
        rows.append(
            {
                "rat": str(event["rat"]),
                "session": session,
                "event_index": event_index,
                "required_core_models_present": int(len(core_set.intersection(present))),
                "required_core_models_total": int(len(core_set)),
                "required_core_complete": not missing,
                "missing_required_core_models": " ".join(missing),
                "trajectory_imm_model": trajectory_imm_model,
                "first_order_imm_model": first_order_imm_model,
                "momentum_model": momentum_model,
                "trajectory_imm_log_evidence": trajectory_value,
                "first_order_imm_log_evidence": first_order_value,
                "momentum_log_evidence": momentum_value,
                "stationary_log_evidence": stationary_value,
                "delta_trajectory_imm_minus_first_order_imm": delta_first,
                "delta_trajectory_imm_minus_momentum": delta_momentum,
                "delta_trajectory_imm_minus_stationary": delta_stationary,
                "trajectory_imm_beats_first_order_imm": bool(delta_first > 0.0),
                "trajectory_imm_beats_momentum": bool(delta_momentum > 0.0),
                "trajectory_imm_confident_vs_first_order_imm": bool(delta_first >= margin_threshold),
                "first_order_imm_confident_vs_trajectory_imm": bool(delta_first <= -margin_threshold),
                "trajectory_imm_ambiguous_vs_first_order_imm": bool(abs(delta_first) < margin_threshold),
                "trajectory_imm_confident_vs_momentum": bool(delta_momentum >= margin_threshold),
                "momentum_confident_vs_trajectory_imm": bool(delta_momentum <= -margin_threshold),
                "trajectory_imm_ambiguous_vs_momentum": bool(abs(delta_momentum) < margin_threshold),
                "best_exact_core_model": best_core_model,
                "best_exact_core_log_evidence": best_core_value,
                "trajectory_imm_rank_in_required_core": rank,
                "trajectory_imm_raw_best_exact_core": bool(best_core_model == trajectory_imm_model),
                "trajectory_imm_confident_exact_core_claim": bool(
                    best_core_model == trajectory_imm_model
                    and np.isfinite(rank)
                    and _second_best_gap(group, trajectory_imm_model, core_set) >= margin_threshold
                ),
                "first_order_imm_raw_best_exact_core": bool(best_core_model == first_order_imm_model),
                "momentum_raw_best_exact_core": bool(best_core_model == momentum_model),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _second_best_gap(group: pd.DataFrame, model: str, models: set[str]) -> float:
    subset = group[group["model"].isin(models)].dropna(subset=["log_evidence"]).copy()
    if subset.empty or model not in set(subset["model"]):
        return float("nan")
    target = float(subset.loc[subset["model"].eq(model), "log_evidence"].iloc[-1])
    others = subset[~subset["model"].eq(model)]["log_evidence"].astype(float)
    if others.empty:
        return float("nan")
    return float(target - others.max())


def _summarize_event_pairs(
    pairs: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    columns = [
        *group_cols,
        "events",
        "complete_core_events",
        "trajectory_imm_wins_vs_first_order_imm",
        "trajectory_imm_win_fraction_vs_first_order_imm",
        "trajectory_imm_confident_vs_first_order_imm",
        "first_order_imm_confident_vs_trajectory_imm",
        "ambiguous_vs_first_order_imm",
        "mean_delta_vs_first_order_imm",
        "median_delta_vs_first_order_imm",
        "min_delta_vs_first_order_imm",
        "trajectory_imm_wins_vs_momentum",
        "trajectory_imm_win_fraction_vs_momentum",
        "trajectory_imm_confident_vs_momentum",
        "momentum_confident_vs_trajectory_imm",
        "ambiguous_vs_momentum",
        "mean_delta_vs_momentum",
        "median_delta_vs_momentum",
        "trajectory_imm_raw_best_exact_core",
        "trajectory_imm_raw_best_exact_core_fraction",
        "trajectory_imm_confident_exact_core_claims",
        "first_order_imm_raw_best_exact_core",
        "momentum_raw_best_exact_core",
        "median_trajectory_imm_rank_in_required_core",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    groups = [((), pairs)] if not group_cols else pairs.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        complete = group[_bool_column(group, "required_core_complete")].copy()
        d_first = complete["delta_trajectory_imm_minus_first_order_imm"].astype(float).dropna()
        d_momentum = complete["delta_trajectory_imm_minus_momentum"].astype(float).dropna()
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": int(len(group)),
                "complete_core_events": int(len(complete)),
                "trajectory_imm_wins_vs_first_order_imm": int((d_first > 0.0).sum()),
                "trajectory_imm_win_fraction_vs_first_order_imm": (
                    float((d_first > 0.0).mean()) if not d_first.empty else 0.0
                ),
                "trajectory_imm_confident_vs_first_order_imm": int(
                    complete["trajectory_imm_confident_vs_first_order_imm"].fillna(False).sum()
                ),
                "first_order_imm_confident_vs_trajectory_imm": int(
                    complete["first_order_imm_confident_vs_trajectory_imm"].fillna(False).sum()
                ),
                "ambiguous_vs_first_order_imm": int(
                    complete["trajectory_imm_ambiguous_vs_first_order_imm"].fillna(False).sum()
                ),
                "mean_delta_vs_first_order_imm": float(d_first.mean()) if not d_first.empty else np.nan,
                "median_delta_vs_first_order_imm": float(d_first.median()) if not d_first.empty else np.nan,
                "min_delta_vs_first_order_imm": float(d_first.min()) if not d_first.empty else np.nan,
                "trajectory_imm_wins_vs_momentum": int((d_momentum > 0.0).sum()),
                "trajectory_imm_win_fraction_vs_momentum": (
                    float((d_momentum > 0.0).mean()) if not d_momentum.empty else 0.0
                ),
                "trajectory_imm_confident_vs_momentum": int(
                    complete["trajectory_imm_confident_vs_momentum"].fillna(False).sum()
                ),
                "momentum_confident_vs_trajectory_imm": int(
                    complete["momentum_confident_vs_trajectory_imm"].fillna(False).sum()
                ),
                "ambiguous_vs_momentum": int(complete["trajectory_imm_ambiguous_vs_momentum"].fillna(False).sum()),
                "mean_delta_vs_momentum": float(d_momentum.mean()) if not d_momentum.empty else np.nan,
                "median_delta_vs_momentum": float(d_momentum.median()) if not d_momentum.empty else np.nan,
                "trajectory_imm_raw_best_exact_core": int(complete["trajectory_imm_raw_best_exact_core"].sum()),
                "trajectory_imm_raw_best_exact_core_fraction": (
                    float(complete["trajectory_imm_raw_best_exact_core"].mean()) if not complete.empty else 0.0
                ),
                "trajectory_imm_confident_exact_core_claims": int(
                    complete["trajectory_imm_confident_exact_core_claim"].fillna(False).sum()
                ),
                "first_order_imm_raw_best_exact_core": int(complete["first_order_imm_raw_best_exact_core"].sum()),
                "momentum_raw_best_exact_core": int(complete["momentum_raw_best_exact_core"].sum()),
                "median_trajectory_imm_rank_in_required_core": (
                    float(complete["trajectory_imm_rank_in_required_core"].median()) if not complete.empty else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def trajectory_imm_superiority_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    return _summarize_event_pairs(pairs)


def rat_trajectory_imm_superiority_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    return _summarize_event_pairs(pairs, group_cols=("rat",))


def leave_one_rat_out_trajectory_imm_superiority_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for rat in sorted(pairs["rat"].dropna().astype(str).unique()):
        retained = pairs[pairs["rat"].astype(str) != rat]
        summary = _summarize_event_pairs(retained)
        if summary.empty:
            continue
        summary.insert(0, "held_out_rat", rat)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("held_out_rat").reset_index(drop=True)


def rat_bootstrap_trajectory_imm_superiority(
    pairs: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    columns = [
        "bootstrap_unit",
        "bootstrap_replicates",
        "random_seed",
        "observed_win_fraction_vs_first_order_imm",
        "win_fraction_vs_first_order_imm_ci95_low",
        "win_fraction_vs_first_order_imm_ci95_high",
        "observed_mean_delta_vs_first_order_imm",
        "mean_delta_vs_first_order_imm_ci95_low",
        "mean_delta_vs_first_order_imm_ci95_high",
        "observed_median_delta_vs_first_order_imm",
        "median_delta_vs_first_order_imm_ci95_low",
        "median_delta_vs_first_order_imm_ci95_high",
        "probability_mean_delta_vs_first_order_imm_positive",
        "probability_median_delta_vs_first_order_imm_positive",
        "observed_raw_best_exact_core_fraction",
        "raw_best_exact_core_fraction_ci95_low",
        "raw_best_exact_core_fraction_ci95_high",
    ]
    complete = pairs[_bool_column(pairs, "required_core_complete")].copy()
    if complete.empty:
        return pd.DataFrame(columns=columns)
    rats = sorted(complete["rat"].dropna().astype(str).unique())
    if not rats:
        return pd.DataFrame(columns=columns)

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        summary = _summarize_event_pairs(frame).iloc[0]
        return {
            "win_fraction": float(summary["trajectory_imm_win_fraction_vs_first_order_imm"]),
            "mean_delta": float(summary["mean_delta_vs_first_order_imm"]),
            "median_delta": float(summary["median_delta_vs_first_order_imm"]),
            "raw_best_fraction": float(summary["trajectory_imm_raw_best_exact_core_fraction"]),
        }

    observed = metrics(complete)
    rng = np.random.default_rng(int(random_seed))
    sampled_metrics: list[dict[str, float]] = []
    for _ in range(int(n_bootstrap)):
        sampled_rats = rng.choice(rats, size=len(rats), replace=True)
        sampled_frames = []
        for sample_index, rat in enumerate(sampled_rats):
            sample = complete[complete["rat"].astype(str).eq(str(rat))].copy()
            sample["_bootstrap_rat"] = f"{sample_index}:{rat}"
            sampled_frames.append(sample)
        sampled_metrics.append(metrics(pd.concat(sampled_frames, ignore_index=True)))
    boot = pd.DataFrame(sampled_metrics)
    return pd.DataFrame(
        [
            {
                "bootstrap_unit": "rat",
                "bootstrap_replicates": int(n_bootstrap),
                "random_seed": int(random_seed),
                "observed_win_fraction_vs_first_order_imm": observed["win_fraction"],
                "win_fraction_vs_first_order_imm_ci95_low": float(boot["win_fraction"].quantile(0.025)),
                "win_fraction_vs_first_order_imm_ci95_high": float(boot["win_fraction"].quantile(0.975)),
                "observed_mean_delta_vs_first_order_imm": observed["mean_delta"],
                "mean_delta_vs_first_order_imm_ci95_low": float(boot["mean_delta"].quantile(0.025)),
                "mean_delta_vs_first_order_imm_ci95_high": float(boot["mean_delta"].quantile(0.975)),
                "observed_median_delta_vs_first_order_imm": observed["median_delta"],
                "median_delta_vs_first_order_imm_ci95_low": float(boot["median_delta"].quantile(0.025)),
                "median_delta_vs_first_order_imm_ci95_high": float(boot["median_delta"].quantile(0.975)),
                "probability_mean_delta_vs_first_order_imm_positive": float((boot["mean_delta"] > 0.0).mean()),
                "probability_median_delta_vs_first_order_imm_positive": float((boot["median_delta"] > 0.0).mean()),
                "observed_raw_best_exact_core_fraction": observed["raw_best_fraction"],
                "raw_best_exact_core_fraction_ci95_low": float(boot["raw_best_fraction"].quantile(0.025)),
                "raw_best_exact_core_fraction_ci95_high": float(boot["raw_best_fraction"].quantile(0.975)),
            }
        ],
        columns=columns,
    )


def _mode_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric trajectory-IMM mode-diagnostic columns if present."""

    mode_pattern = re.compile(r"(mode|mode_mass|mode_probability|mode_posterior|switch|stickiness)", re.I)
    excluded = {"model", "requested_model", "model_family", "diagnostic_state_space_mode"}
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if not mode_pattern.search(str(column)):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            columns.append(str(column))
    return sorted(columns)


def trajectory_imm_mode_readiness(
    scores: pd.DataFrame,
    *,
    trajectory_imm_model: str = DEFAULT_TRAJECTORY_IMM_MODEL,
) -> pd.DataFrame:
    """Summarize whether trajectory-IMM rows expose interpretable mode diagnostics."""

    ok = _success_rows(scores)
    rows = ok[
        ok["model"].astype(str).eq(str(trajectory_imm_model))
        & _bool_column(ok, "evidence_comparable")
    ].copy()
    if rows.empty:
        return pd.DataFrame(
            [
                {
                    "trajectory_imm_model": trajectory_imm_model,
                    "trajectory_imm_rows": 0,
                    "mode_diagnostic_columns": 0,
                    "mode_diagnostic_column_names": "",
                    "mode_diagnostics_present": False,
                    "mode_diagnostics_complete_fraction": 0.0,
                    "interpretability_ready": False,
                }
            ]
        )
    mode_columns = _mode_columns(rows)
    if not mode_columns:
        complete_fraction = 0.0
    else:
        complete_fraction = float(rows[mode_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).mean())
    return pd.DataFrame(
        [
            {
                "trajectory_imm_model": trajectory_imm_model,
                "trajectory_imm_rows": int(len(rows)),
                "mode_diagnostic_columns": int(len(mode_columns)),
                "mode_diagnostic_column_names": " ".join(mode_columns),
                "mode_diagnostics_present": bool(mode_columns),
                "mode_diagnostics_complete_fraction": complete_fraction,
                "interpretability_ready": bool(mode_columns and complete_fraction >= 0.95),
            }
        ]
    )


def trajectory_imm_promotion_gate_summary(
    pairs: pd.DataFrame,
    mode_readiness: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return pass/fail gates for promoting trajectory-IMM to paper-leading row."""

    columns = ["gate", "passed", "observed", "criterion", "details"]
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, details: str = "") -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "details": details,
            }
        )

    summary = trajectory_imm_superiority_summary(pairs)
    if summary.empty:
        add("complete_paired_events_present", False, 0, "complete trajectory-IMM paired events > 0")
    else:
        s = summary.iloc[0]
        add(
            "complete_paired_events_present",
            int(s["complete_core_events"]) > 0,
            int(s["complete_core_events"]),
            "complete required augmented-core events > 0",
        )
        add(
            "trajectory_imm_raw_win_majority_vs_first_order_imm",
            float(s["trajectory_imm_win_fraction_vs_first_order_imm"]) > 0.5,
            f"{float(s['trajectory_imm_win_fraction_vs_first_order_imm']):.6g}",
            "trajectory-IMM raw win fraction vs first-order IMM > 0.5",
        )
        add(
            "trajectory_imm_mean_delta_vs_first_order_imm_positive",
            float(s["mean_delta_vs_first_order_imm"]) > 0.0,
            f"{float(s['mean_delta_vs_first_order_imm']):.6g}",
            "mean trajectory-IMM minus first-order IMM evidence > 0",
        )
        add(
            "trajectory_imm_median_delta_vs_first_order_imm_positive",
            float(s["median_delta_vs_first_order_imm"]) > 0.0,
            f"{float(s['median_delta_vs_first_order_imm']):.6g}",
            "median trajectory-IMM minus first-order IMM evidence > 0",
        )
        add(
            "trajectory_imm_confident_claims_exceed_first_order_imm",
            int(s["trajectory_imm_confident_vs_first_order_imm"])
            > int(s["first_order_imm_confident_vs_trajectory_imm"]),
            f"{int(s['trajectory_imm_confident_vs_first_order_imm'])}/"
            f"{int(s['first_order_imm_confident_vs_trajectory_imm'])}",
            "trajectory-IMM confident claims vs first-order IMM exceed reverse confident claims",
        )
        add(
            "trajectory_imm_beats_exact_sparse_momentum_majority",
            float(s["trajectory_imm_win_fraction_vs_momentum"]) > 0.5,
            f"{float(s['trajectory_imm_win_fraction_vs_momentum']):.6g}",
            "trajectory-IMM raw win fraction vs exact-sparse momentum > 0.5",
        )

    rat = rat_trajectory_imm_superiority_summary(pairs)
    if rat.empty:
        add("all_rats_median_delta_vs_first_order_imm_positive", False, np.nan, "min rat median delta > 0")
    else:
        min_rat_median = float(rat["median_delta_vs_first_order_imm"].min())
        add(
            "all_rats_median_delta_vs_first_order_imm_positive",
            min_rat_median > 0.0,
            f"{min_rat_median:.6g}",
            "every rat median trajectory-IMM minus first-order IMM evidence > 0",
        )

    loo = leave_one_rat_out_trajectory_imm_superiority_summary(pairs)
    if loo.empty:
        add("leave_one_rat_out_median_delta_vs_first_order_imm_positive", False, np.nan, "min LOO median delta > 0")
    else:
        min_loo_median = float(loo["median_delta_vs_first_order_imm"].min())
        add(
            "leave_one_rat_out_median_delta_vs_first_order_imm_positive",
            min_loo_median > 0.0,
            f"{min_loo_median:.6g}",
            "every leave-one-rat-out median trajectory-IMM minus first-order IMM evidence > 0",
        )

    boot = rat_bootstrap_trajectory_imm_superiority(
        pairs,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    if boot.empty:
        add("rat_bootstrap_mean_delta_vs_first_order_imm_ci_positive", False, np.nan, "mean CI95 low > 0")
        add("rat_bootstrap_median_delta_vs_first_order_imm_ci_positive", False, np.nan, "median CI95 low > 0")
    else:
        b = boot.iloc[0]
        add(
            "rat_bootstrap_mean_delta_vs_first_order_imm_ci_positive",
            float(b["mean_delta_vs_first_order_imm_ci95_low"]) > 0.0,
            f"{float(b['mean_delta_vs_first_order_imm_ci95_low']):.6g}",
            "rat-bootstrap mean delta CI95 low > 0",
        )
        add(
            "rat_bootstrap_median_delta_vs_first_order_imm_ci_positive",
            float(b["median_delta_vs_first_order_imm_ci95_low"]) > 0.0,
            f"{float(b['median_delta_vs_first_order_imm_ci95_low']):.6g}",
            "rat-bootstrap median delta CI95 low > 0",
        )

    if mode_readiness.empty:
        add("trajectory_imm_mode_diagnostics_present", False, 0, "mode diagnostics present")
        add("trajectory_imm_mode_diagnostics_interpretability_ready", False, 0, "mode diagnostics complete")
    else:
        m = mode_readiness.iloc[0]
        add(
            "trajectory_imm_mode_diagnostics_present",
            _as_bool(m["mode_diagnostics_present"]),
            int(m["mode_diagnostic_columns"]),
            "trajectory-IMM output exposes numeric mode diagnostics",
            str(m.get("mode_diagnostic_column_names", "")),
        )
        add(
            "trajectory_imm_mode_diagnostics_interpretability_ready",
            _as_bool(m["interpretability_ready"]),
            f"{float(m['mode_diagnostics_complete_fraction']):.6g}",
            "mode diagnostics complete fraction >= 0.95",
        )

    result = pd.DataFrame(rows, columns=columns)
    overall = pd.DataFrame(
        [
            {
                "gate": "overall",
                "passed": bool(result["passed"].all()) if not result.empty else False,
                "observed": f"{int(result['passed'].sum())}/{len(result)} gates passed" if not result.empty else "0/0",
                "criterion": "all trajectory-IMM promotion gates pass",
                "details": "If false, keep trajectory-IMM exploratory instead of paper-leading.",
            }
        ],
        columns=columns,
    )
    return pd.concat([result, overall], ignore_index=True)


def write_trajectory_imm_superiority_outputs(
    scores: pd.DataFrame,
    output: str | Path,
    *,
    trajectory_imm_model: str = DEFAULT_TRAJECTORY_IMM_MODEL,
    first_order_imm_model: str = DEFAULT_FIRST_ORDER_IMM_MODEL,
    momentum_model: str = DEFAULT_MOMENTUM_MODEL,
    required_core_models: tuple[str, ...] = DEFAULT_REQUIRED_AUGMENTED_CORE_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pairs = trajectory_imm_event_pairs(
        scores,
        trajectory_imm_model=trajectory_imm_model,
        first_order_imm_model=first_order_imm_model,
        momentum_model=momentum_model,
        required_core_models=required_core_models,
        margin_threshold=margin_threshold,
    )
    mode = trajectory_imm_mode_readiness(scores, trajectory_imm_model=trajectory_imm_model)
    outputs = {
        "trajectory_imm_superiority_event_pairs.csv": pairs,
        "trajectory_imm_superiority_summary.csv": trajectory_imm_superiority_summary(pairs),
        "rat_trajectory_imm_superiority_summary.csv": rat_trajectory_imm_superiority_summary(pairs),
        "leave_one_rat_out_trajectory_imm_superiority_summary.csv": (
            leave_one_rat_out_trajectory_imm_superiority_summary(pairs)
        ),
        "rat_bootstrap_trajectory_imm_superiority.csv": rat_bootstrap_trajectory_imm_superiority(
            pairs,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        ),
        "trajectory_imm_mode_readiness.csv": mode,
        "trajectory_imm_promotion_gate_summary.csv": trajectory_imm_promotion_gate_summary(
            pairs,
            mode,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", required=True, help="all_sessions_event_model_evidence.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-imm-model", default=DEFAULT_TRAJECTORY_IMM_MODEL)
    parser.add_argument("--first-order-imm-model", default=DEFAULT_FIRST_ORDER_IMM_MODEL)
    parser.add_argument("--momentum-model", default=DEFAULT_MOMENTUM_MODEL)
    parser.add_argument(
        "--required-core-models",
        default=" ".join(DEFAULT_REQUIRED_AUGMENTED_CORE_MODELS),
        help="Whitespace or comma separated exact augmented-core model names.",
    )
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--rat-bootstrap-replicates", type=int, default=DEFAULT_RAT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--rat-bootstrap-random-seed", type=int, default=DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED)
    args = parser.parse_args()

    scores = pd.read_csv(args.event_model_evidence)
    write_trajectory_imm_superiority_outputs(
        scores,
        args.output,
        trajectory_imm_model=args.trajectory_imm_model,
        first_order_imm_model=args.first_order_imm_model,
        momentum_model=args.momentum_model,
        required_core_models=_as_models(args.required_core_models),
        margin_threshold=args.margin_threshold,
        n_bootstrap=args.rat_bootstrap_replicates,
        random_seed=args.rat_bootstrap_random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
