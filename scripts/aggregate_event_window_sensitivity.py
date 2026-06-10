#!/usr/bin/env python3
"""Aggregate all-session event-window sensitivity model-evidence shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_model_evidence_shards import aggregate as aggregate_shards

DEFAULT_MARGIN_THRESHOLD = 5.5
DEFAULT_REQUIRED_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)
DEFAULT_EXACT_TRAJECTORY_MODELS = (
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-exact-sparse",
)


def aggregate_event_window_sensitivity(
    shard_glob: str,
    outdir: Path,
    *,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    core_variant: str = "core",
) -> pd.DataFrame:
    """Aggregate shards and write event-window sensitivity tables."""

    combined = aggregate_shards(shard_glob, outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined = _with_window_variant(combined)
    combined.to_csv(outdir / "all_sessions_event_window_model_evidence.csv", index=False)

    decisions = event_window_family_margin_decisions(
        combined,
        margin_threshold=margin_threshold,
    )
    decisions.to_csv(outdir / "event_window_family_margin_decisions.csv", index=False)
    event_window_family_margin_summary(decisions).to_csv(
        outdir / "event_window_family_margin_summary.csv",
        index=False,
    )
    event_window_family_margin_summary(decisions, group_cols=("event_window_variant", "session")).to_csv(
        outdir / "session_event_window_family_margin_summary.csv",
        index=False,
    )
    event_window_model_summary(combined).to_csv(outdir / "event_window_model_summary.csv", index=False)
    event_window_comparison_to_core(decisions, core_variant=core_variant).to_csv(
        outdir / "event_window_comparison_to_core.csv",
        index=False,
    )
    event_window_observation_normalized_summary(decisions).to_csv(
        outdir / "event_window_observation_normalized_summary.csv",
        index=False,
    )
    event_window_spike_count_summary(decisions).to_csv(
        outdir / "event_window_spike_count_summary.csv",
        index=False,
    )
    event_window_core_matched_attenuation(decisions, core_variant=core_variant).to_csv(
        outdir / "event_window_core_matched_attenuation.csv",
        index=False,
    )
    event_window_control_gate_summary(decisions, core_variant=core_variant).to_csv(
        outdir / "event_window_control_gate_summary.csv",
        index=False,
    )
    event_window_control_gate_summary_v2(decisions, core_variant=core_variant).to_csv(
        outdir / "event_window_control_gate_summary_v2.csv",
        index=False,
    )
    return combined


def _with_window_variant(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "event_window_variant" not in out:
        if "window_index" in out:
            out["event_window_variant"] = "window_" + out["window_index"].astype(str)
        else:
            out["event_window_variant"] = "core"
    return out


def event_window_model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize model evidence separately for each event-window variant."""

    frame = _with_window_variant(frame)
    status_ok = frame["status"].eq("success") if "status" in frame else pd.Series(True, index=frame.index)
    ok = frame[status_ok].copy()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (variant, model), group in ok.groupby(["event_window_variant", "model"], sort=True):
        rows.append(
            {
                "event_window_variant": str(variant),
                "model": str(model),
                "rows": int(len(group)),
                "events": int(group[["session", "event_index"]].drop_duplicates().shape[0]),
                "wins": int(_bool_column(group, "is_best_model").sum()),
                "mean_log_evidence": float(group["log_evidence"].astype(float).mean()),
                "median_log_evidence": float(group["log_evidence"].astype(float).median()),
                "mean_relative_log_evidence": float(
                    _numeric_column(group, "relative_log_evidence").mean()
                ),
                "median_relative_log_evidence": float(
                    _numeric_column(group, "relative_log_evidence").median()
                ),
                "mean_runtime_s": float(_numeric_column(group, "runtime_s").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_window_variant", "wins", "mean_log_evidence"], ascending=[True, False, False])


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


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


def _first_numeric_value(frame: pd.DataFrame, column: str) -> float:
    values = _numeric_column(frame, column).dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denom


def event_window_family_margin_decisions(
    frame: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_REQUIRED_MODELS,
    trajectory_models: tuple[str, ...] = DEFAULT_EXACT_TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Return best exact trajectory versus nontrajectory decisions per window."""

    frame = _with_window_variant(frame)
    required = tuple(str(model) for model in required_models)
    required_set = set(required)
    trajectory_set = set(str(model) for model in trajectory_models)
    status_ok = frame["status"].eq("success") if "status" in frame else pd.Series(True, index=frame.index)
    comparable = _bool_column(frame, "evidence_comparable", default=True)
    ok = frame[status_ok & comparable].copy()
    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(["session", "event_index", "event_window_variant"], sort=True):
        session, event_index, variant = key
        n_spikes = _first_numeric_value(group, "n_spikes")
        n_time = _first_numeric_value(group, "n_time")
        core = group[group["model"].astype(str).isin(required_set)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        complete = not missing
        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        if trajectory.empty or nontrajectory.empty:
            best_trajectory_model = ""
            best_trajectory_log_evidence = np.nan
            best_nontrajectory_model = ""
            best_nontrajectory_log_evidence = np.nan
            delta = np.nan
            raw_win = False
            trajectory_claim = False
            nontrajectory_claim = False
            decision = "incomplete_core"
        else:
            best_trajectory = trajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_nontrajectory = nontrajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_trajectory_log_evidence = float(best_trajectory["log_evidence"])
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_nontrajectory_log_evidence = float(best_nontrajectory["log_evidence"])
            delta = best_trajectory_log_evidence - best_nontrajectory_log_evidence
            raw_win = bool(delta > 0.0)
            trajectory_claim = bool(complete and delta >= float(margin_threshold))
            nontrajectory_claim = bool(complete and delta <= -float(margin_threshold))
            if not complete:
                decision = "incomplete_core"
            elif trajectory_claim:
                decision = "trajectory"
            elif nontrajectory_claim:
                decision = "nontrajectory"
            else:
                decision = "ambiguous"
        rows.append(
            {
                "session": str(session),
                "event_index": int(event_index),
                "event_window_variant": str(variant),
                "n_spikes": n_spikes,
                "n_time": n_time,
                "required_models_present": int(len(present)),
                "required_models_total": int(len(required)),
                "required_models_complete": bool(complete),
                "missing_required_models": " ".join(missing),
                "present_required_models": " ".join(present),
                "margin_threshold": float(margin_threshold),
                "best_trajectory_model": best_trajectory_model,
                "best_trajectory_log_evidence": best_trajectory_log_evidence,
                "best_nontrajectory_model": best_nontrajectory_model,
                "best_nontrajectory_log_evidence": best_nontrajectory_log_evidence,
                "trajectory_minus_nontrajectory_log_evidence": delta,
                "trajectory_raw_win": raw_win,
                "trajectory_confident_claim": trajectory_claim,
                "nontrajectory_confident_claim": nontrajectory_claim,
                "margin_decision": decision,
            }
        )
    return pd.DataFrame(rows)


def _with_normalized_decision_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    out = decisions.copy()
    if out.empty:
        return out
    out["best_trajectory_log_evidence_per_time_bin"] = _safe_divide(
        out["best_trajectory_log_evidence"], out["n_time"]
    )
    out["best_trajectory_log_evidence_per_spike"] = _safe_divide(
        out["best_trajectory_log_evidence"], out["n_spikes"]
    )
    out["trajectory_minus_nontrajectory_log_evidence_per_time_bin"] = _safe_divide(
        out["trajectory_minus_nontrajectory_log_evidence"], out["n_time"]
    )
    out["trajectory_minus_nontrajectory_log_evidence_per_spike"] = _safe_divide(
        out["trajectory_minus_nontrajectory_log_evidence"], out["n_spikes"]
    )
    return out


def event_window_observation_normalized_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-window evidence after normalizing by observation size."""

    columns = [
        "event_window_variant",
        "events",
        "mean_n_spikes",
        "median_n_spikes",
        "mean_n_time",
        "median_n_time",
        "mean_best_trajectory_log_evidence_per_time_bin",
        "median_best_trajectory_log_evidence_per_time_bin",
        "mean_best_trajectory_log_evidence_per_spike",
        "median_best_trajectory_log_evidence_per_spike",
        "mean_family_margin_per_time_bin",
        "median_family_margin_per_time_bin",
        "mean_family_margin_per_spike",
        "median_family_margin_per_spike",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    normalized = _with_normalized_decision_metrics(decisions)
    rows: list[dict[str, object]] = []
    for variant, group in normalized.groupby("event_window_variant", sort=True):
        rows.append(
            {
                "event_window_variant": str(variant),
                "events": int(len(group)),
                "mean_n_spikes": float(_numeric_column(group, "n_spikes").mean()),
                "median_n_spikes": float(_numeric_column(group, "n_spikes").median()),
                "mean_n_time": float(_numeric_column(group, "n_time").mean()),
                "median_n_time": float(_numeric_column(group, "n_time").median()),
                "mean_best_trajectory_log_evidence_per_time_bin": float(
                    _numeric_column(group, "best_trajectory_log_evidence_per_time_bin").mean()
                ),
                "median_best_trajectory_log_evidence_per_time_bin": float(
                    _numeric_column(group, "best_trajectory_log_evidence_per_time_bin").median()
                ),
                "mean_best_trajectory_log_evidence_per_spike": float(
                    _numeric_column(group, "best_trajectory_log_evidence_per_spike").mean()
                ),
                "median_best_trajectory_log_evidence_per_spike": float(
                    _numeric_column(group, "best_trajectory_log_evidence_per_spike").median()
                ),
                "mean_family_margin_per_time_bin": float(
                    _numeric_column(
                        group,
                        "trajectory_minus_nontrajectory_log_evidence_per_time_bin",
                    ).mean()
                ),
                "median_family_margin_per_time_bin": float(
                    _numeric_column(
                        group,
                        "trajectory_minus_nontrajectory_log_evidence_per_time_bin",
                    ).median()
                ),
                "mean_family_margin_per_spike": float(
                    _numeric_column(
                        group,
                        "trajectory_minus_nontrajectory_log_evidence_per_spike",
                    ).mean()
                ),
                "median_family_margin_per_spike": float(
                    _numeric_column(
                        group,
                        "trajectory_minus_nontrajectory_log_evidence_per_spike",
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def event_window_spike_count_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize observation counts by event-window variant."""

    columns = [
        "event_window_variant",
        "events",
        "mean_n_spikes",
        "median_n_spikes",
        "min_n_spikes",
        "max_n_spikes",
        "mean_n_time",
        "median_n_time",
        "min_n_time",
        "max_n_time",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for variant, group in decisions.groupby("event_window_variant", sort=True):
        n_spikes = _numeric_column(group, "n_spikes")
        n_time = _numeric_column(group, "n_time")
        rows.append(
            {
                "event_window_variant": str(variant),
                "events": int(len(group)),
                "mean_n_spikes": float(n_spikes.mean()),
                "median_n_spikes": float(n_spikes.median()),
                "min_n_spikes": float(n_spikes.min()),
                "max_n_spikes": float(n_spikes.max()),
                "mean_n_time": float(n_time.mean()),
                "median_n_time": float(n_time.median()),
                "min_n_time": float(n_time.min()),
                "max_n_time": float(n_time.max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def event_window_core_matched_attenuation(
    decisions: pd.DataFrame,
    *,
    core_variant: str = "core",
) -> pd.DataFrame:
    """Compare each window variant with core after matching events."""

    columns = [
        "event_window_variant",
        "core_variant",
        "matched_events",
        "mean_n_spikes_minus_core",
        "median_n_spikes_minus_core",
        "mean_n_time_minus_core",
        "median_n_time_minus_core",
        "mean_best_trajectory_log_evidence_minus_core",
        "median_best_trajectory_log_evidence_minus_core",
        "fraction_best_trajectory_log_evidence_below_core",
        "mean_best_trajectory_log_evidence_per_time_bin_minus_core",
        "median_best_trajectory_log_evidence_per_time_bin_minus_core",
        "fraction_best_trajectory_log_evidence_per_time_bin_below_core",
        "mean_best_trajectory_log_evidence_per_spike_minus_core",
        "median_best_trajectory_log_evidence_per_spike_minus_core",
        "fraction_best_trajectory_log_evidence_per_spike_below_core",
        "mean_family_margin_minus_core",
        "median_family_margin_minus_core",
        "fraction_family_margin_below_core",
        "mean_family_margin_per_time_bin_minus_core",
        "median_family_margin_per_time_bin_minus_core",
        "fraction_family_margin_per_time_bin_below_core",
        "mean_family_margin_per_spike_minus_core",
        "median_family_margin_per_spike_minus_core",
        "fraction_family_margin_per_spike_below_core",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    normalized = _with_normalized_decision_metrics(decisions)
    index_cols = ["session", "event_index"]
    core = normalized[normalized["event_window_variant"].astype(str).eq(core_variant)].set_index(index_cols)
    rows: list[dict[str, object]] = []
    for variant, group in normalized.groupby("event_window_variant", sort=True):
        current = group.set_index(index_cols)
        matched = current.join(core, lsuffix="_variant", rsuffix="_core", how="inner")
        if matched.empty:
            continue

        def delta(column: str) -> pd.Series:
            return (
                pd.to_numeric(matched[f"{column}_variant"], errors="coerce")
                - pd.to_numeric(matched[f"{column}_core"], errors="coerce")
            )

        def below_core(values: pd.Series) -> float:
            clean = values.dropna()
            return float((clean < 0.0).mean()) if not clean.empty else np.nan

        best_delta = delta("best_trajectory_log_evidence")
        best_per_time_delta = delta("best_trajectory_log_evidence_per_time_bin")
        best_per_spike_delta = delta("best_trajectory_log_evidence_per_spike")
        margin_delta = delta("trajectory_minus_nontrajectory_log_evidence")
        margin_per_time_delta = delta("trajectory_minus_nontrajectory_log_evidence_per_time_bin")
        margin_per_spike_delta = delta("trajectory_minus_nontrajectory_log_evidence_per_spike")
        n_spikes_delta = delta("n_spikes")
        n_time_delta = delta("n_time")
        rows.append(
            {
                "event_window_variant": str(variant),
                "core_variant": core_variant,
                "matched_events": int(len(matched)),
                "mean_n_spikes_minus_core": float(n_spikes_delta.mean()),
                "median_n_spikes_minus_core": float(n_spikes_delta.median()),
                "mean_n_time_minus_core": float(n_time_delta.mean()),
                "median_n_time_minus_core": float(n_time_delta.median()),
                "mean_best_trajectory_log_evidence_minus_core": float(best_delta.mean()),
                "median_best_trajectory_log_evidence_minus_core": float(best_delta.median()),
                "fraction_best_trajectory_log_evidence_below_core": below_core(best_delta),
                "mean_best_trajectory_log_evidence_per_time_bin_minus_core": float(
                    best_per_time_delta.mean()
                ),
                "median_best_trajectory_log_evidence_per_time_bin_minus_core": float(
                    best_per_time_delta.median()
                ),
                "fraction_best_trajectory_log_evidence_per_time_bin_below_core": below_core(
                    best_per_time_delta
                ),
                "mean_best_trajectory_log_evidence_per_spike_minus_core": float(
                    best_per_spike_delta.mean()
                ),
                "median_best_trajectory_log_evidence_per_spike_minus_core": float(
                    best_per_spike_delta.median()
                ),
                "fraction_best_trajectory_log_evidence_per_spike_below_core": below_core(
                    best_per_spike_delta
                ),
                "mean_family_margin_minus_core": float(margin_delta.mean()),
                "median_family_margin_minus_core": float(margin_delta.median()),
                "fraction_family_margin_below_core": below_core(margin_delta),
                "mean_family_margin_per_time_bin_minus_core": float(margin_per_time_delta.mean()),
                "median_family_margin_per_time_bin_minus_core": float(margin_per_time_delta.median()),
                "fraction_family_margin_per_time_bin_below_core": below_core(margin_per_time_delta),
                "mean_family_margin_per_spike_minus_core": float(margin_per_spike_delta.mean()),
                "median_family_margin_per_spike_minus_core": float(margin_per_spike_delta.median()),
                "fraction_family_margin_per_spike_below_core": below_core(margin_per_spike_delta),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("event_window_variant")


def event_window_family_margin_summary(
    decisions: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("event_window_variant",),
) -> pd.DataFrame:
    """Summarize trajectory-family decisions by window variant."""

    columns = [
        *group_cols,
        "events",
        "required_complete_events",
        "incomplete_core_events",
        "margin_threshold",
        "trajectory_raw_wins",
        "nontrajectory_raw_wins",
        "trajectory_raw_win_fraction",
        "trajectory_confident_claims",
        "nontrajectory_confident_claims",
        "ambiguous_events",
        "trajectory_confident_claim_fraction",
        "nontrajectory_confident_claim_fraction",
        "mean_trajectory_minus_nontrajectory_log_evidence",
        "median_trajectory_minus_nontrajectory_log_evidence",
        "most_common_best_trajectory_model",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = [((), decisions)] if not group_cols else decisions.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = pd.to_numeric(group["trajectory_minus_nontrajectory_log_evidence"], errors="coerce").dropna()
        events = int(len(group))
        trajectory_claims = int(_bool_column(group, "trajectory_confident_claim").sum())
        nontrajectory_claims = int(_bool_column(group, "nontrajectory_confident_claim").sum())
        best_trajectory = group["best_trajectory_model"].replace("", pd.NA).dropna().astype(str)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": events,
                "required_complete_events": int(_bool_column(group, "required_models_complete").sum()),
                "incomplete_core_events": int((group["margin_decision"] == "incomplete_core").sum()),
                "margin_threshold": float(group["margin_threshold"].dropna().iloc[0]),
                "trajectory_raw_wins": int((delta > 0.0).sum()),
                "nontrajectory_raw_wins": int((delta < 0.0).sum()),
                "trajectory_raw_win_fraction": float((delta > 0.0).mean()) if not delta.empty else 0.0,
                "trajectory_confident_claims": trajectory_claims,
                "nontrajectory_confident_claims": nontrajectory_claims,
                "ambiguous_events": int((group["margin_decision"] == "ambiguous").sum()),
                "trajectory_confident_claim_fraction": float(trajectory_claims / max(events, 1)),
                "nontrajectory_confident_claim_fraction": float(nontrajectory_claims / max(events, 1)),
                "mean_trajectory_minus_nontrajectory_log_evidence": float(delta.mean()) if not delta.empty else np.nan,
                "median_trajectory_minus_nontrajectory_log_evidence": float(delta.median()) if not delta.empty else np.nan,
                "most_common_best_trajectory_model": (
                    "" if best_trajectory.empty else str(best_trajectory.value_counts().index[0])
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def event_window_comparison_to_core(decisions: pd.DataFrame, *, core_variant: str = "core") -> pd.DataFrame:
    """Compare each window variant with the core SWR window on matched events."""

    if decisions.empty:
        return pd.DataFrame()
    index_cols = ["session", "event_index"]
    core = decisions[decisions["event_window_variant"].astype(str).eq(core_variant)].set_index(index_cols)
    rows: list[dict[str, object]] = []
    for variant, group in decisions.groupby("event_window_variant", sort=True):
        current = group.set_index(index_cols)
        matched = current.join(core, lsuffix="_variant", rsuffix="_core", how="inner")
        if matched.empty:
            continue
        best_delta = (
            pd.to_numeric(matched["best_trajectory_log_evidence_variant"], errors="coerce")
            - pd.to_numeric(matched["best_trajectory_log_evidence_core"], errors="coerce")
        )
        margin_delta = (
            pd.to_numeric(matched["trajectory_minus_nontrajectory_log_evidence_variant"], errors="coerce")
            - pd.to_numeric(matched["trajectory_minus_nontrajectory_log_evidence_core"], errors="coerce")
        )
        rows.append(
            {
                "event_window_variant": str(variant),
                "core_variant": core_variant,
                "matched_events": int(len(matched)),
                "mean_best_trajectory_log_evidence_minus_core": float(best_delta.mean()),
                "median_best_trajectory_log_evidence_minus_core": float(best_delta.median()),
                "mean_family_margin_minus_core": float(margin_delta.mean()),
                "median_family_margin_minus_core": float(margin_delta.median()),
                "trajectory_confident_claims": int(
                    _bool_column(matched, "trajectory_confident_claim_variant").sum()
                ),
                "nontrajectory_confident_claims": int(
                    _bool_column(matched, "nontrajectory_confident_claim_variant").sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("event_window_variant")


def event_window_control_gate_summary(decisions: pd.DataFrame, *, core_variant: str = "core") -> pd.DataFrame:
    """Report coarse event-window sensitivity expectations as gates."""

    summary = event_window_family_margin_summary(decisions)
    by_variant = summary.set_index("event_window_variant") if not summary.empty else pd.DataFrame()
    comparison = event_window_comparison_to_core(decisions, core_variant=core_variant)
    rows: list[dict[str, object]] = []

    core = by_variant.loc[core_variant] if core_variant in by_variant.index else None
    _append_gate(
        rows,
        "core_window_present",
        core is not None,
        "" if core is None else int(core["events"]),
        "core window has scored events",
    )
    if core is not None:
        _append_gate(
            rows,
            "core_window_strong_trajectory_family",
            float(core["trajectory_confident_claim_fraction"]) >= 0.5,
            round(float(core["trajectory_confident_claim_fraction"]), 3),
            "core trajectory confident-claim fraction >= 0.5",
        )
        _append_gate(
            rows,
            "core_window_no_nontrajectory_claims",
            int(core["nontrajectory_confident_claims"]) == 0,
            int(core["nontrajectory_confident_claims"]),
            "core nontrajectory confident claims == 0",
        )

    sensitivity = summary[
        summary["event_window_variant"].astype(str).str.contains("contracted|expanded", case=False, regex=True)
    ]
    _append_gate(
        rows,
        "contracted_expanded_directionally_positive",
        bool(sensitivity.empty)
        or bool((sensitivity["mean_trajectory_minus_nontrajectory_log_evidence"].astype(float) > 0.0).all()),
        "" if sensitivity.empty else round(float(sensitivity["mean_trajectory_minus_nontrajectory_log_evidence"].min()), 3),
        "contracted/expanded mean trajectory-family margin > 0",
    )

    shifted = comparison[
        comparison["event_window_variant"].astype(str).str.contains("shift", case=False, regex=True)
    ]
    _append_gate(
        rows,
        "shifted_windows_reduce_absolute_trajectory_evidence",
        bool(not shifted.empty)
        and bool((shifted["mean_best_trajectory_log_evidence_minus_core"].astype(float) < 0.0).all()),
        "" if shifted.empty else round(float(shifted["mean_best_trajectory_log_evidence_minus_core"].max()), 3),
        "shifted windows have lower mean best-trajectory evidence than core",
    )
    _append_gate(
        rows,
        "shifted_windows_weaken_family_margin",
        bool(not shifted.empty)
        and bool((shifted["mean_family_margin_minus_core"].astype(float) < 0.0).all()),
        "" if shifted.empty else round(float(shifted["mean_family_margin_minus_core"].max()), 3),
        "shifted windows have lower mean trajectory-family margin than core",
    )

    overall = all(bool(row["passed"]) for row in rows)
    rows.append(
        {
            "gate": "overall",
            "passed": overall,
            "observed": f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} gates passed",
            "criterion": "all event-window sensitivity gates pass",
        }
    )
    return pd.DataFrame(rows)


def event_window_control_gate_summary_v2(
    decisions: pd.DataFrame,
    *,
    core_variant: str = "core",
) -> pd.DataFrame:
    """Report observation-aware event-window control checks.

    The v1 gate intentionally remains available for backward compatibility. The
    v2 table treats shifted windows as diagnostics when their spike/time-bin
    counts differ from core, because raw log evidence is not directly comparable
    across different observations.
    """

    summary = event_window_family_margin_summary(decisions)
    normalized = event_window_observation_normalized_summary(decisions)
    matched = event_window_core_matched_attenuation(decisions, core_variant=core_variant)
    by_variant = summary.set_index("event_window_variant") if not summary.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []

    core = by_variant.loc[core_variant] if core_variant in by_variant.index else None
    _append_gate_v2(
        rows,
        "core_window_present",
        "primary",
        core is not None,
        "" if core is None else int(core["events"]),
        "core window has scored events",
    )
    if core is not None:
        _append_gate_v2(
            rows,
            "core_window_strong_trajectory_family",
            "primary",
            float(core["trajectory_confident_claim_fraction"]) >= 0.5,
            round(float(core["trajectory_confident_claim_fraction"]), 3),
            "core trajectory confident-claim fraction >= 0.5",
        )
        _append_gate_v2(
            rows,
            "core_window_no_nontrajectory_claims",
            "primary",
            int(core["nontrajectory_confident_claims"]) == 0,
            int(core["nontrajectory_confident_claims"]),
            "core nontrajectory confident claims == 0",
        )

    boundary = normalized[
        normalized["event_window_variant"].astype(str).str.contains(
            "contracted|expanded",
            case=False,
            regex=True,
        )
    ]
    boundary_positive = bool(boundary.empty) or bool(
        (
            (boundary["mean_family_margin_per_time_bin"].astype(float) > 0.0)
            & (boundary["mean_family_margin_per_spike"].astype(float) > 0.0)
        ).all()
    )
    boundary_observed = (
        ""
        if boundary.empty
        else (
            "min mean margin/time="
            f"{float(boundary['mean_family_margin_per_time_bin'].min()):.3f}; "
            "min mean margin/spike="
            f"{float(boundary['mean_family_margin_per_spike'].min()):.3f}"
        )
    )
    _append_gate_v2(
        rows,
        "contracted_expanded_observation_normalized_positive",
        "primary",
        boundary_positive,
        boundary_observed,
        "contracted/expanded mean family margin per time bin and per spike > 0",
    )

    shifted = matched[
        matched["event_window_variant"].astype(str).str.contains("shift", case=False, regex=True)
    ]
    shifted_has_observation_difference = False
    shifted_observed = ""
    if not shifted.empty:
        max_spike_difference = float(shifted["mean_n_spikes_minus_core"].abs().max())
        max_time_difference = float(shifted["mean_n_time_minus_core"].abs().max())
        shifted_has_observation_difference = bool(max_spike_difference > 0.0 or max_time_difference > 0.0)
        shifted_observed = (
            f"max |mean spikes-core|={max_spike_difference:.3f}; "
            f"max |mean time-core|={max_time_difference:.3f}"
        )
    _append_gate_v2(
        rows,
        "shifted_windows_observation_mismatch_diagnostic",
        "diagnostic",
        bool(not shifted.empty),
        shifted_observed,
        "shifted windows are diagnostic when spike/time-bin observations differ from core",
    )

    shifted_norm = normalized[
        normalized["event_window_variant"].astype(str).str.contains("shift", case=False, regex=True)
    ]
    shifted_norm_observed = ""
    if not shifted_norm.empty:
        shifted_norm_observed = (
            "min mean margin/time="
            f"{float(shifted_norm['mean_family_margin_per_time_bin'].min()):.3f}; "
            "min mean margin/spike="
            f"{float(shifted_norm['mean_family_margin_per_spike'].min()):.3f}"
        )
    _append_gate_v2(
        rows,
        "shifted_windows_normalized_family_margin_diagnostic",
        "diagnostic",
        bool(not shifted_norm.empty),
        shifted_norm_observed,
        "inspect shifted-window normalized family margins; do not use raw evidence as a pass/fail gate",
    )

    shifted_core_matched = shifted
    shifted_core_observed = ""
    if not shifted_core_matched.empty:
        shifted_core_observed = (
            "max mean normalized margin change/time="
            f"{float(shifted_core_matched['mean_family_margin_per_time_bin_minus_core'].max()):.3f}; "
            "max mean normalized margin change/spike="
            f"{float(shifted_core_matched['mean_family_margin_per_spike_minus_core'].max()):.3f}"
        )
    _append_gate_v2(
        rows,
        "shifted_windows_core_matched_attenuation_diagnostic",
        "diagnostic",
        bool(not shifted_core_matched.empty),
        shifted_core_observed,
        "inspect core-matched normalized attenuation and observation-count deltas",
    )

    primary_rows = [row for row in rows if row["gate_type"] == "primary"]
    primary_passed = all(bool(row["passed"]) for row in primary_rows)
    rows.append(
        {
            "gate": "overall_primary",
            "gate_type": "primary",
            "passed": bool(primary_passed),
            "observed": f"{sum(bool(row['passed']) for row in primary_rows)}/{len(primary_rows)} primary gates passed",
            "criterion": "all primary event-window sensitivity gates pass; shifted windows are diagnostics",
        }
    )
    if shifted_has_observation_difference:
        rows.append(
            {
                "gate": "shifted_window_raw_evidence_not_standalone",
                "gate_type": "interpretation",
                "passed": True,
                "observed": shifted_observed,
                "criterion": "raw shifted-window evidence is not a standalone null statistic when observations differ",
            }
        )
    return pd.DataFrame(rows)


def _append_gate(
    rows: list[dict[str, object]],
    gate: str,
    passed: bool,
    observed: object,
    criterion: str,
) -> None:
    rows.append(
        {
            "gate": gate,
            "passed": bool(passed),
            "observed": observed,
            "criterion": criterion,
        }
    )


def _append_gate_v2(
    rows: list[dict[str, object]],
    gate: str,
    gate_type: str,
    passed: bool,
    observed: object,
    criterion: str,
) -> None:
    rows.append(
        {
            "gate": gate,
            "gate_type": gate_type,
            "passed": bool(passed),
            "observed": observed,
            "criterion": criterion,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", default="results/event-window-sensitivity")
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--core-variant", default="core")
    args = parser.parse_args()

    combined = aggregate_event_window_sensitivity(
        args.shard_glob,
        Path(args.output),
        margin_threshold=args.margin_threshold,
        core_variant=args.core_variant,
    )
    out = Path(args.output)
    print("Event-window family-margin summary:")
    print(pd.read_csv(out / "event_window_family_margin_summary.csv").to_string(index=False))
    print("\nEvent-window comparison to core:")
    print(pd.read_csv(out / "event_window_comparison_to_core.csv").to_string(index=False))
    print("\nEvent-window control gates:")
    print(pd.read_csv(out / "event_window_control_gate_summary.csv").to_string(index=False))
    print("\nEvent-window observation-normalized summary:")
    print(pd.read_csv(out / "event_window_observation_normalized_summary.csv").to_string(index=False))
    print("\nEvent-window spike-count summary:")
    print(pd.read_csv(out / "event_window_spike_count_summary.csv").to_string(index=False))
    print("\nEvent-window core-matched attenuation:")
    print(pd.read_csv(out / "event_window_core_matched_attenuation.csv").to_string(index=False))
    print("\nEvent-window control gates v2:")
    print(pd.read_csv(out / "event_window_control_gate_summary_v2.csv").to_string(index=False))
    print(f"\nRows: {len(combined)}")
    if "status" in combined:
        print(f"Failures: {int((combined['status'] != 'success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
