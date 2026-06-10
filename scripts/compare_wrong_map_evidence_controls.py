#!/usr/bin/env python3
"""Compare real-map evidence against wrong-map control evidence.

The wrong-map control is a map-specificity diagnostic, not a second
trajectory-family readiness table.  A wrong map may penalize the stationary
baseline more than trajectory rows, so a within-map trajectory-minus-stationary
margin can survive or increase under a wrong map.  This script therefore reports
absolute real-minus-wrong evidence attenuation as the primary control statistic,
while keeping the family-margin difference-in-differences as a diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MARGIN_THRESHOLD = 5.5
DEFAULT_RAT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED = 1
DEFAULT_REQUIRED_EXACT_CORE_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)
DEFAULT_TRAJECTORY_MODELS = (
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


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
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    return default


def _bool_column(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(lambda value: _as_bool(value, default=default)).astype(bool)


def _success_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "status" in out:
        out = out[out["status"].astype(str).eq("success")].copy()
    out["model"] = out["model"].astype(str)
    out["session"] = out["session"].astype(str)
    out["event_index"] = out["event_index"].astype(int)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    return out.dropna(subset=["log_evidence"])


def _read_evidence(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required evidence columns: {missing}")
    return _success_rows(frame)


def _model_value(group: pd.DataFrame, model: str) -> float:
    row = group[group["model"].astype(str).eq(str(model))]
    if row.empty:
        return float("nan")
    return float(row.iloc[-1]["log_evidence"])


def _best_model(group: pd.DataFrame, model_set: set[str]) -> tuple[str, float]:
    subset = group[group["model"].astype(str).isin(model_set)].dropna(subset=["log_evidence"])
    if subset.empty:
        return "", float("nan")
    row = subset.sort_values("log_evidence", ascending=False).iloc[0]
    return str(row["model"]), float(row["log_evidence"])


def wrong_map_model_evidence_attenuation(real: pd.DataFrame, wrong: pd.DataFrame) -> pd.DataFrame:
    """Return matched per-event/model real-minus-wrong-map evidence deltas."""

    real_ok = _success_rows(real)
    wrong_ok = _success_rows(wrong)
    wrong_cols = ["session", "event_index", "model", "log_evidence"]
    if "map_session" in wrong_ok.columns:
        wrong_cols.append("map_session")
    else:
        wrong_ok["map_session"] = ""
        wrong_cols.append("map_session")
    if "requested_model" in wrong_ok.columns:
        wrong_cols.append("requested_model")

    merged = real_ok[["session", "event_index", "model", "log_evidence"]].merge(
        wrong_ok[wrong_cols],
        on=["session", "event_index", "model"],
        suffixes=("_real_map", "_wrong_map"),
        how="inner",
    )
    if "requested_model" not in merged:
        merged["requested_model"] = merged["model"]
    merged["rat"] = merged["session"].map(_rat_from_session)
    merged["real_minus_wrong_log_evidence"] = (
        merged["log_evidence_real_map"].astype(float) - merged["log_evidence_wrong_map"].astype(float)
    )
    merged["real_map_better"] = merged["real_minus_wrong_log_evidence"] > 0.0
    ordered = [
        "rat",
        "session",
        "event_index",
        "map_session",
        "model",
        "requested_model",
        "log_evidence_real_map",
        "log_evidence_wrong_map",
        "real_minus_wrong_log_evidence",
        "real_map_better",
    ]
    return merged[ordered].sort_values(["session", "event_index", "model"]).reset_index(drop=True)


def wrong_map_model_evidence_attenuation_summary(
    attenuation: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("model",),
) -> pd.DataFrame:
    """Summarize absolute real-minus-wrong evidence attenuation."""

    columns = [
        *group_cols,
        "matched_rows",
        "positive_real_minus_wrong_rows",
        "positive_real_minus_wrong_fraction",
        "mean_real_minus_wrong_log_evidence",
        "median_real_minus_wrong_log_evidence",
        "min_real_minus_wrong_log_evidence",
        "max_real_minus_wrong_log_evidence",
    ]
    if attenuation.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    groups = [((), attenuation)] if not group_cols else attenuation.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = group["real_minus_wrong_log_evidence"].astype(float).dropna()
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "matched_rows": int(len(group)),
                "positive_real_minus_wrong_rows": int((delta > 0.0).sum()),
                "positive_real_minus_wrong_fraction": float((delta > 0.0).mean()) if not delta.empty else 0.0,
                "mean_real_minus_wrong_log_evidence": float(delta.mean()) if not delta.empty else np.nan,
                "median_real_minus_wrong_log_evidence": float(delta.median()) if not delta.empty else np.nan,
                "min_real_minus_wrong_log_evidence": float(delta.min()) if not delta.empty else np.nan,
                "max_real_minus_wrong_log_evidence": float(delta.max()) if not delta.empty else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def wrong_map_family_evidence_attenuation(
    real: pd.DataFrame,
    wrong: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_REQUIRED_EXACT_CORE_MODELS,
    trajectory_models: tuple[str, ...] = DEFAULT_TRAJECTORY_MODELS,
) -> pd.DataFrame:
    """Return event-level family attenuation and margin-DID diagnostics.

    The primary map-specificity statistic is the absolute evidence attenuation
    of the real-map best trajectory model:

        logZ_real(best_trajectory_real) - logZ_wrong(best_trajectory_real)

    The family-margin DID is written too, but should not be the control gate by
    itself because stationary can be penalized more strongly than trajectory
    rows under a wrong map.
    """

    columns = [
        "rat",
        "session",
        "event_index",
        "map_session",
        "required_models_complete_real_map",
        "required_models_complete_wrong_map",
        "required_models_complete_both_maps",
        "missing_required_models_real_map",
        "missing_required_models_wrong_map",
        "best_trajectory_model_real_map",
        "best_trajectory_log_evidence_real_map",
        "same_trajectory_model_log_evidence_wrong_map",
        "best_trajectory_delta_real_minus_wrong",
        "best_trajectory_model_wrong_map",
        "best_trajectory_log_evidence_wrong_map",
        "best_core_model_real_map",
        "best_core_log_evidence_real_map",
        "same_core_model_log_evidence_wrong_map",
        "best_core_delta_real_minus_wrong",
        "best_nontrajectory_model_real_map",
        "best_nontrajectory_log_evidence_real_map",
        "best_nontrajectory_model_wrong_map",
        "best_nontrajectory_log_evidence_wrong_map",
        "stationary_log_evidence_real_map",
        "stationary_log_evidence_wrong_map",
        "stationary_delta_real_minus_wrong",
        "family_margin_real_map",
        "family_margin_wrong_map",
        "family_margin_difference_in_differences",
    ]
    real_ok = _success_rows(real)
    wrong_ok = _success_rows(wrong)
    required = tuple(str(model) for model in required_models)
    required_set = set(required)
    trajectory_set = set(str(model) for model in trajectory_models)
    nontrajectory_set = required_set.difference(trajectory_set)

    real_groups = {
        (str(session), int(event)): group.copy()
        for (session, event), group in real_ok.groupby(["session", "event_index"], sort=False)
    }

    rows: list[dict[str, object]] = []
    for (session, event_index, map_session), wrong_group in wrong_ok.groupby(
        ["session", "event_index", "map_session"], sort=True
    ):
        real_group = real_groups.get((str(session), int(event_index)), pd.DataFrame(columns=real_ok.columns))
        real_core = real_group[real_group["model"].astype(str).isin(required_set)]
        wrong_core = wrong_group[wrong_group["model"].astype(str).isin(required_set)]
        real_present_set = set(real_core["model"].astype(str))
        wrong_present_set = set(wrong_core["model"].astype(str))
        real_missing = tuple(model for model in required if model not in real_present_set)
        wrong_missing = tuple(model for model in required if model not in wrong_present_set)
        real_complete = not real_missing
        wrong_complete = not wrong_missing

        best_traj_real_model, best_traj_real_value = _best_model(real_core, trajectory_set)
        best_traj_wrong_model, best_traj_wrong_value = _best_model(wrong_core, trajectory_set)
        best_core_real_model, best_core_real_value = _best_model(real_core, required_set)
        best_nontraj_real_model, best_nontraj_real_value = _best_model(real_core, nontrajectory_set)
        best_nontraj_wrong_model, best_nontraj_wrong_value = _best_model(wrong_core, nontrajectory_set)

        same_traj_wrong_value = (
            _model_value(wrong_core, best_traj_real_model) if best_traj_real_model else float("nan")
        )
        same_core_wrong_value = (
            _model_value(wrong_core, best_core_real_model) if best_core_real_model else float("nan")
        )
        stationary_real_value = _model_value(real_core, "sorted-spike-state-space-stationary")
        stationary_wrong_value = _model_value(wrong_core, "sorted-spike-state-space-stationary")

        real_margin = best_traj_real_value - best_nontraj_real_value
        wrong_margin = best_traj_wrong_value - best_nontraj_wrong_value
        rows.append(
            {
                "rat": _rat_from_session(session),
                "session": str(session),
                "event_index": int(event_index),
                "map_session": str(map_session),
                "required_models_complete_real_map": bool(real_complete),
                "required_models_complete_wrong_map": bool(wrong_complete),
                "required_models_complete_both_maps": bool(real_complete and wrong_complete),
                "missing_required_models_real_map": " ".join(real_missing),
                "missing_required_models_wrong_map": " ".join(wrong_missing),
                "best_trajectory_model_real_map": best_traj_real_model,
                "best_trajectory_log_evidence_real_map": float(best_traj_real_value),
                "same_trajectory_model_log_evidence_wrong_map": float(same_traj_wrong_value),
                "best_trajectory_delta_real_minus_wrong": float(best_traj_real_value - same_traj_wrong_value),
                "best_trajectory_model_wrong_map": best_traj_wrong_model,
                "best_trajectory_log_evidence_wrong_map": float(best_traj_wrong_value),
                "best_core_model_real_map": best_core_real_model,
                "best_core_log_evidence_real_map": float(best_core_real_value),
                "same_core_model_log_evidence_wrong_map": float(same_core_wrong_value),
                "best_core_delta_real_minus_wrong": float(best_core_real_value - same_core_wrong_value),
                "best_nontrajectory_model_real_map": best_nontraj_real_model,
                "best_nontrajectory_log_evidence_real_map": float(best_nontraj_real_value),
                "best_nontrajectory_model_wrong_map": best_nontraj_wrong_model,
                "best_nontrajectory_log_evidence_wrong_map": float(best_nontraj_wrong_value),
                "stationary_log_evidence_real_map": float(stationary_real_value),
                "stationary_log_evidence_wrong_map": float(stationary_wrong_value),
                "stationary_delta_real_minus_wrong": float(stationary_real_value - stationary_wrong_value),
                "family_margin_real_map": float(real_margin),
                "family_margin_wrong_map": float(wrong_margin),
                "family_margin_difference_in_differences": float(real_margin - wrong_margin),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def wrong_map_family_evidence_attenuation_summary(
    family: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize event-level family attenuation diagnostics."""

    columns = [
        *group_cols,
        "events",
        "complete_family_events",
        "positive_best_trajectory_attenuation_events",
        "best_trajectory_attenuation_fraction",
        "mean_best_trajectory_delta_real_minus_wrong",
        "median_best_trajectory_delta_real_minus_wrong",
        "min_best_trajectory_delta_real_minus_wrong",
        "mean_best_core_delta_real_minus_wrong",
        "median_best_core_delta_real_minus_wrong",
        "mean_stationary_delta_real_minus_wrong",
        "median_stationary_delta_real_minus_wrong",
        "mean_family_margin_real_map",
        "median_family_margin_real_map",
        "mean_family_margin_wrong_map",
        "median_family_margin_wrong_map",
        "mean_family_margin_difference_in_differences",
        "median_family_margin_difference_in_differences",
        "positive_family_margin_difference_in_differences_events",
        "positive_family_margin_difference_in_differences_fraction",
        "most_common_real_map_best_trajectory_model",
        "most_common_wrong_map_best_trajectory_model",
    ]
    if family.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = [((), family)] if not group_cols else family.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        complete = _bool_column(group, "required_models_complete_both_maps")
        usable = group[complete].copy()
        trajectory_delta = usable["best_trajectory_delta_real_minus_wrong"].astype(float).dropna()
        core_delta = usable["best_core_delta_real_minus_wrong"].astype(float).dropna()
        stationary_delta = usable["stationary_delta_real_minus_wrong"].astype(float).dropna()
        margin_real = usable["family_margin_real_map"].astype(float).dropna()
        margin_wrong = usable["family_margin_wrong_map"].astype(float).dropna()
        margin_did = usable["family_margin_difference_in_differences"].astype(float).dropna()
        best_real = usable["best_trajectory_model_real_map"].replace("", pd.NA).dropna().astype(str)
        best_wrong = usable["best_trajectory_model_wrong_map"].replace("", pd.NA).dropna().astype(str)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": int(len(group)),
                "complete_family_events": int(len(usable)),
                "positive_best_trajectory_attenuation_events": int((trajectory_delta > 0.0).sum()),
                "best_trajectory_attenuation_fraction": (
                    float((trajectory_delta > 0.0).mean()) if not trajectory_delta.empty else 0.0
                ),
                "mean_best_trajectory_delta_real_minus_wrong": (
                    float(trajectory_delta.mean()) if not trajectory_delta.empty else np.nan
                ),
                "median_best_trajectory_delta_real_minus_wrong": (
                    float(trajectory_delta.median()) if not trajectory_delta.empty else np.nan
                ),
                "min_best_trajectory_delta_real_minus_wrong": (
                    float(trajectory_delta.min()) if not trajectory_delta.empty else np.nan
                ),
                "mean_best_core_delta_real_minus_wrong": float(core_delta.mean()) if not core_delta.empty else np.nan,
                "median_best_core_delta_real_minus_wrong": (
                    float(core_delta.median()) if not core_delta.empty else np.nan
                ),
                "mean_stationary_delta_real_minus_wrong": (
                    float(stationary_delta.mean()) if not stationary_delta.empty else np.nan
                ),
                "median_stationary_delta_real_minus_wrong": (
                    float(stationary_delta.median()) if not stationary_delta.empty else np.nan
                ),
                "mean_family_margin_real_map": float(margin_real.mean()) if not margin_real.empty else np.nan,
                "median_family_margin_real_map": float(margin_real.median()) if not margin_real.empty else np.nan,
                "mean_family_margin_wrong_map": float(margin_wrong.mean()) if not margin_wrong.empty else np.nan,
                "median_family_margin_wrong_map": float(margin_wrong.median()) if not margin_wrong.empty else np.nan,
                "mean_family_margin_difference_in_differences": (
                    float(margin_did.mean()) if not margin_did.empty else np.nan
                ),
                "median_family_margin_difference_in_differences": (
                    float(margin_did.median()) if not margin_did.empty else np.nan
                ),
                "positive_family_margin_difference_in_differences_events": int((margin_did > 0.0).sum()),
                "positive_family_margin_difference_in_differences_fraction": (
                    float((margin_did > 0.0).mean()) if not margin_did.empty else 0.0
                ),
                "most_common_real_map_best_trajectory_model": (
                    "" if best_real.empty else str(best_real.value_counts().index[0])
                ),
                "most_common_wrong_map_best_trajectory_model": (
                    "" if best_wrong.empty else str(best_wrong.value_counts().index[0])
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def leave_one_rat_out_wrong_map_family_evidence_attenuation(family: pd.DataFrame) -> pd.DataFrame:
    """Return family attenuation summaries after holding out each rat."""

    if family.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for rat in sorted(family["rat"].dropna().astype(str).unique()):
        retained = family[family["rat"].astype(str) != rat]
        summary = wrong_map_family_evidence_attenuation_summary(retained)
        if summary.empty:
            continue
        summary.insert(0, "held_out_rat", rat)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("held_out_rat").reset_index(drop=True)


def rat_bootstrap_wrong_map_family_evidence_attenuation(
    family: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals for absolute map attenuation."""

    columns = [
        "bootstrap_replicates",
        "random_seed",
        "observed_best_trajectory_attenuation_fraction",
        "best_trajectory_attenuation_fraction_ci95_low",
        "best_trajectory_attenuation_fraction_ci95_high",
        "observed_mean_best_trajectory_delta_real_minus_wrong",
        "mean_best_trajectory_delta_ci95_low",
        "mean_best_trajectory_delta_ci95_high",
        "observed_median_best_trajectory_delta_real_minus_wrong",
        "median_best_trajectory_delta_ci95_low",
        "median_best_trajectory_delta_ci95_high",
        "probability_mean_best_trajectory_delta_positive",
        "probability_median_best_trajectory_delta_positive",
        "observed_mean_family_margin_difference_in_differences",
        "mean_family_margin_did_ci95_low",
        "mean_family_margin_did_ci95_high",
    ]
    if family.empty or "rat" not in family:
        return pd.DataFrame(columns=columns)
    complete = family[
        _bool_column(family, "required_models_complete_both_maps")
        & family["best_trajectory_delta_real_minus_wrong"].notna()
    ].copy()
    if complete.empty:
        return pd.DataFrame(columns=columns)
    rats = sorted(complete["rat"].dropna().astype(str).unique())
    if not rats:
        return pd.DataFrame(columns=columns)

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        summary = wrong_map_family_evidence_attenuation_summary(frame).iloc[0]
        return {
            "fraction": float(summary["best_trajectory_attenuation_fraction"]),
            "mean": float(summary["mean_best_trajectory_delta_real_minus_wrong"]),
            "median": float(summary["median_best_trajectory_delta_real_minus_wrong"]),
            "mean_did": float(summary["mean_family_margin_difference_in_differences"]),
        }

    observed = metrics(complete)
    rng = np.random.default_rng(int(random_seed))
    samples: list[dict[str, float]] = []
    for _ in range(int(n_bootstrap)):
        sampled_rats = rng.choice(rats, size=len(rats), replace=True)
        sampled_frames = []
        for sample_index, rat in enumerate(sampled_rats):
            sample = complete[complete["rat"].astype(str).eq(str(rat))].copy()
            sample["_bootstrap_rat"] = f"{sample_index}:{rat}"
            sampled_frames.append(sample)
        samples.append(metrics(pd.concat(sampled_frames, ignore_index=True)))
    boot = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "bootstrap_replicates": int(n_bootstrap),
                "random_seed": int(random_seed),
                "observed_best_trajectory_attenuation_fraction": observed["fraction"],
                "best_trajectory_attenuation_fraction_ci95_low": float(boot["fraction"].quantile(0.025)),
                "best_trajectory_attenuation_fraction_ci95_high": float(boot["fraction"].quantile(0.975)),
                "observed_mean_best_trajectory_delta_real_minus_wrong": observed["mean"],
                "mean_best_trajectory_delta_ci95_low": float(boot["mean"].quantile(0.025)),
                "mean_best_trajectory_delta_ci95_high": float(boot["mean"].quantile(0.975)),
                "observed_median_best_trajectory_delta_real_minus_wrong": observed["median"],
                "median_best_trajectory_delta_ci95_low": float(boot["median"].quantile(0.025)),
                "median_best_trajectory_delta_ci95_high": float(boot["median"].quantile(0.975)),
                "probability_mean_best_trajectory_delta_positive": float((boot["mean"] > 0.0).mean()),
                "probability_median_best_trajectory_delta_positive": float((boot["median"] > 0.0).mean()),
                "observed_mean_family_margin_difference_in_differences": observed["mean_did"],
                "mean_family_margin_did_ci95_low": float(boot["mean_did"].quantile(0.025)),
                "mean_family_margin_did_ci95_high": float(boot["mean_did"].quantile(0.975)),
            }
        ],
        columns=columns,
    )


def wrong_map_control_gate_summary(
    family: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return pass/fail gates for map-sensitive absolute evidence attenuation."""

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

    summary = wrong_map_family_evidence_attenuation_summary(family)
    if summary.empty:
        add("complete_family_map_pairs_present", False, 0, "complete family map pairs > 0")
        result = pd.DataFrame(rows, columns=columns)
    else:
        row = summary.iloc[0]
        add(
            "complete_family_map_pairs_present",
            int(row["complete_family_events"]) > 0,
            int(row["complete_family_events"]),
            "complete family map pairs > 0",
        )
        add(
            "best_trajectory_mean_real_minus_wrong_positive",
            float(row["mean_best_trajectory_delta_real_minus_wrong"]) > 0.0,
            f"{float(row['mean_best_trajectory_delta_real_minus_wrong']):.6g}",
            "mean best-trajectory real-minus-wrong evidence > 0",
        )
        add(
            "best_trajectory_median_real_minus_wrong_positive",
            float(row["median_best_trajectory_delta_real_minus_wrong"]) > 0.0,
            f"{float(row['median_best_trajectory_delta_real_minus_wrong']):.6g}",
            "median best-trajectory real-minus-wrong evidence > 0",
        )

        rat = wrong_map_family_evidence_attenuation_summary(family, group_cols=("rat",))
        if rat.empty:
            add("all_rats_best_trajectory_mean_real_minus_wrong_positive", False, np.nan, "min rat mean > 0")
            add("all_rats_best_trajectory_median_real_minus_wrong_positive", False, np.nan, "min rat median > 0")
        else:
            min_rat_mean = float(rat["mean_best_trajectory_delta_real_minus_wrong"].min())
            min_rat_median = float(rat["median_best_trajectory_delta_real_minus_wrong"].min())
            add(
                "all_rats_best_trajectory_mean_real_minus_wrong_positive",
                min_rat_mean > 0.0,
                f"{min_rat_mean:.6g}",
                "min rat mean best-trajectory real-minus-wrong evidence > 0",
            )
            add(
                "all_rats_best_trajectory_median_real_minus_wrong_positive",
                min_rat_median > 0.0,
                f"{min_rat_median:.6g}",
                "min rat median best-trajectory real-minus-wrong evidence > 0",
            )

        leave_one = leave_one_rat_out_wrong_map_family_evidence_attenuation(family)
        if leave_one.empty:
            add("leave_one_rat_out_mean_real_minus_wrong_positive", False, np.nan, "min retained mean > 0")
            add("leave_one_rat_out_median_real_minus_wrong_positive", False, np.nan, "min retained median > 0")
        else:
            min_leave_mean = float(leave_one["mean_best_trajectory_delta_real_minus_wrong"].min())
            min_leave_median = float(leave_one["median_best_trajectory_delta_real_minus_wrong"].min())
            add(
                "leave_one_rat_out_mean_real_minus_wrong_positive",
                min_leave_mean > 0.0,
                f"{min_leave_mean:.6g}",
                "min leave-one-rat-out mean best-trajectory real-minus-wrong evidence > 0",
            )
            add(
                "leave_one_rat_out_median_real_minus_wrong_positive",
                min_leave_median > 0.0,
                f"{min_leave_median:.6g}",
                "min leave-one-rat-out median best-trajectory real-minus-wrong evidence > 0",
            )

        bootstrap = rat_bootstrap_wrong_map_family_evidence_attenuation(
            family,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        if bootstrap.empty:
            add("rat_bootstrap_mean_real_minus_wrong_ci_positive", False, np.nan, "mean CI95 low > 0")
            add("rat_bootstrap_median_real_minus_wrong_ci_positive", False, np.nan, "median CI95 low > 0")
        else:
            boot = bootstrap.iloc[0]
            add(
                "rat_bootstrap_mean_real_minus_wrong_ci_positive",
                float(boot["mean_best_trajectory_delta_ci95_low"]) > 0.0,
                f"{float(boot['mean_best_trajectory_delta_ci95_low']):.6g}",
                "mean best-trajectory real-minus-wrong CI95 low > 0",
            )
            add(
                "rat_bootstrap_median_real_minus_wrong_ci_positive",
                float(boot["median_best_trajectory_delta_ci95_low"]) > 0.0,
                f"{float(boot['median_best_trajectory_delta_ci95_low']):.6g}",
                "median best-trajectory real-minus-wrong CI95 low > 0",
            )

        add(
            "stationary_attenuation_reported",
            pd.notna(row["mean_stationary_delta_real_minus_wrong"]),
            f"{float(row['mean_stationary_delta_real_minus_wrong']):.6g}",
            "stationary attenuation is reported, not used as sole null",
        )
        add(
            "family_margin_difference_in_differences_reported",
            pd.notna(row["mean_family_margin_difference_in_differences"]),
            f"{float(row['mean_family_margin_difference_in_differences']):.6g}",
            "family-margin DID is diagnostic and is not required to be positive",
        )
        result = pd.DataFrame(rows, columns=columns)

    overall = pd.DataFrame(
        [
            {
                "gate": "overall",
                "passed": bool(result["passed"].all()) if not result.empty else False,
                "observed": f"{int(result['passed'].sum())}/{len(result)} gates passed" if not result.empty else "0/0",
                "criterion": "all absolute attenuation gates pass",
                "details": "family-margin difference-in-differences is diagnostic only",
            }
        ],
        columns=columns,
    )
    return pd.concat([result, overall], ignore_index=True)


def write_wrong_map_comparison_outputs(
    real: pd.DataFrame,
    wrong: pd.DataFrame,
    output: str | Path,
    *,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    attenuation = wrong_map_model_evidence_attenuation(real, wrong)
    family = wrong_map_family_evidence_attenuation(real, wrong)
    family_did = family[
        [
            "rat",
            "session",
            "event_index",
            "map_session",
            "family_margin_real_map",
            "family_margin_wrong_map",
            "family_margin_difference_in_differences",
            "best_trajectory_delta_real_minus_wrong",
            "stationary_delta_real_minus_wrong",
        ]
    ].copy()

    outputs = {
        "wrong_map_model_evidence_attenuation.csv": attenuation,
        "wrong_map_model_evidence_attenuation_summary.csv": wrong_map_model_evidence_attenuation_summary(
            attenuation
        ),
        "wrong_map_family_evidence_attenuation.csv": family,
        "wrong_map_family_evidence_attenuation_summary.csv": wrong_map_family_evidence_attenuation_summary(
            family
        ),
        "rat_wrong_map_family_evidence_attenuation.csv": wrong_map_family_evidence_attenuation_summary(
            family,
            group_cols=("rat",),
        ),
        "leave_one_rat_out_wrong_map_family_evidence_attenuation.csv": (
            leave_one_rat_out_wrong_map_family_evidence_attenuation(family)
        ),
        "rat_bootstrap_wrong_map_family_evidence_attenuation.csv": (
            rat_bootstrap_wrong_map_family_evidence_attenuation(
                family,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )
        ),
        "wrong_map_margin_difference_in_differences.csv": family_did,
        "wrong_map_control_gate_summary.csv": wrong_map_control_gate_summary(
            family,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-event-model-evidence", required=True)
    parser.add_argument("--wrong-map-event-model-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--rat-bootstrap-replicates", type=int, default=DEFAULT_RAT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--rat-bootstrap-random-seed", type=int, default=DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED)
    args = parser.parse_args()

    real = _read_evidence(args.real_event_model_evidence)
    wrong = _read_evidence(args.wrong_map_event_model_evidence)
    write_wrong_map_comparison_outputs(
        real,
        wrong,
        args.output,
        margin_threshold=args.margin_threshold,
        n_bootstrap=args.rat_bootstrap_replicates,
        random_seed=args.rat_bootstrap_random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
