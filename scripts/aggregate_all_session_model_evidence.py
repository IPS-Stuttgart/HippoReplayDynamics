#!/usr/bin/env python3
"""Aggregate all-session event-sharded model-evidence outputs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmark_model_evidence import _add_evidence_columns, _counts, _ensure_evidence_support_columns, _summary, _write
from hipporeplayimm.advanced_result_diagnostics import paired_model_margin_decisions
from model_evidence_settings import _validate_constant_settings

DEFAULT_MARGIN_POSITIVE_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
DEFAULT_MARGIN_REFERENCE_MODEL = "sorted-spike-state-space-diffusion"
DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD = 5.5
DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS = (0.0, 1.0, 3.0, 5.5, 10.0, 20.0)
DEFAULT_RAT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED = 1
DEFAULT_PAPER_MIN_RATS = 4
DEFAULT_PAPER_MIN_SESSIONS = 8
DEFAULT_PAPER_MIN_PAIRED_EVENTS_PER_SESSION = 5
DEFAULT_PAPER_MIN_RAW_WIN_FRACTION = 0.5
DEFAULT_PAPER_MIN_CONFIDENT_CLAIM_FRACTION = 0.5
DEFAULT_PAPER_MIN_FULL_CORE_EXACT_MODELS = 5
DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS = (
    "sorted-spike-state-space-stationary",
    DEFAULT_MARGIN_REFERENCE_MODEL,
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    DEFAULT_MARGIN_POSITIVE_MODEL,
)
DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS = (
    DEFAULT_MARGIN_REFERENCE_MODEL,
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    DEFAULT_MARGIN_POSITIVE_MODEL,
)


def _load_score_files(shard_glob: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(shard_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No model-evidence shard CSVs matched: {shard_glob}")
    return paths


def _load_combined(shard_glob: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _load_score_files(shard_glob):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_shard_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("All model-evidence shard CSVs were empty.")

    combined = pd.concat(frames, ignore_index=True)
    duplicate_key = ["session", "event_index", "model"]
    duplicates = combined.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        duplicate_rows = combined.loc[duplicates, duplicate_key + ["source_shard_file"]]
        raise ValueError(
            "All-session model-evidence shards contain duplicate event/model rows:\n"
            + duplicate_rows.head(20).to_string(index=False)
        )
    _validate_constant_settings(combined)
    return _add_evidence_columns(combined.drop(columns=["source_shard_file"]))


def session_model_evidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_evidence_support_columns(df)
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    ok = ok.copy()
    if "is_best_truncated_lower_bound" not in ok:
        ok["is_best_truncated_lower_bound"] = False
    if "truncated_relative_log_evidence" not in ok:
        ok["truncated_relative_log_evidence"] = np.nan
    out = ok.groupby(["session", "model", "model_family", "evidence_support", "evidence_comparable"], as_index=False).agg(
        events=("event_index", "count"), wins=("is_best_model", "sum"),
        truncated_lower_bound_wins=("is_best_truncated_lower_bound", "sum"),
        mean_log_evidence=("log_evidence", "mean"), median_log_evidence=("log_evidence", "median"),
        mean_relative_log_evidence=("relative_log_evidence", "mean"),
        median_relative_log_evidence=("relative_log_evidence", "median"),
        mean_model_probability=("model_probability", "mean"),
        median_model_probability=("model_probability", "median"),
        mean_truncated_relative_log_evidence=("truncated_relative_log_evidence", "mean"),
        median_truncated_relative_log_evidence=("truncated_relative_log_evidence", "median"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    out["win_fraction"] = out["wins"] / out["events"].clip(lower=1)
    out["truncated_lower_bound_win_fraction"] = out["truncated_lower_bound_wins"] / out["events"].clip(lower=1)
    return out.sort_values(
        ["session", "evidence_comparable", "wins", "truncated_lower_bound_wins", "mean_log_evidence"],
        ascending=[True, False, False, False, False],
    )


def session_best_model_counts(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    base = ok.drop_duplicates(["session", "event_index"])
    rows: list[dict[str, object]] = []
    for session, session_frame in base.groupby("session", sort=True):
        for col in (
            "best_model",
            "best_trajectory_model",
            "best_nontrajectory_model",
            "best_truncated_lower_bound_model",
        ):
            if col not in session_frame:
                continue
            values = session_frame[col].dropna().astype(str)
            values = values[values != ""]
            if values.empty:
                continue
            counts = values.value_counts().rename_axis("model").reset_index(name="events")
            counts["session"] = session
            counts["comparison"] = col
            rows.extend(counts.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=["session", "comparison", "model", "events"])
    return pd.DataFrame(rows)[["session", "comparison", "model", "events"]]


def random_effects_model_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact random-effects-style model probability table by session.

    Each session votes for the exact-comparable model with the largest summed
    log evidence across successfully scored events. Truncated lower-bound rows
    are excluded from the random-effects and fixed-effects probability columns.
    """

    df = _ensure_evidence_support_columns(df)
    ok = df[(df["status"] == "success") & df["evidence_comparable"]].copy()
    if ok.empty:
        return pd.DataFrame()
    per_session = ok.groupby(["session", "model"], as_index=False).agg(
        session_events=("event_index", "nunique"),
        session_log_evidence=("log_evidence", "sum"),
        session_mean_log_evidence=("log_evidence", "mean"),
        session_wins=("is_best_model", "sum"),
    )
    models = sorted(per_session["model"].unique())
    sessions = sorted(per_session["session"].unique())
    session_winners: dict[str, str] = {}
    for session, group in per_session.groupby("session", sort=True):
        winner = str(group.sort_values("session_log_evidence", ascending=False).iloc[0]["model"])
        session_winners[session] = winner

    win_counts = {model: 0 for model in models}
    for winner in session_winners.values():
        win_counts[winner] += 1
    fixed_log_evidence = per_session.groupby("model")["session_log_evidence"].sum().reindex(models).to_numpy(dtype=float)
    fixed_probs = np.exp(fixed_log_evidence - logsumexp(fixed_log_evidence))
    n_sessions = len(sessions)
    n_models = len(models)
    rows = []
    for idx, model in enumerate(models):
        rows.append(
            {
                "model": model,
                "sessions": n_sessions,
                "session_win_count": int(win_counts[model]),
                "random_effects_probability": float((1.0 + win_counts[model]) / (n_models + n_sessions)),
                "fixed_effects_log_evidence": float(fixed_log_evidence[idx]),
                "fixed_effects_probability": float(fixed_probs[idx]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["random_effects_probability", "fixed_effects_probability"], ascending=[False, False]
    )


def paired_momentum_diffusion_margin_decisions(
    df: pd.DataFrame,
    *,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Return calibrated exact-sparse momentum-vs-diffusion event decisions."""

    return paired_model_margin_decisions(
        df,
        positive_model=DEFAULT_MARGIN_POSITIVE_MODEL,
        reference_model=DEFAULT_MARGIN_REFERENCE_MODEL,
        margin_threshold=margin_threshold,
        group_cols=("session", "event_index"),
    )


def paired_momentum_diffusion_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibrated exact-sparse momentum-vs-diffusion decisions."""

    return _paired_margin_summary(decisions, group_cols=())


def paired_momentum_diffusion_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize paired momentum-vs-diffusion decisions across margin thresholds."""

    rows = []
    for threshold in thresholds:
        decisions = paired_momentum_diffusion_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(paired_momentum_diffusion_margin_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def session_paired_momentum_diffusion_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibrated exact-sparse momentum-vs-diffusion decisions by session."""

    return _paired_margin_summary(decisions, group_cols=("session",))


def session_paired_momentum_diffusion_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize paired threshold sensitivity by session."""

    rows = []
    for threshold in thresholds:
        decisions = paired_momentum_diffusion_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(session_paired_momentum_diffusion_margin_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "session"]).reset_index(drop=True)


def rat_paired_momentum_diffusion_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibrated exact-sparse momentum-vs-diffusion decisions by rat."""

    return _paired_margin_summary(_with_rat(decisions), group_cols=("rat",))


def leave_one_rat_out_paired_momentum_diffusion_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibrated momentum-vs-diffusion decisions after excluding each rat."""

    return _leave_one_rat_out_summary(decisions, paired_momentum_diffusion_margin_summary)


def leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize paired threshold sensitivity after excluding each rat."""

    rows = []
    for threshold in thresholds:
        decisions = paired_momentum_diffusion_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(leave_one_rat_out_paired_momentum_diffusion_margin_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "held_out_rat"]).reset_index(drop=True)


def rat_bootstrap_paired_momentum_diffusion_margin_summary(
    decisions: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals for paired momentum decisions."""

    return _rat_bootstrap_margin_summary(
        decisions,
        delta_col="positive_minus_reference_log_evidence",
        positive_claim_col="positive_model_claimed",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals across paired margin thresholds."""

    rows = []
    for threshold in thresholds:
        decisions = paired_momentum_diffusion_margin_decisions(df, margin_threshold=float(threshold))
        summary = rat_bootstrap_paired_momentum_diffusion_margin_summary(
            decisions,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        rows.append(_insert_margin_threshold(summary, threshold))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def exact_sparse_momentum_core_margins(
    df: pd.DataFrame,
    *,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Return exact-sparse momentum margins against the best other exact model.

    The paired diffusion table answers the calibrated primary contrast. This
    table answers the full-core question: whether exact-sparse momentum remains
    best after adding stationary, fragmented, first-order IMM, and other
    exact-comparable alternatives.
    """

    columns = [
        "session",
        "event_index",
        "positive_model",
        "positive_log_evidence",
        "positive_exact_rank",
        "positive_is_exact_best",
        "positive_confident_core_claim",
        "best_other_exact_model",
        "best_other_exact_log_evidence",
        "positive_minus_best_other_exact_log_evidence",
        "margin_threshold",
        "exact_models_compared",
    ]
    df = _ensure_evidence_support_columns(df)
    ok = df[(df["status"] == "success") & df["evidence_comparable"].fillna(False).astype(bool)].copy()
    if ok.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(["session", "event_index"], sort=False):
        session, event_index = key
        group = group.dropna(subset=["log_evidence"]).copy()
        if group.empty:
            continue
        positive = group[group["model"].astype(str).eq(DEFAULT_MARGIN_POSITIVE_MODEL)]
        if positive.empty:
            continue
        positive_row = positive.iloc[-1]
        positive_value = float(positive_row["log_evidence"])
        ranked = group.sort_values("log_evidence", ascending=False).reset_index(drop=True)
        positive_matches = ranked["model"].astype(str).eq(DEFAULT_MARGIN_POSITIVE_MODEL)
        if not positive_matches.any():
            continue
        positive_rank = int(np.flatnonzero(positive_matches.to_numpy())[0] + 1)
        others = ranked[~positive_matches]
        if others.empty:
            best_other_model = ""
            best_other_value = np.nan
            delta = np.inf
        else:
            best_other = others.iloc[0]
            best_other_model = str(best_other["model"])
            best_other_value = float(best_other["log_evidence"])
            delta = positive_value - best_other_value
        rows.append(
            {
                "session": str(session),
                "event_index": int(event_index),
                "positive_model": DEFAULT_MARGIN_POSITIVE_MODEL,
                "positive_log_evidence": positive_value,
                "positive_exact_rank": positive_rank,
                "positive_is_exact_best": bool(positive_rank == 1),
                "positive_confident_core_claim": bool(delta >= float(margin_threshold)),
                "best_other_exact_model": best_other_model,
                "best_other_exact_log_evidence": float(best_other_value),
                "positive_minus_best_other_exact_log_evidence": float(delta),
                "margin_threshold": float(margin_threshold),
                "exact_models_compared": int(len(ranked)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def exact_sparse_momentum_core_margin_summary(margins: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact-sparse momentum full-core margins across sessions."""

    return _core_margin_summary(margins, group_cols=())


def exact_sparse_momentum_core_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize full-core exact-sparse momentum decisions across thresholds."""

    rows = []
    for threshold in thresholds:
        margins = exact_sparse_momentum_core_margins(df, margin_threshold=float(threshold))
        rows.append(exact_sparse_momentum_core_margin_summary(margins))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def session_exact_sparse_momentum_core_margin_summary(margins: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact-sparse momentum full-core margins by session."""

    return _core_margin_summary(margins, group_cols=("session",))


def session_exact_sparse_momentum_core_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize full-core threshold sensitivity by session."""

    rows = []
    for threshold in thresholds:
        margins = exact_sparse_momentum_core_margins(df, margin_threshold=float(threshold))
        rows.append(session_exact_sparse_momentum_core_margin_summary(margins))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "session"]).reset_index(drop=True)


def rat_exact_sparse_momentum_core_margin_summary(margins: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact-sparse momentum full-core margins by rat."""

    return _core_margin_summary(_with_rat(margins), group_cols=("rat",))


def leave_one_rat_out_exact_sparse_momentum_core_margin_summary(margins: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact-sparse momentum full-core margins after excluding each rat."""

    return _leave_one_rat_out_summary(margins, exact_sparse_momentum_core_margin_summary)


def leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize full-core threshold sensitivity after excluding each rat."""

    rows = []
    for threshold in thresholds:
        margins = exact_sparse_momentum_core_margins(df, margin_threshold=float(threshold))
        rows.append(leave_one_rat_out_exact_sparse_momentum_core_margin_summary(margins))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "held_out_rat"]).reset_index(drop=True)


def rat_bootstrap_exact_sparse_momentum_core_margin_summary(
    margins: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals for full-core momentum margins."""

    return _rat_bootstrap_margin_summary(
        margins,
        delta_col="positive_minus_best_other_exact_log_evidence",
        positive_claim_col="positive_confident_core_claim",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals across full-core thresholds."""

    rows = []
    for threshold in thresholds:
        margins = exact_sparse_momentum_core_margins(df, margin_threshold=float(threshold))
        summary = rat_bootstrap_exact_sparse_momentum_core_margin_summary(
            margins,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        rows.append(_insert_margin_threshold(summary, threshold))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def paper_readiness_gate_summary(
    df: pd.DataFrame,
    *,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
    min_rats: int = DEFAULT_PAPER_MIN_RATS,
    min_sessions: int = DEFAULT_PAPER_MIN_SESSIONS,
    min_paired_events_per_session: int = DEFAULT_PAPER_MIN_PAIRED_EVENTS_PER_SESSION,
    min_raw_win_fraction: float = DEFAULT_PAPER_MIN_RAW_WIN_FRACTION,
    min_confident_claim_fraction: float = DEFAULT_PAPER_MIN_CONFIDENT_CLAIM_FRACTION,
    min_full_core_exact_models: int = DEFAULT_PAPER_MIN_FULL_CORE_EXACT_MODELS,
    required_full_core_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
) -> pd.DataFrame:
    """Return explicit pass/fail gates for the calibrated momentum evidence."""

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

    status_failures = int((df["status"] != "success").sum()) if "status" in df else 0
    add("no_scoring_failures", status_failures == 0, status_failures, "status failures == 0")

    paired_decisions = paired_momentum_diffusion_margin_decisions(df, margin_threshold=margin_threshold)
    paired_summary = paired_momentum_diffusion_margin_summary(paired_decisions)
    if paired_summary.empty:
        add("paired_events_present", False, 0, "paired events > 0")
        return pd.DataFrame(rows, columns=columns)

    paired = paired_summary.iloc[0]
    add("paired_events_present", int(paired["events"]) > 0, int(paired["events"]), "paired events > 0")
    observed_sessions = int(paired_decisions["session"].dropna().astype(str).nunique())
    observed_rats = int(paired_decisions["session"].dropna().map(_rat_from_session).nunique())
    per_session_events = paired_decisions.groupby("session", sort=False).size()
    min_observed_events_per_session = int(per_session_events.min()) if not per_session_events.empty else 0
    add(
        "minimum_rats_present",
        observed_rats >= int(min_rats),
        observed_rats,
        f"observed rats >= {int(min_rats)}",
    )
    add(
        "minimum_sessions_present",
        observed_sessions >= int(min_sessions),
        observed_sessions,
        f"observed sessions >= {int(min_sessions)}",
    )
    add(
        "minimum_paired_events_per_session",
        min_observed_events_per_session >= int(min_paired_events_per_session),
        min_observed_events_per_session,
        f"min paired events/session >= {int(min_paired_events_per_session)}",
    )
    add(
        "paired_no_confident_diffusion_claims",
        int(paired["reference_model_claims"]) == 0,
        int(paired["reference_model_claims"]),
        "reference_model_claims == 0",
    )
    add(
        "paired_confident_momentum_claims_present",
        int(paired["positive_model_claims"]) > 0,
        int(paired["positive_model_claims"]),
        "positive_model_claims > 0",
        f"margin_threshold={float(margin_threshold):g}",
    )
    add(
        "paired_raw_momentum_win_majority",
        float(paired["positive_raw_win_fraction"]) > float(min_raw_win_fraction),
        f"{float(paired['positive_raw_win_fraction']):.6g}",
        f"positive_raw_win_fraction > {float(min_raw_win_fraction):g}",
    )
    add(
        "paired_confident_momentum_claim_majority",
        float(paired["positive_claim_fraction"]) > float(min_confident_claim_fraction),
        f"{float(paired['positive_claim_fraction']):.6g}",
        f"positive_claim_fraction > {float(min_confident_claim_fraction):g}",
        f"margin_threshold={float(margin_threshold):g}",
    )

    session = session_paired_momentum_diffusion_margin_summary(paired_decisions)
    if session.empty:
        add("all_sessions_have_confident_momentum_claims", False, 0, "min session positive_model_claims > 0")
        add("all_sessions_have_no_confident_diffusion_claims", False, 0, "max session reference_model_claims == 0")
    else:
        min_session_positive = int(session["positive_model_claims"].min())
        max_session_reference = int(session["reference_model_claims"].max())
        add(
            "all_sessions_have_confident_momentum_claims",
            min_session_positive > 0,
            min_session_positive,
            "min session positive_model_claims > 0",
        )
        add(
            "all_sessions_have_no_confident_diffusion_claims",
            max_session_reference == 0,
            max_session_reference,
            "max session reference_model_claims == 0",
        )

    leave_one = leave_one_rat_out_paired_momentum_diffusion_margin_summary(paired_decisions)
    if leave_one.empty:
        add("leave_one_rat_out_mean_delta_positive", False, np.nan, "min leave-one-rat-out mean delta > 0")
        add("leave_one_rat_out_median_delta_positive", False, np.nan, "min leave-one-rat-out median delta > 0")
    else:
        min_leave_one_mean = float(leave_one["mean_positive_minus_reference_log_evidence"].min())
        min_leave_one_median = float(leave_one["median_positive_minus_reference_log_evidence"].min())
        add(
            "leave_one_rat_out_mean_delta_positive",
            min_leave_one_mean > 0.0,
            f"{min_leave_one_mean:.6g}",
            "min leave-one-rat-out mean delta > 0",
        )
        add(
            "leave_one_rat_out_median_delta_positive",
            min_leave_one_median > 0.0,
            f"{min_leave_one_median:.6g}",
            "min leave-one-rat-out median delta > 0",
        )

    bootstrap = rat_bootstrap_paired_momentum_diffusion_margin_summary(
        paired_decisions,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    if bootstrap.empty:
        add("rat_bootstrap_mean_delta_ci_positive", False, np.nan, "mean_delta_ci95_low > 0")
        add("rat_bootstrap_median_delta_ci_positive", False, np.nan, "median_delta_ci95_low > 0")
        add("rat_bootstrap_claim_fraction_ci_nonzero", False, np.nan, "positive_claim_fraction_ci95_low > 0")
    else:
        boot = bootstrap.iloc[0]
        add(
            "rat_bootstrap_mean_delta_ci_positive",
            float(boot["mean_delta_ci95_low"]) > 0.0,
            f"{float(boot['mean_delta_ci95_low']):.6g}",
            "mean_delta_ci95_low > 0",
        )
        add(
            "rat_bootstrap_median_delta_ci_positive",
            float(boot["median_delta_ci95_low"]) > 0.0,
            f"{float(boot['median_delta_ci95_low']):.6g}",
            "median_delta_ci95_low > 0",
        )
        add(
            "rat_bootstrap_claim_fraction_ci_nonzero",
            float(boot["positive_claim_fraction_ci95_low"]) > 0.0,
            f"{float(boot['positive_claim_fraction_ci95_low']):.6g}",
            "positive_claim_fraction_ci95_low > 0",
        )

    core_margins = exact_sparse_momentum_core_margins(df, margin_threshold=margin_threshold)
    required_core_coverage = _required_full_core_model_coverage(df, required_full_core_models)
    core_summary = exact_sparse_momentum_core_margin_summary(core_margins)
    if core_summary.empty:
        add(
            "full_core_required_exact_models_present",
            False,
            required_core_coverage["observed"],
            "all events include required exact core models",
            str(required_core_coverage["details"]),
        )
        add(
            "full_core_min_exact_models_compared",
            False,
            0,
            f"min exact_models_compared >= {int(min_full_core_exact_models)}",
        )
        add("full_core_exact_sparse_claims_present", False, 0, "positive_confident_core_claims > 0")
        add(
            "full_core_exact_sparse_best_majority",
            False,
            np.nan,
            f"positive_exact_best_fraction > {float(min_raw_win_fraction):g}",
        )
        add(
            "full_core_confident_exact_sparse_claim_majority",
            False,
            np.nan,
            f"positive_confident_core_claim_fraction > {float(min_confident_claim_fraction):g}",
        )
    else:
        core = core_summary.iloc[0]
        min_core_models = int(core_margins["exact_models_compared"].min()) if not core_margins.empty else 0
        add(
            "full_core_required_exact_models_present",
            bool(required_core_coverage["passed"]),
            required_core_coverage["observed"],
            "all events include required exact core models",
            str(required_core_coverage["details"]),
        )
        add(
            "full_core_min_exact_models_compared",
            min_core_models >= int(min_full_core_exact_models),
            min_core_models,
            f"min exact_models_compared >= {int(min_full_core_exact_models)}",
        )
        add(
            "full_core_exact_sparse_claims_present",
            int(core["positive_confident_core_claims"]) > 0,
            int(core["positive_confident_core_claims"]),
            "positive_confident_core_claims > 0",
        )
        add(
            "full_core_exact_sparse_best_majority",
            float(core["positive_exact_best_fraction"]) > float(min_raw_win_fraction),
            f"{float(core['positive_exact_best_fraction']):.6g}",
            f"positive_exact_best_fraction > {float(min_raw_win_fraction):g}",
        )
        add(
            "full_core_confident_exact_sparse_claim_majority",
            float(core["positive_confident_core_claim_fraction"]) > float(min_confident_claim_fraction),
            f"{float(core['positive_confident_core_claim_fraction']):.6g}",
            f"positive_confident_core_claim_fraction > {float(min_confident_claim_fraction):g}",
            f"margin_threshold={float(margin_threshold):g}",
        )

    result = pd.DataFrame(rows, columns=columns)
    overall_pass = bool(result["passed"].all())
    overall = pd.DataFrame(
        [
            {
                "gate": "overall",
                "passed": overall_pass,
                "observed": f"{int(result['passed'].sum())}/{len(result)} gates passed",
                "criterion": "all gates pass",
                "details": f"margin_threshold={float(margin_threshold):g}",
            }
        ],
        columns=columns,
    )
    return pd.concat([result, overall], ignore_index=True)


def required_full_core_model_coverage_table(
    df: pd.DataFrame,
    required_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
) -> pd.DataFrame:
    """Return per-event coverage for the named exact/comparable core models."""

    columns = [
        "session",
        "event_index",
        "required_models_present",
        "required_models_total",
        "required_models_complete",
        "missing_required_models",
        "present_required_models",
        "exact_models_compared",
        "exact_comparable_models",
    ]
    required = tuple(str(model) for model in required_models)
    df = _ensure_evidence_support_columns(df)
    if "model" not in df or "session" not in df or "event_index" not in df:
        return pd.DataFrame(columns=columns)

    status_ok = df["status"].eq("success") if "status" in df else pd.Series(True, index=df.index)
    comparable = df["evidence_comparable"].fillna(False).astype(bool)
    ok = df[status_ok & comparable].copy()
    if ok.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(["session", "event_index"], sort=True):
        session, event_index = key
        models = sorted(set(group["model"].dropna().astype(str)))
        model_set = set(models)
        present = tuple(model for model in required if model in model_set)
        missing = tuple(model for model in required if model not in model_set)
        rows.append(
            {
                "session": str(session),
                "event_index": int(event_index),
                "required_models_present": int(len(present)),
                "required_models_total": int(len(required)),
                "required_models_complete": not missing,
                "missing_required_models": " ".join(missing),
                "present_required_models": " ".join(present),
                "exact_models_compared": int(len(models)),
                "exact_comparable_models": " ".join(models),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def exact_core_model_claim_decisions(
    df: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Return per-event winner and margin decisions among required exact core models."""

    columns = [
        "session",
        "event_index",
        "required_models_present",
        "required_models_total",
        "required_models_complete",
        "missing_required_models",
        "present_required_models",
        "margin_threshold",
        "best_core_model",
        "best_core_log_evidence",
        "runner_up_core_model",
        "runner_up_core_log_evidence",
        "best_minus_runner_up_log_evidence",
        "claim_model",
    ]
    required = tuple(str(model) for model in required_models)
    required_set = set(required)
    df = _ensure_evidence_support_columns(df)
    if "model" not in df or "session" not in df or "event_index" not in df:
        return pd.DataFrame(columns=columns)

    status_ok = df["status"].eq("success") if "status" in df else pd.Series(True, index=df.index)
    comparable = df["evidence_comparable"].fillna(False).astype(bool)
    ok = df[status_ok & comparable].copy()
    if ok.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(["session", "event_index"], sort=True):
        session, event_index = key
        core = group[group["model"].astype(str).isin(required_set)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        complete = not missing
        if core.empty:
            best_model = ""
            best_value = np.nan
            runner_up_model = ""
            runner_up_value = np.nan
            margin = np.nan
            claim_model = "incomplete_core"
        else:
            ranked = core.sort_values("log_evidence", ascending=False).reset_index(drop=True)
            best = ranked.iloc[0]
            best_model = str(best["model"])
            best_value = float(best["log_evidence"])
            if len(ranked) > 1:
                runner_up = ranked.iloc[1]
                runner_up_model = str(runner_up["model"])
                runner_up_value = float(runner_up["log_evidence"])
                margin = best_value - runner_up_value
            else:
                runner_up_model = ""
                runner_up_value = np.nan
                margin = np.inf
            if not complete:
                claim_model = "incomplete_core"
            elif margin >= float(margin_threshold):
                claim_model = best_model
            else:
                claim_model = "ambiguous"
        rows.append(
            {
                "session": str(session),
                "event_index": int(event_index),
                "required_models_present": int(len(present)),
                "required_models_total": int(len(required)),
                "required_models_complete": bool(complete),
                "missing_required_models": " ".join(missing),
                "present_required_models": " ".join(present),
                "margin_threshold": float(margin_threshold),
                "best_core_model": best_model,
                "best_core_log_evidence": float(best_value),
                "runner_up_core_model": runner_up_model,
                "runner_up_core_log_evidence": float(runner_up_value),
                "best_minus_runner_up_log_evidence": float(margin),
                "claim_model": claim_model,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def exact_core_model_claim_summary(
    decisions: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize raw and calibrated claims for each required exact core model."""

    columns = [
        *group_cols,
        "model",
        "events",
        "required_complete_events",
        "incomplete_core_events",
        "ambiguous_events",
        "margin_threshold",
        "raw_best_events",
        "raw_best_fraction",
        "confident_claims",
        "confident_claim_fraction",
        "mean_winning_margin_when_best",
        "median_winning_margin_when_best",
    ]
    required = tuple(str(model) for model in required_models)
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = [((), decisions)] if not group_cols else decisions.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        events = int(len(group))
        complete = group["required_models_complete"].fillna(False).astype(bool)
        required_complete_events = int(complete.sum())
        incomplete_core_events = int((group["claim_model"].astype(str) == "incomplete_core").sum())
        ambiguous_events = int((group["claim_model"].astype(str) == "ambiguous").sum())
        threshold = float(group["margin_threshold"].dropna().iloc[0]) if "margin_threshold" in group else np.nan
        for model in required:
            raw_best = group["best_core_model"].astype(str).eq(model)
            confident = group["claim_model"].astype(str).eq(model)
            winning_margins = group.loc[raw_best, "best_minus_runner_up_log_evidence"].astype(float)
            row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
            row.update(
                {
                    "model": model,
                    "events": events,
                    "required_complete_events": required_complete_events,
                    "incomplete_core_events": incomplete_core_events,
                    "ambiguous_events": ambiguous_events,
                    "margin_threshold": threshold,
                    "raw_best_events": int(raw_best.sum()),
                    "raw_best_fraction": float(raw_best.mean()) if events else 0.0,
                    "confident_claims": int(confident.sum()),
                    "confident_claim_fraction": float(confident.mean()) if events else 0.0,
                    "mean_winning_margin_when_best": (
                        float(winning_margins.mean()) if not winning_margins.empty else np.nan
                    ),
                    "median_winning_margin_when_best": (
                        float(winning_margins.median()) if not winning_margins.empty else np.nan
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def session_exact_core_model_claim_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact core model claims by session."""

    return exact_core_model_claim_summary(decisions, group_cols=("session",))


def exact_trajectory_dynamics_summary(
    decisions: pd.DataFrame,
    *,
    trajectory_models: tuple[str, ...] = DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize exact trajectory-vs-static decisions from exact core claims."""

    columns = [
        *group_cols,
        "events",
        "required_complete_events",
        "incomplete_core_events",
        "ambiguous_events",
        "margin_threshold",
        "trajectory_raw_best_events",
        "nontrajectory_raw_best_events",
        "trajectory_raw_best_fraction",
        "trajectory_confident_claims",
        "nontrajectory_confident_claims",
        "trajectory_confident_claim_fraction",
        "most_common_trajectory_claim_model",
        "most_common_nontrajectory_claim_model",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    trajectory_set = set(str(model) for model in trajectory_models)
    rows: list[dict[str, object]] = []
    groups = [((), decisions)] if not group_cols else decisions.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        events = int(len(group))
        complete = group["required_models_complete"].fillna(False).astype(bool)
        raw_trajectory = group["best_core_model"].fillna("").astype(str).isin(trajectory_set)
        claim_model = group["claim_model"].fillna("").astype(str)
        trajectory_claims = claim_model.isin(trajectory_set)
        nontrajectory_claims = ~claim_model.isin((*trajectory_set, "ambiguous", "incomplete_core", ""))
        trajectory_claim_values = claim_model[trajectory_claims]
        nontrajectory_claim_values = claim_model[nontrajectory_claims]
        threshold = float(group["margin_threshold"].dropna().iloc[0]) if "margin_threshold" in group else np.nan
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": events,
                "required_complete_events": int(complete.sum()),
                "incomplete_core_events": int((claim_model == "incomplete_core").sum()),
                "ambiguous_events": int((claim_model == "ambiguous").sum()),
                "margin_threshold": threshold,
                "trajectory_raw_best_events": int(raw_trajectory.sum()),
                "nontrajectory_raw_best_events": int((~raw_trajectory).sum()),
                "trajectory_raw_best_fraction": float(raw_trajectory.mean()) if events else 0.0,
                "trajectory_confident_claims": int(trajectory_claims.sum()),
                "nontrajectory_confident_claims": int(nontrajectory_claims.sum()),
                "trajectory_confident_claim_fraction": float(trajectory_claims.mean()) if events else 0.0,
                "most_common_trajectory_claim_model": (
                    "" if trajectory_claim_values.empty else str(trajectory_claim_values.value_counts().index[0])
                ),
                "most_common_nontrajectory_claim_model": (
                    "" if nontrajectory_claim_values.empty else str(nontrajectory_claim_values.value_counts().index[0])
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def exact_trajectory_dynamics_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics claims across margin thresholds."""

    rows = []
    for threshold in thresholds:
        decisions = exact_core_model_claim_decisions(df, margin_threshold=float(threshold))
        rows.append(exact_trajectory_dynamics_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def exact_trajectory_nontrajectory_margin_decisions(
    df: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
    trajectory_models: tuple[str, ...] = DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Return per-event best-trajectory-vs-best-nontrajectory margins.

    Exact-core winner tables can understate the broader trajectory-dynamics
    claim when trajectory models split support among themselves. This table asks
    the family-level question directly: does the best exact trajectory row beat
    the best required exact nontrajectory row?
    """

    columns = [
        "session",
        "event_index",
        "required_models_present",
        "required_models_total",
        "required_models_complete",
        "missing_required_models",
        "present_required_models",
        "margin_threshold",
        "best_trajectory_model",
        "best_trajectory_log_evidence",
        "best_nontrajectory_model",
        "best_nontrajectory_log_evidence",
        "trajectory_minus_nontrajectory_log_evidence",
        "trajectory_raw_win",
        "trajectory_confident_claim",
        "nontrajectory_confident_claim",
        "margin_decision",
    ]
    required = tuple(str(model) for model in required_models)
    required_set = set(required)
    trajectory_set = set(str(model) for model in trajectory_models)
    df = _ensure_evidence_support_columns(df)
    if "model" not in df or "session" not in df or "event_index" not in df:
        return pd.DataFrame(columns=columns)

    status_ok = df["status"].eq("success") if "status" in df else pd.Series(True, index=df.index)
    comparable = df["evidence_comparable"].fillna(False).astype(bool)
    ok = df[status_ok & comparable].copy()
    if ok.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(["session", "event_index"], sort=True):
        session, event_index = key
        core = group[group["model"].astype(str).isin(required_set)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        complete = not missing
        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        if trajectory.empty or nontrajectory.empty:
            best_trajectory_model = ""
            best_trajectory_value = np.nan
            best_nontrajectory_model = ""
            best_nontrajectory_value = np.nan
            delta = np.nan
            raw_win = False
            trajectory_claim = False
            nontrajectory_claim = False
            margin_decision = "incomplete_core"
        else:
            best_trajectory = trajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_nontrajectory = nontrajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_trajectory_value = float(best_trajectory["log_evidence"])
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_nontrajectory_value = float(best_nontrajectory["log_evidence"])
            delta = best_trajectory_value - best_nontrajectory_value
            raw_win = bool(delta > 0.0)
            trajectory_claim = bool(complete and delta >= float(margin_threshold))
            nontrajectory_claim = bool(complete and delta <= -float(margin_threshold))
            if not complete:
                margin_decision = "incomplete_core"
            elif trajectory_claim:
                margin_decision = "trajectory"
            elif nontrajectory_claim:
                margin_decision = "nontrajectory"
            else:
                margin_decision = "ambiguous"
        rows.append(
            {
                "session": str(session),
                "event_index": int(event_index),
                "required_models_present": int(len(present)),
                "required_models_total": int(len(required)),
                "required_models_complete": bool(complete),
                "missing_required_models": " ".join(missing),
                "present_required_models": " ".join(present),
                "margin_threshold": float(margin_threshold),
                "best_trajectory_model": best_trajectory_model,
                "best_trajectory_log_evidence": float(best_trajectory_value),
                "best_nontrajectory_model": best_nontrajectory_model,
                "best_nontrajectory_log_evidence": float(best_nontrajectory_value),
                "trajectory_minus_nontrajectory_log_evidence": float(delta),
                "trajectory_raw_win": raw_win,
                "trajectory_confident_claim": trajectory_claim,
                "nontrajectory_confident_claim": nontrajectory_claim,
                "margin_decision": margin_decision,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def exact_trajectory_nontrajectory_margin_summary(
    decisions: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize best-trajectory-vs-best-nontrajectory margins."""

    columns = [
        *group_cols,
        "events",
        "required_complete_events",
        "incomplete_core_events",
        "margin_threshold",
        "trajectory_raw_wins",
        "nontrajectory_raw_wins",
        "raw_ties",
        "trajectory_raw_win_fraction",
        "trajectory_confident_claims",
        "nontrajectory_confident_claims",
        "ambiguous_events",
        "trajectory_confident_claim_fraction",
        "nontrajectory_confident_claim_fraction",
        "ambiguous_fraction",
        "mean_trajectory_minus_nontrajectory_log_evidence",
        "median_trajectory_minus_nontrajectory_log_evidence",
        "min_trajectory_minus_nontrajectory_log_evidence",
        "max_trajectory_minus_nontrajectory_log_evidence",
        "most_common_best_trajectory_model",
        "most_common_best_nontrajectory_model",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = [((), decisions)] if not group_cols else decisions.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = group["trajectory_minus_nontrajectory_log_evidence"].astype(float).dropna()
        events = int(len(group))
        complete = group["required_models_complete"].fillna(False).astype(bool)
        trajectory_claims = int(group["trajectory_confident_claim"].fillna(False).astype(bool).sum())
        nontrajectory_claims = int(group["nontrajectory_confident_claim"].fillna(False).astype(bool).sum())
        ambiguous = int((group["margin_decision"] == "ambiguous").sum())
        best_trajectory = group["best_trajectory_model"].replace("", pd.NA).dropna().astype(str)
        best_nontrajectory = group["best_nontrajectory_model"].replace("", pd.NA).dropna().astype(str)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": events,
                "required_complete_events": int(complete.sum()),
                "incomplete_core_events": int((group["margin_decision"] == "incomplete_core").sum()),
                "margin_threshold": float(group["margin_threshold"].dropna().iloc[0]),
                "trajectory_raw_wins": int((delta > 0.0).sum()),
                "nontrajectory_raw_wins": int((delta < 0.0).sum()),
                "raw_ties": int((delta == 0.0).sum()),
                "trajectory_raw_win_fraction": float((delta > 0.0).mean()) if not delta.empty else 0.0,
                "trajectory_confident_claims": trajectory_claims,
                "nontrajectory_confident_claims": nontrajectory_claims,
                "ambiguous_events": ambiguous,
                "trajectory_confident_claim_fraction": float(trajectory_claims / max(events, 1)),
                "nontrajectory_confident_claim_fraction": float(nontrajectory_claims / max(events, 1)),
                "ambiguous_fraction": float(ambiguous / max(events, 1)),
                "mean_trajectory_minus_nontrajectory_log_evidence": float(delta.mean()) if not delta.empty else np.nan,
                "median_trajectory_minus_nontrajectory_log_evidence": (
                    float(delta.median()) if not delta.empty else np.nan
                ),
                "min_trajectory_minus_nontrajectory_log_evidence": float(delta.min()) if not delta.empty else np.nan,
                "max_trajectory_minus_nontrajectory_log_evidence": float(delta.max()) if not delta.empty else np.nan,
                "most_common_best_trajectory_model": (
                    "" if best_trajectory.empty else str(best_trajectory.value_counts().index[0])
                ),
                "most_common_best_nontrajectory_model": (
                    "" if best_nontrajectory.empty else str(best_nontrajectory.value_counts().index[0])
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def exact_trajectory_nontrajectory_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize best-trajectory-vs-best-nontrajectory margins across thresholds."""

    rows = []
    for threshold in thresholds:
        decisions = exact_trajectory_nontrajectory_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(exact_trajectory_nontrajectory_margin_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def session_exact_trajectory_nontrajectory_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize best-trajectory-vs-best-nontrajectory threshold sensitivity by session."""

    rows = []
    for threshold in thresholds:
        decisions = exact_trajectory_nontrajectory_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(exact_trajectory_nontrajectory_margin_summary(decisions, group_cols=("session",)))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "session"]).reset_index(drop=True)


def rat_exact_trajectory_nontrajectory_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize trajectory-vs-nontrajectory margins by rat."""

    return exact_trajectory_nontrajectory_margin_summary(_with_rat(decisions), group_cols=("rat",))


def rat_exact_trajectory_nontrajectory_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize trajectory-vs-nontrajectory threshold sensitivity by rat."""

    rows = []
    for threshold in thresholds:
        decisions = exact_trajectory_nontrajectory_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(rat_exact_trajectory_nontrajectory_margin_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "rat"]).reset_index(drop=True)


def leave_one_rat_out_exact_trajectory_nontrajectory_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize trajectory-vs-nontrajectory margins after excluding each rat."""

    return _leave_one_rat_out_summary(decisions, exact_trajectory_nontrajectory_margin_summary)


def leave_one_rat_out_exact_trajectory_nontrajectory_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize trajectory-vs-nontrajectory threshold sensitivity after excluding each rat."""

    rows = []
    for threshold in thresholds:
        decisions = exact_trajectory_nontrajectory_margin_decisions(df, margin_threshold=float(threshold))
        rows.append(leave_one_rat_out_exact_trajectory_nontrajectory_margin_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "held_out_rat"]).reset_index(drop=True)


def rat_bootstrap_exact_trajectory_nontrajectory_margin_summary(
    decisions: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals for trajectory-vs-nontrajectory margins."""

    complete = decisions[
        decisions["required_models_complete"].fillna(False).astype(bool)
        & decisions["trajectory_minus_nontrajectory_log_evidence"].notna()
    ].copy()
    return _rat_bootstrap_margin_summary(
        complete,
        delta_col="trajectory_minus_nontrajectory_log_evidence",
        positive_claim_col="trajectory_confident_claim",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def rat_bootstrap_exact_trajectory_nontrajectory_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals for trajectory-vs-nontrajectory thresholds."""

    rows = []
    for threshold in thresholds:
        decisions = exact_trajectory_nontrajectory_margin_decisions(df, margin_threshold=float(threshold))
        summary = rat_bootstrap_exact_trajectory_nontrajectory_margin_summary(
            decisions,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        rows.append(_insert_margin_threshold(summary, threshold))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def exact_trajectory_nontrajectory_gate_summary(
    df: pd.DataFrame,
    *,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
    min_raw_win_fraction: float = DEFAULT_PAPER_MIN_RAW_WIN_FRACTION,
    min_confident_claim_fraction: float = DEFAULT_PAPER_MIN_CONFIDENT_CLAIM_FRACTION,
) -> pd.DataFrame:
    """Return pass/fail gates for the exact trajectory-family claim."""

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

    decisions = exact_trajectory_nontrajectory_margin_decisions(df, margin_threshold=margin_threshold)
    summary = exact_trajectory_nontrajectory_margin_summary(decisions)
    if summary.empty:
        add("complete_family_margin_events_present", False, 0, "required_complete_events > 0")
        result = pd.DataFrame(rows, columns=columns)
    else:
        row = summary.iloc[0]
        complete_events = int(row["required_complete_events"])
        add(
            "complete_family_margin_events_present",
            complete_events > 0,
            complete_events,
            "required_complete_events > 0",
        )
        add(
            "trajectory_family_raw_win_majority",
            float(row["trajectory_raw_win_fraction"]) > float(min_raw_win_fraction),
            f"{float(row['trajectory_raw_win_fraction']):.6g}",
            f"trajectory_raw_win_fraction > {float(min_raw_win_fraction):g}",
        )
        add(
            "trajectory_family_confident_claim_majority",
            float(row["trajectory_confident_claim_fraction"]) > float(min_confident_claim_fraction),
            f"{float(row['trajectory_confident_claim_fraction']):.6g}",
            f"trajectory_confident_claim_fraction > {float(min_confident_claim_fraction):g}",
            f"margin_threshold={float(margin_threshold):g}",
        )
        add(
            "no_confident_nontrajectory_claims",
            int(row["nontrajectory_confident_claims"]) == 0,
            int(row["nontrajectory_confident_claims"]),
            "nontrajectory_confident_claims == 0",
        )

        session = exact_trajectory_nontrajectory_margin_summary(decisions, group_cols=("session",))
        if session.empty:
            add("all_sessions_have_trajectory_family_claims", False, 0, "min session trajectory claims > 0")
            add(
                "all_sessions_have_trajectory_family_claim_majority",
                False,
                np.nan,
                "min session trajectory claim fraction > 0.5",
            )
            add("all_sessions_have_no_nontrajectory_claims", False, 0, "max session nontrajectory claims == 0")
            add(
                "all_sessions_family_mean_delta_positive",
                False,
                np.nan,
                "min session mean delta > 0",
            )
            add(
                "all_sessions_family_median_delta_positive",
                False,
                np.nan,
                "min session median delta > 0",
            )
        else:
            min_session_trajectory = int(session["trajectory_confident_claims"].min())
            min_session_claim_fraction = float(session["trajectory_confident_claim_fraction"].min())
            max_session_nontrajectory = int(session["nontrajectory_confident_claims"].max())
            min_session_mean = float(session["mean_trajectory_minus_nontrajectory_log_evidence"].min())
            min_session_median = float(session["median_trajectory_minus_nontrajectory_log_evidence"].min())
            add(
                "all_sessions_have_trajectory_family_claims",
                min_session_trajectory > 0,
                min_session_trajectory,
                "min session trajectory_confident_claims > 0",
            )
            add(
                "all_sessions_have_trajectory_family_claim_majority",
                min_session_claim_fraction > float(min_confident_claim_fraction),
                f"{min_session_claim_fraction:.6g}",
                f"min session trajectory claim fraction > {float(min_confident_claim_fraction):g}",
            )
            add(
                "all_sessions_have_no_nontrajectory_claims",
                max_session_nontrajectory == 0,
                max_session_nontrajectory,
                "max session nontrajectory_confident_claims == 0",
            )
            add(
                "all_sessions_family_mean_delta_positive",
                min_session_mean > 0.0,
                f"{min_session_mean:.6g}",
                "min session mean delta > 0",
            )
            add(
                "all_sessions_family_median_delta_positive",
                min_session_median > 0.0,
                f"{min_session_median:.6g}",
                "min session median delta > 0",
            )

        rat = rat_exact_trajectory_nontrajectory_margin_summary(decisions)
        if rat.empty:
            add(
                "all_rats_have_trajectory_family_claim_majority",
                False,
                np.nan,
                "min rat trajectory claim fraction > 0.5",
            )
            add(
                "all_rats_have_no_nontrajectory_claims",
                False,
                np.nan,
                "max rat nontrajectory claims == 0",
            )
            add(
                "all_rats_family_mean_delta_positive",
                False,
                np.nan,
                "min rat mean delta > 0",
            )
            add(
                "all_rats_family_median_delta_positive",
                False,
                np.nan,
                "min rat median delta > 0",
            )
        else:
            min_rat_claim_fraction = float(rat["trajectory_confident_claim_fraction"].min())
            max_rat_nontrajectory_claims = int(rat["nontrajectory_confident_claims"].max())
            min_rat_mean = float(rat["mean_trajectory_minus_nontrajectory_log_evidence"].min())
            min_rat_median = float(rat["median_trajectory_minus_nontrajectory_log_evidence"].min())
            add(
                "all_rats_have_trajectory_family_claim_majority",
                min_rat_claim_fraction > float(min_confident_claim_fraction),
                f"{min_rat_claim_fraction:.6g}",
                f"min rat trajectory claim fraction > {float(min_confident_claim_fraction):g}",
            )
            add(
                "all_rats_have_no_nontrajectory_claims",
                max_rat_nontrajectory_claims == 0,
                max_rat_nontrajectory_claims,
                "max rat nontrajectory claims == 0",
            )
            add(
                "all_rats_family_mean_delta_positive",
                min_rat_mean > 0.0,
                f"{min_rat_mean:.6g}",
                "min rat mean delta > 0",
            )
            add(
                "all_rats_family_median_delta_positive",
                min_rat_median > 0.0,
                f"{min_rat_median:.6g}",
                "min rat median delta > 0",
            )

        leave_one = leave_one_rat_out_exact_trajectory_nontrajectory_margin_summary(decisions)
        if leave_one.empty:
            add(
                "leave_one_rat_out_family_claim_majority",
                False,
                np.nan,
                "min leave-one-rat-out trajectory claim fraction > 0.5",
            )
            add(
                "leave_one_rat_out_no_nontrajectory_claims",
                False,
                np.nan,
                "max leave-one-rat-out nontrajectory claims == 0",
            )
            add(
                "leave_one_rat_out_family_mean_delta_positive",
                False,
                np.nan,
                "min leave-one-rat-out mean delta > 0",
            )
            add(
                "leave_one_rat_out_family_median_delta_positive",
                False,
                np.nan,
                "min leave-one-rat-out median delta > 0",
            )
        else:
            min_leave_one_claim_fraction = float(leave_one["trajectory_confident_claim_fraction"].min())
            max_leave_one_nontrajectory_claims = int(leave_one["nontrajectory_confident_claims"].max())
            min_leave_one_mean = float(leave_one["mean_trajectory_minus_nontrajectory_log_evidence"].min())
            min_leave_one_median = float(leave_one["median_trajectory_minus_nontrajectory_log_evidence"].min())
            add(
                "leave_one_rat_out_family_claim_majority",
                min_leave_one_claim_fraction > float(min_confident_claim_fraction),
                f"{min_leave_one_claim_fraction:.6g}",
                f"min leave-one-rat-out trajectory claim fraction > {float(min_confident_claim_fraction):g}",
            )
            add(
                "leave_one_rat_out_no_nontrajectory_claims",
                max_leave_one_nontrajectory_claims == 0,
                max_leave_one_nontrajectory_claims,
                "max leave-one-rat-out nontrajectory claims == 0",
            )
            add(
                "leave_one_rat_out_family_mean_delta_positive",
                min_leave_one_mean > 0.0,
                f"{min_leave_one_mean:.6g}",
                "min leave-one-rat-out mean delta > 0",
            )
            add(
                "leave_one_rat_out_family_median_delta_positive",
                min_leave_one_median > 0.0,
                f"{min_leave_one_median:.6g}",
                "min leave-one-rat-out median delta > 0",
            )

        bootstrap = rat_bootstrap_exact_trajectory_nontrajectory_margin_summary(
            decisions,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        if bootstrap.empty:
            add("rat_bootstrap_family_claim_fraction_ci_majority", False, np.nan, "positive_claim_fraction_ci95_low > 0.5")
            add("rat_bootstrap_family_mean_delta_ci_positive", False, np.nan, "mean_delta_ci95_low > 0")
            add("rat_bootstrap_family_median_delta_ci_positive", False, np.nan, "median_delta_ci95_low > 0")
        else:
            boot = bootstrap.iloc[0]
            add(
                "rat_bootstrap_family_claim_fraction_ci_majority",
                float(boot["positive_claim_fraction_ci95_low"]) > float(min_confident_claim_fraction),
                f"{float(boot['positive_claim_fraction_ci95_low']):.6g}",
                f"positive_claim_fraction_ci95_low > {float(min_confident_claim_fraction):g}",
            )
            add(
                "rat_bootstrap_family_mean_delta_ci_positive",
                float(boot["mean_delta_ci95_low"]) > 0.0,
                f"{float(boot['mean_delta_ci95_low']):.6g}",
                "mean_delta_ci95_low > 0",
            )
            add(
                "rat_bootstrap_family_median_delta_ci_positive",
                float(boot["median_delta_ci95_low"]) > 0.0,
                f"{float(boot['median_delta_ci95_low']):.6g}",
                "median_delta_ci95_low > 0",
            )
        result = pd.DataFrame(rows, columns=columns)

    overall_pass = bool(result["passed"].all()) if not result.empty else False
    overall = pd.DataFrame(
        [
            {
                "gate": "overall",
                "passed": overall_pass,
                "observed": f"{int(result['passed'].sum())}/{len(result)} gates passed",
                "criterion": "all gates pass",
                "details": f"margin_threshold={float(margin_threshold):g}",
            }
        ],
        columns=columns,
    )
    return pd.concat([result, overall], ignore_index=True)


def session_exact_trajectory_dynamics_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics claims by session."""

    return exact_trajectory_dynamics_summary(decisions, group_cols=("session",))


def session_exact_trajectory_dynamics_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics threshold sensitivity by session."""

    rows = []
    for threshold in thresholds:
        decisions = exact_core_model_claim_decisions(df, margin_threshold=float(threshold))
        rows.append(session_exact_trajectory_dynamics_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "session"]).reset_index(drop=True)


def rat_exact_trajectory_dynamics_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics claims by rat."""

    return exact_trajectory_dynamics_summary(_with_rat(decisions), group_cols=("rat",))


def rat_exact_trajectory_dynamics_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics threshold sensitivity by rat."""

    rows = []
    for threshold in thresholds:
        decisions = exact_core_model_claim_decisions(df, margin_threshold=float(threshold))
        rows.append(rat_exact_trajectory_dynamics_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "rat"]).reset_index(drop=True)


def leave_one_rat_out_exact_trajectory_dynamics_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics claims after excluding each rat."""

    return _leave_one_rat_out_summary(decisions, exact_trajectory_dynamics_summary)


def leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize exact trajectory-dynamics threshold sensitivity after excluding each rat."""

    rows = []
    for threshold in thresholds:
        decisions = exact_core_model_claim_decisions(df, margin_threshold=float(threshold))
        rows.append(leave_one_rat_out_exact_trajectory_dynamics_summary(decisions))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["margin_threshold", "held_out_rat"]).reset_index(drop=True)


def rat_bootstrap_exact_trajectory_dynamics_summary(
    decisions: pd.DataFrame,
    *,
    trajectory_models: tuple[str, ...] = DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals for exact trajectory-dynamics claims."""

    columns = [
        "bootstrap_unit",
        "bootstrap_replicates",
        "random_seed",
        "observed_events",
        "observed_rats",
        "observed_required_complete_fraction",
        "required_complete_fraction_ci95_low",
        "required_complete_fraction_ci95_high",
        "observed_trajectory_raw_best_fraction",
        "trajectory_raw_best_fraction_ci95_low",
        "trajectory_raw_best_fraction_ci95_high",
        "observed_trajectory_confident_claim_fraction",
        "trajectory_confident_claim_fraction_ci95_low",
        "trajectory_confident_claim_fraction_ci95_high",
        "probability_trajectory_confident_claim_fraction_gt_half",
        "observed_nontrajectory_confident_claim_fraction",
        "nontrajectory_confident_claim_fraction_ci95_low",
        "nontrajectory_confident_claim_fraction_ci95_high",
        "probability_nontrajectory_confident_claim_fraction_eq_0",
        "observed_ambiguous_fraction",
        "ambiguous_fraction_ci95_low",
        "ambiguous_fraction_ci95_high",
        "observed_incomplete_core_fraction",
        "incomplete_core_fraction_ci95_low",
        "incomplete_core_fraction_ci95_high",
    ]
    decisions = _with_rat(decisions)
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    missing = [column for column in ("rat", "best_core_model", "claim_model") if column not in decisions]
    if missing:
        raise KeyError(f"decisions is missing required columns: {missing}")

    observed = _trajectory_bootstrap_metrics(decisions, trajectory_models=trajectory_models)
    rat_counts = _trajectory_bootstrap_count_table(decisions, trajectory_models=trajectory_models)
    rats = rat_counts["rat"].tolist()
    rng = np.random.default_rng(int(random_seed))
    count_columns = [
        "event_count",
        "required_complete_count",
        "trajectory_raw_best_count",
        "trajectory_confident_claim_count",
        "nontrajectory_confident_claim_count",
        "ambiguous_count",
        "incomplete_core_count",
    ]
    count_array = rat_counts[count_columns].to_numpy(dtype=float)
    sampled = rng.integers(0, len(rats), size=(int(n_bootstrap), len(rats)))
    replicate_counts = count_array[sampled].sum(axis=1)
    denominators = replicate_counts[:, 0]
    replicates = pd.DataFrame(
        {
            "required_complete_fraction": replicate_counts[:, 1] / denominators,
            "trajectory_raw_best_fraction": replicate_counts[:, 2] / denominators,
            "trajectory_confident_claim_fraction": replicate_counts[:, 3] / denominators,
            "nontrajectory_confident_claim_fraction": replicate_counts[:, 4] / denominators,
            "ambiguous_fraction": replicate_counts[:, 5] / denominators,
            "incomplete_core_fraction": replicate_counts[:, 6] / denominators,
        }
    )

    def ci(metric: str, q: float) -> float:
        return float(np.nanquantile(replicates[metric].to_numpy(dtype=float), q))

    row = {
        "bootstrap_unit": "rat",
        "bootstrap_replicates": int(n_bootstrap),
        "random_seed": int(random_seed),
        "observed_events": int(len(decisions)),
        "observed_rats": int(len(rats)),
        "observed_required_complete_fraction": observed["required_complete_fraction"],
        "required_complete_fraction_ci95_low": ci("required_complete_fraction", 0.025),
        "required_complete_fraction_ci95_high": ci("required_complete_fraction", 0.975),
        "observed_trajectory_raw_best_fraction": observed["trajectory_raw_best_fraction"],
        "trajectory_raw_best_fraction_ci95_low": ci("trajectory_raw_best_fraction", 0.025),
        "trajectory_raw_best_fraction_ci95_high": ci("trajectory_raw_best_fraction", 0.975),
        "observed_trajectory_confident_claim_fraction": observed["trajectory_confident_claim_fraction"],
        "trajectory_confident_claim_fraction_ci95_low": ci("trajectory_confident_claim_fraction", 0.025),
        "trajectory_confident_claim_fraction_ci95_high": ci("trajectory_confident_claim_fraction", 0.975),
        "probability_trajectory_confident_claim_fraction_gt_half": float(
            (replicates["trajectory_confident_claim_fraction"] > 0.5).mean()
        ),
        "observed_nontrajectory_confident_claim_fraction": observed["nontrajectory_confident_claim_fraction"],
        "nontrajectory_confident_claim_fraction_ci95_low": ci("nontrajectory_confident_claim_fraction", 0.025),
        "nontrajectory_confident_claim_fraction_ci95_high": ci("nontrajectory_confident_claim_fraction", 0.975),
        "probability_nontrajectory_confident_claim_fraction_eq_0": float(
            (replicates["nontrajectory_confident_claim_fraction"] == 0.0).mean()
        ),
        "observed_ambiguous_fraction": observed["ambiguous_fraction"],
        "ambiguous_fraction_ci95_low": ci("ambiguous_fraction", 0.025),
        "ambiguous_fraction_ci95_high": ci("ambiguous_fraction", 0.975),
        "observed_incomplete_core_fraction": observed["incomplete_core_fraction"],
        "incomplete_core_fraction_ci95_low": ci("incomplete_core_fraction", 0.025),
        "incomplete_core_fraction_ci95_high": ci("incomplete_core_fraction", 0.975),
    }
    return pd.DataFrame([row], columns=columns)


def rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return rat-cluster bootstrap intervals across trajectory-dynamics thresholds."""

    rows = []
    for threshold in thresholds:
        decisions = exact_core_model_claim_decisions(df, margin_threshold=float(threshold))
        summary = rat_bootstrap_exact_trajectory_dynamics_summary(
            decisions,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        rows.append(_insert_margin_threshold(summary, threshold))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("margin_threshold").reset_index(drop=True)


def exact_trajectory_dynamics_gate_summary(
    df: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
    trajectory_models: tuple[str, ...] = DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
    min_raw_win_fraction: float = DEFAULT_PAPER_MIN_RAW_WIN_FRACTION,
    min_confident_claim_fraction: float = DEFAULT_PAPER_MIN_CONFIDENT_CLAIM_FRACTION,
) -> pd.DataFrame:
    """Return pass/fail gates for the broader exact trajectory-dynamics claim."""

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

    trajectory_set = set(str(model) for model in trajectory_models)
    decisions = exact_core_model_claim_decisions(
        df,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    coverage = _required_full_core_model_coverage(df, required_models)
    failures = int((df["status"] != "success").sum()) if "status" in df else 0
    add("no_scoring_failures", failures == 0, failures, "status failures == 0")
    add("exact_core_events_present", not decisions.empty, int(len(decisions)), "exact core events > 0")
    add(
        "required_exact_core_models_present",
        bool(coverage["passed"]),
        coverage["observed"],
        "all events include required exact core models",
        str(coverage["details"]),
    )

    if decisions.empty:
        add("exact_trajectory_raw_best_majority", False, 0.0, "trajectory raw best fraction > 0.5")
        add(
            "exact_trajectory_confident_claim_majority",
            False,
            0.0,
            "trajectory confident claim fraction > 0.5",
            f"margin_threshold={float(margin_threshold):g}",
        )
        add("no_confident_static_or_other_core_claims", False, 0, "nontrajectory confident claims == 0")
    else:
        raw_best_trajectory = decisions["best_core_model"].fillna("").astype(str).isin(trajectory_set)
        confident_trajectory = decisions["claim_model"].fillna("").astype(str).isin(trajectory_set)
        nontrajectory_claims = decisions[
            ~decisions["claim_model"].fillna("").astype(str).isin((*trajectory_set, "ambiguous", "incomplete_core"))
        ]
        raw_fraction = float(raw_best_trajectory.mean())
        claim_fraction = float(confident_trajectory.mean())
        add(
            "exact_trajectory_raw_best_majority",
            raw_fraction > float(min_raw_win_fraction),
            raw_fraction,
            f"trajectory raw best fraction > {float(min_raw_win_fraction):g}",
            "trajectory_models=" + " ".join(str(model) for model in trajectory_models),
        )
        add(
            "exact_trajectory_confident_claim_majority",
            claim_fraction > float(min_confident_claim_fraction),
            claim_fraction,
            f"trajectory confident claim fraction > {float(min_confident_claim_fraction):g}",
            f"margin_threshold={float(margin_threshold):g}",
        )
        add(
            "no_confident_static_or_other_core_claims",
            int(len(nontrajectory_claims)) == 0,
            int(len(nontrajectory_claims)),
            "nontrajectory confident claims == 0",
        )

    result = pd.DataFrame(rows, columns=columns)
    overall = pd.DataFrame(
        [
            {
                "gate": "overall",
                "passed": bool(result["passed"].all()),
                "observed": f"{int(result['passed'].sum())}/{len(result)} gates passed",
                "criterion": "all gates pass",
                "details": f"margin_threshold={float(margin_threshold):g}",
            }
        ],
        columns=columns,
    )
    return pd.concat([result, overall], ignore_index=True)


def _required_full_core_model_coverage(
    df: pd.DataFrame,
    required_models: tuple[str, ...],
) -> dict[str, object]:
    required = tuple(str(model) for model in required_models)
    if not required:
        return {"passed": True, "observed": "0/0", "details": "required_models="}
    table = required_full_core_model_coverage_table(df, required)
    if table.empty:
        return {
            "passed": False,
            "observed": f"0/{len(required)}",
            "details": "missing=" + " ".join(required),
        }

    missing_union: set[str] = set()
    for missing in table["missing_required_models"].dropna().astype(str):
        missing_union.update(model for model in missing.split() if model)
    min_present = int(table["required_models_present"].min())
    missing = " ".join(sorted(missing_union))
    return {
        "passed": bool(table["required_models_complete"].fillna(False).astype(bool).all()),
        "observed": f"{min_present}/{len(required)}",
        "details": "missing=" + missing if missing else "required_models=" + " ".join(required),
    }


def _insert_margin_threshold(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = summary.copy()
    insert_at = min(3, len(out.columns))
    out.insert(insert_at, "margin_threshold", float(threshold))
    return out


def _with_rat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "rat" not in out:
        if "session" in out:
            out["rat"] = out["session"].map(_rat_from_session)
        else:
            out["rat"] = pd.Series(dtype=str)
    return out


def _rat_from_session(session: object) -> str:
    return str(session).replace("\\", "/").split("/", 1)[0]


def _leave_one_rat_out_summary(frame: pd.DataFrame, summary_func) -> pd.DataFrame:
    frame = _with_rat(frame)
    summary_columns = list(summary_func(frame).columns)
    columns = ["held_out_rat", "included_rats", *summary_columns]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    rats = sorted(frame["rat"].dropna().astype(str).unique())
    for held_out_rat in rats:
        retained = frame[frame["rat"].astype(str) != held_out_rat].copy()
        summary = summary_func(retained)
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "held_out_rat": held_out_rat,
                "included_rats": " ".join(sorted(retained["rat"].dropna().astype(str).unique())),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _rat_bootstrap_margin_summary(
    frame: pd.DataFrame,
    *,
    delta_col: str,
    positive_claim_col: str,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    columns = [
        "bootstrap_unit",
        "bootstrap_replicates",
        "random_seed",
        "observed_events",
        "observed_rats",
        "observed_positive_raw_win_fraction",
        "positive_raw_win_fraction_ci95_low",
        "positive_raw_win_fraction_ci95_high",
        "observed_positive_claim_fraction",
        "positive_claim_fraction_ci95_low",
        "positive_claim_fraction_ci95_high",
        "observed_mean_delta",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "probability_mean_delta_gt_0",
        "observed_median_delta",
        "median_delta_ci95_low",
        "median_delta_ci95_high",
        "probability_median_delta_gt_0",
    ]
    frame = _with_rat(frame)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in ("rat", delta_col, positive_claim_col) if column not in frame]
    if missing:
        raise KeyError(f"frame is missing required columns: {missing}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    observed = _bootstrap_margin_metrics(frame, delta_col=delta_col, positive_claim_col=positive_claim_col)
    rats = sorted(frame["rat"].dropna().astype(str).unique())
    by_rat = {rat: frame[frame["rat"].astype(str).eq(rat)] for rat in rats}
    rng = np.random.default_rng(int(random_seed))
    replicate_rows: list[dict[str, float]] = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        sample = pd.concat([by_rat[rat] for rat in sampled], ignore_index=True)
        replicate_rows.append(
            _bootstrap_margin_metrics(sample, delta_col=delta_col, positive_claim_col=positive_claim_col)
        )
    replicates = pd.DataFrame(replicate_rows)

    def ci(metric: str, q: float) -> float:
        return float(np.nanquantile(replicates[metric].to_numpy(dtype=float), q))

    row = {
        "bootstrap_unit": "rat",
        "bootstrap_replicates": int(n_bootstrap),
        "random_seed": int(random_seed),
        "observed_events": int(len(frame)),
        "observed_rats": int(len(rats)),
        "observed_positive_raw_win_fraction": observed["positive_raw_win_fraction"],
        "positive_raw_win_fraction_ci95_low": ci("positive_raw_win_fraction", 0.025),
        "positive_raw_win_fraction_ci95_high": ci("positive_raw_win_fraction", 0.975),
        "observed_positive_claim_fraction": observed["positive_claim_fraction"],
        "positive_claim_fraction_ci95_low": ci("positive_claim_fraction", 0.025),
        "positive_claim_fraction_ci95_high": ci("positive_claim_fraction", 0.975),
        "observed_mean_delta": observed["mean_delta"],
        "mean_delta_ci95_low": ci("mean_delta", 0.025),
        "mean_delta_ci95_high": ci("mean_delta", 0.975),
        "probability_mean_delta_gt_0": float((replicates["mean_delta"] > 0.0).mean()),
        "observed_median_delta": observed["median_delta"],
        "median_delta_ci95_low": ci("median_delta", 0.025),
        "median_delta_ci95_high": ci("median_delta", 0.975),
        "probability_median_delta_gt_0": float((replicates["median_delta"] > 0.0).mean()),
    }
    return pd.DataFrame([row], columns=columns)


def _bootstrap_margin_metrics(
    frame: pd.DataFrame,
    *,
    delta_col: str,
    positive_claim_col: str,
) -> dict[str, float]:
    delta = frame[delta_col].astype(float)
    return {
        "positive_raw_win_fraction": float((delta > 0.0).mean()),
        "positive_claim_fraction": float(frame[positive_claim_col].fillna(False).astype(bool).mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
    }


def _trajectory_bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    trajectory_models: tuple[str, ...],
) -> dict[str, float]:
    trajectory_set = set(str(model) for model in trajectory_models)
    claim_model = frame["claim_model"].fillna("").astype(str)
    return {
        "required_complete_fraction": float(frame["required_models_complete"].fillna(False).astype(bool).mean()),
        "trajectory_raw_best_fraction": float(frame["best_core_model"].fillna("").astype(str).isin(trajectory_set).mean()),
        "trajectory_confident_claim_fraction": float(claim_model.isin(trajectory_set).mean()),
        "nontrajectory_confident_claim_fraction": float(
            (~claim_model.isin((*trajectory_set, "ambiguous", "incomplete_core", ""))).mean()
        ),
        "ambiguous_fraction": float((claim_model == "ambiguous").mean()),
        "incomplete_core_fraction": float((claim_model == "incomplete_core").mean()),
    }


def _trajectory_bootstrap_count_table(
    frame: pd.DataFrame,
    *,
    trajectory_models: tuple[str, ...],
) -> pd.DataFrame:
    trajectory_set = set(str(model) for model in trajectory_models)
    rows = []
    for rat, group in frame.groupby("rat", sort=True):
        claim_model = group["claim_model"].fillna("").astype(str)
        rows.append(
            {
                "rat": str(rat),
                "event_count": int(len(group)),
                "required_complete_count": int(group["required_models_complete"].fillna(False).astype(bool).sum()),
                "trajectory_raw_best_count": int(
                    group["best_core_model"].fillna("").astype(str).isin(trajectory_set).sum()
                ),
                "trajectory_confident_claim_count": int(claim_model.isin(trajectory_set).sum()),
                "nontrajectory_confident_claim_count": int(
                    (~claim_model.isin((*trajectory_set, "ambiguous", "incomplete_core", ""))).sum()
                ),
                "ambiguous_count": int((claim_model == "ambiguous").sum()),
                "incomplete_core_count": int((claim_model == "incomplete_core").sum()),
            }
        )
    return pd.DataFrame(rows)


def _core_margin_summary(margins: pd.DataFrame, *, group_cols: tuple[str, ...]) -> pd.DataFrame:
    columns = [
        *group_cols,
        "events",
        "positive_model",
        "margin_threshold",
        "positive_exact_best_events",
        "non_positive_exact_best_events",
        "positive_exact_best_fraction",
        "positive_confident_core_claims",
        "ambiguous_or_other_best_events",
        "positive_confident_core_claim_fraction",
        "mean_positive_minus_best_other_exact_log_evidence",
        "median_positive_minus_best_other_exact_log_evidence",
        "min_positive_minus_best_other_exact_log_evidence",
        "max_positive_minus_best_other_exact_log_evidence",
        "most_common_best_other_exact_model",
    ]
    if margins.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = [((), margins)] if not group_cols else margins.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = group["positive_minus_best_other_exact_log_evidence"].astype(float)
        events = int(len(group))
        exact_best = int(group["positive_is_exact_best"].fillna(False).astype(bool).sum())
        confident = int(group["positive_confident_core_claim"].fillna(False).astype(bool).sum())
        best_other = group["best_other_exact_model"].fillna("").astype(str)
        best_other = best_other[best_other != ""]
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": events,
                "positive_model": str(group["positive_model"].dropna().iloc[0]),
                "margin_threshold": float(group["margin_threshold"].dropna().iloc[0]),
                "positive_exact_best_events": exact_best,
                "non_positive_exact_best_events": int(events - exact_best),
                "positive_exact_best_fraction": float(exact_best / max(events, 1)),
                "positive_confident_core_claims": confident,
                "ambiguous_or_other_best_events": int(events - confident),
                "positive_confident_core_claim_fraction": float(confident / max(events, 1)),
                "mean_positive_minus_best_other_exact_log_evidence": float(delta.mean()),
                "median_positive_minus_best_other_exact_log_evidence": float(delta.median()),
                "min_positive_minus_best_other_exact_log_evidence": float(delta.min()),
                "max_positive_minus_best_other_exact_log_evidence": float(delta.max()),
                "most_common_best_other_exact_model": "" if best_other.empty else str(best_other.value_counts().index[0]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _paired_margin_summary(decisions: pd.DataFrame, *, group_cols: tuple[str, ...]) -> pd.DataFrame:
    columns = [
        *group_cols,
        "events",
        "positive_model",
        "reference_model",
        "margin_threshold",
        "positive_raw_wins",
        "reference_raw_wins",
        "raw_ties",
        "positive_raw_win_fraction",
        "positive_model_claims",
        "reference_model_claims",
        "ambiguous_events",
        "positive_claim_fraction",
        "reference_claim_fraction",
        "ambiguous_fraction",
        "mean_positive_minus_reference_log_evidence",
        "median_positive_minus_reference_log_evidence",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = [((), decisions)] if not group_cols else decisions.groupby(list(group_cols), sort=True)
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = group["positive_minus_reference_log_evidence"].astype(float)
        events = int(len(group))
        positive_claims = int(group["positive_model_claimed"].fillna(False).astype(bool).sum())
        reference_claims = int((group["margin_decision"] == group["reference_model"]).sum())
        ambiguous = int((group["margin_decision"] == "ambiguous").sum())
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "events": events,
                "positive_model": str(group["positive_model"].dropna().iloc[0]),
                "reference_model": str(group["reference_model"].dropna().iloc[0]),
                "margin_threshold": float(group["margin_threshold"].dropna().iloc[0]),
                "positive_raw_wins": int((delta > 0.0).sum()),
                "reference_raw_wins": int((delta < 0.0).sum()),
                "raw_ties": int((delta == 0.0).sum()),
                "positive_raw_win_fraction": float((delta > 0.0).mean()),
                "positive_model_claims": positive_claims,
                "reference_model_claims": reference_claims,
                "ambiguous_events": ambiguous,
                "positive_claim_fraction": float(positive_claims / max(events, 1)),
                "reference_claim_fraction": float(reference_claims / max(events, 1)),
                "ambiguous_fraction": float(ambiguous / max(events, 1)),
                "mean_positive_minus_reference_log_evidence": float(delta.mean()),
                "median_positive_minus_reference_log_evidence": float(delta.median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def aggregate_all_sessions(shard_glob: str, outdir: Path) -> pd.DataFrame:
    combined = _load_combined(shard_glob)
    outdir.mkdir(parents=True, exist_ok=True)
    paired_decisions = paired_momentum_diffusion_margin_decisions(combined)
    core_margins = exact_sparse_momentum_core_margins(combined)
    exact_core_decisions = exact_core_model_claim_decisions(combined)
    trajectory_nontrajectory_decisions = exact_trajectory_nontrajectory_margin_decisions(combined)

    _write(combined, outdir)
    combined.to_csv(outdir / "all_sessions_event_model_evidence.csv", index=False)
    _summary(combined).to_csv(outdir / "all_sessions_model_evidence_summary.csv", index=False)
    _counts(combined).to_csv(outdir / "all_sessions_best_model_counts.csv", index=False)
    session_model_evidence_summary(combined).to_csv(outdir / "session_model_evidence_summary.csv", index=False)
    session_best_model_counts(combined).to_csv(outdir / "session_best_model_counts.csv", index=False)
    random_effects_model_probabilities(combined).to_csv(outdir / "random_effects_model_probabilities.csv", index=False)
    paper_readiness_gate_summary(combined).to_csv(outdir / "paper_readiness_gate_summary.csv", index=False)
    exact_trajectory_dynamics_gate_summary(combined).to_csv(
        outdir / "exact_trajectory_dynamics_gate_summary.csv",
        index=False,
    )
    exact_trajectory_dynamics_threshold_sensitivity(combined).to_csv(
        outdir / "exact_trajectory_dynamics_threshold_sensitivity.csv",
        index=False,
    )
    session_exact_trajectory_dynamics_threshold_sensitivity(combined).to_csv(
        outdir / "session_exact_trajectory_dynamics_threshold_sensitivity.csv",
        index=False,
    )
    rat_exact_trajectory_dynamics_threshold_sensitivity(combined).to_csv(
        outdir / "rat_exact_trajectory_dynamics_threshold_sensitivity.csv",
        index=False,
    )
    leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity(combined).to_csv(
        outdir / "leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity.csv",
        index=False,
    )
    rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity(combined).to_csv(
        outdir / "rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity.csv",
        index=False,
    )
    trajectory_nontrajectory_decisions.to_csv(
        outdir / "exact_trajectory_nontrajectory_margin_decisions.csv",
        index=False,
    )
    exact_trajectory_nontrajectory_margin_summary(trajectory_nontrajectory_decisions).to_csv(
        outdir / "exact_trajectory_nontrajectory_margin_summary.csv",
        index=False,
    )
    exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_csv(
        outdir / "exact_trajectory_nontrajectory_threshold_sensitivity.csv",
        index=False,
    )
    session_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_csv(
        outdir / "session_exact_trajectory_nontrajectory_threshold_sensitivity.csv",
        index=False,
    )
    rat_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_csv(
        outdir / "rat_exact_trajectory_nontrajectory_threshold_sensitivity.csv",
        index=False,
    )
    leave_one_rat_out_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_csv(
        outdir / "leave_one_rat_out_exact_trajectory_nontrajectory_threshold_sensitivity.csv",
        index=False,
    )
    rat_bootstrap_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_csv(
        outdir / "rat_bootstrap_exact_trajectory_nontrajectory_threshold_sensitivity.csv",
        index=False,
    )
    exact_trajectory_nontrajectory_gate_summary(combined).to_csv(
        outdir / "exact_trajectory_nontrajectory_gate_summary.csv",
        index=False,
    )
    required_full_core_model_coverage_table(combined).to_csv(
        outdir / "required_full_core_model_coverage.csv",
        index=False,
    )
    exact_core_decisions.to_csv(outdir / "exact_core_model_claim_decisions.csv", index=False)
    exact_core_model_claim_summary(exact_core_decisions).to_csv(
        outdir / "exact_core_model_claim_summary.csv",
        index=False,
    )
    session_exact_core_model_claim_summary(exact_core_decisions).to_csv(
        outdir / "session_exact_core_model_claim_summary.csv",
        index=False,
    )
    paired_decisions.to_csv(outdir / "paired_momentum_diffusion_margin_decisions.csv", index=False)
    paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "paired_momentum_diffusion_margin_summary.csv",
        index=False,
    )
    paired_momentum_diffusion_threshold_sensitivity(combined).to_csv(
        outdir / "paired_momentum_diffusion_threshold_sensitivity.csv",
        index=False,
    )
    session_paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "session_paired_momentum_diffusion_margin_summary.csv",
        index=False,
    )
    session_paired_momentum_diffusion_threshold_sensitivity(combined).to_csv(
        outdir / "session_paired_momentum_diffusion_threshold_sensitivity.csv",
        index=False,
    )
    rat_paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "rat_paired_momentum_diffusion_margin_summary.csv",
        index=False,
    )
    leave_one_rat_out_paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "leave_one_rat_out_paired_momentum_diffusion_margin_summary.csv",
        index=False,
    )
    leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity(combined).to_csv(
        outdir / "leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity.csv",
        index=False,
    )
    rat_bootstrap_paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "rat_bootstrap_paired_momentum_diffusion_margin_summary.csv",
        index=False,
    )
    rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity(combined).to_csv(
        outdir / "rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity.csv",
        index=False,
    )
    core_margins.to_csv(outdir / "exact_sparse_momentum_core_margins.csv", index=False)
    exact_sparse_momentum_core_margin_summary(core_margins).to_csv(
        outdir / "exact_sparse_momentum_core_margin_summary.csv",
        index=False,
    )
    exact_sparse_momentum_core_threshold_sensitivity(combined).to_csv(
        outdir / "exact_sparse_momentum_core_threshold_sensitivity.csv",
        index=False,
    )
    session_exact_sparse_momentum_core_margin_summary(core_margins).to_csv(
        outdir / "session_exact_sparse_momentum_core_margin_summary.csv",
        index=False,
    )
    session_exact_sparse_momentum_core_threshold_sensitivity(combined).to_csv(
        outdir / "session_exact_sparse_momentum_core_threshold_sensitivity.csv",
        index=False,
    )
    rat_exact_sparse_momentum_core_margin_summary(core_margins).to_csv(
        outdir / "rat_exact_sparse_momentum_core_margin_summary.csv",
        index=False,
    )
    leave_one_rat_out_exact_sparse_momentum_core_margin_summary(core_margins).to_csv(
        outdir / "leave_one_rat_out_exact_sparse_momentum_core_margin_summary.csv",
        index=False,
    )
    leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity(combined).to_csv(
        outdir / "leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity.csv",
        index=False,
    )
    rat_bootstrap_exact_sparse_momentum_core_margin_summary(core_margins).to_csv(
        outdir / "rat_bootstrap_exact_sparse_momentum_core_margin_summary.csv",
        index=False,
    )
    rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity(combined).to_csv(
        outdir / "rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity.csv",
        index=False,
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate all-session event-sharded model-evidence outputs.")
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", default="results/all-session-model-evidence")
    args = parser.parse_args()

    combined = aggregate_all_sessions(args.shard_glob, Path(args.output))
    print(_summary(combined).to_string(index=False))
    print("\nSession summary:")
    print(session_model_evidence_summary(combined).to_string(index=False))
    print("\nRandom-effects model probabilities:")
    print(random_effects_model_probabilities(combined).to_string(index=False))
    print("\nPaper readiness gate summary:")
    print(paper_readiness_gate_summary(combined).to_string(index=False))
    print("\nExact trajectory dynamics gate summary:")
    print(exact_trajectory_dynamics_gate_summary(combined).to_string(index=False))
    print("\nExact trajectory dynamics threshold sensitivity:")
    print(exact_trajectory_dynamics_threshold_sensitivity(combined).to_string(index=False))
    print("\nSession exact trajectory dynamics threshold sensitivity:")
    print(session_exact_trajectory_dynamics_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat exact trajectory dynamics threshold sensitivity:")
    print(rat_exact_trajectory_dynamics_threshold_sensitivity(combined).to_string(index=False))
    print("\nLeave-one-rat-out exact trajectory dynamics threshold sensitivity:")
    print(leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat-bootstrap exact trajectory dynamics threshold sensitivity:")
    print(rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity(combined).to_string(index=False))
    trajectory_nontrajectory_decisions = exact_trajectory_nontrajectory_margin_decisions(combined)
    print("\nExact trajectory-vs-nontrajectory margin summary:")
    print(exact_trajectory_nontrajectory_margin_summary(trajectory_nontrajectory_decisions).to_string(index=False))
    print("\nExact trajectory-vs-nontrajectory threshold sensitivity:")
    print(exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_string(index=False))
    print("\nSession exact trajectory-vs-nontrajectory threshold sensitivity:")
    print(session_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat exact trajectory-vs-nontrajectory threshold sensitivity:")
    print(rat_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_string(index=False))
    print("\nLeave-one-rat-out exact trajectory-vs-nontrajectory threshold sensitivity:")
    print(leave_one_rat_out_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat-bootstrap exact trajectory-vs-nontrajectory threshold sensitivity:")
    print(rat_bootstrap_exact_trajectory_nontrajectory_threshold_sensitivity(combined).to_string(index=False))
    print("\nExact trajectory-vs-nontrajectory gate summary:")
    print(exact_trajectory_nontrajectory_gate_summary(combined).to_string(index=False))
    print("\nRequired full-core model coverage:")
    print(required_full_core_model_coverage_table(combined).to_string(index=False))
    exact_core_decisions = exact_core_model_claim_decisions(combined)
    print("\nExact core model claim summary:")
    print(exact_core_model_claim_summary(exact_core_decisions).to_string(index=False))
    print("\nSession exact core model claim summary:")
    print(session_exact_core_model_claim_summary(exact_core_decisions).to_string(index=False))
    decisions = paired_momentum_diffusion_margin_decisions(combined)
    print("\nPaired exact-sparse momentum-vs-diffusion margin summary:")
    print(paired_momentum_diffusion_margin_summary(decisions).to_string(index=False))
    print("\nPaired exact-sparse momentum-vs-diffusion threshold sensitivity:")
    print(paired_momentum_diffusion_threshold_sensitivity(combined).to_string(index=False))
    print("\nSession paired exact-sparse momentum-vs-diffusion threshold sensitivity:")
    print(session_paired_momentum_diffusion_threshold_sensitivity(combined).to_string(index=False))
    core_margins = exact_sparse_momentum_core_margins(combined)
    print("\nExact-sparse momentum full-core margin summary:")
    print(exact_sparse_momentum_core_margin_summary(core_margins).to_string(index=False))
    print("\nExact-sparse momentum full-core threshold sensitivity:")
    print(exact_sparse_momentum_core_threshold_sensitivity(combined).to_string(index=False))
    print("\nSession exact-sparse momentum full-core threshold sensitivity:")
    print(session_exact_sparse_momentum_core_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat paired exact-sparse momentum-vs-diffusion margin summary:")
    print(rat_paired_momentum_diffusion_margin_summary(decisions).to_string(index=False))
    print("\nLeave-one-rat-out paired exact-sparse momentum-vs-diffusion margin summary:")
    print(leave_one_rat_out_paired_momentum_diffusion_margin_summary(decisions).to_string(index=False))
    print("\nLeave-one-rat-out paired exact-sparse momentum-vs-diffusion threshold sensitivity:")
    print(leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity(combined).to_string(index=False))
    print("\nLeave-one-rat-out exact-sparse momentum full-core threshold sensitivity:")
    print(leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat-bootstrap paired exact-sparse momentum-vs-diffusion margin summary:")
    print(rat_bootstrap_paired_momentum_diffusion_margin_summary(decisions).to_string(index=False))
    print("\nRat-bootstrap paired exact-sparse momentum-vs-diffusion threshold sensitivity:")
    print(rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity(combined).to_string(index=False))
    print("\nRat-bootstrap exact-sparse momentum full-core threshold sensitivity:")
    print(rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity(combined).to_string(index=False))
    print(f"\nRows: {len(combined)}")
    if "status" in combined:
        print(f"Failures: {int((combined['status'] != 'success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
