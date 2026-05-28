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


def session_paired_momentum_diffusion_margin_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibrated exact-sparse momentum-vs-diffusion decisions by session."""

    return _paired_margin_summary(decisions, group_cols=("session",))


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

    _write(combined, outdir)
    combined.to_csv(outdir / "all_sessions_event_model_evidence.csv", index=False)
    _summary(combined).to_csv(outdir / "all_sessions_model_evidence_summary.csv", index=False)
    _counts(combined).to_csv(outdir / "all_sessions_best_model_counts.csv", index=False)
    session_model_evidence_summary(combined).to_csv(outdir / "session_model_evidence_summary.csv", index=False)
    session_best_model_counts(combined).to_csv(outdir / "session_best_model_counts.csv", index=False)
    random_effects_model_probabilities(combined).to_csv(outdir / "random_effects_model_probabilities.csv", index=False)
    paired_decisions.to_csv(outdir / "paired_momentum_diffusion_margin_decisions.csv", index=False)
    paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "paired_momentum_diffusion_margin_summary.csv",
        index=False,
    )
    session_paired_momentum_diffusion_margin_summary(paired_decisions).to_csv(
        outdir / "session_paired_momentum_diffusion_margin_summary.csv",
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
    decisions = paired_momentum_diffusion_margin_decisions(combined)
    print("\nPaired exact-sparse momentum-vs-diffusion margin summary:")
    print(paired_momentum_diffusion_margin_summary(decisions).to_string(index=False))
    print(f"\nRows: {len(combined)}")
    if "status" in combined:
        print(f"Failures: {int((combined['status'] != 'success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
