#!/usr/bin/env python3
"""Aggregate all-session event-sharded model-evidence outputs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmark_model_evidence import _add_evidence_columns, _counts, _summary, _write


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
    return _add_evidence_columns(combined.drop(columns=["source_shard_file"]))


def session_model_evidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    out = ok.groupby(["session", "model", "model_family"], as_index=False).agg(
        events=("event_index", "count"),
        wins=("is_best_model", "sum"),
        mean_log_evidence=("log_evidence", "mean"),
        median_log_evidence=("log_evidence", "median"),
        mean_relative_log_evidence=("relative_log_evidence", "mean"),
        median_relative_log_evidence=("relative_log_evidence", "median"),
        mean_model_probability=("model_probability", "mean"),
        median_model_probability=("model_probability", "median"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    out["win_fraction"] = out["wins"] / out["events"].clip(lower=1)
    return out.sort_values(["session", "wins", "mean_log_evidence"], ascending=[True, False, False])


def session_best_model_counts(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    base = ok.drop_duplicates(["session", "event_index"])
    rows: list[dict[str, object]] = []
    for session, session_frame in base.groupby("session", sort=True):
        for col in ("best_model", "best_trajectory_model", "best_nontrajectory_model"):
            counts = session_frame[col].value_counts().rename_axis("model").reset_index(name="events")
            counts["session"] = session
            counts["comparison"] = col
            rows.extend(counts.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=["session", "comparison", "model", "events"])
    return pd.DataFrame(rows)[["session", "comparison", "model", "events"]]


def random_effects_model_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact random-effects-style model probability table by session.

    Each session votes for the model with the largest summed log evidence across
    its successfully scored events. The reported random-effects probability is a
    Dirichlet(1) posterior mean over these session-level model wins. The table
    also includes a fixed-effects posterior over the summed session log evidence.
    """

    ok = df[df["status"] == "success"].copy()
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


def aggregate_all_sessions(shard_glob: str, outdir: Path) -> pd.DataFrame:
    combined = _load_combined(shard_glob)
    outdir.mkdir(parents=True, exist_ok=True)

    # Backwards-compatible single-run outputs.
    _write(combined, outdir)

    # Explicit all-session outputs for paper-side consumption.
    combined.to_csv(outdir / "all_sessions_event_model_evidence.csv", index=False)
    _summary(combined).to_csv(outdir / "all_sessions_model_evidence_summary.csv", index=False)
    _counts(combined).to_csv(outdir / "all_sessions_best_model_counts.csv", index=False)
    session_model_evidence_summary(combined).to_csv(outdir / "session_model_evidence_summary.csv", index=False)
    session_best_model_counts(combined).to_csv(outdir / "session_best_model_counts.csv", index=False)
    random_effects_model_probabilities(combined).to_csv(outdir / "random_effects_model_probabilities.csv", index=False)
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
    print(f"\nRows: {len(combined)}")
    if "status" in combined:
        print(f"Failures: {int((combined['status'] != 'success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
