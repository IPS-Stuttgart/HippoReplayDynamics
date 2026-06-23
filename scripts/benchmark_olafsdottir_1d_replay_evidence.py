#!/usr/bin/env python3
"""Run the provisional Olafsdottir 1D Z-track replay evidence smoke.

This script is intentionally an end-to-end workflow smoke, not a biological
claim generator. It builds the bridge Pfeiffer/Foster-style session from a
Track1/SleepPOST Olafsdottir day pair, scores a small number of SleepPOST replay
candidate events with the exact core state-space models, and writes compact
Olafsdottir-specific summary tables for 1D-vs-2D comparison planning.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    _coerce_bool_series,
    ensure_evidence_support_columns,
)


DEFAULT_EXTRACTED_ROOT = Path("/home/github-runner/.cache/datasets/olafsdottir2016/extracted")
DEFAULT_SESSION = "R2142/ZTrack20140806"
DEFAULT_MODELS: tuple[str, ...] = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)
STATIONARY_MODEL = "sorted-spike-state-space-stationary"
DIFFUSION_MODEL = "sorted-spike-state-space-diffusion"
FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM_MODEL = "sorted-spike-state-space-first-order-imm"
MOMENTUM_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
TRAJECTORY_MODELS = {
    DIFFUSION_MODEL,
    FRAGMENTED_MODEL,
    FIRST_ORDER_IMM_MODEL,
    MOMENTUM_MODEL,
}
REQUIRED_OUTPUTS: tuple[str, ...] = (
    "olafsdottir_1d_event_model_evidence.csv",
    "olafsdottir_1d_family_margin_decisions.csv",
    "olafsdottir_1d_family_margin_summary.csv",
    "olafsdottir_1d_exact_core_model_claim_summary.csv",
    "olafsdottir_1d_paired_momentum_diffusion_summary.csv",
    "olafsdottir_1d_session_summary.csv",
    "olafsdottir_1d_control_gate_summary.csv",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _models_arg(models: str | Sequence[str]) -> str:
    if isinstance(models, str):
        return models
    return " ".join(models)


def build_prepare_command(args: argparse.Namespace, derived_root: Path | None = None) -> list[str]:
    output_root = Path(derived_root if derived_root is not None else args.derived_root)
    return [
        sys.executable,
        str(_repo_root() / "scripts" / "prepare_olafsdottir_ztrack_sessions.py"),
        "--extracted-root",
        str(args.extracted_root),
        "--output-root",
        str(output_root),
        "--sessions",
        str(args.session),
        "--tetrode-mode",
        str(args.tetrode_mode),
        "--lfp-detector-mode",
        str(args.lfp_detector_mode),
        "--min-event-spikes",
        str(args.min_event_spikes),
        "--min-event-active-cells",
        str(args.min_event_active_cells),
        "--lfp-channels",
        str(args.lfp_channels),
        "--ripple-high-threshold-z",
        str(args.ripple_high_threshold_z),
        "--ripple-low-threshold-z",
        str(args.ripple_low_threshold_z),
    ]


def build_benchmark_command(args: argparse.Namespace, derived_root: Path, benchmark_output: Path) -> list[str]:
    command = [
        sys.executable,
        str(_repo_root() / "scripts" / "benchmark_model_evidence.py"),
        "--dataset-root",
        str(derived_root),
        "--session",
        str(args.session),
        "--events",
        str(args.events),
        "--models",
        _models_arg(args.models),
        "--bin-size-cm",
        str(args.bin_size_cm),
        "--min-speed-cm-s",
        str(args.min_speed_cm_s),
        "--time-bin-s",
        str(args.time_bin_s),
        "--output",
        str(benchmark_output),
        "--continue-on-error",
    ]
    if args.max_events is not None:
        command.extend(["--max-events", str(args.max_events)])
    return command


def run_command(command: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def _script_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def run_workflow(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    output = Path(args.output)
    derived_root = Path(args.derived_root) if args.derived_root is not None else output.parent / "olafsdottir-1d-derived"
    benchmark_output = output / "_benchmark_model_evidence"
    output.mkdir(parents=True, exist_ok=True)
    env = _script_env(repo_root)

    run_command(build_prepare_command(args, derived_root), cwd=repo_root, env=env)
    run_command(build_benchmark_command(args, derived_root, benchmark_output), cwd=repo_root, env=env)

    event_model_evidence = benchmark_output / "event_model_evidence.csv"
    if not event_model_evidence.is_file():
        raise FileNotFoundError(f"Benchmark did not write {event_model_evidence}")
    scores = pd.read_csv(event_model_evidence)
    conversion_summary = derived_root / "olafsdottir_ztrack_conversion_summary.csv"
    write_olafsdottir_outputs(
        scores,
        output,
        session=args.session,
        max_events=args.max_events,
        margin_threshold=args.margin_threshold,
        models=_models_arg(args.models).replace(",", " ").split(),
        derived_root=derived_root,
        benchmark_output=benchmark_output,
        conversion_summary=conversion_summary if conversion_summary.is_file() else None,
    )
    _copy_auxiliary_files(benchmark_output, output)


def _copy_auxiliary_files(benchmark_output: Path, output: Path) -> None:
    for path in benchmark_output.glob("*.csv"):
        target = output / path.name
        if target.name.startswith("olafsdottir_1d_") or target.name == "event_model_evidence.csv":
            continue
        shutil.copyfile(path, target)


def write_olafsdottir_outputs(
    scores: pd.DataFrame,
    output: str | Path,
    *,
    session: str,
    max_events: int | None,
    margin_threshold: float,
    models: Sequence[str] | None = None,
    derived_root: str | Path | None = None,
    benchmark_output: str | Path | None = None,
    conversion_summary: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    evidence = _normalise_evidence_table(scores)
    evidence.to_csv(output_path / "olafsdottir_1d_event_model_evidence.csv", index=False)

    decisions = family_margin_decisions(evidence, margin_threshold=margin_threshold)
    family_summary = family_margin_summary(decisions, margin_threshold=margin_threshold)
    exact_claims = exact_core_model_claim_summary(evidence, margin_threshold=margin_threshold)
    paired = paired_momentum_diffusion_summary(evidence, margin_threshold=margin_threshold)
    session_summary = olafsdottir_session_summary(
        evidence,
        session=session,
        max_events=max_events,
        models=models or DEFAULT_MODELS,
        derived_root=derived_root,
        benchmark_output=benchmark_output,
        conversion_summary=conversion_summary,
    )
    tables = {
        "olafsdottir_1d_family_margin_decisions.csv": decisions,
        "olafsdottir_1d_family_margin_summary.csv": family_summary,
        "olafsdottir_1d_exact_core_model_claim_summary.csv": exact_claims,
        "olafsdottir_1d_paired_momentum_diffusion_summary.csv": paired,
        "olafsdottir_1d_session_summary.csv": session_summary,
    }
    for name, table in tables.items():
        table.to_csv(output_path / name, index=False)
    gate_summary = control_gate_summary(
        evidence,
        decisions,
        output_path,
        margin_threshold=margin_threshold,
        conversion_summary=conversion_summary,
    )
    gate_summary.to_csv(output_path / "olafsdottir_1d_control_gate_summary.csv", index=False)
    return {
        "event_model_evidence": evidence,
        "family_margin_decisions": decisions,
        "family_margin_summary": family_summary,
        "exact_core_model_claim_summary": exact_claims,
        "paired_momentum_diffusion_summary": paired,
        "session_summary": session_summary,
        "control_gate_summary": gate_summary,
    }


def _normalise_evidence_table(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if "status" not in out:
        out["status"] = "success"
    if "model_family" not in out:
        out["model_family"] = out["model"].map(model_family)
    out = ensure_evidence_support_columns(out)
    if "evidence_comparable" in out:
        out["evidence_comparable"] = _coerce_bool_series(out["evidence_comparable"])
    if "n_spikes" not in out:
        out["n_spikes"] = np.nan
    if "duration_s" not in out:
        out["duration_s"] = np.nan
    return out


def model_family(model: object) -> str:
    name = str(model)
    if name in TRAJECTORY_MODELS:
        return "trajectory"
    if name == STATIONARY_MODEL or name == "stationary":
        return "nontrajectory"
    return "other"


def _exact_success_rows(scores: pd.DataFrame) -> pd.DataFrame:
    frame = _normalise_evidence_table(scores)
    status = frame["status"].astype(str).str.lower()
    comparable = _coerce_bool_series(frame["evidence_comparable"])
    support = frame["evidence_support"].astype(str).eq(EXACT_EVIDENCE_SUPPORT)
    return frame[status.eq("success") & comparable & support].copy()


def _event_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("session", "event_index", "event_id", "candidate_id") if column in frame.columns]


def family_margin_decisions(scores: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    exact = _exact_success_rows(scores)
    columns = [
        "session",
        "event_index",
        "best_trajectory_model",
        "best_trajectory_log_evidence",
        "best_nontrajectory_model",
        "best_nontrajectory_log_evidence",
        "trajectory_minus_nontrajectory_margin",
        "trajectory_family_claim",
        "trajectory_confident_claim",
        "nontrajectory_confident_claim",
        "ambiguous_claim",
        "complete_exact_core",
        "n_models_compared",
        "n_spikes",
        "duration_s",
    ]
    if exact.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    group_columns = _event_group_columns(exact)
    if "session" not in group_columns or "event_index" not in group_columns:
        raise ValueError("scores must contain session and event_index columns")
    for keys, group in exact.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        trajectory = group[group["model"].isin(TRAJECTORY_MODELS)]
        nontrajectory = group[group["model"].eq(STATIONARY_MODEL)]
        if trajectory.empty or nontrajectory.empty:
            continue
        best_traj = trajectory.loc[trajectory["log_evidence"].astype(float).idxmax()]
        best_nontraj = nontrajectory.loc[nontrajectory["log_evidence"].astype(float).idxmax()]
        margin = float(best_traj["log_evidence"]) - float(best_nontraj["log_evidence"])
        if margin >= margin_threshold:
            claim = "trajectory_confident"
        elif margin <= -margin_threshold:
            claim = "nontrajectory_confident"
        else:
            claim = "ambiguous"
        models_present = set(group["model"].astype(str))
        rows.append(
            {
                **base,
                "best_trajectory_model": str(best_traj["model"]),
                "best_trajectory_log_evidence": float(best_traj["log_evidence"]),
                "best_nontrajectory_model": str(best_nontraj["model"]),
                "best_nontrajectory_log_evidence": float(best_nontraj["log_evidence"]),
                "trajectory_minus_nontrajectory_margin": margin,
                "trajectory_family_claim": claim,
                "trajectory_confident_claim": claim == "trajectory_confident",
                "nontrajectory_confident_claim": claim == "nontrajectory_confident",
                "ambiguous_claim": claim == "ambiguous",
                "complete_exact_core": set(DEFAULT_MODELS).issubset(models_present),
                "n_models_compared": int(group["model"].nunique()),
                "n_spikes": _event_first(group, "n_spikes"),
                "duration_s": _event_first(group, "duration_s"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def family_margin_summary(decisions: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    columns = [
        "scope",
        "events",
        "complete_exact_core_events",
        "trajectory_confident_claims",
        "nontrajectory_confident_claims",
        "ambiguous_events",
        "mean_trajectory_minus_nontrajectory_margin",
        "median_trajectory_minus_nontrajectory_margin",
        "min_trajectory_minus_nontrajectory_margin",
        "max_trajectory_minus_nontrajectory_margin",
        "margin_threshold",
    ]
    if decisions.empty:
        return pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "events": 0,
                    "complete_exact_core_events": 0,
                    "trajectory_confident_claims": 0,
                    "nontrajectory_confident_claims": 0,
                    "ambiguous_events": 0,
                    "mean_trajectory_minus_nontrajectory_margin": np.nan,
                    "median_trajectory_minus_nontrajectory_margin": np.nan,
                    "min_trajectory_minus_nontrajectory_margin": np.nan,
                    "max_trajectory_minus_nontrajectory_margin": np.nan,
                    "margin_threshold": float(margin_threshold),
                }
            ],
            columns=columns,
        )
    margins = decisions["trajectory_minus_nontrajectory_margin"].astype(float)
    row = {
        "scope": "overall",
        "events": int(len(decisions)),
        "complete_exact_core_events": int(decisions["complete_exact_core"].map(bool).sum()),
        "trajectory_confident_claims": int(decisions["trajectory_confident_claim"].map(bool).sum()),
        "nontrajectory_confident_claims": int(decisions["nontrajectory_confident_claim"].map(bool).sum()),
        "ambiguous_events": int(decisions["ambiguous_claim"].map(bool).sum()),
        "mean_trajectory_minus_nontrajectory_margin": float(margins.mean()),
        "median_trajectory_minus_nontrajectory_margin": float(margins.median()),
        "min_trajectory_minus_nontrajectory_margin": float(margins.min()),
        "max_trajectory_minus_nontrajectory_margin": float(margins.max()),
        "margin_threshold": float(margin_threshold),
    }
    return pd.DataFrame([row], columns=columns)


def exact_core_model_claim_summary(scores: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    exact = _exact_success_rows(scores)
    columns = [
        "model",
        "model_family",
        "events_scored",
        "raw_best_events",
        "confident_best_events",
        "mean_log_evidence",
        "median_log_evidence",
        "margin_threshold",
    ]
    rows = [
        {
            "model": model,
            "model_family": model_family(model),
            "events_scored": 0,
            "raw_best_events": 0,
            "confident_best_events": 0,
            "mean_log_evidence": np.nan,
            "median_log_evidence": np.nan,
            "margin_threshold": float(margin_threshold),
        }
        for model in DEFAULT_MODELS
    ]
    if exact.empty:
        return pd.DataFrame(rows, columns=columns)

    group_columns = _event_group_columns(exact)
    raw_best: dict[str, int] = {model: 0 for model in DEFAULT_MODELS}
    confident_best: dict[str, int] = {model: 0 for model in DEFAULT_MODELS}
    for _keys, group in exact.groupby(group_columns, dropna=False, sort=True):
        exact_core = group[group["model"].isin(DEFAULT_MODELS)].copy()
        if exact_core.empty:
            continue
        exact_core = exact_core.sort_values("log_evidence", ascending=False)
        best = str(exact_core.iloc[0]["model"])
        raw_best[best] = raw_best.get(best, 0) + 1
        if len(exact_core) > 1:
            margin = float(exact_core.iloc[0]["log_evidence"]) - float(exact_core.iloc[1]["log_evidence"])
        else:
            margin = np.inf
        if margin >= margin_threshold:
            confident_best[best] = confident_best.get(best, 0) + 1

    summary = exact[exact["model"].isin(DEFAULT_MODELS)].groupby("model", as_index=False).agg(
        events_scored=("event_index", "count"),
        mean_log_evidence=("log_evidence", "mean"),
        median_log_evidence=("log_evidence", "median"),
    )
    by_model = summary.set_index("model")
    rows = []
    for model in DEFAULT_MODELS:
        if model in by_model.index:
            model_row = by_model.loc[model]
            events = int(model_row["events_scored"])
            mean_log = float(model_row["mean_log_evidence"])
            median_log = float(model_row["median_log_evidence"])
        else:
            events = 0
            mean_log = np.nan
            median_log = np.nan
        rows.append(
            {
                "model": model,
                "model_family": model_family(model),
                "events_scored": events,
                "raw_best_events": int(raw_best.get(model, 0)),
                "confident_best_events": int(confident_best.get(model, 0)),
                "mean_log_evidence": mean_log,
                "median_log_evidence": median_log,
                "margin_threshold": float(margin_threshold),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paired_momentum_diffusion_summary(scores: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    exact = _exact_success_rows(scores)
    columns = [
        "paired_events",
        "momentum_raw_wins",
        "diffusion_raw_wins",
        "momentum_confident_wins",
        "diffusion_confident_wins",
        "ambiguous_events",
        "mean_delta_momentum_minus_diffusion",
        "median_delta_momentum_minus_diffusion",
        "min_delta_momentum_minus_diffusion",
        "max_delta_momentum_minus_diffusion",
        "margin_threshold",
    ]
    deltas: list[float] = []
    group_columns = _event_group_columns(exact)
    for _keys, group in exact.groupby(group_columns, dropna=False, sort=True):
        by_model = group.set_index("model")["log_evidence"]
        if MOMENTUM_MODEL not in by_model.index or DIFFUSION_MODEL not in by_model.index:
            continue
        deltas.append(float(by_model[MOMENTUM_MODEL]) - float(by_model[DIFFUSION_MODEL]))
    if not deltas:
        return pd.DataFrame(
            [
                {
                    "paired_events": 0,
                    "momentum_raw_wins": 0,
                    "diffusion_raw_wins": 0,
                    "momentum_confident_wins": 0,
                    "diffusion_confident_wins": 0,
                    "ambiguous_events": 0,
                    "mean_delta_momentum_minus_diffusion": np.nan,
                    "median_delta_momentum_minus_diffusion": np.nan,
                    "min_delta_momentum_minus_diffusion": np.nan,
                    "max_delta_momentum_minus_diffusion": np.nan,
                    "margin_threshold": float(margin_threshold),
                }
            ],
            columns=columns,
        )
    arr = np.asarray(deltas, dtype=float)
    row = {
        "paired_events": int(arr.size),
        "momentum_raw_wins": int((arr > 0).sum()),
        "diffusion_raw_wins": int((arr < 0).sum()),
        "momentum_confident_wins": int((arr >= margin_threshold).sum()),
        "diffusion_confident_wins": int((arr <= -margin_threshold).sum()),
        "ambiguous_events": int((np.abs(arr) < margin_threshold).sum()),
        "mean_delta_momentum_minus_diffusion": float(arr.mean()),
        "median_delta_momentum_minus_diffusion": float(np.median(arr)),
        "min_delta_momentum_minus_diffusion": float(arr.min()),
        "max_delta_momentum_minus_diffusion": float(arr.max()),
        "margin_threshold": float(margin_threshold),
    }
    return pd.DataFrame([row], columns=columns)


def olafsdottir_session_summary(
    scores: pd.DataFrame,
    *,
    session: str,
    max_events: int | None,
    models: Sequence[str],
    derived_root: str | Path | None,
    benchmark_output: str | Path | None,
    conversion_summary: str | Path | None,
) -> pd.DataFrame:
    frame = _normalise_evidence_table(scores)
    successful = frame["status"].astype(str).str.lower().eq("success")
    event_count = int(frame["event_index"].nunique()) if "event_index" in frame else 0
    conversion_events = np.nan
    included_cells = np.nan
    if conversion_summary is not None and Path(conversion_summary).is_file():
        summary = pd.read_csv(conversion_summary)
        if not summary.empty:
            conversion_events = _numeric_first(summary, "ripple_events")
            included_cells = _numeric_first(summary, "included_cells")
    row = {
        "session": session,
        "environment_type": "1D_Z_track",
        "first_milestone": "end-to-end ingestion -> linearization -> event detection -> evidence scoring -> summary tables",
        "biological_interpretation": "workflow_smoke_only",
        "events_scored": event_count,
        "score_rows": int(len(frame)),
        "successful_rows": int(successful.sum()),
        "failed_rows": int((~successful).sum()),
        "models_requested": " ".join(models),
        "models_observed": " ".join(sorted(frame["model"].dropna().astype(str).unique())),
        "max_events": "" if max_events is None else int(max_events),
        "derived_root": "" if derived_root is None else str(derived_root),
        "benchmark_output": "" if benchmark_output is None else str(benchmark_output),
        "conversion_summary": "" if conversion_summary is None else str(conversion_summary),
        "conversion_ripple_events": conversion_events,
        "included_cells": included_cells,
    }
    return pd.DataFrame([row])


def control_gate_summary(
    scores: pd.DataFrame,
    decisions: pd.DataFrame,
    output: Path,
    *,
    margin_threshold: float,
    conversion_summary: str | Path | None,
) -> pd.DataFrame:
    successful = scores["status"].astype(str).str.lower().eq("success") if "status" in scores else pd.Series([], dtype=bool)
    models_present = set(scores["model"].dropna().astype(str)) if "model" in scores else set()
    complete_events = int(decisions["complete_exact_core"].map(bool).sum()) if not decisions.empty else 0
    event_count = int(scores["event_index"].nunique()) if "event_index" in scores else 0
    gates = [
        (
            "derived_session_conversion_available",
            conversion_summary is not None and Path(conversion_summary).is_file(),
            "Bridge-derived session conversion summary is present.",
        ),
        ("event_model_evidence_nonempty", not scores.empty, "Exact-core model evidence table is nonempty."),
        ("exact_core_models_present", set(DEFAULT_MODELS).issubset(models_present), "All five requested exact-core 1D models are present."),
        (
            "all_scored_events_have_exact_core",
            event_count > 0 and complete_events == event_count,
            "Every scored event has stationary, diffusion, fragmented, first-order IMM, and exact-sparse momentum rows.",
        ),
        ("no_failed_model_rows", len(successful) > 0 and bool(successful.all()), "No model row failed during the workflow smoke."),
        (
            "summary_tables_written",
            all((output / name).is_file() for name in REQUIRED_OUTPUTS if name != "olafsdottir_1d_control_gate_summary.csv"),
            "All Olafsdottir 1D workflow summary tables were written.",
        ),
        (
            "biological_claim_not_assessed",
            True,
            "This gate records an end-to-end workflow milestone only; trajectory/IMM biology is not claimed here.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "gate": name,
                "passed": bool(passed),
                "value": str(passed).lower(),
                "margin_threshold": float(margin_threshold),
                "note": note,
            }
            for name, passed, note in gates
        ]
    )


def _event_first(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def _numeric_first(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    parser.add_argument("--derived-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/olafsdottir-1d-evidence"))
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--events", default="all")
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--models", default=" ".join(DEFAULT_MODELS))
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--tetrode-mode", choices=("hippocampus", "all"), default="hippocampus")
    parser.add_argument("--lfp-detector-mode", choices=("mean-envelope", "per-channel-union", "per-channel-consensus"), default="mean-envelope")
    parser.add_argument("--lfp-channels", default="1-4")
    parser.add_argument("--ripple-high-threshold-z", type=float, default=2.25)
    parser.add_argument("--ripple-low-threshold-z", type=float, default=0.75)
    parser.add_argument("--min-event-spikes", type=int, default=5)
    parser.add_argument("--min-event-active-cells", type=int, default=3)
    parser.add_argument("--bin-size-cm", type=float, default=5.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=4.0)
    parser.add_argument("--time-bin-s", type=float, default=0.02)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_workflow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
