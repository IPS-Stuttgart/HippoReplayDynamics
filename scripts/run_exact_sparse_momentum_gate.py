"""Run and aggregate the exact-sparse momentum viability gate.

The gate is deliberately small: by default it simulates one diffusion and one
momentum event per Pfeiffer/Foster open-field session, scores exact/comparable
state-space rows plus lower-bound audit rows, and writes a pass/fail dashboard.

Example
-------
python scripts/run_exact_sparse_momentum_gate.py data/DataSetFromPfeifferFoster \\
  --output results/exact-sparse-gate \\
  --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd


DEFAULT_SESSIONS = (
    "Rat1/Open1",
    "Rat1/Open2",
    "Rat2/Open1",
    "Rat2/Open2",
    "Rat3/Open1",
    "Rat3/Open2",
    "Rat4/Open1",
    "Rat4/Open2",
)

DIFFUSION_MODEL = "sorted-spike-state-space-diffusion"
EXACT_SPARSE_MOMENTUM_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM_MODEL = "sorted-spike-state-space-first-order-imm"
CANDIDATE_MOMENTUM_MODEL = "sorted-spike-state-space-momentum"
CANDIDATE_IMM_MODEL = "sorted-spike-state-space-imm"

DEFAULT_SCORING_MODELS = (
    DIFFUSION_MODEL,
    EXACT_SPARSE_MOMENTUM_MODEL,
    FRAGMENTED_MODEL,
    FIRST_ORDER_IMM_MODEL,
    CANDIDATE_MOMENTUM_MODEL,
    CANDIDATE_IMM_MODEL,
)

REQUIRED_EXACT_MODELS = frozenset(
    {
        DIFFUSION_MODEL,
        EXACT_SPARSE_MOMENTUM_MODEL,
        FRAGMENTED_MODEL,
        FIRST_ORDER_IMM_MODEL,
    }
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        sessions = parse_sessions(args.sessions)
        for session_index, session in enumerate(sessions):
            session_seed = int(args.random_seed) + int(session_index)
            session_output = output / safe_session_id(session)
            if args.skip_existing and _session_has_scores(session_output):
                print(f"[skip] {session}: existing event scores found in {session_output}")
                continue
            command = build_simulation_command(
                args,
                session=session,
                session_output=session_output,
                random_seed=session_seed,
            )
            if args.dry_run:
                print(" ".join(_quote_for_display(part) for part in command))
                continue
            run_one_session(
                command,
                session=session,
                session_output=session_output,
                repo_root=Path(args.repo_root),
                continue_after_failure=args.keep_going,
            )

    if args.dry_run:
        return 0

    status = aggregate_gate_results(
        output,
        min_momentum_recovery=float(args.min_momentum_recovery),
        min_diffusion_recovery=float(args.min_diffusion_recovery),
        max_first_order_imm_best_fraction=float(args.max_first_order_imm_best_fraction),
    )
    dashboard = output / "exact_sparse_momentum_gate.md"
    print(dashboard)
    print("PASS" if status["gate_passed"] else "FAIL")
    return 0 if status["gate_passed"] else 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reduced exact-sparse momentum simulation-recovery gate."
    )
    parser.add_argument("dataset_root", help="Path to DataSetFromPfeifferFoster.")
    parser.add_argument(
        "--output",
        default="results/exact-sparse-gate",
        help="Gate output directory.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to add ./src to PYTHONPATH for subprocesses.",
    )
    parser.add_argument(
        "--sessions",
        default=" ".join(DEFAULT_SESSIONS),
        help="Whitespace/comma-separated sessions, or 'all'.",
    )
    parser.add_argument("--events", default="run")
    parser.add_argument("--max-template-events", type=int, default=5)
    parser.add_argument("--events-per-model", type=int, default=1)
    parser.add_argument("--true-models", default="diffusion momentum")
    parser.add_argument("--models", default=" ".join(DEFAULT_SCORING_MODELS))
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--time-bin-ms", type=float, default=3.0)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument(
        "--state-space-momentum-initial-sigma-cm-sqrt-s",
        type=float,
        default=85.0,
    )
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.060)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument(
        "--state-space-momentum-candidate-top-k",
        type=int,
        default=128,
        help="Candidate-pruned audit support size; exact-sparse does not use this.",
    )
    parser.add_argument(
        "--state-space-momentum-predicted-candidate-top-k",
        type=int,
        default=8,
        help="Candidate-pruned audit support augmentation; exact-sparse does not use this.",
    )
    parser.add_argument(
        "--min-momentum-recovery",
        type=float,
        default=0.70,
        help="Minimum exact-surrogate recovery accuracy for true momentum events.",
    )
    parser.add_argument(
        "--min-diffusion-recovery",
        type=float,
        default=0.70,
        help="Minimum expected-model recovery accuracy for true diffusion events.",
    )
    parser.add_argument(
        "--max-first-order-imm-best-fraction",
        type=float,
        default=0.80,
        help="Fail if first-order IMM is the best exact model above this fraction.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter used for hipporeplayimm CLI subprocesses.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Pass --continue-on-error to simulate-recovery so model failures are recorded.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later sessions if a subprocess exits non-zero.",
    )
    parser.add_argument(
        "--rerun-existing",
        dest="skip_existing",
        action="store_false",
        help="Rerun sessions even if simulation_recovery_event_scores.csv already exists.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run simulations; reaggregate existing session outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args(argv)


def parse_sessions(spec: str) -> list[str]:
    value = spec.strip()
    if value.lower() == "all":
        return list(DEFAULT_SESSIONS)
    sessions = [item.strip() for item in re.split(r"[\s,]+", value) if item.strip()]
    if not sessions:
        raise ValueError("at least one session is required")
    return sessions


def safe_session_id(session: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", session).strip("_")


def _session_has_scores(session_output: Path) -> bool:
    return (session_output / "simulation_recovery_event_scores.csv").exists()


def build_simulation_command(
    args: argparse.Namespace,
    *,
    session: str,
    session_output: Path,
    random_seed: int,
) -> list[str]:
    command = [
        str(args.python_executable),
        "-c",
        "from hipporeplayimm.cli import main; raise SystemExit(main())",
        "simulate-recovery",
        str(args.dataset_root),
        "--session",
        session,
        "--events",
        str(args.events),
        "--max-template-events",
        str(args.max_template_events),
        "--events-per-model",
        str(args.events_per_model),
        "--true-models",
        str(args.true_models),
        "--models",
        str(args.models),
        "--random-seed",
        str(random_seed),
        "--time-bin-ms",
        _float_arg(args.time_bin_ms),
        "--spike-rate-scale",
        _float_arg(args.spike_rate_scale),
        "--bin-size-cm",
        _float_arg(args.bin_size_cm),
        "--smoothing-sigma-bins",
        _float_arg(args.smoothing_sigma_bins),
        "--min-speed-cm-s",
        _float_arg(args.min_speed_cm_s),
        "--state-space-diffusion-sigma-cm-sqrt-s",
        _float_arg(args.state_space_diffusion_sigma_cm_sqrt_s),
        "--state-space-momentum-sigma-cm-sqrt-s",
        _float_arg(args.state_space_momentum_sigma_cm_sqrt_s),
        "--state-space-momentum-initial-sigma-cm-sqrt-s",
        _float_arg(args.state_space_momentum_initial_sigma_cm_sqrt_s),
        "--state-space-momentum-velocity-decay-tau-s",
        _float_arg(args.state_space_momentum_velocity_decay_tau_s),
        "--true-state-space-diffusion-sigma-cm-sqrt-s",
        _float_arg(args.state_space_diffusion_sigma_cm_sqrt_s),
        "--true-state-space-momentum-sigma-cm-sqrt-s",
        _float_arg(args.state_space_momentum_sigma_cm_sqrt_s),
        "--true-state-space-momentum-initial-sigma-cm-sqrt-s",
        _float_arg(args.state_space_momentum_initial_sigma_cm_sqrt_s),
        "--true-state-space-momentum-velocity-decay-tau-s",
        _float_arg(args.state_space_momentum_velocity_decay_tau_s),
        "--state-space-max-step-sigma",
        _float_arg(args.state_space_max_step_sigma),
        "--state-space-imm-mode-stickiness",
        _float_arg(args.state_space_imm_mode_stickiness),
        "--state-space-momentum-candidate-top-k",
        str(args.state_space_momentum_candidate_top_k),
        "--state-space-momentum-predicted-candidate-top-k",
        str(args.state_space_momentum_predicted_candidate_top_k),
        "--output",
        str(session_output),
    ]
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def _float_arg(value: object) -> str:
    return f"{float(value):.12g}"


def _quote_for_display(value: str) -> str:
    if re.search(r"[\s'\"$]", value):
        return repr(value)
    return value


def run_one_session(
    command: list[str],
    *,
    session: str,
    session_output: Path,
    repo_root: Path,
    continue_after_failure: bool,
) -> None:
    session_output.mkdir(parents=True, exist_ok=True)
    log_path = session_output / "run.log"
    env = _subprocess_env(repo_root)
    print(f"[run] {session}")
    print(" ".join(_quote_for_display(part) for part in command))
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"session: {session}\n")
        log_file.write("command: " + " ".join(_quote_for_display(part) for part in command) + "\n")
        log_file.flush()
        process = subprocess.Popen(  # noqa: S603 - command is constructed from explicit args.
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        elapsed = time.perf_counter() - start
        log_file.write(f"\nreturn_code: {return_code}\nelapsed_s: {elapsed:.3f}\n")
    if return_code != 0 and not continue_after_failure:
        raise subprocess.CalledProcessError(return_code, command)


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str((repo_root / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def aggregate_gate_results(
    output: str | Path,
    *,
    min_momentum_recovery: float = 0.70,
    min_diffusion_recovery: float = 0.70,
    max_first_order_imm_best_fraction: float = 0.80,
) -> dict[str, object]:
    output_dir = Path(output)
    scores = read_gate_event_scores(output_dir)
    if scores.empty:
        status = _empty_status(
            min_momentum_recovery=min_momentum_recovery,
            min_diffusion_recovery=min_diffusion_recovery,
            max_first_order_imm_best_fraction=max_first_order_imm_best_fraction,
        )
        _write_gate_outputs(output_dir, scores, pd.DataFrame(), pd.DataFrame(), status)
        return status

    event_summary = build_event_summary(scores)
    session_summary = build_session_summary(event_summary)
    runtime_summary = build_runtime_summary(scores)

    momentum = event_summary[event_summary["true_model"].eq("momentum")]
    diffusion = event_summary[event_summary["true_model"].eq("diffusion")]
    momentum_accuracy = _mean_bool(momentum["exact_surrogate_recovered"])
    diffusion_accuracy = _mean_bool(diffusion["recovered_expected_model"])
    first_order_fraction = _mean_bool(
        event_summary["best_model"].eq(FIRST_ORDER_IMM_MODEL)
    )
    required_failures = _required_failure_count(scores)
    missing_required = _missing_required_score_count(scores)
    status = {
        "gate_passed": bool(
            momentum_accuracy >= min_momentum_recovery
            and diffusion_accuracy >= min_diffusion_recovery
            and first_order_fraction <= max_first_order_imm_best_fraction
            and required_failures == 0
            and missing_required == 0
        ),
        "momentum_exact_surrogate_recovered_events": int(
            _bool_series(momentum["exact_surrogate_recovered"]).sum()
        ),
        "momentum_events": int(momentum.shape[0]),
        "momentum_exact_surrogate_recovery_accuracy": momentum_accuracy,
        "momentum_recovery_threshold": float(min_momentum_recovery),
        "diffusion_recovered_events": int(
            _bool_series(diffusion["recovered_expected_model"]).sum()
        ),
        "diffusion_events": int(diffusion.shape[0]),
        "diffusion_recovery_accuracy": diffusion_accuracy,
        "diffusion_recovery_threshold": float(min_diffusion_recovery),
        "first_order_imm_best_fraction": first_order_fraction,
        "max_first_order_imm_best_fraction": float(max_first_order_imm_best_fraction),
        "required_exact_model_failures": int(required_failures),
        "missing_required_exact_model_scores": int(missing_required),
        "total_model_failures": int(scores["status"].ne("success").sum())
        if "status" in scores
        else 0,
        "scored_events": int(event_summary.shape[0]),
        "scored_sessions": int(event_summary["session"].nunique()),
        "exact_sparse_model": EXACT_SPARSE_MOMENTUM_MODEL,
    }
    _write_gate_outputs(output_dir, scores, event_summary, session_summary, status)
    runtime_summary.to_csv(
        output_dir / "exact_sparse_momentum_gate_runtime_summary.csv",
        index=False,
    )
    return status


def _empty_status(
    *,
    min_momentum_recovery: float,
    min_diffusion_recovery: float,
    max_first_order_imm_best_fraction: float,
) -> dict[str, object]:
    return {
        "gate_passed": False,
        "reason": "no_event_scores_found",
        "momentum_exact_surrogate_recovered_events": 0,
        "momentum_events": 0,
        "momentum_exact_surrogate_recovery_accuracy": 0.0,
        "momentum_recovery_threshold": float(min_momentum_recovery),
        "diffusion_recovered_events": 0,
        "diffusion_events": 0,
        "diffusion_recovery_accuracy": 0.0,
        "diffusion_recovery_threshold": float(min_diffusion_recovery),
        "first_order_imm_best_fraction": 0.0,
        "max_first_order_imm_best_fraction": float(max_first_order_imm_best_fraction),
        "required_exact_model_failures": 0,
        "missing_required_exact_model_scores": 0,
        "total_model_failures": 0,
        "scored_events": 0,
        "scored_sessions": 0,
        "exact_sparse_model": EXACT_SPARSE_MOMENTUM_MODEL,
    }


def read_gate_event_scores(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("*/simulation_recovery_event_scores.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["gate_output_session_dir"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_event_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (session, event_index), group in scores.groupby(
        ["session", "event_index"],
        sort=False,
        dropna=False,
    ):
        first = group.iloc[0]
        true_model = str(first.get("true_model", ""))
        expected_model = str(first.get("expected_model", ""))
        surrogate_model = str(
            first.get("expected_exact_surrogate_model", expected_model)
        )
        best_model = _event_best_model(group)
        recovered_expected = _event_bool(
            group,
            "recovered_expected_model",
            fallback=best_model == expected_model,
        )
        exact_surrogate_recovered = _event_bool(
            group,
            "exact_surrogate_recovered_expected_model",
            fallback=best_model == surrogate_model,
        )
        rows.append(
            {
                "session": session,
                "event_index": int(event_index),
                "true_model": true_model,
                "expected_model": expected_model,
                "expected_exact_surrogate_model": surrogate_model,
                "best_model": best_model,
                "recovered_expected_model": bool(recovered_expected),
                "exact_surrogate_recovered": bool(exact_surrogate_recovered),
                "n_time": _first_numeric(first, "n_time"),
                "n_spikes": _first_numeric(first, "n_spikes"),
                "diffusion_log_evidence": _model_log_evidence(group, DIFFUSION_MODEL),
                "exact_sparse_momentum_log_evidence": _model_log_evidence(
                    group,
                    EXACT_SPARSE_MOMENTUM_MODEL,
                ),
                "first_order_imm_log_evidence": _model_log_evidence(
                    group,
                    FIRST_ORDER_IMM_MODEL,
                ),
                "fragmented_log_evidence": _model_log_evidence(
                    group,
                    FRAGMENTED_MODEL,
                ),
                "required_exact_model_failures": int(
                    _required_failure_count(group)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["exact_sparse_minus_diffusion"] = (
        frame["exact_sparse_momentum_log_evidence"] - frame["diffusion_log_evidence"]
    )
    frame["exact_sparse_minus_first_order_imm"] = (
        frame["exact_sparse_momentum_log_evidence"] - frame["first_order_imm_log_evidence"]
    )
    frame["exact_sparse_minus_fragmented"] = (
        frame["exact_sparse_momentum_log_evidence"] - frame["fragmented_log_evidence"]
    )
    return frame


def _event_best_model(group: pd.DataFrame) -> str:
    if "best_model" in group:
        values = group["best_model"].dropna().astype(str)
        values = values[values.str.len() > 0]
        if not values.empty:
            return str(values.iloc[0])
    scored = _successful_finite_rows(group)
    if "evidence_comparable" in scored:
        scored = scored[_bool_series(scored["evidence_comparable"])]
    if scored.empty:
        return ""
    values = pd.to_numeric(scored["log_evidence"], errors="coerce")
    return str(scored.iloc[int(values.argmax())]["model"])


def _successful_finite_rows(group: pd.DataFrame) -> pd.DataFrame:
    scored = group.copy()
    if "status" in scored:
        scored = scored[scored["status"].eq("success")]
    if "log_evidence" in scored:
        scored = scored[pd.to_numeric(scored["log_evidence"], errors="coerce").notna()]
    return scored


def _event_bool(group: pd.DataFrame, column: str, *, fallback: bool) -> bool:
    if column not in group:
        return bool(fallback)
    values = group[column].dropna()
    if values.empty:
        return bool(fallback)
    return _coerce_bool(values.iloc[0])


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        return bool(math.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_coerce_bool).astype(bool)


def _first_numeric(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _model_log_evidence(group: pd.DataFrame, model: str) -> float:
    model_rows = (
        group[group["model"].astype(str).eq(model)]
        if "model" in group
        else group.iloc[0:0]
    )
    model_rows = _successful_finite_rows(model_rows)
    if model_rows.empty or "log_evidence" not in model_rows:
        return float("nan")
    values = pd.to_numeric(model_rows["log_evidence"], errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def build_session_summary(event_summary: pd.DataFrame) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    rows = []
    for (session, true_model), group in event_summary.groupby(
        ["session", "true_model"],
        sort=False,
    ):
        rows.append(
            {
                "session": session,
                "true_model": true_model,
                "events": int(group.shape[0]),
                "recovered_expected_events": int(
                    _bool_series(group["recovered_expected_model"]).sum()
                ),
                "recovery_accuracy": _mean_bool(group["recovered_expected_model"]),
                "exact_surrogate_recovered_events": int(
                    _bool_series(group["exact_surrogate_recovered"]).sum()
                ),
                "exact_surrogate_recovery_accuracy": _mean_bool(
                    group["exact_surrogate_recovered"]
                ),
                "best_model": _mode_or_empty(group["best_model"]),
                "mean_exact_sparse_minus_diffusion": float(
                    pd.to_numeric(
                        group["exact_sparse_minus_diffusion"],
                        errors="coerce",
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_runtime_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or "model" not in scores:
        return pd.DataFrame()
    frame = scores.copy()
    if "status" not in frame:
        frame["status"] = "success"
    for column in ("runtime_s", "n_time", "n_spikes"):
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    model_col = "model"
    summary = frame.groupby(model_col, sort=False).agg(
        rows=(model_col, "count"),
        failures=("status", lambda values: int(values.ne("success").sum())),
        mean_runtime_s=("runtime_s", "mean"),
        max_runtime_s=("runtime_s", "max"),
        mean_n_time=("n_time", "mean"),
        mean_n_spikes=("n_spikes", "mean"),
    )
    return summary.reset_index()


def _mean_bool(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(_bool_series(values).mean())


def _mode_or_empty(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    clean = clean[clean.str.len() > 0]
    if clean.empty:
        return ""
    return str(clean.value_counts().index[0])


def _required_failure_count(frame: pd.DataFrame) -> int:
    if frame.empty or "status" not in frame:
        return 0
    model_values = (
        frame["model"].astype(str)
        if "model" in frame
        else pd.Series("", index=frame.index)
    )
    requested_values = (
        frame["requested_model"].astype(str)
        if "requested_model" in frame
        else pd.Series("", index=frame.index)
    )
    required = model_values.isin(REQUIRED_EXACT_MODELS) | requested_values.isin(
        REQUIRED_EXACT_MODELS
    )
    return int((required & frame["status"].ne("success")).sum())


def _missing_required_score_count(scores: pd.DataFrame) -> int:
    if scores.empty:
        return len(REQUIRED_EXACT_MODELS)
    missing = 0
    for _, group in scores.groupby(["session", "event_index"], sort=False, dropna=False):
        model_values = set(group["model"].dropna().astype(str)) if "model" in group else set()
        requested_values = (
            set(group["requested_model"].dropna().astype(str))
            if "requested_model" in group
            else set()
        )
        observed = model_values | requested_values
        missing += len(REQUIRED_EXACT_MODELS - observed)
    return int(missing)


def _write_gate_outputs(
    output_dir: Path,
    scores: pd.DataFrame,
    event_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    status: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "exact_sparse_momentum_gate_event_scores.csv", index=False)
    event_summary.to_csv(
        output_dir / "exact_sparse_momentum_gate_event_summary.csv",
        index=False,
    )
    session_summary.to_csv(
        output_dir / "exact_sparse_momentum_gate_session_summary.csv",
        index=False,
    )
    (output_dir / "exact_sparse_momentum_gate_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "exact_sparse_momentum_gate.md").write_text(
        _dashboard_markdown(status, session_summary, event_summary),
        encoding="utf-8",
    )


def _dashboard_markdown(
    status: dict[str, object],
    session_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> str:
    verdict = "PASS" if status.get("gate_passed") else "FAIL"
    lines = [
        "# Exact-sparse momentum viability gate",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Gate metrics",
        "",
        f"- Momentum exact-surrogate recovery: "
        f"{status.get('momentum_exact_surrogate_recovered_events', 0)}/"
        f"{status.get('momentum_events', 0)} "
        f"({float(status.get('momentum_exact_surrogate_recovery_accuracy', 0.0)):.3f}); "
        f"threshold {float(status.get('momentum_recovery_threshold', 0.0)):.3f}",
        f"- Diffusion recovery: {status.get('diffusion_recovered_events', 0)}/"
        f"{status.get('diffusion_events', 0)} "
        f"({float(status.get('diffusion_recovery_accuracy', 0.0)):.3f}); "
        f"threshold {float(status.get('diffusion_recovery_threshold', 0.0)):.3f}",
        f"- First-order IMM best fraction: "
        f"{float(status.get('first_order_imm_best_fraction', 0.0)):.3f}; "
        f"maximum {float(status.get('max_first_order_imm_best_fraction', 0.0)):.3f}",
        f"- Required exact-model failures: "
        f"{status.get('required_exact_model_failures', 0)}",
        f"- Missing required exact-model scores: "
        f"{status.get('missing_required_exact_model_scores', 0)}",
        f"- Total model failures, including lower-bound audit rows: "
        f"{status.get('total_model_failures', 0)}",
        "",
        "## Session summary",
        "",
        _frame_for_markdown(session_summary),
        "",
        "## Event summary",
        "",
        _frame_for_markdown(event_summary),
        "",
    ]
    return "\n".join(lines)


def _frame_for_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return "```text\n" + frame.to_string(index=False) + "\n```"


if __name__ == "__main__":
    raise SystemExit(main())
