#!/usr/bin/env python3
"""Session-scoped held-out replay benchmark for Pfeiffer/Foster data.

This script benchmarks explicit ripple event ids for one session. It avoids the
repository-wide loader so incomplete unrelated session folders do not break a
focused paper-analysis run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    _add_relative_metrics,
    _build_models,
    _score_train_joint_model,
    _heldout_split_seed,
    _session_mark_diagnostics,
    _split_cells,
    bootstrap_delta_ci,
)
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, build_emissions, fit_place_field_encoding


_REQUIRED_SESSION_FILES = (
    "Position_Data.mat",
    "Ripple_Events.mat",
    "Spike_Data.mat",
    "Epochs.mat",
)


def parse_event_ids(spec: str) -> list[int]:
    """Parse comma-separated event ids and inclusive ranges, e.g. 0-25,30."""
    event_ids: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid descending event range: {token}")
            event_ids.extend(range(start, end + 1))
        else:
            event_ids.append(int(token))
    if not event_ids:
        raise ValueError("At least one event id is required.")
    return sorted(dict.fromkeys(event_ids))


def _session_path(dataset_root: str | Path, session_id: str) -> Path:
    """Resolve a Rat/Open session ID to one session directory."""
    parts = session_id.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', e.g. 'Rat1/Open1'")
    return Path(dataset_root) / parts[0] / parts[1]


def _validate_session_files(session_path: Path) -> None:
    missing = [name for name in _REQUIRED_SESSION_FILES if not (session_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Requested session {session_path} is missing required file(s): {', '.join(missing)}"
        )


def _failure_row(
    *,
    session: str,
    event_id: int,
    model_name: str,
    runtime_s: float,
    error: Exception,
) -> dict[str, object]:
    return {
        "status": "failure",
        "session": session,
        "event_index": int(event_id),
        "model": model_name,
        "heldout_log_likelihood": np.nan,
        "joint_log_likelihood": np.nan,
        "train_log_likelihood": np.nan,
        "test_spikes": 0,
        "n_time": 0,
        "runtime_s": runtime_s,
        "error": f"{type(error).__name__}: {error}",
    }


def score_explicit_events(args: argparse.Namespace) -> BenchmarkResult:
    """Run held-out likelihood benchmark for explicit event ids in one session."""
    event_ids = parse_event_ids(args.events)
    session_path = _session_path(args.dataset_root, args.session)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Requested session directory does not exist: {session_path}")
    _validate_session_files(session_path)

    session = load_replay_session(session_path)
    invalid = [event_id for event_id in event_ids if event_id < 0 or event_id >= session.ripple_count]
    if invalid:
        raise IndexError(
            f"Event ids outside available range 0..{session.ripple_count - 1}: {invalid}"
        )

    heldout_cell_splits = int(getattr(args, "heldout_cell_splits", 1))
    if heldout_cell_splits < 1:
        raise ValueError("heldout_cell_splits must be >= 1")
    if heldout_cell_splits > 1:
        split_frames = []
        for split_index in range(heldout_cell_splits):
            split_args = argparse.Namespace(**vars(args))
            split_args.heldout_cell_splits = 1
            split_args.random_seed = _heldout_split_seed(args.random_seed, split_index)
            split_result = score_explicit_events(split_args)
            split_frame = split_result.rows.copy()
            if not split_frame.empty:
                split_frame["heldout_split_index"] = int(split_index)
                split_frame["heldout_split_seed"] = int(split_args.random_seed)
                split_frame["benchmark_heldout_cell_splits"] = int(heldout_cell_splits)
            split_frames.append(split_frame)
        frame = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
        if not frame.empty:
            frame = frame.sort_values(["heldout_split_index", "event_index", "model"]).reset_index(drop=True)
        return BenchmarkResult(frame)

    config = BenchmarkConfig(
        emissions=EmissionConfig(time_bin_s=args.time_bin_s, spike_rate_scale=args.spike_rate_scale),
        test_cell_fraction=args.test_cell_fraction,
        heldout_cell_splits=heldout_cell_splits,
        candidate_top_k=args.candidate_top_k,
        pyrecest_particles=args.pyrecest_particles,
        random_seed=args.random_seed,
        models=tuple(args.models),
    )

    encoding = fit_place_field_encoding(session, config.encoding)
    train_cells, test_cells = _split_cells(
        encoding.cell_ids,
        config.test_cell_fraction,
        config.random_seed,
    )
    if test_cells.size == 0 or train_cells.size == 0:
        raise ValueError(
            f"Cannot split cells into train/test. Found train={train_cells.size}, test={test_cells.size}."
        )

    train_encoding = encoding.select_cells(train_cells)
    joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))
    model_objects = _build_models(config, session=session)

    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        train_emissions = build_emissions(session, train_encoding, int(event_id), config.emissions)
        joint_emissions = build_emissions(session, joint_encoding, int(event_id), config.emissions)
        if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
            continue

        for model_name, model in model_objects.items():
            start_time = time.perf_counter()
            try:
                train_score, joint_score = _score_train_joint_model(
                    model,
                    train_emissions,
                    joint_emissions,
                    encoding.bin_centers,
                )
                runtime_s = time.perf_counter() - start_time
                heldout = joint_score.log_likelihood - train_score.log_likelihood
                test_spikes = int(joint_emissions.n_spikes - train_emissions.n_spikes)
                bits_per_spike = float(heldout / np.log(2.0) / max(test_spikes, 1))

                rows.append(
                    {
                        "status": "success",
                        "session": session.session_id,
                        "event_index": int(event_id),
                        "model": joint_score.model_name,
                        "requested_model": model_name,
                        "heldout_log_likelihood": float(heldout),
                        "heldout_bits_per_spike": bits_per_spike,
                        "joint_log_likelihood": float(joint_score.log_likelihood),
                        "train_log_likelihood": float(train_score.log_likelihood),
                        "test_spikes": test_spikes,
                        "n_time": int(train_emissions.n_time),
                        "runtime_s": runtime_s,
                        "error": "",
                        **_session_mark_diagnostics(session),
                        **{
                            f"diagnostic_{key}": value
                            for key, value in joint_score.diagnostics.items()
                        },
                    }
                )
                print(f"Benchmarked {session.session_id} event {event_id} with {model_name}")
            except Exception as exc:
                runtime_s = time.perf_counter() - start_time
                rows.append(
                    _failure_row(
                        session=session.session_id,
                        event_id=event_id,
                        model_name=model_name,
                        runtime_s=runtime_s,
                        error=exc,
                    )
                )
                print(
                    f"Failed {session.session_id} event {event_id} with {model_name}: {exc}",
                    flush=True,
                )
                if not args.continue_on_error:
                    raise

    frame = pd.DataFrame(rows)
    if not frame.empty:
        success_mask = frame["status"] == "success"
        success = frame.loc[success_mask].copy()
        failures = frame.loc[~success_mask].copy()
        if not success.empty:
            success = _add_relative_metrics(success)
        frame = pd.concat([success, failures], ignore_index=True)
        frame = frame.sort_values(["event_index", "model"]).reset_index(drop=True)

    return BenchmarkResult(frame)


def write_outputs(result: BenchmarkResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_scores_path = output_dir / "event_scores.csv"
    summary_path = output_dir / "summary.csv"
    result.rows.to_csv(event_scores_path, index=False)
    result.summary().to_csv(summary_path, index=False)

    if not result.rows.empty:
        success = result.rows[result.rows["status"] == "success"]
        pivot_index = ["session", "event_index"]
        for column in ("heldout_split_index", "heldout_split_seed"):
            if column in success.columns:
                pivot_index.append(column)
        for metric in (
            "heldout_log_likelihood",
            "heldout_bits_per_spike",
            "delta_vs_best_static",
            "bits_per_spike_vs_best_static",
        ):
            if metric in success.columns:
                pivot = success.pivot_table(
                    index=pivot_index,
                    columns="model",
                    values=metric,
                    aggfunc="first",
                ).reset_index()
                pivot.to_csv(output_dir / f"event_model_pivot_{metric}.csv", index=False)

    print(f"Wrote event scores: {event_scores_path}")
    print(f"Wrote summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a session-scoped held-out replay benchmark.")
    parser.add_argument("--dataset-root", required=True, help="Path to DataSetFromPfeifferFoster.")
    parser.add_argument("--session", required=True, help="Session ID, e.g. Rat1/Open1.")
    parser.add_argument("--events", default="0-25", help="Event ids/ranges, e.g. 0-25,30.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random", "stationary", "diffusion", "momentum", "imm"],
        choices=(
            "random",
            "stationary",
            "diffusion",
            "momentum",
            "imm",
            "pyrecest-goal-particle",
            "pyrecest-goal-particle-imm",
        ),
    )
    parser.add_argument("--candidate-top-k", default=64, type=int)
    parser.add_argument("--pyrecest-particles", default=512, type=int)
    parser.add_argument("--test-cell-fraction", default=0.25, type=float)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--heldout-cell-splits", default=1, type=int)
    parser.add_argument("--time-bin-s", default=0.02, type=float)
    parser.add_argument("--spike-rate-scale", default=1.0, type=float)
    parser.add_argument("--output", default="results/heldout-benchmark")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and keep processing remaining event/model pairs.",
    )
    args = parser.parse_args()

    result = score_explicit_events(args)
    print(result.summary().to_string(index=False))
    successes = result.rows[result.rows.get("status", "") == "success"] if not result.rows.empty else result.rows
    if not successes.empty and "imm" in set(successes["model"]):
        ci_low, ci_high = bootstrap_delta_ci(successes, model="imm")
        print(f"IMM mean delta bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    write_outputs(result, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
