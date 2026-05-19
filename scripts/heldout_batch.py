#!/usr/bin/env python3
"""Session-scoped held-out likelihood benchmark for Pfeiffer/Foster replay data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _add_relative_metrics,
    _benchmark_config_metadata,
    _build_models,
    _score_train_joint_model,
    _session_mark_diagnostics,
    _split_cells,
)
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, build_emissions, fit_place_field_encoding

_REQUIRED_FILES = ("Position_Data.mat", "Ripple_Events.mat", "Spike_Data.mat", "Epochs.mat")


def parse_events(spec: str) -> list[int]:
    event_ids: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", maxsplit=1))
            if end < start:
                raise ValueError(f"Invalid descending event range: {token}")
            event_ids.extend(range(start, end + 1))
        else:
            event_ids.append(int(token))
    if not event_ids:
        raise ValueError("At least one event id is required.")
    return sorted(dict.fromkeys(event_ids))


def session_path(dataset_root: str | Path, session_id: str) -> Path:
    parts = session_id.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', e.g. 'Rat1/Open1'")
    return Path(dataset_root) / parts[0] / parts[1]


def validate_session(path: Path) -> None:
    missing = [name for name in _REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing required file(s): {', '.join(missing)}")


def score_event_model(
    *,
    session,
    event_id: int,
    model_name: str,
    model,
    train_encoding,
    joint_encoding,
    bin_centers: np.ndarray,
    emissions: EmissionConfig,
    metadata: dict[str, object],
) -> dict[str, object]:
    train_emissions = build_emissions(session, train_encoding, int(event_id), emissions)
    joint_emissions = build_emissions(session, joint_encoding, int(event_id), emissions)
    if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
        raise ValueError("Event produced zero emission bins.")

    train_score, joint_score = _score_train_joint_model(
        model,
        train_emissions,
        joint_emissions,
        bin_centers,
    )

    heldout = float(joint_score.log_likelihood - train_score.log_likelihood)
    test_spikes = int(joint_emissions.n_spikes - train_emissions.n_spikes)
    denom = max(float(test_spikes), 1.0)
    row = {
        "status": "success",
        "session": session.session_id,
        "event_index": int(event_id),
        "event_id": int(event_id),
        "model": joint_score.model_name,
        "requested_model": model_name,
        "heldout_log_likelihood": heldout,
        "heldout_log_likelihood_per_spike": heldout / denom,
        "heldout_bits_per_spike": heldout / np.log(2.0) / denom,
        "joint_log_likelihood": float(joint_score.log_likelihood),
        "train_log_likelihood": float(train_score.log_likelihood),
        "test_spikes": test_spikes,
        "train_spikes": int(train_emissions.n_spikes),
        "joint_spikes": int(joint_emissions.n_spikes),
        "n_time": int(train_emissions.n_time),
        **metadata,
        "error": "",
        **_session_mark_diagnostics(session),
    }
    row.update({f"diagnostic_{key}": value for key, value in joint_score.diagnostics.items()})
    return row


def failure_row(
    session: str,
    event_id: int,
    model_name: str,
    error: Exception,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "status": "failure",
        "session": session,
        "event_index": int(event_id),
        "event_id": int(event_id),
        "model": model_name,
        "heldout_log_likelihood": np.nan,
        "heldout_log_likelihood_per_spike": np.nan,
        "heldout_bits_per_spike": np.nan,
        "joint_log_likelihood": np.nan,
        "train_log_likelihood": np.nan,
        "test_spikes": 0,
        "requested_model": model_name,
        "train_spikes": 0,
        "joint_spikes": 0,
        "n_time": 0,
        "runtime_s": np.nan,
        "error": f"{type(error).__name__}: {error}",
    }
    if metadata:
        row.update(metadata)
    return row


def _format_cell_ids(cell_ids: np.ndarray) -> str:
    return ",".join(str(int(cell_id)) for cell_id in np.asarray(cell_ids, dtype=int))


def _heldout_batch_metadata(
    config: BenchmarkConfig,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
) -> dict[str, object]:
    """Return provenance metadata needed for faithful post-hoc decoding."""

    metadata = {
        "train_cell_ids": _format_cell_ids(train_cells),
        "test_cell_ids": _format_cell_ids(test_cells),
    }
    metadata.update(_benchmark_config_metadata(config))
    return metadata


def model_summary(success: pd.DataFrame) -> pd.DataFrame:
    if success.empty:
        return pd.DataFrame()
    return (
        success.groupby("model", as_index=False)
        .agg(
            events=("heldout_log_likelihood", "count"),
            mean_heldout_log_likelihood=("heldout_log_likelihood", "mean"),
            median_heldout_log_likelihood=("heldout_log_likelihood", "median"),
            mean_heldout_bits_per_spike=("heldout_bits_per_spike", "mean"),
            mean_delta_vs_best_static=("delta_vs_best_static", "mean"),
            median_delta_vs_best_static=("delta_vs_best_static", "median"),
            mean_bits_per_spike_vs_best_static=("bits_per_spike_vs_best_static", "mean"),
            mean_test_spikes=("test_spikes", "mean"),
            mean_n_time=("n_time", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
        )
        .sort_values("model")
    )


def paired_delta_summary(success: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    if success.empty or reference_model not in set(success["model"]):
        return pd.DataFrame()
    pivot = success.pivot_table(
        index=["session", "event_index"],
        columns="model",
        values="heldout_log_likelihood",
        aggfunc="first",
    )
    spikes = success.pivot_table(
        index=["session", "event_index"],
        columns="model",
        values="test_spikes",
        aggfunc="first",
    )
    rows = []
    for model in sorted(col for col in pivot.columns if col != reference_model):
        paired = pivot[[reference_model, model]].dropna()
        for key, values in paired.iterrows():
            delta = float(values[model] - values[reference_model])
            n_spikes = max(float(spikes.loc[key, model]), 1.0)
            rows.append(
                {
                    "session": key[0],
                    "event_index": int(key[1]),
                    "event_id": int(key[1]),
                    "model": model,
                    "reference_model": reference_model,
                    "delta_heldout_log_likelihood": delta,
                    "delta_bits_per_spike": delta / np.log(2.0) / n_spikes,
                }
            )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise
    return (
        pairwise.groupby(["reference_model", "model"], as_index=False)
        .agg(
            events=("delta_heldout_log_likelihood", "count"),
            mean_delta_heldout_log_likelihood=("delta_heldout_log_likelihood", "mean"),
            median_delta_heldout_log_likelihood=("delta_heldout_log_likelihood", "median"),
            mean_delta_bits_per_spike=("delta_bits_per_spike", "mean"),
            fraction_events_improved=("delta_heldout_log_likelihood", lambda x: float((x > 0).mean())),
        )
        .sort_values(["reference_model", "model"])
    )


def run(args: argparse.Namespace) -> None:
    events = parse_events(args.events)
    path = session_path(args.dataset_root, args.session)
    if not path.is_dir():
        raise FileNotFoundError(f"Requested session directory does not exist: {path}")
    validate_session(path)

    session = load_replay_session(path)
    encoding = fit_place_field_encoding(session)
    train_cells, test_cells = _split_cells(encoding.cell_ids, args.test_cell_fraction, args.random_seed)
    if train_cells.size == 0 or test_cells.size == 0:
        raise ValueError("Held-out split produced an empty train or test cell set.")

    train_encoding = encoding.select_cells(train_cells)
    joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))
    emission_config = EmissionConfig(time_bin_s=args.time_bin_s, spike_rate_scale=args.spike_rate_scale)
    config = BenchmarkConfig(
        candidate_top_k=args.candidate_top_k,
        pyrecest_particles=args.pyrecest_particles,
        random_seed=args.random_seed,
        test_cell_fraction=args.test_cell_fraction,
        emissions=emission_config,
        models=tuple(args.models),
    )
    models = _build_models(config, session=session)
    row_metadata = _heldout_batch_metadata(config, train_cells, test_cells)

    rows = []
    for event_id in events:
        if event_id < 0 or event_id >= session.ripple_count:
            raise IndexError(f"Event id {event_id} outside available range 0..{session.ripple_count - 1}")
        for model_name in args.models:
            start = time.perf_counter()
            try:
                row = score_event_model(
                    session=session,
                    event_id=event_id,
                    model_name=model_name,
                    model=models[model_name],
                    train_encoding=train_encoding,
                    joint_encoding=joint_encoding,
                    bin_centers=encoding.bin_centers,
                    emissions=emission_config,
                    metadata=row_metadata,
                )
                row["runtime_s"] = float(time.perf_counter() - start)
                rows.append(row)
                print(f"Scored {args.session} event {event_id} with {model_name}: {row['heldout_log_likelihood']:.3f}")
            except Exception as exc:
                rows.append(failure_row(args.session, event_id, model_name, exc, metadata=row_metadata))
                print(f"Failed {args.session} event {event_id} with {model_name}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    success = raw[raw["status"] == "success"].copy()
    if not success.empty:
        success = _add_relative_metrics(success)
    event_scores = pd.concat([success, raw[raw["status"] != "success"]], ignore_index=True, sort=False)

    event_scores.to_csv(output / "event_scores.csv", index=False)
    success.to_csv(output / "successful_event_scores.csv", index=False)
    model_summary(success).to_csv(output / "summary.csv", index=False)
    paired_delta_summary(success, args.reference_model).to_csv(output / "paired_delta_summary.csv", index=False)
    pd.DataFrame(
        [{"split": "train", "cell_id": int(cell)} for cell in train_cells]
        + [{"split": "test", "cell_id": int(cell)} for cell in test_cells]
    ).to_csv(output / "cell_split.csv", index=False)

    print(f"Wrote held-out outputs to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a session-scoped held-out likelihood benchmark.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="0-25")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random", "stationary", "diffusion", "momentum", "imm"],
        choices=("random", "stationary", "diffusion", "momentum", "imm", "pyrecest-goal-particle", "pyrecest-goal-particle-imm"),
    )
    parser.add_argument("--reference-model", default="diffusion")
    parser.add_argument("--candidate-top-k", default=64, type=int)
    parser.add_argument("--pyrecest-particles", default=512, type=int)
    parser.add_argument("--time-bin-s", default=0.02, type=float)
    parser.add_argument("--spike-rate-scale", default=1.0, type=float)
    parser.add_argument("--test-cell-fraction", default=0.25, type=float)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--output", default="results/heldout-batch")
    parser.add_argument("--continue-on-error", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
